#!/usr/bin/env python3
"""
Online KL-anchored GRPO for Mica Looped Transformer using Mica-as-Critic.

100% homegrown RL loop:
  - Policy : LoopedTransformer initialized from Noir SFT (3.69M)
  - Ref    : Frozen copy of the Noir SFT checkpoint (KL anchor)
  - Critic : Frozen MicaCritic trained from scratch on noir pairs + human clicks (3.69M)

Usage:
  uv run python src/rl_grpo.py \
    --policy-ckpt models/experiments/looped_noir/best.pt \
    --critic-ckpt models/experiments/looped_critic/best.pt \
    --out-dir models/experiments/looped_rl
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import random
import re
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from model_critic import MicaCritic
from model_looped import LoopedTransformer


def parse_args():
    p = argparse.ArgumentParser(description="KL-anchored GRPO with Mica-as-Critic")
    p.add_argument("--policy-ckpt", default="models/experiments/looped_noir/best.pt")
    p.add_argument("--critic-ckpt", default="models/experiments/looped_critic/best.pt")
    p.add_argument("--corpus", default="corpus/corpus.txt")
    p.add_argument("--tokenizer", default="mica_tokenizer.json")
    p.add_argument("--out-dir", default="models/experiments/looped_rl")
    p.add_argument("--log-csv", default="logs/looped/rl_grpo.csv")

    # GRPO hyperparameters
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4, help="Rollouts per prompt (G)")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--kl-beta", type=float, default=0.04, help="KL penalty weight")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--sample-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_prompts(corpus_path: str, max_prompts: int = 8000) -> list[str]:
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    prompts = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if not para:
            continue
        for sent in re.split(r"(?<=[.?!])\s+", para):
            words = sent.split()
            if len(words) >= 12:
                k = random.randint(4, 7)
                prompts.append(" ".join(words[:k]))
                if len(prompts) >= max_prompts:
                    return prompts
    return prompts


def guardrail_reward(text: str, n_new_tokens: int, natural_stop: bool) -> float:
    """Cheap deterministic anti-hacking guardrails."""
    bonus = 0.0
    words = [w.lower().strip(".,!?;:\"'") for w in text.split() if w.strip()]

    # Penalize immediate EOS / ultra-short rollouts
    if n_new_tokens < 6:
        bonus -= 1.0

    # Penalize repeated bigrams/trigrams ("a chair in a chair")
    if len(words) >= 6:
        bigrams = list(zip(words, words[1:]))
        uniq_ratio = len(set(bigrams)) / max(len(bigrams), 1)
        if uniq_ratio < 0.85:
            bonus -= (0.85 - uniq_ratio) * 4.0

    # Small reward for clean sentence ending
    if natural_stop and text.rstrip().endswith((".", "!", "?", '"')):
        bonus += 0.25

    return bonus


@torch.no_grad()
def sample_rollouts(
    model: LoopedTransformer,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
):
    """Generate G completions in parallel for a single prompt."""
    model.eval()
    eos_id = tokenizer.token_to_id("<|eos|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    idx = torch.tensor([prompt_ids] * group_size, dtype=torch.long, device=device)
    finished = [False] * group_size
    gen_lists: list[list[int]] = [[] for _ in range(group_size)]
    natural_stops = [False] * group_size

    for step in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size :])
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)  # (G, 1)

        all_done = True
        for g in range(group_size):
            if finished[g]:
                continue
            tok = nxt[g, 0].item()
            if tok in (eos_id, eot_id):
                finished[g] = True
                natural_stops[g] = True
            else:
                gen_lists[g].append(tok)
                all_done = False
        idx = torch.cat([idx, nxt], dim=1)
        if all_done:
            break

    return gen_lists, natural_stops


def completion_logprobs(
    model: LoopedTransformer,
    full_ids: list[int],
    prompt_len: int,
    device: str,
) -> torch.Tensor:
    """Per-token log P(y_t | x, y_<t) for completion tokens only -> (L_comp,)."""
    seq = full_ids[-model.config.block_size :]
    shift = len(full_ids) - len(seq)
    eff_prompt_len = max(1, prompt_len - shift)

    inp = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([seq[1:]], dtype=torch.long, device=device)

    # Full-sequence logits via targets path (we only need logits)
    b = model
    x = b.drop(b.wte(inp))
    x = b.prelude(x, start_pos=0, loop_idx=0)
    x_pre = x
    for k in range(b.config.n_loops):
        x = x + b.inj_gate[k] * x_pre + b.loop_embed.weight[k]
        if b.folded:
            x = b.core[0](x, start_pos=0, loop_idx=k)
        else:
            x = b.core[k](x, start_pos=0, loop_idx=0)
    x = b.coda(x, start_pos=0, loop_idx=0)
    logits = b.lm_head(b.ln_f(x))  # (1, T-1, V)

    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T-1,)
    return tok_lp[eff_prompt_len - 1 :]


def save_policy(path: Path, model: LoopedTransformer, step: int, mean_reward: float):
    torch.save(
        {
            "config": dataclasses.asdict(model.config),
            "folded": model.folded,
            "layout": model.layout,
            "step": step,
            "mean_reward": mean_reward,
            "model": model.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompts = load_prompts(args.corpus)
    print(f"Loaded {len(prompts):,} noir prompts from {args.corpus}")

    policy = LoopedTransformer.from_checkpoint(args.policy_ckpt, device=device)
    # Disable dropout on policy during RL so log_pi matches ref (KL=0 at step 0)
    policy.config.dropout = 0.0
    for m in policy.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if hasattr(m, "attn_dropout"):
            m.attn_dropout = 0.0
    ref = LoopedTransformer.from_checkpoint(args.policy_ckpt, device=device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    critic = MicaCritic.from_checkpoint(args.critic_ckpt, device=device)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    csv_f = open(log_csv, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step", "mean_reward", "critic_reward", "kl", "loss"])

    eval_prompts = [
        "The fat man leaned back",
        "The rain came down hard on",
        "She walked into the room",
        "I lit a cigarette",
    ]

    best_reward = -float("inf")
    running_r, running_cr, running_kl, running_loss = 0.0, 0.0, 0.0, 0.0

    for step in range(1, args.steps + 1):
        step_prompts = [random.choice(prompts) for _ in range(args.prompts_per_step)]

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        n_valid = 0
        step_rewards = []
        step_critic_r = []
        step_kls = []

        for prompt in step_prompts:
            p_ids = tokenizer.encode(prompt).ids
            if not p_ids:
                continue
            gen_lists, nat_stops = sample_rollouts(
                policy, tokenizer, p_ids, args.group_size,
                args.max_new_tokens, args.temperature, args.top_k, device,
            )

            # Score all G rollouts with MicaCritic + guardrails
            full_seqs = [p_ids + g for g in gen_lists]
            max_l = max(len(s) for s in full_seqs)
            c_ids = torch.zeros((args.group_size, max_l), dtype=torch.long, device=device)
            c_mask = torch.zeros((args.group_size, max_l), dtype=torch.float32, device=device)
            for g, s in enumerate(full_seqs):
                c_ids[g, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                c_mask[g, : len(s)] = 1.0

            with torch.no_grad():
                c_scores = critic(c_ids, c_mask).cpu().tolist()

            rewards = []
            for g in range(args.group_size):
                txt = tokenizer.decode(full_seqs[g])
                gr = guardrail_reward(txt, len(gen_lists[g]), nat_stops[g])
                rewards.append(c_scores[g] + gr)

            r_t = torch.tensor(rewards, dtype=torch.float32)
            r_std = r_t.std().item()
            if r_std < 1e-6:
                advs = [0.0] * args.group_size
            else:
                r_mean = r_t.mean().item()
                advs = [(r - r_mean) / (r_std + 1e-6) for r in rewards]

            step_rewards.extend(rewards)
            step_critic_r.extend(c_scores)

            # Policy gradient + KL anchor on completion tokens
            policy.train()
            for g in range(args.group_size):
                if len(gen_lists[g]) == 0:
                    continue
                log_pi = completion_logprobs(policy, full_seqs[g], len(p_ids), device)
                with torch.no_grad():
                    log_ref = completion_logprobs(ref, full_seqs[g], len(p_ids), device)

                # Non-negative Schulman KL estimator
                diff = log_ref - log_pi
                kl = (torch.exp(diff) - diff - 1.0).mean()
                pg = -(advs[g] * log_pi).mean()

                total_loss = total_loss + pg + args.kl_beta * kl
                step_kls.append(kl.item())
                n_valid += 1

        if n_valid > 0:
            loss = total_loss / n_valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            loss_val = loss.item()
        else:
            loss_val = 0.0

        mr = sum(step_rewards) / max(len(step_rewards), 1)
        mcr = sum(step_critic_r) / max(len(step_critic_r), 1)
        mkl = sum(step_kls) / max(len(step_kls), 1)

        running_r += mr
        running_cr += mcr
        running_kl += mkl
        running_loss += loss_val

        if step % args.log_every == 0 or step == 1:
            denom = 1 if step == 1 else args.log_every
            avg_r = running_r / denom
            avg_cr = running_cr / denom
            avg_kl = running_kl / denom
            avg_l = running_loss / denom
            print(
                f"step {step:>4}/{args.steps} | reward={avg_r:+.3f} (critic={avg_cr:+.3f}) | "
                f"kl={avg_kl:.4f} | loss={avg_l:+.4f}"
            )
            writer.writerow([step, f"{avg_r:.4f}", f"{avg_cr:.4f}", f"{avg_kl:.5f}", f"{avg_l:.4f}"])
            csv_f.flush()
            if step > 20 and avg_r > best_reward:
                best_reward = avg_r
                save_policy(out_dir / "best.pt", policy, step, best_reward)
            running_r = running_cr = running_kl = running_loss = 0.0

        if step % args.sample_every == 0:
            print("\n── LIVE SAMPLES (RL vs Ref) ──")
            for ep in eval_prompts:
                e_ids = tokenizer.encode(ep).ids
                rl_gen, _ = sample_rollouts(policy, tokenizer, e_ids, 1, 40, 0.7, 40, device)
                rf_gen, _ = sample_rollouts(ref, tokenizer, e_ids, 1, 40, 0.7, 40, device)
                print(f"  [RL ] {tokenizer.decode(e_ids + rl_gen[0]).replace(chr(10), ' ')}")
                print(f"  [Ref] {tokenizer.decode(e_ids + rf_gen[0]).replace(chr(10), ' ')}")
            print()

    save_policy(out_dir / "final.pt", policy, args.steps, best_reward)
    csv_f.close()
    print(f"\nDone! Saved {out_dir / 'final.pt'} and {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
