#!/usr/bin/env python3
"""
Mica Live Interleaved Pretraining — Co-Evolving Critic + RL DURING Model Creation.

100% from scratch — zero external models, zero old checkpoints, zero old rankings.
Direct apples-to-apples test against passive 35k-step Gutenberg pretraining:

  Stage 1 — Gutenberg Warmup (5,000 steps):
    3,500 unfolded (1 + 4 + 1) -> 1,500 soft_fold -> hard_fold into 1 + *4 + 1 (3.69M)
  Stage 2 — Critic Birth (at step 5,000):
    Clone freshly folded backbone into MicaCritic + 120-step bootstrap on Gutenberg
  Stage 3 — Interleaved Gutenberg Pretraining (30,000 CE steps = 600 cycles x 50 CE):
    Each cycle:
      - Listen    : 50 Gutenberg CE steps (wiki_train.bin)
      - Speak     : 4 prompts x 4 rollouts = 16 fresh drafts on real Gutenberg prompts
      - Critic    : 4 Bradley-Terry steps (60% live self-play buffer + 40% corrupted Gutenberg)
      - Reinforce : 2 GRPO steps anchored to start-of-cycle policy (trust-region KL)

Total: 35,000 Gutenberg CE steps + 2,400 Critic steps + 1,200 RL steps (~12h on M1).

Usage:
  uv run python src/train_interleaved.py
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import dataclasses
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from model_critic import MicaCritic
from model_looped import (
    LoopedConfig,
    LoopedTransformer,
    hard_fold_core,
    soft_tie_loss,
)


def parse_args():
    p = argparse.ArgumentParser(description="Live Interleaved Gutenberg Pretraining + Co-Evolving Critic + RL")
    # Data
    p.add_argument("--train-bin", default="data/wiki_train.bin")
    p.add_argument("--val-bin", default="data/wiki_val.bin")
    p.add_argument("--tokenizer", default="mica_tokenizer.json")
    p.add_argument("--out-dir", default="models/experiments/interleaved_gutenberg")
    p.add_argument("--log-csv", default="logs/looped/interleaved_gutenberg.csv")

    # Architecture (3.69M looped — exact match to passive pretrain baseline)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--ffn-hidden", type=int, default=704)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--attn-type", choices=["multires", "mha"], default="multires")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-kickstart", action="store_true", help="Start folded 1+*4+1 from step 0")

    # Stage 1: Gutenberg warmup (5,000 steps)
    p.add_argument("--unfolded-steps", type=int, default=3500)
    p.add_argument("--soft-fold-steps", type=int, default=1500)
    p.add_argument("--tie-weight", type=float, default=500.0)
    p.add_argument("--warmup-lr", type=float, default=1.2e-3)
    p.add_argument("--batch-size", type=int, default=32)

    # Stage 2: Critic bootstrap
    p.add_argument("--critic-warmup-steps", type=int, default=120)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--critic-ema", type=float, default=0.95, help="EMA blend of policy weights into critic backbone each cycle (1.0 = off)")
    p.add_argument("--critic-batch-size", type=int, default=24)
    p.add_argument("--critic-max-len", type=int, default=96)
    p.add_argument("--buffer-size", type=int, default=512, help="Rolling live self-play pair buffer")

    # Stage 3: Interleaved Pretraining (600 cycles x 50 CE = 30,000 CE steps -> 35k total)
    p.add_argument("--cycles", type=int, default=600)
    p.add_argument("--ce-per-cycle", type=int, default=50)
    p.add_argument("--critic-per-cycle", type=int, default=4)
    p.add_argument("--rl-per-cycle", type=int, default=4)
    p.add_argument("--prompts-per-cycle", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=44)
    p.add_argument("--rl-temp", type=float, default=0.8)
    p.add_argument("--ce-lr", type=float, default=1.0e-3)
    p.add_argument("--ce-min-lr", type=float, default=1e-5)
    p.add_argument("--rl-lr", type=float, default=None, help="Fixed RL LR (default: tracks --rl-lr-ratio * ce_lr)")
    p.add_argument("--rl-lr-ratio", type=float, default=0.25, help="RL LR as fraction of current CE LR")
    p.add_argument("--kl-beta", type=float, default=0.15)

    # Resume into Stage 3 without losing warmup/progress
    p.add_argument("--resume-policy", default=None, help="Resume Stage 3 policy from best.pt/final.pt")
    p.add_argument("--resume-critic", default=None, help="Resume Stage 3 critic from critic_best.pt")
    p.add_argument("--reset-schedule", action="store_true", help="Reset step/cycle/best_val to 0 when starting a new domain (e.g. Noir)")
    p.add_argument("--contrast-bin", default=None, help="Optional out-of-domain .bin (e.g. data/wiki_val.bin) as critic voice negatives")

    # Eval / logging
    p.add_argument("--eval-every", type=int, default=500, help="Eval cadence during warmup")
    p.add_argument("--eval-every-cycles", type=int, default=10, help="Eval cadence during Stage 3 (10x50 = 500 steps)")
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_dropout(model: torch.nn.Module, p: float):
    if hasattr(model, "config"):
        model.config.dropout = p
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
        if hasattr(m, "attn_dropout"):
            m.attn_dropout = p


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_optimizer(model, lr: float, weight_decay: float = 0.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not name.endswith("wte.weight") and not name.endswith("loop_embed.weight"):
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def cosine_lr(step: int, total_steps: int, warmup: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= total_steps:
        return min_lr
    prog = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def sample_real_passage(
    data: np.memmap,
    tokenizer: Tokenizer,
    eos_id: int | None,
    eot_id: int | None,
    rng: random.Random,
    min_words: int = 12,
    max_words: int = 50,
) -> str:
    """Sample a clean sentence-aligned passage directly from a token .bin file."""
    specials = {t for t in (eos_id, eot_id) if t is not None}
    max_start = len(data) - 160
    for _ in range(10):
        start = rng.randint(0, max_start)
        chunk = data[start : start + 140].tolist()
        # Start right after a sentence/paragraph boundary if one exists early
        offset = 0
        for i, tok in enumerate(chunk[:60]):
            if tok in specials:
                offset = i + 1
                break
        # Collect tokens across 1-2 sentences up to ~64 tokens
        picked = []
        for tok in chunk[offset : offset + 72]:
            if tok in specials:
                if len(picked) >= 16:
                    break
                continue
            picked.append(tok)
        txt = " ".join(tokenizer.decode(picked).split())
        words = txt.split()
        if min_words <= len(words) <= max_words:
            return txt
    # Fallback
    toks = [int(t) for t in data[start : start + 56] if int(t) not in specials]
    return " ".join(tokenizer.decode(toks).split())


def corrupt_passage(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 6:
        return text
    mode = rng.choice(["repeat", "shuffle", "truncate", "drop"])
    if mode == "repeat":
        span_len = rng.randint(2, min(5, len(words) // 2))
        pos = rng.randint(2, len(words) - span_len)
        span = words[pos : pos + span_len]
        reps = rng.randint(2, 3)
        return " ".join((words[:pos] + span * reps + words[pos + span_len :])[: len(words) + 8])
    elif mode == "shuffle":
        win = rng.randint(5, min(10, len(words)))
        pos = rng.randint(0, len(words) - win)
        chunk = words[pos : pos + win]
        rng.shuffle(chunk)
        return " ".join(words[:pos] + chunk + words[pos + win :])
    elif mode == "truncate":
        cut = rng.randint(3, max(4, len(words) // 2))
        return " ".join(words[:cut]).rstrip(".!?\"'")
    else:
        kept = [w for w in words if rng.random() > 0.35]
        return " ".join(kept) if len(kept) >= 3 else " ".join(words[:3])


def encode_critic_batch(pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_len: int, device: str):
    c_list = [tokenizer.encode(c).ids[:max_len] or [0] for c, _ in pairs]
    r_list = [tokenizer.encode(r).ids[:max_len] or [0] for _, r in pairs]

    def pad(seqs):
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.float32)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1.0
        return ids.to(device), mask.to(device)

    c_ids, c_mask = pad(c_list)
    r_ids, r_mask = pad(r_list)
    return c_ids, c_mask, r_ids, r_mask


def guardrail_reward(text: str, n_new_tokens: int, natural_stop: bool) -> float:
    bonus = 0.0
    words = [w.lower().strip(".,!?;:\"'") for w in text.split() if w.strip()]
    if n_new_tokens < 6:
        bonus -= 1.0
    if len(words) >= 6:
        bigrams = list(zip(words, words[1:]))
        uniq_ratio = len(set(bigrams)) / max(len(bigrams), 1)
        if uniq_ratio < 0.85:
            bonus -= (0.85 - uniq_ratio) * 4.0
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
    device: str,
    top_k: int = 40,
):
    model.eval()
    eos_id = tokenizer.token_to_id("<|eos|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    idx = torch.tensor([prompt_ids] * group_size, dtype=torch.long, device=device)
    finished = [False] * group_size
    gen_lists: list[list[int]] = [[] for _ in range(group_size)]
    natural_stops = [False] * group_size

    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size :])
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

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
    seq = full_ids[-model.config.block_size :]
    shift = len(full_ids) - len(seq)
    eff_prompt_len = max(1, prompt_len - shift)

    inp = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([seq[1:]], dtype=torch.long, device=device)

    x = model.drop(model.wte(inp))
    x = model.prelude(x, start_pos=0, loop_idx=0)
    x_pre = x
    for k in range(model.config.n_loops):
        x = x + model.inj_gate[k] * x_pre + model.loop_embed.weight[k]
        if model.folded:
            x = model.core[0](x, start_pos=0, loop_idx=k)
        else:
            x = model.core[k](x, start_pos=0, loop_idx=0)
    x = model.coda(x, start_pos=0, loop_idx=0)
    logits = model.lm_head(model.ln_f(x))

    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)
    return tok_lp[eff_prompt_len - 1 :]


def save_policy(path: Path, model: LoopedTransformer, step: int, val_loss: float):
    torch.save(
        {
            "config": dataclasses.asdict(model.config),
            "folded": model.folded,
            "layout": model.layout,
            "step": step,
            "val_loss": val_loss,
            "model": model.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    contrast_data = np.memmap(args.contrast_bin, dtype=np.uint16, mode="r") if args.contrast_bin else None
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|eos|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    warmup_total = args.unfolded_steps + args.soft_fold_steps
    interleave_ce_total = args.cycles * args.ce_per_cycle

    resuming = args.resume_policy is not None
    if resuming:
        ckpt = torch.load(args.resume_policy, map_location=device, weights_only=False)
        cfg = LoopedConfig(**ckpt["config"])
        cfg.device = device
        cfg.dropout = args.dropout
        model = LoopedTransformer(cfg, folded=ckpt.get("folded", True)).to(device)
        model.load_state_dict(ckpt["model"])
        if args.reset_schedule:
            warmup_total = 0
            global_step = 0
            best_val = float("inf")
            start_cycle = 1
            print(
                f"Starting fresh Stage 3 domain from {args.resume_policy} "
                f"(cycles 1..{args.cycles}, layout={model.layout})"
            )
        else:
            global_step = int(ckpt.get("step", warmup_total))
            best_val = float(ckpt.get("val_loss", float("inf")))
            start_cycle = max(1, (global_step - warmup_total) // max(args.ce_per_cycle, 1) + 1)
            print(
                f"Resuming Stage 3 from {args.resume_policy} at step {global_step} "
                f"(cycle {start_cycle}/{args.cycles}, best_val={best_val:.4f}, layout={model.layout})"
            )
    else:
        cfg = LoopedConfig(
            vocab_size=tokenizer.get_vocab_size(),
            block_size=args.block_size,
            n_embd=args.n_embd,
            ffn_hidden=args.ffn_hidden,
            n_loops=args.n_loops,
            attn_type=args.attn_type,
            dropout=args.dropout,
            device=device,
        )
        model = LoopedTransformer(cfg, folded=args.no_kickstart).to(device)
        optimizer = build_optimizer(model, args.warmup_lr, weight_decay=0.1)
        global_step = warmup_total
        best_val = float("inf")
        start_cycle = 1

        print(
            f"From-Zero Interleaved Pretraining | Device={device} | Start={model.layout} "
            f"({model.num_params()/1e6:.2f}M)"
        )
        print(f"Data: train={len(train_data):,} tok ({args.train_bin}), val={len(val_data):,} tok ({args.val_bin})")
        print(f"Stage 1 (Warmup):      {args.unfolded_steps} unfolded + {args.soft_fold_steps} soft_fold = {warmup_total} steps")

    print(
        f"Stage 3 (Interleaved): cycles {start_cycle}..{args.cycles} x "
        f"({args.ce_per_cycle} CE + {args.critic_per_cycle} critic + {args.rl_per_cycle} RL, "
        f"rl_lr={'fixed ' + str(args.rl_lr) if args.rl_lr else f'{args.rl_lr_ratio}x ce_lr'})"
    )

    append_csv = resuming and (not args.reset_schedule) and log_csv.exists()
    csv_f = open(log_csv, "a" if append_csv else "w", newline="")
    writer = csv.writer(csv_f)
    if not append_csv:
        writer.writerow(["step", "stage", "layout", "lr", "train_ce", "tie_loss", "val_ce", "val_ppl", "critic_acc", "reward", "kl"])

    eval_prompts = [
        "The rain came down hard on",
        "He looked across the room and",
        "She opened the door slowly",
    ]

    # ── STAGE 1: Grammar Warmup on Gutenberg ──────────────────────────────────
    if not resuming:
        t_log = time.perf_counter()
        steps_since_log = 0
        running_ce = 0.0

        for step in range(warmup_total):
            if args.no_kickstart or step < args.unfolded_steps:
                stage = "warmup" if args.no_kickstart else "unfolded"
                tie_w = 0.0
            else:
                stage = "soft_fold"
                frac = (step - args.unfolded_steps + 1) / max(args.soft_fold_steps, 1)
                tie_w = args.tie_weight * frac

            lr = cosine_lr(step, warmup_total, min(400, warmup_total // 4), args.warmup_lr, args.warmup_lr * 0.25)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            xb, yb = get_batch(train_data, args.batch_size, cfg.block_size, device)
            optimizer.zero_grad(set_to_none=True)
            _, ce_loss = model(xb, yb)
            loss = ce_loss + tie_w * soft_tie_loss(model) if (tie_w > 0 and not model.folded) else ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_ce += ce_loss.item()
            steps_since_log += 1

            if (step + 1) % args.log_every == 0 or step == 0:
                if device == "mps":
                    torch.mps.synchronize()
                dt = time.perf_counter() - t_log
                tok_s = (steps_since_log * args.batch_size * cfg.block_size) / max(dt, 1e-6)
                print(f"step {step+1:>5}/{warmup_total} [{stage:<9}] lr={lr:.2e} ce={running_ce/steps_since_log:.4f} {tok_s:,.0f} tok/s")
                running_ce = 0.0
                steps_since_log = 0
                t_log = time.perf_counter()

            if (step + 1) % args.eval_every == 0 or (step + 1) == warmup_total:
                losses = estimate_loss(model, train_data, val_data, args.batch_size, cfg.block_size, device, args.eval_iters)
                ppl = math.exp(min(losses["val"], 20.0))
                print(f"── WARMUP EVAL step {step+1} ({model.layout}): train={losses['train']:.4f} val={losses['val']:.4f} (ppl {ppl:.2f})")
                writer.writerow([step + 1, stage, model.layout, f"{lr:.6g}", f"{losses['train']:.4f}", f"{tie_w:.1f}", f"{losses['val']:.4f}", f"{ppl:.2f}", "", "", ""])
                csv_f.flush()
                t_log = time.perf_counter()

        # Hard fold into 1 + *K + 1 right at the end of warmup
        post_val = 0.0
        if not model.folded:
            pre = estimate_loss(model, train_data, val_data, args.batch_size, cfg.block_size, device, eval_iters=20)
            hard_fold_core(model, optimizer=optimizer)
            post = estimate_loss(model, train_data, val_data, args.batch_size, cfg.block_size, device, eval_iters=20)
            post_val = post["val"]
            print(
                f"\n>>> HARD FOLD at step {warmup_total}: {model.layout} ({model.num_params()/1e6:.2f}M) | "
                f"val {pre['val']:.4f} -> {post['val']:.4f} (delta={post['val']-pre['val']:+.4f})\n"
            )
        save_policy(out_dir / "after_warmup.pt", model, warmup_total, post_val)

    # ── STAGE 2: Critic Birth or Resume ───────────────────────────────────────
    if args.resume_critic:
        critic = MicaCritic.from_checkpoint(args.resume_critic, device=device)
        set_dropout(critic, args.dropout)
        print(f">>> Resumed MicaCritic from {args.resume_critic}")
    else:
        print(f">>> Initializing fresh MicaCritic from step-{global_step} looped weights...")
        critic = MicaCritic(copy.deepcopy(model.config), folded=True).to(device)
        critic.backbone.load_state_dict(model.state_dict())
        set_dropout(critic, args.dropout)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=0.05)

    selfplay_buf: collections.deque[tuple[str, str]] = collections.deque(maxlen=args.buffer_size)

    def make_critic_batch(batch_sz: int) -> list[tuple[str, str]]:
        pairs = []
        for _ in range(batch_sz):
            r = rng.random()
            if len(selfplay_buf) >= 8 and r < 0.55:
                pairs.append(rng.choice(selfplay_buf))
            elif contrast_data is not None and r < 0.75:
                p = sample_real_passage(train_data, tokenizer, eos_id, eot_id, rng)
                g = sample_real_passage(contrast_data, tokenizer, eos_id, eot_id, rng)
                pairs.append((p, g))
            else:
                p = sample_real_passage(train_data, tokenizer, eos_id, eot_id, rng)
                pairs.append((p, corrupt_passage(p, rng)))
        return pairs

    # Seed self-play buffer with initial rollouts
    for _ in range(12):
        real_p = sample_real_passage(train_data, tokenizer, eos_id, eot_id, rng)
        words = real_p.split()
        prompt = " ".join(words[: rng.randint(4, min(7, max(4, len(words) // 2)))])
        p_ids = tokenizer.encode(prompt).ids
        if not p_ids:
            continue
        gens, _ = sample_rollouts(model, tokenizer, p_ids, 2, args.max_new_tokens, args.rl_temp, device)
        for g in gens:
            fake = " ".join(tokenizer.decode(p_ids + g).split())
            if fake != real_p and len(fake.split()) >= 4:
                selfplay_buf.append((real_p, fake))

    if not args.resume_critic:
        critic.train()
        for _ in range(args.critic_warmup_steps):
            batch = make_critic_batch(args.critic_batch_size)
            c_ids, c_mask, r_ids, r_mask = encode_critic_batch(batch, tokenizer, args.critic_max_len, device)
            critic_opt.zero_grad(set_to_none=True)
            rc = critic(c_ids, c_mask)
            rr = critic(r_ids, r_mask)
            loss = (-F.logsigmoid(rc - rr)).mean() + 1e-3 * (rc.pow(2).mean() + rr.pow(2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

    @torch.no_grad()
    def eval_critic_quick() -> float:
        critic.eval()
        val_b = []
        for _ in range(96):
            p = sample_real_passage(val_data, tokenizer, eos_id, eot_id, rng)
            val_b.append((p, corrupt_passage(p, rng)))
        c_ids, c_mask, r_ids, r_mask = encode_critic_batch(val_b, tokenizer, args.critic_max_len, device)
        acc = (critic(c_ids, c_mask) > critic(r_ids, r_mask)).float().mean().item()
        critic.train()
        return acc

    print(f">>> Critic ready | quick val acc = {eval_critic_quick()*100:.1f}%\n")

    # ── STAGE 3: Live Interleaved Pretraining (CE + Co-Evolving Critic + RL) ──
    set_dropout(model, args.dropout)
    ce_opt = build_optimizer(model, args.ce_lr, weight_decay=0.1)
    init_rl_lr = args.rl_lr if args.rl_lr is not None else args.ce_lr * args.rl_lr_ratio
    rl_opt = torch.optim.AdamW(model.parameters(), lr=init_rl_lr, weight_decay=0.01)

    ref_model = LoopedTransformer(copy.deepcopy(model.config), folded=True).to(device)
    set_dropout(ref_model, 0.0)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    for cycle in range(start_cycle, args.cycles + 1):
        t_cyc = time.perf_counter()

        # ── 1. LISTEN: Gutenberg CE steps ────────────────────────────────
        set_dropout(model, args.dropout)
        model.train()
        cyc_ce = 0.0
        for _ in range(args.ce_per_cycle):
            ce_step = global_step - warmup_total
            lr = cosine_lr(ce_step, interleave_ce_total, min(500, interleave_ce_total // 10), args.ce_lr, args.ce_min_lr)
            for pg in ce_opt.param_groups:
                pg["lr"] = lr

            xb, yb = get_batch(train_data, args.batch_size, cfg.block_size, device)
            ce_opt.zero_grad(set_to_none=True)
            _, ce_loss = model(xb, yb)
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ce_opt.step()
            cyc_ce += ce_loss.item()
            global_step += 1
        cyc_ce /= max(args.ce_per_cycle, 1)

        # ── 2. SPEAK: Fresh rollouts on real Gutenberg prompts ───────────
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()

        rollout_groups = []
        for _ in range(args.prompts_per_cycle):
            real_p = sample_real_passage(train_data, tokenizer, eos_id, eot_id, rng)
            words = real_p.split()
            k = rng.randint(4, min(7, max(4, len(words) // 2)))
            p_ids = tokenizer.encode(" ".join(words[:k])).ids
            if not p_ids:
                continue
            gens, nat_stops = sample_rollouts(
                model, tokenizer, p_ids, args.group_size,
                args.max_new_tokens, args.rl_temp, device,
            )
            rollout_groups.append((real_p, p_ids, gens, nat_stops))
            for g in gens:
                fake = " ".join(tokenizer.decode(p_ids + g).split())
                if fake != real_p and len(fake.split()) >= 4:
                    selfplay_buf.append((real_p, fake))

        # ── 3. CRITIC CO-EVOLVES: EMA backbone sync + Bradley-Terry steps
        if args.critic_ema < 1.0:
            with torch.no_grad():
                tau = args.critic_ema
                for p_c, p_m in zip(critic.backbone.parameters(), model.parameters()):
                    p_c.data.mul_(tau).add_(p_m.data, alpha=1.0 - tau)

        critic.train()
        for _ in range(args.critic_per_cycle):
            batch = make_critic_batch(args.critic_batch_size)
            c_ids, c_mask, r_ids, r_mask = encode_critic_batch(batch, tokenizer, args.critic_max_len, device)
            critic_opt.zero_grad(set_to_none=True)
            rc = critic(c_ids, c_mask)
            rr = critic(r_ids, r_mask)
            c_loss = (-F.logsigmoid(rc - rr)).mean() + 1e-3 * (rc.pow(2).mean() + rr.pow(2).mean())
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

        # ── 4. REINFORCE: Trust-region GRPO using live critic ────────────
        cur_rl_lr = args.rl_lr if args.rl_lr is not None else lr * args.rl_lr_ratio
        for pg in rl_opt.param_groups:
            pg["lr"] = cur_rl_lr
        critic.eval()
        set_dropout(model, 0.0)
        cyc_rewards, cyc_kls = [], []

        for rl_iter in range(args.rl_per_cycle):
            if rl_iter > 0:
                rollout_groups = []
                for _ in range(args.prompts_per_cycle):
                    real_p = sample_real_passage(train_data, tokenizer, eos_id, eot_id, rng)
                    words = real_p.split()
                    k = rng.randint(4, min(7, max(4, len(words) // 2)))
                    p_ids = tokenizer.encode(" ".join(words[:k])).ids
                    if not p_ids:
                        continue
                    gens, nat_stops = sample_rollouts(
                        model, tokenizer, p_ids, args.group_size,
                        args.max_new_tokens, args.rl_temp, device,
                    )
                    rollout_groups.append((real_p, p_ids, gens, nat_stops))

            rl_opt.zero_grad(set_to_none=True)
            total_rl_loss = torch.zeros((), device=device)
            n_valid = 0

            for _, p_ids, gens, nat_stops in rollout_groups:
                full_seqs = [p_ids + g for g in gens]
                max_l = max(len(s) for s in full_seqs)
                c_ids = torch.zeros((len(full_seqs), max_l), dtype=torch.long, device=device)
                c_mask = torch.zeros((len(full_seqs), max_l), dtype=torch.float32, device=device)
                for g, s in enumerate(full_seqs):
                    c_ids[g, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                    c_mask[g, : len(s)] = 1.0

                with torch.no_grad():
                    c_scores = critic(c_ids, c_mask).cpu().tolist()

                rewards = [
                    c_scores[g] + guardrail_reward(tokenizer.decode(full_seqs[g]), len(gens[g]), nat_stops[g])
                    for g in range(len(full_seqs))
                ]
                r_t = torch.tensor(rewards, dtype=torch.float32)
                r_std = r_t.std().item()
                advs = [0.0] * len(rewards) if r_std < 1e-6 else [(r - r_t.mean().item()) / (r_std + 1e-6) for r in rewards]
                cyc_rewards.extend(rewards)

                model.train()
                for g in range(len(full_seqs)):
                    if len(gens[g]) == 0:
                        continue
                    log_pi = completion_logprobs(model, full_seqs[g], len(p_ids), device)
                    with torch.no_grad():
                        log_ref = completion_logprobs(ref_model, full_seqs[g], len(p_ids), device)
                    diff = log_ref - log_pi
                    kl = (torch.exp(diff) - diff - 1.0).mean()
                    pg = -(advs[g] * log_pi).mean()
                    total_rl_loss = total_rl_loss + pg + args.kl_beta * kl
                    cyc_kls.append(kl.item())
                    n_valid += 1

            if n_valid > 0:
                (total_rl_loss / n_valid).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                rl_opt.step()

        dt_cyc = time.perf_counter() - t_cyc
        mean_r = sum(cyc_rewards) / max(len(cyc_rewards), 1)
        mean_kl = sum(cyc_kls) / max(len(cyc_kls), 1)

        if cycle == 1 or cycle % 5 == 0:
            print(
                f"cycle {cycle:>3}/{args.cycles} (step {global_step:>5}) | "
                f"lr={lr:.2e} | ce={cyc_ce:.4f} | reward={mean_r:+.3f} | kl={mean_kl:.4f} | {dt_cyc:.1f}s/cyc"
            )

        # ── Periodic Eval + Samples (every 500 CE steps by default) ──────
        if cycle % args.eval_every_cycles == 0 or cycle == args.cycles:
            losses = estimate_loss(model, train_data, val_data, args.batch_size, cfg.block_size, device, args.eval_iters)
            ppl = math.exp(min(losses["val"], 20.0))
            c_acc = eval_critic_quick()
            print(
                f"\n── EVAL cycle {cycle} (step {global_step}): "
                f"train={losses['train']:.4f}  val={losses['val']:.4f} (ppl {ppl:.2f}) | "
                f"critic_acc={c_acc*100:.1f}%"
            )
            for ep in eval_prompts:
                e_ids = tokenizer.encode(ep).ids
                s_gen, _ = sample_rollouts(model, tokenizer, e_ids, 1, 40, 0.7, device)
                print(f"   sample: {tokenizer.decode(e_ids + s_gen[0]).replace(chr(10), ' ')}")
            print()

            writer.writerow([
                global_step, "interleaved", model.layout, f"{lr:.6g}",
                f"{losses['train']:.4f}", "0.0", f"{losses['val']:.4f}", f"{ppl:.2f}",
                f"{c_acc:.3f}", f"{mean_r:.4f}", f"{mean_kl:.5f}",
            ])
            csv_f.flush()

            if losses["val"] < best_val:
                best_val = losses["val"]
                save_policy(out_dir / "best.pt", model, global_step, best_val)
                critic.save(str(out_dir / "critic_best.pt"), step=global_step, acc=c_acc)
                print(f"   ★ New best val={best_val:.4f} -> {out_dir / 'best.pt'}\n")

    save_policy(out_dir / "final.pt", model, global_step, best_val)
    critic.save(str(out_dir / "critic_final.pt"), step=global_step)
    csv_f.close()
    print(f"\nDone! Best val={best_val:.4f} | Saved to {out_dir}")


if __name__ == "__main__":
    main()
