#!/usr/bin/env python3
"""
Train Mica-as-Critic — 100% from-scratch Bradley-Terry pairwise reward model.

Builds homegrown (chosen, rejected) pairs with zero external models:
  1. corrupt   : real noir vs. corrupted noir (repetition, word shuffle, truncation)
  2. gutenberg : real noir vs. Gutenberg prose (teaches noir voice)
  3. rollout   : real noir continuation vs. Mica's own rollout on the same prompt
  4. human     : your own rankings from dpo_rankings.json (upweighted)

Usage:
  uv run python src/train_critic.py \
    --lm-ckpt models/experiments/looped_noir/best.pt \
    --out-dir models/experiments/looped_critic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from model_critic import MicaCritic
from model_looped import LoopedTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Train Mica-as-Critic reward model")
    p.add_argument("--lm-ckpt", default="models/experiments/looped_noir/best.pt")
    p.add_argument("--corpus", default="corpus/corpus.txt")
    p.add_argument("--gutenberg-bin", default="data/wiki_val.bin")
    p.add_argument("--rankings", default="dpo_rankings.json")
    p.add_argument("--tokenizer", default="mica_tokenizer.json")
    p.add_argument("--out-dir", default="models/experiments/looped_critic")

    # Dataset sizes
    p.add_argument("--n-corrupt", type=int, default=4000)
    p.add_argument("--n-gutenberg", type=int, default=2000)
    p.add_argument("--n-rollout", type=int, default=1200)
    p.add_argument("--refresh-rollouts", action="store_true", help="Regenerate cached Mica rollouts")
    p.add_argument("--human-weight", type=float, default=3.0, help="Loss multiplier for human pairs")

    # Training
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--score-reg", type=float, default=1e-3, help="L2 regularization on raw scores")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Corpus & negative builders ────────────────────────────────────────────────

def load_noir_passages(path: str, min_words: int = 12, max_words: int = 60) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    passages = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if not para:
            continue
        sents = re.split(r"(?<=[.?!])\s+", para)
        buf = []
        wc = 0
        for s in sents:
            w = len(s.split())
            if wc + w > max_words and wc >= min_words:
                passages.append(" ".join(buf))
                buf, wc = [s], w
            else:
                buf.append(s)
                wc += w
        if min_words <= wc <= max_words:
            passages.append(" ".join(buf))
    return passages


def corrupt_passage(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 6:
        return text
    mode = rng.choice(["repeat", "shuffle", "truncate", "drop"])
    if mode == "repeat":
        # Inject an n-gram repetition loop ("a chair in a chair")
        span_len = rng.randint(2, min(5, len(words) // 2))
        pos = rng.randint(2, len(words) - span_len)
        span = words[pos : pos + span_len]
        reps = rng.randint(2, 3)
        out = words[:pos] + span * reps + words[pos + span_len :]
        return " ".join(out[: len(words) + 8])
    elif mode == "shuffle":
        # Scramble a contiguous window of 5-10 words
        win = rng.randint(5, min(10, len(words)))
        pos = rng.randint(0, len(words) - win)
        chunk = words[pos : pos + win]
        rng.shuffle(chunk)
        return " ".join(words[:pos] + chunk + words[pos + win :])
    elif mode == "truncate":
        # Cut mid-sentence without punctuation
        cut = rng.randint(3, max(4, len(words) // 2))
        return " ".join(words[:cut]).rstrip(".!?\"'")
    else:
        # Randomly drop ~35% of words
        kept = [w for w in words if rng.random() > 0.35]
        return " ".join(kept) if len(kept) >= 3 else " ".join(words[:3])


def sample_gutenberg_passages(
    bin_path: str, tokenizer: Tokenizer, n: int, rng: random.Random
) -> list[str]:
    if not os.path.exists(bin_path):
        return []
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    out = []
    max_start = len(data) - 100
    for _ in range(n):
        start = rng.randint(0, max_start)
        toks = [int(t) for t in data[start : start + 70] if int(t) not in (eot_id, eos_id)]
        txt = " ".join(tokenizer.decode(toks).split())
        if len(txt.split()) >= 8:
            out.append(txt)
    return out


@torch.no_grad()
def generate_rollout_pairs(
    lm_ckpt: str,
    tokenizer: Tokenizer,
    passages: list[str],
    n_pairs: int,
    device: str,
    rng: random.Random,
    cache_path: Path,
    refresh: bool = False,
    batch_size: int = 32,
) -> list[dict]:
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached) >= n_pairs:
            print(f"Loaded {len(cached)} cached rollout pairs from {cache_path}")
            return cached[:n_pairs]

    print(f"Generating {n_pairs} Mica rollout pairs for critic training...")
    model = LoopedTransformer.from_checkpoint(lm_ckpt, device=device)
    model.eval()
    eos_id = tokenizer.token_to_id("<|eos|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    chosen_samples = rng.sample(passages, min(n_pairs, len(passages)))
    pairs = []

    for i in range(0, len(chosen_samples), batch_size):
        chunk = chosen_samples[i : i + batch_size]
        prompts = []
        reals = []
        for p in chunk:
            words = p.split()
            k = rng.randint(4, min(7, max(4, len(words) // 3)))
            prompts.append(" ".join(words[:k]))
            reals.append(p)

        # Generate one continuation per prompt (varied temps for diversity)
        for prompt, real in zip(prompts, reals):
            temp = rng.choice([0.65, 0.75, 0.85, 0.95])
            p_ids = tokenizer.encode(prompt).ids
            idx = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = []
            max_new = rng.randint(24, 56)
            for step in range(max_new):
                logits, _ = model(idx[:, -model.config.block_size :])
                logits = logits[:, -1, :] / temp
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                tok = nxt.item()
                if step >= 5 and tok in (eos_id, eot_id):
                    break
                gen.append(tok)
                idx = torch.cat([idx, nxt], dim=1)
            fake = " ".join(tokenizer.decode(p_ids + gen).split())
            if fake != real and len(fake.split()) >= 5:
                pairs.append({"chosen": real, "rejected": fake, "kind": "rollout"})

        if (i + batch_size) % 256 == 0 or (i + batch_size) >= len(chosen_samples):
            print(f"  rollouts: {min(i + batch_size, len(chosen_samples))}/{len(chosen_samples)}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Cached {len(pairs)} rollout pairs -> {cache_path}")
    return pairs


def load_human_pairs(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = []
    for item in raw:
        prompt = item.get("prompt", "")
        goods = item.get("good", [])
        bads = item.get("bad", [])
        for g in goods:
            for b in bads:
                pairs.append({
                    "chosen": " ".join((prompt + g).split()),
                    "rejected": " ".join((prompt + b).split()),
                    "kind": "human",
                })
    return pairs


# ── Batch encoding & evaluation ───────────────────────────────────────────────

def encode_batch(
    pairs: list[dict],
    tokenizer: Tokenizer,
    max_len: int,
    device: str,
    human_weight: float,
):
    c_ids_list, r_ids_list, weights = [], [], []
    for p in pairs:
        c = tokenizer.encode(p["chosen"]).ids[:max_len] or [0]
        r = tokenizer.encode(p["rejected"]).ids[:max_len] or [0]
        c_ids_list.append(c)
        r_ids_list.append(r)
        weights.append(human_weight if p["kind"] == "human" else 1.0)

    def pad(seqs):
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.float32)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1.0
        return ids.to(device), mask.to(device)

    c_ids, c_mask = pad(c_ids_list)
    r_ids, r_mask = pad(r_ids_list)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return c_ids, c_mask, r_ids, r_mask, w


@torch.no_grad()
def evaluate_critic(
    critic: MicaCritic,
    val_pairs: list[dict],
    tokenizer: Tokenizer,
    max_len: int,
    device: str,
    batch_size: int = 64,
):
    critic.eval()
    by_kind: dict[str, list[float]] = {}
    total_loss = 0.0
    total_n = 0

    for i in range(0, len(val_pairs), batch_size):
        chunk = val_pairs[i : i + batch_size]
        c_ids, c_mask, r_ids, r_mask, _ = encode_batch(chunk, tokenizer, max_len, device, 1.0)
        rc = critic(c_ids, c_mask)
        rr = critic(r_ids, r_mask)
        diff = rc - rr
        total_loss += (-F.logsigmoid(diff)).sum().item()
        total_n += len(chunk)
        wins = (diff > 0).float().cpu().tolist()
        for p, win in zip(chunk, wins):
            by_kind.setdefault(p["kind"], []).append(win)

    critic.train()
    overall_acc = sum(sum(v) for v in by_kind.values()) / max(total_n, 1)
    kind_acc = {k: sum(v) / max(len(v), 1) for k, v in sorted(by_kind.items())}
    return total_loss / max(total_n, 1), overall_acc, kind_acc


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    passages = load_noir_passages(args.corpus)
    rng.shuffle(passages)
    print(f"Loaded {len(passages):,} noir passages from {args.corpus}")

    # Split passages 90/10 so val pairs never share chosen text with train
    split = int(len(passages) * 0.9)
    tr_pass, va_pass = passages[:split], passages[split:]

    def make_pairs(src_passages, n_corr, n_gut):
        out = []
        for _ in range(n_corr):
            p = rng.choice(src_passages)
            out.append({"chosen": p, "rejected": corrupt_passage(p, rng), "kind": "corrupt"})
        guts = sample_gutenberg_passages(args.gutenberg_bin, tokenizer, n_gut, rng)
        for g in guts:
            p = rng.choice(src_passages)
            out.append({"chosen": p, "rejected": g, "kind": "gutenberg"})
        return out

    train_pairs = make_pairs(tr_pass, int(args.n_corrupt * 0.9), int(args.n_gutenberg * 0.9))
    val_pairs = make_pairs(va_pass, int(args.n_corrupt * 0.1), int(args.n_gutenberg * 0.1))

    # Self-generated Mica rollouts
    if args.n_rollout > 0:
        rollouts = generate_rollout_pairs(
            args.lm_ckpt, tokenizer, passages, args.n_rollout, device, rng,
            cache_path=out_dir / "rollout_pairs.json",
            refresh=args.refresh_rollouts,
        )
        rng.shuffle(rollouts)
        r_split = int(len(rollouts) * 0.9)
        train_pairs.extend(rollouts[:r_split])
        val_pairs.extend(rollouts[r_split:])

    # Human taste clicks
    human_pairs = load_human_pairs(args.rankings)
    if human_pairs:
        rng.shuffle(human_pairs)
        h_split = int(len(human_pairs) * 0.85)
        train_pairs.extend(human_pairs[:h_split])
        val_pairs.extend(human_pairs[h_split:])
        print(f"Loaded {len(human_pairs)} human preference pairs from {args.rankings}")

    print(f"Total pairs: train={len(train_pairs):,}, val={len(val_pairs):,}")

    critic = MicaCritic.from_lm_checkpoint(args.lm_ckpt, device=device, dropout=args.dropout)
    print(f"Initialized MicaCritic from {args.lm_ckpt} ({critic.num_params()/1e6:.2f}M params)")

    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    for step in range(1, args.steps + 1):
        # Cosine LR with 100-step warmup
        if step <= 100:
            lr = args.lr * step / 100
        else:
            prog = (step - 100) / max(args.steps - 100, 1)
            lr = 1e-6 + 0.5 * (args.lr - 1e-6) * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = [rng.choice(train_pairs) for _ in range(args.batch_size)]
        c_ids, c_mask, r_ids, r_mask, w = encode_batch(
            batch, tokenizer, args.max_len, device, args.human_weight
        )

        optimizer.zero_grad(set_to_none=True)
        rc = critic(c_ids, c_mask)
        rr = critic(r_ids, r_mask)
        bt_loss = (-F.logsigmoid(rc - rr) * w).mean()
        reg_loss = args.score_reg * (rc.pow(2).mean() + rr.pow(2).mean())
        loss = bt_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_acc, kind_acc = evaluate_critic(
                critic, val_pairs, tokenizer, args.max_len, device
            )
            breakdown = "  ".join(f"{k}={v*100:.1f}%" for k, v in kind_acc.items())
            print(
                f"step {step:>4}/{args.steps} | lr={lr:.2e} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%  [{breakdown}]"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                critic.save(str(out_dir / "best.pt"), step=step, val_acc=val_acc, val_loss=val_loss)
                print(f"  ★ New best critic -> {out_dir / 'best.pt'} ({val_acc*100:.1f}%)")

    critic.save(str(out_dir / "final.pt"), step=args.steps, val_acc=best_acc)
    print(f"\nDone! Best val pairwise accuracy: {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()
