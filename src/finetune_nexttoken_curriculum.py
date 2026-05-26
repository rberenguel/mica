#!/usr/bin/env python3
"""
Fine-tune a pretrained model with pure next-token prediction and curriculum mixing.

Strategy: start with mostly Gutenberg, slowly ramp to 100% noir over most of training.
No teacher. No masking. No cosine loss. Just standard causal LM with curriculum mixing.

Usage:
    uv run python src/finetune_nexttoken_curriculum.py \
        --ckpt models/experiments/teacher_student_hybrid_folding/ckpt.pt \
        --noir-train data/train.bin --noir-val data/val.bin \
        --gb-train data/wiki_train.bin --gb-val data/wiki_val.bin \
        --steps 20000 --ramp-steps 16000 \
        --out-dir models/experiments/teacher_student_hybrid_folding/noir_finetune_v3
"""
import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_mixed_batch(noir_data, gb_data, batch_size, block_size, device, noir_ratio=0.5):
    if torch.rand(1).item() < noir_ratio:
        return get_batch(noir_data, batch_size, block_size, device)
    else:
        return get_batch(gb_data, batch_size, block_size, device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_iters=200):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
#  LR schedule
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", MicaConfig())

    if "student" in ckpt:
        sd = ckpt["student"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Handle the mask token if present
    wte_shape = sd.get("transformer.wte.weight", sd.get("transformer.wte.weight", None))
    if wte_shape is not None:
        actual_vocab = wte_shape.shape[0]
        if actual_vocab != config.vocab_size:
            print(f"  Checkpoint vocab {actual_vocab} vs config {config.vocab_size}")
            config.vocab_size = actual_vocab

    model = OrigamiTransformer(config).to(device)
    model.load_state_dict(sd, strict=False)

    if "segments" in ckpt:
        restore_segments(model, ckpt["segments"])

    # Re-tie
    model.lm_head.weight = model.transformer.wte.weight

    print(f"  Layout: {model.layout}")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model, config


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--noir-train", type=str, default="data/train.bin")
    parser.add_argument("--noir-val",   type=str, default="data/val.bin")
    parser.add_argument("--gb-train",   type=str, default="data/wiki_train.bin")
    parser.add_argument("--gb-val",     type=str, default="data/wiki_val.bin")
    parser.add_argument("--noir-ratio", type=float, default=1.0,
                        help="Final noir ratio at end of ramp (default 1.0 = 100% noir)")
    parser.add_argument("--start-noir-ratio", type=float, default=0.05,
                        help="Starting noir ratio (default 0.05 = 5% noir)")
    parser.add_argument("--ramp-steps", type=int, default=16_000,
                        help="Steps over which noir ratio ramps from start to target")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout",    type=float, default=0.05)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--steps",      type=int, default=20_000)
    parser.add_argument("--warmup",     type=int, default=500)
    parser.add_argument("--max-lr",     type=float, default=5e-5)
    parser.add_argument("--min-lr",     type=float, default=1e-6)
    parser.add_argument("--grad-clip",  type=float, default=1.0)
    parser.add_argument("--log-every",  type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=2_000)
    parser.add_argument("--out-dir", type=str,
                        default="models/experiments/teacher_student_hybrid_folding/noir_finetune_v3")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.upper()}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    final_path = os.path.join(args.out_dir, "final.pt")

    # ── Load model ───────────────────────────────────────────────────────────
    model, config = load_model(args.ckpt, device)
    config.dropout = args.dropout

    # ── Data ───────────────────────────────────────────────────────────────
    noir_train = np.memmap(args.noir_train, dtype=np.uint16, mode="r")
    noir_val   = np.memmap(args.noir_val,   dtype=np.uint16, mode="r")
    gb_train   = np.memmap(args.gb_train,   dtype=np.uint16, mode="r")
    gb_val     = np.memmap(args.gb_val,     dtype=np.uint16, mode="r")

    print(f"\nNoir:      {len(noir_train):,} train / {len(noir_val):,} val")
    print(f"Gutenberg: {len(gb_train):,} train / {len(gb_val):,} val")
    print(f"Curriculum: {args.start_noir_ratio*100:.0f}% → {args.noir_ratio*100:.0f}% noir over {args.ramp_steps:,} steps")
    print(f"Total steps: {args.steps:,}")
    if args.ramp_steps < args.steps:
        print(f"After ramp: {args.noir_ratio*100:.0f}% noir for remaining {args.steps - args.ramp_steps:,} steps")
    else:
        print(f"Ramp extends to end of training")

    # ── Optimizer ──────────────────────────────────────────────────────────
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    t0 = time.perf_counter()
    loss_acc = 0.0

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Curriculum mix ratio
        if step < args.ramp_steps:
            ratio = args.start_noir_ratio + (args.noir_ratio - args.start_noir_ratio) * (step / args.ramp_steps)
        else:
            ratio = args.noir_ratio

        # Mixed batch
        X, Y = get_mixed_batch(noir_train, gb_train, args.batch_size, args.block_size,
                               device, ratio)

        # Standard next-token loss
        _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_acc += loss.item()

        # ── Logging ────────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f"Step {step + 1:>5} (noir={ratio:.0%}): loss {loss_acc/n:.4f}  "
                  f"lr {lr:.2e}  ({dt:.1f}s / {n})")
            loss_acc = 0.0

        # ── Eval ───────────────────────────────────────────────────────────
        if (step + 1) % args.eval_every == 0:
            noir_loss = estimate_loss(model, noir_val, args.batch_size, args.block_size, device, eval_iters=100)
            gb_loss   = estimate_loss(model, gb_val,   args.batch_size, args.block_size, device, eval_iters=100)
            print(f"  → noir {noir_loss:.4f}  gb {gb_loss:.4f}  "
                  f"(ppl noir {math.exp(noir_loss):.1f}  gb {math.exp(gb_loss):.1f})")
            model.train()

        # ── Checkpoint ─────────────────────────────────────────────────────
        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "step": step + 1,
                "layout": model.layout,
                "segments": model.segments,
            }, ckpt_path)
            print(f"  → checkpoint saved")

    # ── Final save ─────────────────────────────────────────────────────────
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "layout": model.layout,
        "segments": model.segments,
    }, final_path)
    print(f"\nFine-tuning complete. Saved to {final_path}")


if __name__ == "__main__":
    main()
