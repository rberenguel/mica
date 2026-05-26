#!/usr/bin/env python3
"""
Fine-tune a Teacher-Student pretrained model for next-token prediction.

Loads the student encoder from a teacher-student checkpoint, attaches an LM head
(using the standard OrigamiTransformer architecture), and fine-tunes on the noir
corpus (or evaluates on Gutenberg without training).

Usage — quick eval (no training, just see what perplexity the latents give):
    uv run python src/finetune_teacher_student.py \
        --student-ckpt models/experiments/teacher_student/ckpt_.pt \
        --eval-only

Usage — fine-tune on noir:
    uv run python src/finetune_teacher_student.py \
        --student-ckpt models/experiments/teacher_student/ckpt_.pt \
        --steps 15000 --lr 5e-5
"""
import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from config import MicaConfig
from model_origami import OrigamiTransformer


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=200):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def load_student_into_origami(ckpt_path: str, device: str):
    """
    Load a LatentOrigamiTransformer checkpoint into a standard OrigamiTransformer.
    The <MASK> token embedding (last row of wte) is discarded; lm_head is fresh init.
    """
    print(f"Loading student weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Use the exact config from training (n_layer, n_embd, ffn_ratio, etc.)
    config = ckpt.get("config", MicaConfig())
    config.dropout = 0.0  # eval / fresh start
    model = OrigamiTransformer(config)
    model.to(device)

    # The student's wte has vocab_size+1 rows (includes <MASK> token).
    # We need to slice off the MASK token embedding before loading into the standard model.
    student_sd = ckpt["student"]
    adapted_sd = {}
    for k, v in student_sd.items():
        if k == "transformer.wte.weight":
            adapted_sd[k] = v[:config.vocab_size, :]  # discard MASK embedding
        else:
            adapted_sd[k] = v

    # strict=False because the student blocks have max_loops=1 LNs while
    # OrigamiTransformer blocks have max_loops=config.max_loops LNs.
    missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
    if missing:
        print(f"  Missing keys (expected — loop-specific LNs, lm_head, depth_embed): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    # Re-establish the tied weight (load_state_dict overwrote it)
    model.lm_head.weight = model.transformer.wte.weight

    print(f"  Loaded. Layout={model.layout}  Params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model, config


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt", type=str, required=True,
                        help="Path to teacher-student checkpoint (e.g. ckpt_.pt)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run eval on Gutenberg + Noir without training")
    parser.add_argument("--data", type=str, default="data/train.bin",
                        help="Training data for fine-tuning (default: noir train)")
    parser.add_argument("--val", type=str, default="data/val.bin")
    parser.add_argument("--gb-train", type=str, default="data/wiki_train.bin",
                        help="Gutenberg train for eval-only comparison")
    parser.add_argument("--gb-val", type=str, default="data/wiki_val.bin")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--out", type=str, default="models/experiments/teacher_student/finetuned_noir.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.upper()}")

    # ── Load pretrained student into standard OrigamiTransformer ──────────────
    model, config = load_student_into_origami(args.student_ckpt, device)
    config.dropout = args.dropout

    # ── Eval-only mode ───────────────────────────────────────────────────────
    if args.eval_only:
        print("\n=== EVAL-ONLY: measuring raw next-token perplexity ===\n")

        for name, train_path, val_path in [
            ("Gutenberg", args.gb_train, args.gb_val),
            ("Noir", args.data, args.val),
        ]:
            if not os.path.exists(train_path):
                print(f"Skipping {name}: {train_path} not found")
                continue
            train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
            val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
            losses = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, device, eval_iters=200)
            print(f"{name:12s}  train loss: {losses['train']:.4f}  |  val loss: {losses['val']:.4f}  |  ppl: {math.exp(losses['val']):.2f}")
        return

    # ── Fine-tune on noir (or whichever --data) ──────────────────────────────
    train_data = np.memmap(args.data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val, dtype=np.uint16, mode="r")
    print(f"Fine-tuning corpus: {args.data}")
    print(f"  Train: {len(train_data):,} tokens  |  Val: {len(val_data):,} tokens")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.train()
    t0 = time.perf_counter()
    loss_acc = 0.0

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        X, Y = get_batch(train_data, args.batch_size, args.block_size, device)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_acc += loss.item()

        if (step + 1) % 100 == 0:
            avg = loss_acc / 100
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f"Step {step + 1:>5}: loss {avg:.4f}  lr {lr:.2e}  ({dt:.1f}s / 100 batches)")
            loss_acc = 0.0

        if (step + 1) % args.eval_every == 0:
            losses = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, device, eval_iters=100)
            print(f"  → train {losses['train']:.4f}  val {losses['val']:.4f}  ppl {math.exp(losses['val']):.2f}")
            model.train()

        if (step + 1) % args.ckpt_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "step": step + 1,
                "layout": model.layout,
                "segments": model.segments,
            }
            torch.save(ckpt, args.out)
            print(f"  → saved to {args.out}")

    # Final save
    torch.save({
        "model": model.state_dict(),
        "layout": model.layout,
        "segments": model.segments,
    }, args.out)
    print(f"\nFine-tuning complete. Saved to {args.out}")


if __name__ == "__main__":
    main()
