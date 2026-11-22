#!/usr/bin/env python3
"""
Train Mica Looped Transformer (Prelude → Looped Core → Coda).

Two modes:
  1. pretrain (default) — Origami Kickstart on Gutenberg:
       Stage 1 (unfolded):  1 + K + 1 independent blocks
       Stage 2 (soft-fold): smooth L2 pull on shared core weights (norms stay distinct)
       Stage 3 (looped):    hard_fold_core() -> permanent 1 + *K + 1 looped training
  2. finetune — Standard low-LR noir fine-tuning on the folded looped model.

Examples:
  # Phase 1: Gutenberg pretraining (~35k steps, ~12h on M1)
  uv run python src/train_looped.py

  # Quick smoke test (200 steps)
  uv run python src/train_looped.py --unfolded-steps 80 --soft-fold-steps 40 --looped-steps 80 \
    --eval-every 40 --log-every 20 --out-dir models/experiments/looped_smoke

  # Phase 2: Noir fine-tuning from Phase 1 best checkpoint
  uv run python src/train_looped.py --mode finetune \
    --init-from models/experiments/looped/best.pt \
    --data data/train.bin --val data/val.bin \
    --out-dir models/experiments/looped_noir \
    --looped-steps 8000 --lr 1e-4 --min-lr 1e-6 --dropout 0.15
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from model_looped import (
    LoopedConfig,
    LoopedTransformer,
    hard_fold_core,
    soft_tie_loss,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Mica Looped Transformer")
    p.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--data", default=None, help="Training .bin path")
    p.add_argument("--val", default=None, help="Validation .bin path")
    p.add_argument("--tokenizer", default="mica_tokenizer.json")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--log-csv", default=None)

    # Architecture
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--ffn-hidden", type=int, default=704)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--attn-type", choices=["multires", "mha"], default="multires")
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=None)

    # Kickstart / schedule
    p.add_argument("--no-kickstart", action="store_true", help="Start looped from step 0")
    p.add_argument("--unfolded-steps", type=int, default=6000)
    p.add_argument("--soft-fold-steps", type=int, default=2000)
    p.add_argument("--looped-steps", type=int, default=27000)
    p.add_argument("--tie-weight", type=float, default=500.0, help="Peak soft-tie L2 weight")

    # Optimization
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Checkpointing / eval
    p.add_argument("--init-from", default=None, help="Load weights for finetune")
    p.add_argument("--resume", default=None, help="Resume full training state from ckpt.pt")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-prompt", default="The rain came down hard on")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


@torch.no_grad()
def quick_sample(model, tokenizer, prompt: str, device: str, max_new: int = 40, temp: float = 0.7):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    gen = []
    for _ in range(max_new):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temp, 1e-5)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        tok = nxt.item()
        if len(gen) >= 8 and tok in (eot_id, eos_id):
            break
        gen.append(tok)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return tokenizer.decode(ids + gen).replace("\n", " ")


def build_optimizer(model: LoopedTransformer, lr: float, weight_decay: float):
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
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def save_weights(path: Path, model: LoopedTransformer, step: int, val_loss: float):
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


def save_ckpt(path: Path, model, optimizer, step: int, stage: str, best_val: float):
    torch.save(
        {
            "config": dataclasses.asdict(model.config),
            "folded": model.folded,
            "layout": model.layout,
            "step": step,
            "stage": stage,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Mode-dependent defaults
    if args.mode == "pretrain":
        data_path = args.data or "data/wiki_train.bin"
        val_path = args.val or "data/wiki_val.bin"
        out_dir = Path(args.out_dir or "models/experiments/looped")
        log_csv = Path(args.log_csv or "logs/looped/pretrain.csv")
        dropout = args.dropout if args.dropout is not None else 0.1
        max_lr = args.lr if args.lr is not None else 1.2e-3
        min_lr = args.min_lr if args.min_lr is not None else 1e-5
        warmup = args.warmup if args.warmup is not None else 500
    else:
        data_path = args.data or "data/train.bin"
        val_path = args.val or "data/val.bin"
        out_dir = Path(args.out_dir or "models/experiments/looped_noir")
        log_csv = Path(args.log_csv or "logs/looped/finetune.csv")
        dropout = args.dropout if args.dropout is not None else 0.15
        max_lr = args.lr if args.lr is not None else 1e-4
        min_lr = args.min_lr if args.min_lr is not None else 1e-6
        warmup = args.warmup if args.warmup is not None else 200

    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(data_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Missing data files: {data_path} / {val_path}. "
            "Run prepare_wiki.py (Gutenberg) or prepare.py (Noir) first."
        )

    train_data = np.memmap(data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Build or load model
    start_step = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = LoopedConfig(**ckpt["config"])
        cfg.device = device
        cfg.dropout = dropout
        model = LoopedTransformer(cfg, folded=ckpt["folded"]).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = build_optimizer(model, max_lr, args.weight_decay)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from {args.resume} at step {start_step} (layout={model.layout}, best_val={best_val:.4f})")
    elif args.mode == "finetune":
        if not args.init_from:
            raise ValueError("--mode finetune requires --init-from <checkpoint>")
        model = LoopedTransformer.from_checkpoint(args.init_from, device=device)
        model.config.dropout = dropout
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout
            if hasattr(m, "attn_dropout"):
                m.attn_dropout = dropout
        optimizer = build_optimizer(model, max_lr, args.weight_decay)
        print(f"Loaded {args.init_from} for noir fine-tuning (layout={model.layout}, params={model.num_params()/1e6:.2f}M)")
    else:
        cfg = LoopedConfig(
            vocab_size=tokenizer.get_vocab_size(),
            block_size=args.block_size,
            n_embd=args.n_embd,
            ffn_hidden=args.ffn_hidden,
            n_loops=args.n_loops,
            attn_type=args.attn_type,
            n_head=args.n_head,
            dropout=dropout,
            device=device,
        )
        start_folded = args.no_kickstart
        model = LoopedTransformer(cfg, folded=start_folded).to(device)
        optimizer = build_optimizer(model, max_lr, args.weight_decay)

    # Determine step boundaries
    if args.mode == "finetune" or args.no_kickstart:
        unfolded_end = 0
        soft_fold_end = 0
        total_steps = args.looped_steps
    else:
        unfolded_end = args.unfolded_steps
        soft_fold_end = args.unfolded_steps + args.soft_fold_steps
        total_steps = soft_fold_end + args.looped_steps

    print(
        f"Mode={args.mode} | Device={device} | Layout={model.layout} | "
        f"Params={model.num_params()/1e6:.2f}M | Tokens/step={args.batch_size * model.config.block_size:,}"
    )
    print(
        f"Data: train={len(train_data):,} tok, val={len(val_data):,} tok | "
        f"Steps: unfolded={unfolded_end}, soft_fold={soft_fold_end - unfolded_end}, "
        f"looped={total_steps - soft_fold_end} (total={total_steps})"
    )

    write_header = not log_csv.exists() or start_step == 0
    csv_f = open(log_csv, "a" if not write_header else "w", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["step", "stage", "layout", "lr", "train_ce", "tie_loss", "val_ce", "val_ppl", "tok_s"])

    t_log = time.perf_counter()
    steps_since_log = 0
    running_ce = 0.0
    running_tie = 0.0

    for step in range(start_step, total_steps):
        # ── Stage transitions ────────────────────────────────────────────
        if step < unfolded_end:
            stage = "unfolded"
            lr = cosine_lr(step, soft_fold_end, warmup, max_lr, max_lr * 0.25)
            tie_w = 0.0
        elif step < soft_fold_end:
            stage = "soft_fold"
            lr = cosine_lr(step, soft_fold_end, warmup, max_lr, max_lr * 0.25)
            frac = (step - unfolded_end + 1) / max(args.soft_fold_steps, 1)
            tie_w = args.tie_weight * frac
        else:
            stage = "looped" if args.mode == "pretrain" else "finetune"
            # Trigger hard fold right at the boundary
            if not model.folded:
                pre_losses = estimate_loss(
                    model, train_data, val_data, args.batch_size,
                    model.config.block_size, device, eval_iters=20,
                )
                print(
                    f"\n>>> HARD FOLD at step {step}: {model.layout} -> 1 + *{model.config.n_loops} + 1 "
                    f"(pre-fold val={pre_losses['val']:.4f})"
                )
                hard_fold_core(model, optimizer=optimizer)
                post_losses = estimate_loss(
                    model, train_data, val_data, args.batch_size,
                    model.config.block_size, device, eval_iters=20,
                )
                print(
                    f">>> Fold complete! Params={model.num_params()/1e6:.2f}M | "
                    f"post-fold val={post_losses['val']:.4f} "
                    f"(delta={post_losses['val'] - pre_losses['val']:+.4f})\n"
                )
                # Reset optimizer cleanly for the folded parameter set
                optimizer = build_optimizer(model, max_lr * 0.85, args.weight_decay)
                t_log = time.perf_counter()

            loop_step = step - soft_fold_end
            loop_total = total_steps - soft_fold_end
            loop_max_lr = max_lr if (args.mode == "finetune" or args.no_kickstart) else max_lr * 0.85
            lr = cosine_lr(loop_step, loop_total, warmup, loop_max_lr, min_lr)
            tie_w = 0.0

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Forward / backward ───────────────────────────────────────────
        xb, yb = get_batch(train_data, args.batch_size, model.config.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, ce_loss = model(xb, yb)

        if tie_w > 0.0:
            t_loss = soft_tie_loss(model)
            loss = ce_loss + tie_w * t_loss
            running_tie += t_loss.item()
        else:
            loss = ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_ce += ce_loss.item()
        steps_since_log += 1

        # ── Logging ──────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0 or step == 0:
            if device == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t_log
            tok_s = (steps_since_log * args.batch_size * model.config.block_size) / max(dt, 1e-6)
            avg_ce = running_ce / steps_since_log
            avg_tie = running_tie / steps_since_log
            tie_str = f"  tie={avg_tie:.5f}" if tie_w > 0 else ""
            print(
                f"step {step+1:>6}/{total_steps} [{stage:<9}] "
                f"lr={lr:.2e}  ce={avg_ce:.4f}{tie_str}  {tok_s:,.0f} tok/s"
            )
            running_ce = 0.0
            running_tie = 0.0
            steps_since_log = 0
            t_log = time.perf_counter()

        # ── Evaluation ───────────────────────────────────────────────────
        if (step + 1) % args.eval_every == 0 or (step + 1) == total_steps:
            losses = estimate_loss(
                model, train_data, val_data, args.batch_size,
                model.config.block_size, device, args.eval_iters,
            )
            val_ppl = math.exp(min(losses["val"], 20.0))
            sample = quick_sample(model, tokenizer, args.sample_prompt, device)
            print(
                f"── EVAL step {step+1} ({model.layout}): "
                f"train={losses['train']:.4f}  val={losses['val']:.4f} (ppl {val_ppl:.2f})"
            )
            print(f"   sample: {sample}")

            writer.writerow([
                step + 1, stage, model.layout, f"{lr:.6g}",
                f"{losses['train']:.4f}", f"{tie_w:.3f}",
                f"{losses['val']:.4f}", f"{val_ppl:.2f}", "",
            ])
            csv_f.flush()

            # Track best val once in the final stage (folded)
            if model.folded and losses["val"] < best_val:
                best_val = losses["val"]
                save_weights(out_dir / "best.pt", model, step + 1, best_val)
                print(f"   ★ New best val={best_val:.4f} -> {out_dir / 'best.pt'}")
            t_log = time.perf_counter()

        # ── Periodic checkpoint ──────────────────────────────────────────
        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(out_dir / "ckpt.pt", model, optimizer, step + 1, stage, best_val)

    # Final save
    final_losses = estimate_loss(
        model, train_data, val_data, args.batch_size,
        model.config.block_size, device, args.eval_iters,
    )
    save_weights(out_dir / "final.pt", model, total_steps, final_losses["val"])
    save_ckpt(out_dir / "ckpt.pt", model, optimizer, total_steps, stage, best_val)
    csv_f.close()
    print(f"\nDone! Final val={final_losses['val']:.4f} | Best val={best_val:.4f}")
    print(f"Saved: {out_dir / 'final.pt'} and {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
