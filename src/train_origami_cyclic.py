#!/usr/bin/env python3
"""
Cyclical Origami Training — arbitrary layout curriculum.

Phases are specified as layout:step-count pairs.  Each phase starts with a
linear LR warmup, then cosine decay.  Structural transitions (fold/unfold)
happen between phases via apply_layout().

Usage:
  # Default n=6 experiment (15k steps per phase)
  uv run python train_origami_cyclic.py --n-layer 6

  # Custom phases
  uv run python train_origami_cyclic.py --n-layer 6 \
    --phase "[6]:15k" --phase "[*2 4]:15k" --phase "[*2 *2 2]:15k"

  # Override LR
  uv run python train_origami_cyclic.py --n-layer 6 --lr 1e-3 --min-lr 1e-5
"""
import argparse
import math
import os
import csv
import time
import numpy as np
import torch
from config import MicaConfig
from model_origami import (
    OrigamiTransformer, parse_layout, apply_layout, restore_segments,
    fuse_blocks_smart, _fuse_on_fold
)


def _parse_steps(s: str) -> int:
    s = s.strip().lower()
    if s.endswith('k'):
        return int(float(s[:-1]) * 1000)
    if s.endswith('m'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _parse_phase(s: str):
    """Parse 'layout:steps' or 'layout:steps:max_lr'."""
    parts = s.split(':')
    layout = parts[0].strip()
    steps = _parse_steps(parts[1])
    max_lr = float(parts[2]) if len(parts) > 2 else None
    return {"layout": layout, "steps": steps, "max_lr": max_lr}


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-layer",    type=int,   default=6,     help="Total transformer depth")
parser.add_argument("--phase",      action="append", default=None,
                    help="Layout:steps[:lr]  e.g. '[6]:15000' or '*2 4:15k:1e-3'")
parser.add_argument("--lr",         type=float, default=1e-3,  help="Max learning rate")
parser.add_argument("--min-lr",     type=float, default=1e-5,  help="Min learning rate")
parser.add_argument("--warmup",     type=int,   default=1200,  help="Steps of linear LR warmup per phase")
parser.add_argument("--batch-size", type=int,   default=16,    help="Batch size")
parser.add_argument("--device",     default=None)
args = parser.parse_args()

N = args.n_layer

# Default curriculum for n=6
if args.phase is None:
    PHASES = [
        {"layout": "6",        "steps": 15000, "max_lr": args.lr},
        {"layout": "*2 4",    "steps": 15000, "max_lr": args.lr},
        {"layout": "*2 *2 2", "steps": 15000, "max_lr": args.lr},
        {"layout": "2 *2 2", "steps": 15000, "max_lr": args.lr},
        {"layout": "6",        "steps": 15000, "max_lr": args.lr},
    ]
else:
    PHASES = [_parse_phase(p) for p in args.phase]
    # Fill in missing max_lr from --lr
    for p in PHASES:
        if p["max_lr"] is None:
            p["max_lr"] = args.lr

# Validate all layouts
for p in PHASES:
    parse_layout(p["layout"], N)

TOTAL_STEPS = sum(p["steps"] for p in PHASES)
CHECKPOINT  = f"mica_origami_cyclic_n{N}_ckpt.pt"
WEIGHTS_OUT = f"mica_origami_cyclic_n{N}.pt"
LOG_CSV     = f"logs/origami/train_log_n{N}.csv"

eval_interval = max(100, int(500 * (args.warmup / 1200)))

config = MicaConfig()
config.n_layer = N
if args.device:
    config.device = args.device
print(f"Initializing OrigamiTransformer ({N} layers) on {config.device.upper()}...")
model = OrigamiTransformer(config)
model.to(config.device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  layout={model.layout}")

train_data = np.memmap("data/wiki_train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap("data/wiki_val.bin",   dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - config.block_size, (args.batch_size,))
    x    = torch.stack([torch.from_numpy(data[i  :i  +config.block_size].astype(np.int64)) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i+1:i+1+config.block_size].astype(np.int64)) for i in ix])
    return x.to(config.device), y.to(config.device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def make_optimizer(max_lr):
    param_dict     = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": 0.1},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=max_lr, betas=(0.9, 0.95),
    )


def get_lr(step_in_phase, phase_steps, max_lr, min_lr):
    if step_in_phase < args.warmup:
        return max_lr * (step_in_phase + 1) / args.warmup
    ratio = (step_in_phase - args.warmup) / max(1, phase_steps - args.warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ── Resume ────────────────────────────────────────────────────────────────────
start_phase = 0
optimizer = make_optimizer(PHASES[0]["max_lr"])
global_step = 0

if os.path.exists(CHECKPOINT):
    print(f"\nResuming from {CHECKPOINT}...")
    ckpt = torch.load(CHECKPOINT, map_location=config.device)
    model.load_state_dict(ckpt['model'], strict=False)

    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    elif 'bottom_loops' in ckpt:
        from model_origami import restore_folds
        restore_folds(model, ckpt['bottom_loops'], ckpt.get('top_loops', 1))

    global_step = ckpt['step']
    saved_phase = ckpt.get('phase', 0)
    print(f"  Resumed at step {global_step}, phase {saved_phase}, layout={model.layout}")

    cum_steps = sum(PHASES[i]['steps'] for i in range(saved_phase + 1))
    if global_step >= cum_steps:
        if saved_phase + 1 < len(PHASES):
            print(f"  Phase {chr(65+saved_phase)} complete. Transitioning...")
            target = PHASES[saved_phase + 1]["layout"]
            # Try to load old optimizer state for fusion, then create fresh one
            try:
                _fuse_on_fold(model, target, optimizer)
            except Exception as e:
                print(f"  Fusion skipped (no optimizer state): {e}")
            apply_layout(model, target)
            optimizer = make_optimizer(PHASES[saved_phase + 1]["max_lr"])
            start_phase = saved_phase + 1
            print(f"  Layout now: {model.layout}")
        else:
            print("Training already complete.")
            exit(0)
    else:
        start_phase = saved_phase
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            print(f"  Optimizer state incompatible ({e}), starting fresh optimizer.")
            optimizer = make_optimizer(PHASES[saved_phase]['max_lr'])
else:
    if os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)

# ── Timing ────────────────────────────────────────────────────────────────────
last_eval_time = time.perf_counter()

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\nCyclic curriculum ({TOTAL_STEPS:,} total steps, n_layer={N}):")
for i, p in enumerate(PHASES):
    lr_info = f"LR {p['max_lr']:.0e}→{args.min_lr:.0e}"
    print(f"  Phase {chr(65+i)}: {p['layout']:>8}  {p['steps']:,} steps  {lr_info}")

for phase_idx, phase in enumerate(PHASES):
    if phase_idx < start_phase:
        continue

    # Apply layout for this phase (no-op if already correct)
    target_layout = phase["layout"]
    if model.layout != target_layout:
        print(f"\n  Transition: {model.layout} → {target_layout}")
        _fuse_on_fold(model, target_layout, optimizer)
        apply_layout(model, target_layout)
        print(f"  Layout now: {model.layout}")
        optimizer = make_optimizer(phase["max_lr"])

    phase_steps = phase["steps"]
    phase_max_lr = phase["max_lr"]

    print(f"\n{'='*60}")
    print(f"Phase {chr(65+phase_idx)}: {model.layout}  ({phase_steps:,} steps)")
    print(f"{'='*60}")

    # Mid-phase resume offset
    resume_offset = 0
    if phase_idx == start_phase and os.path.exists(CHECKPOINT):
        resume_offset = global_step - sum(PHASES[i]['steps'] for i in range(phase_idx))
        if resume_offset > 0:
            print(f"  Fast-forwarding to step_in_phase {resume_offset}...")

    for step_in_phase in range(resume_offset, phase_steps):
        if global_step % eval_interval == 0 or step_in_phase == phase_steps - 1:
            losses = estimate_loss()
            lr = optimizer.param_groups[0]["lr"]
            now = time.perf_counter()
            block_secs = round(now - last_eval_time)
            last_eval_time = now
            print(f"Step {global_step:>6} [P{chr(65+phase_idx)} {model.layout}]: "
                  f"train {losses['train']:.4f}  val {losses['val']:.4f}  lr {lr:.2e}  ({block_secs}s)")
            with open(LOG_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, phase_idx, model.layout,
                                 f"{losses['train']:.6f}", f"{losses['val']:.6f}",
                                 f"{lr:.6e}"])

        lr = get_lr(step_in_phase, phase_steps, phase_max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        X, Y = get_batch("train")
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1

        if global_step % 500 == 0 and global_step > 0:
            torch.save({
                'model':        model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'step':         global_step,
                'phase':        phase_idx,
                'layout':       model.layout,
                'segments':     model.segments,
            }, CHECKPOINT)

# ── Final save ───────────────────────────────────────────────────────────────
torch.save({
    'model':        model.state_dict(),
    'layout':       model.layout,
    'segments':     model.segments,
}, WEIGHTS_OUT)
print(f"\nCyclic training complete. Saved to {WEIGHTS_OUT}  layout={model.layout}")
