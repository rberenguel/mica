#!/usr/bin/env python3
"""
Origami phase 2 — cyclic fine-tune on noir corpus with fold/unfold regularisation.

Folds before training to act as a compression regulariser, then progressively
unfolds. Prevents overfitting on the small (6.6M token) noir corpus.

Usage:
  uv run python train_origami2_cyclic.py
  uv run python train_origami2_cyclic.py --lr 5e-5 --steps-per-phase 2000
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
    restore_folds, fuse_blocks_smart, _fuse_on_fold
)

parser = argparse.ArgumentParser()
parser.add_argument("--weights",       default="mica_origami_v2_phase1.pt",
                    help="Phase-1 checkpoint to load")
parser.add_argument("--phase",         action="append", default=None,
                    help="Layout:steps  e.g. '[*2 *2 6]:2000'")
parser.add_argument("--lr",            type=float, default=5e-5)
parser.add_argument("--min-lr",        type=float, default=1e-6)
parser.add_argument("--warmup",        type=int,   default=200)
parser.add_argument("--dropout",       type=float, default=0.2)
parser.add_argument("--batch-size",    type=int,   default=16)
parser.add_argument("--device",        default=None)
args = parser.parse_args()

CHECKPOINT  = "mica_origami_v2_ckpt.pt"
WEIGHTS_OUT = "mica_origami_v2.pt"
LOG_CSV     = "logs/origami/train_log_v2.csv"

# Default cyclic fine-tune: gentle start, single fold as regulariser.
# Double-folding on top of domain shift is too violent for tiny corpus.
if args.phase is None:
    PHASES = [
        {"layout": "*2 8",  "steps": 8000},  # single fold — main learning
        {"layout": "10",    "steps": 4000},  # unfold — recover full capacity
    ]
else:
    def _parse(p):
        layout, steps = p.split(":")
        return {"layout": layout.strip(), "steps": int(steps.strip())}
    PHASES = [_parse(p) for p in args.phase]

config = MicaConfig()
config.dropout = args.dropout
if args.device:
    config.device = args.device

print(f"Initializing OrigamiTransformer on {config.device.upper()}...")
model = OrigamiTransformer(config)
model.to(config.device)

# Load phase-1 weights
c = torch.load(args.weights, map_location=config.device)
model.load_state_dict(c['model'], strict=False)
if 'segments' in c:
    restore_segments(model, c['segments'])
else:
    restore_folds(model, c.get('bottom_loops', 1), c.get('top_loops', 1))
print(f"Loaded {args.weights}  layout={model.layout}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap("data/val.bin",   dtype=np.uint16, mode="r")
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


# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL_STEPS = sum(p["steps"] for p in PHASES)
eval_interval = max(50, int(100 * (args.warmup / 200)))

print(f"\nFine-tune curriculum ({TOTAL_STEPS:,} steps):")
for i, p in enumerate(PHASES):
    print(f"  Phase {chr(65+i)}: {p['layout']:>8}  {p['steps']:,} steps  LR {args.lr:.0e}→{args.min_lr:.0e}")

if os.path.exists(LOG_CSV):
    os.remove(LOG_CSV)

optimizer = make_optimizer(args.lr)
global_step = 0
last_eval_time = time.perf_counter()

for phase_idx, phase in enumerate(PHASES):
    target_layout = phase["layout"]
    if model.layout != target_layout:
        print(f"\n  Transition: {model.layout} → {target_layout}")
        _fuse_on_fold(model, target_layout, optimizer)
        apply_layout(model, target_layout)
        print(f"  Layout now: {model.layout}")
        optimizer = make_optimizer(args.lr)

    phase_steps = phase["steps"]
    print(f"\n{'='*60}")
    print(f"Phase {chr(65+phase_idx)}: {model.layout}  ({phase_steps:,} steps)")
    print(f"{'='*60}")

    for step_in_phase in range(phase_steps):
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

        lr = get_lr(step_in_phase, phase_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        X, Y = get_batch("train")
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if global_step % 500 == 0:
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
print(f"\nFine-tune complete. Saved to {WEIGHTS_OUT}  layout={model.layout}")
