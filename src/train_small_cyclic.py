#!/usr/bin/env python3
"""
Small-model overfit experiment — n=5, ffn_ratio=2, cyclic on Gutenberg.

Half the layers, half the FFN width. ~2.5M params.
Train way longer (300K steps) to see how low a tiny model can go.

Usage:
  uv run python src/train_small_cyclic.py
  uv run python src/train_small_cyclic.py --lr 1e-3 --phase "[5]:40k" --phase "[*2 3]:40k"
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
    _fuse_on_fold
)


def _parse_steps(s: str) -> int:
    s = s.strip().lower()
    if s.endswith('k'):
        return int(float(s[:-1]) * 1000)
    if s.endswith('m'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _parse_phase(s: str):
    parts = s.split(':')
    layout = parts[0].strip()
    steps = _parse_steps(parts[1])
    max_lr = float(parts[2]) if len(parts) > 2 else None
    return {"layout": layout, "steps": steps, "max_lr": max_lr}


parser = argparse.ArgumentParser()
parser.add_argument("--n-layer",    type=int,   default=5,     help="Total transformer depth")
parser.add_argument("--ffn-ratio",  type=int,   default=2,     help="MLP expansion ratio (2=small, 4=std)")
parser.add_argument("--phase",      action="append", default=None,
                    help="Layout:steps[:lr]  e.g. '[5]:60k' or '*2 3:60k:1e-3'")
parser.add_argument("--lr",         type=float, default=1e-3,  help="Max learning rate")
parser.add_argument("--min-lr",     type=float, default=1e-5,  help="Min learning rate")
parser.add_argument("--warmup",     type=int,   default=400,   help="Steps of linear LR warmup per phase (short for 10k phases)")
parser.add_argument("--batch-size", type=int,   default=16,    help="Batch size")
parser.add_argument("--dropout",    type=float, default=0.05,  help="Dropout (low for overfit)")
parser.add_argument("--cycles",     type=int,   default=3,     help="Number of fold/unfold cycles (default 3)")
parser.add_argument("--device",     default=None)
args = parser.parse_args()

N = args.n_layer

# Violent wrapping curriculum: 10k steps per phase, fold/unfold cycles.
# Warm-up unfold → (fold → unfold) × cycles → wrap fold → cool-down unfold.
# The wrap fold means the last unfold is treated like the warm-up: fold again.
if args.phase is None:
    PHASES = []
    # Warm-up unfold
    PHASES.append({"layout": "5",       "steps": 10000, "max_lr": args.lr})
    # Cycles: fold → unfold
    for c in range(args.cycles):
        PHASES.append({"layout": "*2 *2 1", "steps": 10000, "max_lr": args.lr})
        PHASES.append({"layout": "5",       "steps": 10000, "max_lr": args.lr})
    # Wrap fold: after the last unfold, fold again as if it were warm-up
    PHASES.append({"layout": "*2 *2 1", "steps": 10000, "max_lr": args.lr})
    # Final cool-down unfold at lower LR
    PHASES.append({"layout": "5",       "steps": 10000, "max_lr": args.lr * 0.3})
else:
    PHASES = [_parse_phase(p) for p in args.phase]
    for p in PHASES:
        if p["max_lr"] is None:
            p["max_lr"] = args.lr

for p in PHASES:
    parse_layout(p["layout"], N)

TOTAL_STEPS = sum(p["steps"] for p in PHASES)
CKPT_NAME   = f"mica_small_n{N}_ckpt.pt"
OUT_NAME    = f"mica_small_n{N}.pt"
LOG_CSV     = f"logs/small/train_log_small_n{N}.csv"

eval_interval = max(100, int(500 * (args.warmup / 1200)))

config = MicaConfig()
config.n_layer = N
config.ffn_ratio = args.ffn_ratio
config.dropout = args.dropout
if args.device:
    config.device = args.device

print(f"Initializing OrigamiTransformer ({N} layers, ffn_ratio={config.ffn_ratio}, dropout={config.dropout}) on {config.device.upper()}...")
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


# ── Resume ───────────────────────────────────────────────────────────────────
start_phase = 0
optimizer = make_optimizer(PHASES[0]["max_lr"])
global_step = 0

os.makedirs("logs/small", exist_ok=True)

if os.path.exists(CKPT_NAME):
    print(f"\nResuming from {CKPT_NAME}...")
    ckpt = torch.load(CKPT_NAME, map_location=config.device)
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    global_step = ckpt['step']
    saved_phase = ckpt.get('phase', 0)
    print(f"  Resumed at step {global_step}, phase {saved_phase}, layout={model.layout}")

    cum_steps = sum(PHASES[i]['steps'] for i in range(saved_phase + 1))
    if global_step >= cum_steps:
        if saved_phase + 1 < len(PHASES):
            print(f"  Phase {chr(65+saved_phase)} complete. Transitioning...")
            target = PHASES[saved_phase + 1]["layout"]
            try:
                _fuse_on_fold(model, target, optimizer)
            except Exception as e:
                print(f"  Fusion skipped: {e}")
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
            print(f"  Optimizer state incompatible ({e}), starting fresh.")
            optimizer = make_optimizer(PHASES[saved_phase]['max_lr'])
else:
    if os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)

last_eval_time = time.perf_counter()

print(f"\nSmall-model curriculum ({TOTAL_STEPS:,} total steps, n_layer={N}, ffn_ratio={config.ffn_ratio}):")
for i, p in enumerate(PHASES):
    lr_info = f"LR {p['max_lr']:.0e}→{args.min_lr:.0e}"
    print(f"  Phase {chr(65+i)}: {p['layout']:>8}  {p['steps']:,} steps  {lr_info}")

for phase_idx, phase in enumerate(PHASES):
    if phase_idx < start_phase:
        continue

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
            }, CKPT_NAME)

# ── Final save ───────────────────────────────────────────────────────────────
torch.save({
    'model':        model.state_dict(),
    'layout':       model.layout,
    'segments':     model.segments,
}, OUT_NAME)
print(f"\nSmall-model training complete. Saved to {OUT_NAME}  layout={model.layout}")
