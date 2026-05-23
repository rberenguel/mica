#!/usr/bin/env python3
"""
Extend noir fine-tune on the small model with cosine LR.

Loads mica_small_n5_noir_ckpt.pt (or mica_small_n5_noir.pt),
continues training on noir at lower LR with cosine decay.

Usage:
  uv run python src/train_small_noir_extend.py --steps 30000 --lr 5e-5
"""
import argparse
import math
import os
import numpy as np
import torch
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments, restore_folds

parser = argparse.ArgumentParser()
parser.add_argument("--weights",     default="mica_small_n5_noir.pt", help="Starting weights")
parser.add_argument("--ckpt",        default="mica_small_n5_noir_ckpt.pt", help="Checkpoint to resume from")
parser.add_argument("--output",      default="mica_small_n5_noir_extended.pt")
parser.add_argument("--lr",          type=float, default=5e-5,  help="Max LR")
parser.add_argument("--min-lr",      type=float, default=1e-7,  help="Min LR")
parser.add_argument("--steps",       type=int,   default=30000, help="Additional steps")
parser.add_argument("--warmup",      type=int,   default=500,   help="Warmup steps")
parser.add_argument("--batch-size",  type=int,   default=16)
args = parser.parse_args()

max_iters     = args.steps
eval_interval = 100
batch_size    = args.batch_size
max_lr        = args.lr
min_lr        = args.min_lr
warmup_iters  = args.warmup

config = MicaConfig()
config.n_layer   = 5
config.ffn_ratio = 2
config.dropout   = 0.05

print(f"Initializing small OrigamiTransformer on {config.device.upper()}...")
model = OrigamiTransformer(config)

ckpt       = None
start_step = 0

if os.path.exists(args.ckpt):
    print(f"Resuming from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=config.device)
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    else:
        restore_folds(model, ckpt.get('bottom_loops', 1), ckpt.get('top_loops', 1))
    start_step = ckpt['step'] + 1
    print(f"  Resumed at step {start_step}, layout={model.layout}")
elif os.path.exists(args.weights):
    print(f"Loading from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=config.device)
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    else:
        restore_folds(model, ckpt.get('bottom_loops', 1), ckpt.get('top_loops', 1))
    print(f"  Layout restored: {model.layout}")
else:
    print("No checkpoint found.")
    exit(1)

model.to(config.device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  layout={model.layout}")

train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap("data/val.bin",   dtype=np.uint16, mode="r")


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - config.block_size, (batch_size,))
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


def get_lr(step):
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


param_dict     = {n: p for n, p in model.named_parameters() if p.requires_grad}
decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay_params,   "weight_decay": 0.1},
     {"params": nodecay_params, "weight_decay": 0.0}],
    lr=max_lr, betas=(0.9, 0.95),
)
if ckpt is not None and 'optimizer' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
    except Exception as e:
        print(f"  Optimizer state incompatible ({e}), starting fresh.")

print(f"Extending noir fine-tune ({max_iters} steps @ LR {max_lr:.0e}→{min_lr:.0e})...")

for step in range(start_step, max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Step {step:>6} [{model.layout}]: train {losses['train']:.4f}  val {losses['val']:.4f}  lr {lr:.2e}")

    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    X, Y = get_batch("train")
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0 and step > start_step:
        torch.save({
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'step':         step,
            'layout':       model.layout,
            'segments':     model.segments,
        }, args.ckpt)

torch.save({
    'model':        model.state_dict(),
    'layout':       model.layout,
    'segments':     model.segments,
}, args.output)
print(f"\nExtended fine-tune complete. Saved to {args.output}")
