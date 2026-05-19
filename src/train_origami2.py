#!/usr/bin/env python3
"""
Origami phase 2 — fine-tune on the expanded noir corpus.

Loads mica_origami_v2_phase1.pt (already folded to [2+6+2]), restores the
hard ties, then fine-tunes on noir at 1e-4 for 10k steps.
No further folding.

Saves mica_origami_v2.pt on completion.
"""
import math
import os
import numpy as np
import torch
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments, restore_folds

PHASE1_CHECKPOINT = "mica_origami_v2_phase1.pt"
CHECKPOINT        = "mica_origami_v2_ckpt.pt"
WEIGHTS_OUT       = "mica_origami_v2.pt"

max_iters     = 10_000
eval_interval = 100
warmup_iters  = 200
batch_size    = 16
max_lr        = 1e-4
min_lr        = 1e-6

config = MicaConfig()
print(f"Initializing OrigamiTransformer on {config.device.upper()}...")
model = OrigamiTransformer(config)

ckpt       = None
start_step = 0

if os.path.exists(CHECKPOINT):
    print(f"Resuming from {CHECKPOINT}...")
    ckpt = torch.load(CHECKPOINT, map_location=config.device)
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    else:
        restore_folds(model, ckpt['bottom_loops'], ckpt['top_loops'])
    start_step = ckpt['step'] + 1
    print(f"  Resumed at step {start_step}, layout={model.layout}")
elif os.path.exists(PHASE1_CHECKPOINT):
    print(f"Loading phase-1 weights from {PHASE1_CHECKPOINT}...")
    ckpt = torch.load(PHASE1_CHECKPOINT, map_location=config.device)
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    else:
        restore_folds(model, ckpt.get('bottom_loops', 1), ckpt.get('top_loops', 1))
    print(f"  Layout restored: {model.layout}")
else:
    print("No phase-1 checkpoint found — training from scratch.")

model.to(config.device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  layout={model.layout}")

train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap("data/val.bin",   dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")


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

print(f"Starting phase 2 ({max_iters} steps)...")

for step in range(start_step, max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {step:>6} [{model.layout}]: train {losses['train']:.4f}  val {losses['val']:.4f}  lr {get_lr(step):.2e}")

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
        }, CHECKPOINT)

torch.save({
    'model':        model.state_dict(),
    'layout':       model.layout,
    'segments':     model.segments,
}, WEIGHTS_OUT)
print(f"\nPhase 2 complete. Saved to {WEIGHTS_OUT}  layout={model.layout}")
