#!/usr/bin/env python3
"""
Origami folding v2 — phase 1 pre-training on the Gutenberg corpus.

Trains from scratch with a single fold to [2+6+2]:

  Steps 0            – WARMUP_STEPS:          all 10 layers independent  [1+8+1]
  WARMUP             – WARMUP+SOFT:           soft-tie outer pair, grads independent
  WARMUP+SOFT        – +POST_FOLD1:           hard fold → [2+6+2]  (final layout)

Final model: 1 shared block looping 2× (with loop-specific LayerNorms) /
             6 independent bridge blocks /
             1 shared block looping 2× (with loop-specific LayerNorms).
Physical layers: 8.  Functional depth: 10.

Key difference from v1: each OrigamiBlock has per-loop LayerNorm parameters,
so shared weights operate in different normalised spaces per iteration.

Saves mica_origami_v2_phase1.pt on completion.
Mid-run checkpoint: mica_origami_v2_ckpt.pt.
Next: uv run python train_origami2.py
"""
import math
import os
import numpy as np
import torch
from config import MicaConfig
from model_origami import OrigamiTransformer, soft_tie, apply_fold, restore_folds, restore_segments

CHECKPOINT  = "mica_origami_v2_ckpt.pt"
WEIGHTS_OUT = "mica_origami_v2_phase1.pt"

# ── Curriculum ────────────────────────────────────────────────────────────────
WARMUP_STEPS     = 20_000  # [1+8+1] all independent — let manifold stabilise
SOFT_TIE_STEPS   =    500  # soft-tie window before hard fold
POST_FOLD1_STEPS = 29_500  # [2+6+2] final layout

FOLD1_SOFT_AT = WARMUP_STEPS
FOLD1_HARD_AT = WARMUP_STEPS + SOFT_TIE_STEPS
MAX_ITERS     = FOLD1_HARD_AT + POST_FOLD1_STEPS

# ── Hypers ────────────────────────────────────────────────────────────────────
eval_interval = 500
lr_warmup     = 1_000
batch_size    = 16
max_lr        = 1e-3
min_lr        = 1e-5

# ── Model ─────────────────────────────────────────────────────────────────────
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
else:
    print("Starting from scratch.")

model.to(config.device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  layout={model.layout}")

# ── Data ──────────────────────────────────────────────────────────────────────
train_data = np.memmap("data/wiki_train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap("data/wiki_val.bin",   dtype=np.uint16, mode="r")
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
    if step < lr_warmup:
        return max_lr * (step + 1) / lr_warmup
    ratio = (step - lr_warmup) / max(1, MAX_ITERS - lr_warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def make_optimizer():
    param_dict     = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": 0.1},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=max_lr, betas=(0.9, 0.95),
    )


optimizer = make_optimizer()
if ckpt is not None and 'optimizer' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
    except Exception as e:
        print(f"  Optimizer state incompatible ({e}), starting fresh.")

n = config.n_layer
print(f"Curriculum: warmup {WARMUP_STEPS:,} | soft {SOFT_TIE_STEPS} | fold1 {POST_FOLD1_STEPS:,}  (total {MAX_ITERS:,})")
print(f"Starting at step {start_step}...")

# ── Training loop ─────────────────────────────────────────────────────────────
for step in range(start_step, MAX_ITERS):

    # Fold schedule
    if step == FOLD1_SOFT_AT and model.bottom_loops == 1:
        print(f"\nStep {step}: soft-tying first outer pair (h[1]↔h[0], h[{n-2}]↔h[{n-1}])...")
        soft_tie(model.transformer.h[1],     model.transformer.h[0])
        soft_tie(model.transformer.h[n - 2], model.transformer.h[n - 1])

    if step == FOLD1_HARD_AT and model.bottom_loops == 1:
        print(f"\nStep {step}: hard fold  {model.layout} → ", end="")
        apply_fold(model)
        optimizer = make_optimizer()
        print(f"{model.layout}  ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    # Eval
    if step % eval_interval == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"Step {step:>6} [{model.layout}]: train {losses['train']:.4f}  val {losses['val']:.4f}  lr {get_lr(step):.2e}")

    # Train step
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    X, Y = get_batch("train")
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Mid-run checkpoint
    if step % 500 == 0 and step > start_step:
        torch.save({
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'step':         step,
            'layout':       model.layout,
            'segments':     model.segments,
        }, CHECKPOINT)

# ── Final save ────────────────────────────────────────────────────────────────
torch.save({
    'model':        model.state_dict(),
    'layout':       model.layout,
    'segments':     model.segments,
}, WEIGHTS_OUT)
print(f"\nPhase 1 complete. Saved to {WEIGHTS_OUT}  layout={model.layout}")
print("Next: uv run python train_origami2.py")
