#!/usr/bin/env python3
"""
Phase 1: pre-train on Simple Wikipedia to learn English grammar and semantics.
Saves checkpoint to mica_phase1.pt when done.
Run prepare_wiki.py first.
"""
import math
import numpy as np
import torch
from config import MicaConfig
from model import MicaTransformer

# ~2 passes through Simple Wikipedia (~50-70M tokens)
max_iters     = 50000
eval_interval = 500
warmup_iters  = 1000
max_lr        = 1e-3
min_lr        = 1e-5
batch_size    = 16
CHECKPOINT    = "mica_phase1.pt"

config = MicaConfig()
print(f"Initializing Mica Transformer on {config.device.upper()}...")
model = MicaTransformer(config)
model.to(config.device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

train_data = np.memmap("wiki_train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap("wiki_val.bin",   dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+config.block_size].astype(np.int64)) for i in ix])
    return x.to(config.device), y.to(config.device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step):
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

param_dict    = {n: p for n, p in model.named_parameters() if p.requires_grad}
decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params= [p for n, p in param_dict.items() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay_params, "weight_decay": 0.1},
     {"params": nodecay_params, "weight_decay": 0.0}],
    lr=max_lr, betas=(0.9, 0.95),
)

print(f"Starting Phase 1 training ({max_iters} steps)...")
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {step:>6}: train {losses['train']:.4f}  val {losses['val']:.4f}  lr {get_lr(step):.2e}")

    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    X, Y = get_batch("train")
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.save(model.state_dict(), CHECKPOINT)
print(f"\nPhase 1 complete. Checkpoint saved to {CHECKPOINT}.")
print("Next: uv run python train.py")
