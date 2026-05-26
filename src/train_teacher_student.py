#!/usr/bin/env python3
"""
Teacher-Student Self-Supervised Pretraining (data2vec-style)
=============================================================
Phase 1 grammar pretraining on Gutenberg (or mixed) corpus.

Investigates whether a masked latent-prediction objective (student predicts
teacher EMA latents) can learn stable representations on Mica's existing
Origami blocks before we introduce folding or an LM head.

Layout: [1 1 1 1 1] — 5 independent layers, NO folding.
Model:  5 layers × 192 dim (~2.44 M params per network).

If this experiment works, the next steps are:
  1. Attach an LM head and do next-token fine-tuning on noir (Phase 2).
  2. Introduce origami folding post-pretraining and observe transfer.
  3. Compare against the standard next-token pretraining baseline.

Usage:
    uv run python src/train_teacher_student.py
    uv run python src/train_teacher_student.py --steps 150000 --eval-every 2000
"""
import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from config import MicaConfig
from model_origami import OrigamiBlock


# ═══════════════════════════════════════════════════════════════════════════════
#  Architecture — Latent Transformer (no LM head)
# ═══════════════════════════════════════════════════════════════════════════════

class LatentOrigamiTransformer(nn.Module):
    """
    Transformer encoder that returns continuous latent vectors [B, T, D].
    Built from OrigamiBlocks with max_loops=1 (no folding, no depth embeddings).
    """
    def __init__(self, config, mask_token_id: int):
        super().__init__()
        self.config = config
        self.mask_token_id = mask_token_id
        n = config.n_layer

        # +1 in the embedding table to hold the artificial <MASK> token
        vocab_size = config.vocab_size + 1

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([OrigamiBlock(config, max_loops=1) for _ in range(n)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        # No folding for the initial experiment: [1 1 1 1 1]
        # Future: can use model_origami.parse_layout/apply_layout if folding is desired.
        self.segments = [(i, 1) for i in range(n)]
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, start_pos: int = 0):
        b, t = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        h = self.transformer.h
        for phys_idx, loop_count in self.segments:
            block = h[phys_idx]
            for loop_idx in range(loop_count):
                x = block(x, start_pos, loop_idx=loop_idx)
        x = self.transformer.ln_f(x)
        return x  # (B, T, n_embd)


class StudentPredictor(nn.Module):
    """
    Small projection head on the STUDENT only.
    This asymmetry (student has predictor, teacher does not) is the
    canonical anti-collapse mechanism from BYOL / data2vec.
    """
    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    """
    Loads pre-tokenised sequences from a uint16 memory-mapped binary.
    Each sample is a contiguous block of `block_size` token IDs.
    """
    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size].astype(np.int64)
        return torch.from_numpy(chunk)


class RandomChunkSampler(Sampler):
    """
    Samples random indices without materialising the full index list.
    Essential for very large memory-mapped corpora.
    """
    def __init__(self, dataset: Dataset, num_samples: int | None = None):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)

    def __iter__(self):
        n = len(self.dataset)
        for _ in range(self.num_samples):
            yield torch.randint(0, n, (1,)).item()

    def __len__(self):
        return self.num_samples


def get_dataloader(data_path: str, batch_size: int, block_size: int,
                   shuffle: bool = True, num_samples: int | None = None):
    ds = TextDataset(data_path, block_size)
    sampler = RandomChunkSampler(ds, num_samples) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Masking
# ═══════════════════════════════════════════════════════════════════════════════

def mask_spans(x: torch.Tensor, mask_token_id: int, mask_prob: float = 0.15,
               min_span: int = 3, max_span: int = 10):
    """
    Span masking: replace contiguous spans of tokens with `mask_token_id`.
    Spans are randomly placed; total masked fraction is approximately `mask_prob`.
    Returns (masked_x, mask_bool_tensor).
    """
    b, t = x.shape
    masked_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)

    for i in range(b):
        # Target number of masked tokens for this sequence
        n_masked = int(t * mask_prob)
        masked_count = 0
        attempts = 0
        while masked_count < n_masked and attempts < t:
            span_len = torch.randint(min_span, max_span + 1, (1,)).item()
            start = torch.randint(0, max(1, t - span_len + 1), (1,)).item()
            # Only mask positions not already masked
            span_mask = torch.zeros(t, dtype=torch.bool, device=x.device)
            span_mask[start:start + span_len] = True
            new_mask = span_mask & ~mask[i]
            if new_mask.any():
                masked_x[i, new_mask] = mask_token_id
                mask[i] |= new_mask
                masked_count += new_mask.sum().item()
            attempts += span_len
    return masked_x, mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Teacher EMA update
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, tau: float = 0.999):
    """In-place EMA: θ_teacher ← τ·θ_teacher + (1−τ)·θ_student."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(tau).add_(s_param.data, alpha=1.0 - tau)


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def fixed_normalize(x: torch.Tensor, eps: float = 1e-6):
    """Instance-wise LayerNorm without learnable affine parameters."""
    return F.layer_norm(x, x.shape[-1:], weight=None, bias=None, eps=eps)


def cosine_loss(student_out: torch.Tensor, teacher_out: torch.Tensor,
                mask: torch.Tensor, temp: float = 0.1):
    """
    Cosine-similarity loss on masked positions.
    We want cos(s, t) → 1, so loss = (1 - cos) / temp.
    temp scales the sharpness (lower = harder, more contrastive).
    """
    s = student_out[mask]   # (N_masked, D)
    t = teacher_out[mask]
    s = F.normalize(s, dim=-1, eps=1e-6)
    t = F.normalize(t, dim=-1, eps=1e-6)
    cos = (s * t).sum(dim=-1)  # (N_masked,)
    return ((1.0 - cos) / temp).mean()


@torch.no_grad()
def estimate_loss(student, predictor, teacher, val_loader, mask_token_id, mask_prob, device, max_batches=50, temp=0.1, min_span=3, max_span=10):
    student.eval()
    predictor.eval()
    teacher.eval()
    losses = []
    for i, x in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device)
        masked_x, mask = mask_spans(x, mask_token_id, mask_prob, min_span, max_span)
        s_latent = student(masked_x)
        t_latent = teacher(x)
        # Fixed normalization + student predictor (anti-collapse)
        s_norm = fixed_normalize(predictor(s_latent))
        t_norm = fixed_normalize(t_latent)
        if mask.any():
            loss = cosine_loss(s_norm, t_norm, mask, temp=temp)
            losses.append(loss.item())
    student.train()
    predictor.train()
    teacher.train()
    return sum(losses) / len(losses) if losses else float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
#  LR schedule
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data",  type=str, default="data/wiki_train.bin")
    parser.add_argument("--val",   type=str, default="data/wiki_val.bin")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)

    # Model
    parser.add_argument("--n-layer",   type=int, default=5)
    parser.add_argument("--n-embd",    type=int, default=192)
    parser.add_argument("--ffn-ratio", type=int, default=2)
    parser.add_argument("--dropout",   type=float, default=0.05)
    parser.add_argument("--device",    type=str, default=None)

    # Masking
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--min-span",  type=int, default=3,
                        help="Minimum span length for masking")
    parser.add_argument("--max-span",  type=int, default=10,
                        help="Maximum span length for masking")
    parser.add_argument("--temp",      type=float, default=0.1,
                        help="Cosine loss temperature (lower = sharper)")

    # Training
    parser.add_argument("--steps",      type=int, default=100_000,
                        help="Total training steps (default 100k for Gutenberg pretraining)")
    parser.add_argument("--warmup",     type=int, default=2_000)
    parser.add_argument("--max-lr",     type=float, default=1e-3)
    parser.add_argument("--min-lr",     type=float, default=1e-5)
    parser.add_argument("--tau",        type=float, default=0.99,
                        help="EMA decay (lower = faster teacher update). Default 0.99 (was 0.999)")
    parser.add_argument("--grad-clip",  type=float, default=1.0)
    parser.add_argument("--log-every",  type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=2_000,
                        help="Evaluation interval in steps")
    parser.add_argument("--ckpt-every", type=int, default=10_000,
                        help="Checkpoint save interval in steps")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from (defaults to out_dir/ckpt.pt if found)")
    parser.add_argument("--fresh",      action="store_true",
                        help="Ignore any existing checkpoint and start from scratch")
    parser.add_argument("--out-dir",    type=str, default="models/experiments/teacher_student")

    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    final_path = os.path.join(args.out_dir, "final.pt")

    # ── Config ───────────────────────────────────────────────────────────────
    config = MicaConfig()
    config.n_layer   = args.n_layer
    config.n_embd    = args.n_embd
    config.ffn_ratio = args.ffn_ratio
    config.dropout   = args.dropout
    if args.device:
        config.device = args.device
    device = config.device

    # The artificial <MASK> token sits at the end of the vocabulary
    mask_token_id = config.vocab_size  # e.g. 5000

    if config.n_embd != 192:
        print(f"WARNING: MultiResolutionAttention hard-codes 192 dims; n_embd={config.n_embd} will fail.")

    print(f"Device: {device.upper()}")
    print(f"Model:  {config.n_layer} layers, {config.n_embd} dim, ffn_ratio={config.ffn_ratio}")
    print(f"Layout: [1 1 1 1 1] (no folding)")
    print(f"Mask token ID: {mask_token_id}  (embedding table size = {config.vocab_size + 1})")
    print(f"Loss: cosine similarity  (temp={args.temp})")
    print(f"Masking: span  ({args.min_span}-{args.max_span} tokens, {args.mask_prob*100:.0f}% coverage)")
    print(f"EMA tau: {args.tau}")

    # ── Models ───────────────────────────────────────────────────────────────
    student = LatentOrigamiTransformer(config, mask_token_id).to(device)
    teacher = LatentOrigamiTransformer(config, mask_token_id).to(device)
    # Student-only predictor: BYOL/data2vec anti-collapse asymmetry
    predictor = StudentPredictor(config.n_embd, hidden=config.n_embd, dropout=config.dropout).to(device)
    # Start teacher from student weights
    teacher.load_state_dict(student.state_dict())
    # Teacher is frozen — no gradients
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in student.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"Parameters per network: {n_params / 1e6:.2f}M")
    print(f"Predictor params:       {pred_params / 1e6:.2f}M")
    print(f"Total (student+teacher+predictor): {(2 * n_params + pred_params) / 1e6:.2f}M")

    # Fixed normalization (no learnable parameters — prevents collapse)
    # We use F.layer_norm with weight=None, bias=None at compute time.

    # ── Optimiser ──────────────────────────────────────────────────────────
    # Optimise student + predictor only (teacher is EMA, norm is fixed)
    optim_params = list(student.parameters()) + list(predictor.parameters())
    decay = [p for p in optim_params if p.dim() >= 2]
    no_decay = [p for p in optim_params if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95),
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if args.fresh:
        resume_path = None
        print("--fresh: starting from scratch (ignoring any checkpoint)")
    else:
        resume_path = args.resume if args.resume else (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed at step {start_step}")

    # ── Data loaders ───────────────────────────────────────────────────────
    # Train loader is effectively infinite (RandomChunkSampler cycles);
    # we don't need num_samples because we break by step count.
    train_loader = get_dataloader(args.data, args.batch_size, args.block_size, shuffle=True)
    val_loader = get_dataloader(args.val, args.batch_size, args.block_size, shuffle=False)

    train_tokens = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0
    tokens_per_step = args.batch_size * args.block_size
    total_tokens = tokens_per_step * args.steps
    print(f"Corpus:  ~{train_tokens:,} tokens")
    print(f"Tokens/step: {tokens_per_step:,}  |  Total steps: {args.steps:,}  |  Total tokens: ~{total_tokens:,}")
    print(f"Eval every {args.eval_every:,} steps  |  CKPT every {args.ckpt_every:,} steps")
    print(f"LR: {args.max_lr:.0e} → {args.min_lr:.0e}  (warmup {args.warmup:,})")

    # ── Training loop ──────────────────────────────────────────────────────
    student.train()
    teacher.train()

    batch_iter = iter(train_loader)
    loss_acc = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Fetch batch (with cycling loader)
        try:
            x = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            x = next(batch_iter)

        x = x.to(device)

        # Mask student input; teacher sees the raw tokens
        masked_x, mask = mask_spans(x, mask_token_id, args.mask_prob, args.min_span, args.max_span)

        # Forward
        s_latent = student(masked_x)
        with torch.no_grad():
            t_latent = teacher(x)

        # Fixed normalization + student predictor (anti-collapse asymmetry)
        s_norm = fixed_normalize(predictor(s_latent))
        t_norm = fixed_normalize(t_latent)

        # Cosine similarity loss on masked positions
        if mask.any():
            loss = cosine_loss(s_norm, t_norm, mask, temp=args.temp)
        else:
            loss = torch.tensor(0.0, device=device)

        # Student step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
        optimizer.step()

        # Teacher EMA update
        update_teacher_ema(student, teacher, tau=args.tau)

        loss_acc += loss.item()

        # ── Logging ─────────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            avg_loss = loss_acc / args.log_every
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f"Step {step + 1:>6}: loss {avg_loss:.6f}  lr {lr:.2e}  ({dt:.1f}s / {args.log_every} batches)")
            loss_acc = 0.0

        # ── Evaluation ──────────────────────────────────────────────────────
        if (step + 1) % args.eval_every == 0:
            val_loss = estimate_loss(
                student, predictor, teacher, val_loader,
                mask_token_id, args.mask_prob, device, temp=args.temp,
                min_span=args.min_span, max_span=args.max_span,
            )
            print(f"  → val loss {val_loss:.6f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": config,
            }, ckpt_path)
            print(f"  → checkpoint saved to {ckpt_path}")

    # ── Final save ─────────────────────────────────────────────────────────
    torch.save({
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "predictor": predictor.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete. Saved to {final_path}")


if __name__ == "__main__":
    main()
