#!/usr/bin/env python3
"""
Teacher-Student Hybrid: Next-Token CE + Latent Cosine Regulariser
==================================================================
Phase 1 pretraining on Gutenberg. The student is trained on two objectives:

  1. Primary — Next-token cross-entropy on UNMASKED positions (standard GPT).
  2. Secondary — Cosine similarity between student and teacher LATENTS on
     MASKED positions (data2vec-style regulariser).

The next-token loss anchors the model to actual language structure.
The latent loss prevents overfitting to local statistics and encourages
masked positions to align with full-context representations.

Teacher is updated by EMA of the student. Teacher logits are not used.

Architecture: standard OrigamiTransformer (5 layers, [1 1 1 1 1], no folding).

Usage:
    uv run python src/train_teacher_student_hybrid.py \
      --steps 30000 --warmup 2000 \
      --eval-every 1000 --ckpt-every 5000 --log-every 500 \
      --tau 0.99 --temp 0.1 --latent-weight 0.1 \
      --fresh
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
from model_origami import OrigamiTransformer, OrigamiBlock


# ═══════════════════════════════════════════════════════════════════════════════
#  Latent-only Transformer (for teacher — no logits needed, but we keep the head
#  so EMA copies all student parameters including the LM head)
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherTransformer(OrigamiTransformer):
    """Identical to OrigamiTransformer but we only use its ln_f output as latents."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size].astype(np.int64)
        return torch.from_numpy(chunk)


class RandomChunkSampler(Sampler):
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
    """Span masking: replace contiguous spans with <MASK>."""
    b, t = x.shape
    masked_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)

    for i in range(b):
        n_masked = int(t * mask_prob)
        masked_count = 0
        attempts = 0
        while masked_count < n_masked and attempts < t:
            span_len = torch.randint(min_span, max_span + 1, (1,)).item()
            start = torch.randint(0, max(1, t - span_len + 1), (1,)).item()
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
#  Loss helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fixed_normalize(x: torch.Tensor, eps: float = 1e-6):
    return F.layer_norm(x, x.shape[-1:], weight=None, bias=None, eps=eps)


def cosine_loss(student_out: torch.Tensor, teacher_out: torch.Tensor,
                mask: torch.Tensor, temp: float = 0.1):
    s = student_out[mask]
    t = teacher_out[mask]
    s = F.normalize(s, dim=-1, eps=1e-6)
    t = F.normalize(t, dim=-1, eps=1e-6)
    cos = (s * t).sum(dim=-1)
    return ((1.0 - cos) / temp).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Teacher EMA
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, tau: float = 0.99):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(tau).add_(s_param.data, alpha=1.0 - tau)


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, val_loader, mask_token_id, mask_prob, device, eval_iters=100):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        try:
            x = next(iter(val_loader))
        except StopIteration:
            break
        x = x.to(device)
        # For eval we just do next-token prediction on unmasked input
        logits, loss = model(x[:, :-1], x[:, 1:].contiguous())
        losses[k] = loss.item()
    model.train()
    return losses[:k+1].mean().item()


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
    parser.add_argument("--min-span",  type=int, default=3)
    parser.add_argument("--max-span",  type=int, default=10)

    # Loss weights
    parser.add_argument("--temp",          type=float, default=0.1)
    parser.add_argument("--latent-weight", type=float, default=0.1,
                        help="Weight of the latent cosine loss relative to CE loss")

    # Training
    parser.add_argument("--steps",      type=int, default=30_000)
    parser.add_argument("--warmup",     type=int, default=2_000)
    parser.add_argument("--max-lr",     type=float, default=1e-3)
    parser.add_argument("--min-lr",     type=float, default=1e-5)
    parser.add_argument("--tau",        type=float, default=0.99)
    parser.add_argument("--grad-clip",  type=float, default=1.0)
    parser.add_argument("--log-every",  type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--ckpt-every", type=int, default=5_000)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--fresh",      action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Load model weights but create a fresh optimizer (for fine-tuning)")
    parser.add_argument("--reset-step", action="store_true",
                        help="Reset step counter to 0 (restart LR schedule, for fine-tuning)")
    parser.add_argument("--reset-teacher", action="store_true",
                        help="Copy student weights into teacher at start (both co-evolve on new domain)")
    parser.add_argument("--out-dir",    type=str, default="models/experiments/teacher_student_hybrid")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    final_path = os.path.join(args.out_dir, "final.pt")

    config = MicaConfig()
    config.n_layer   = args.n_layer
    config.n_embd    = args.n_embd
    config.ffn_ratio = args.ffn_ratio
    config.dropout   = args.dropout
    if args.device:
        config.device = args.device
    device = config.device

    # MASK token at the end of vocab
    mask_token_id = config.vocab_size

    print(f"Device: {device.upper()}")
    print(f"Model:  {config.n_layer} layers, {config.n_embd} dim, ffn_ratio={config.ffn_ratio}")
    print(f"Layout: [1 1 1 1 1] (no folding)")
    print(f"Loss:   CE (unmasked) + {args.latent_weight} × cosine (masked, temp={args.temp})")
    print(f"Mask:   span ({args.min_span}-{args.max_span}), {args.mask_prob*100:.0f}% coverage")
    print(f"EMA:    tau={args.tau}")

    # ── Models ───────────────────────────────────────────────────────────────
    # Student: standard OrigamiTransformer with LM head
    student = OrigamiTransformer(config).to(device)
    # Teacher: identical architecture, EMA-updated
    teacher = OrigamiTransformer(config).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in student.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M per network")
    print(f"Total:      {2 * n_params / 1e6:.2f}M")

    # ── Optimiser ──────────────────────────────────────────────────────────
    decay = [p for n, p in student.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in student.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95),
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if args.fresh:
        resume_path = None
        print("--fresh: starting from scratch")
    else:
        resume_path = args.resume if args.resume else (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        if args.reset_teacher:
            teacher.load_state_dict(student.state_dict())
            print("  Teacher reset to student (both co-evolve on new domain)")
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  Optimizer state restored")
        else:
            print("  Optimizer state reset (fresh AdamW)")
        if args.reset_step:
            start_step = 0
            print("  Step counter reset to 0 (LR schedule restarts)")
        else:
            start_step = ckpt.get("step", 0)
            print(f"  Resumed at step {start_step}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = get_dataloader(args.data, args.batch_size, args.block_size, shuffle=True)
    val_loader = get_dataloader(args.val, args.batch_size, args.block_size, shuffle=False)

    train_tokens = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0
    tokens_per_step = args.batch_size * args.block_size
    total_tokens = tokens_per_step * args.steps
    print(f"Corpus:  ~{train_tokens:,} tokens")
    print(f"Tokens/step: {tokens_per_step:,}  |  Steps: {args.steps:,}  |  Total: ~{total_tokens:,}")
    print(f"Eval every {args.eval_every:,}  |  CKPT every {args.ckpt_every:,}")

    # ── Training loop ──────────────────────────────────────────────────────
    student.train()
    teacher.train()

    batch_iter = iter(train_loader)
    ce_acc = 0.0
    lat_acc = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        try:
            x = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            x = next(batch_iter)

        x = x.to(device)

        # Mask spans for student; teacher sees full sequence
        masked_x, mask = mask_spans(x, mask_token_id, args.mask_prob, args.min_span, args.max_span)

        # Student forward: full sequence logits + latents
        # OrigamiTransformer returns logits and loss when targets are provided.
        # We need the full-sequence logits and the pre-head latent vectors.
        # HACK: forward manually to get both logits and ln_f output.
        b, t = masked_x.size()
        s_emb = student.transformer.drop(student.transformer.wte(masked_x))
        s_h = s_emb
        for phys_idx, loop_count in student.segments:
            block = student.transformer.h[phys_idx]
            for loop_idx in range(loop_count):
                s_h = block(s_h, start_pos=0, loop_idx=loop_idx)
        s_latent = student.transformer.ln_f(s_h)  # (B, T, D)
        s_logits = student.lm_head(s_latent)       # (B, T, vocab)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_emb = teacher.transformer.drop(teacher.transformer.wte(x))
            t_h = t_emb
            for phys_idx, loop_count in teacher.segments:
                block = teacher.transformer.h[phys_idx]
                for loop_idx in range(loop_count):
                    t_h = block(t_h, start_pos=0, loop_idx=loop_idx)
            t_latent = teacher.transformer.ln_f(t_h)

        # Targets for next-token prediction: shift by 1
        targets = x[:, 1:].contiguous()  # (B, T-1)
        s_logits_shift = s_logits[:, :-1, :].contiguous()  # (B, T-1, V)
        mask_shift = mask[:, 1:].contiguous()  # mask aligned with targets

        # ── Primary loss: CE on UNMASKED positions ──────────────────────────
        # We want next-token prediction everywhere EXCEPT where the input was masked
        # (the student shouldn't be evaluated on predicting a token it never saw).
        ce_mask = ~mask_shift  # predict next token only from unmasked context
        if ce_mask.any():
            ce_loss = F.cross_entropy(
                s_logits_shift.view(-1, s_logits_shift.size(-1)),
                targets.view(-1),
                reduction='none',
            )
            ce_loss = ce_loss[ce_mask.view(-1)].mean()
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # ── Secondary loss: cosine on masked positions ──────────────────────
        # Align student latents with teacher latents at positions that were masked
        s_norm = fixed_normalize(s_latent[:, 1:, :])  # align with targets
        t_norm = fixed_normalize(t_latent[:, 1:, :])
        if mask_shift.any():
            lat_loss = cosine_loss(s_norm, t_norm, mask_shift, temp=args.temp)
        else:
            lat_loss = torch.tensor(0.0, device=device)

        loss = ce_loss + args.latent_weight * lat_loss

        # ── Backward ────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        # Teacher EMA
        update_teacher_ema(student, teacher, tau=args.tau)

        ce_acc += ce_loss.item()
        lat_acc += lat_loss.item()

        # ── Logging ─────────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f"Step {step + 1:>6}: ce {ce_acc/n:.4f}  lat {lat_acc/n:.4f}  "
                  f"lr {lr:.2e}  ({dt:.1f}s / {n} batches)")
            ce_acc = 0.0
            lat_acc = 0.0

        # ── Eval ────────────────────────────────────────────────────────────
        if (step + 1) % args.eval_every == 0:
            val_loss = estimate_loss(student, val_loader, mask_token_id, args.mask_prob, device, eval_iters=100)
            print(f"  → val loss {val_loss:.4f}  (ppl {math.exp(val_loss):.1f})")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": config,
            }, ckpt_path)
            print(f"  → checkpoint saved to {ckpt_path}")

    # ── Final save ─────────────────────────────────────────────────────────
    torch.save({
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete. Saved to {final_path}")


if __name__ == "__main__":
    main()
