#!/usr/bin/env python3
"""
Teacher-Student Hybrid + Origami Folding with Smooth Fold Transitions
========================================================================

Loads a teacher-student hybrid checkpoint and runs a cyclic folding curriculum.

Key innovation: **Soft fold warmup**. When entering a folded phase, instead of
instantly aliasing blocks (which nukes learning), we gradually interpolate the
target blocks toward their average over the first 1/3 of the phase. Only after
the warmup do we apply the hard fold.

Curriculum (default for 5 layers, matching cromulent noir):
    [1 1 1 1 1] → [*2 1 1 1] → [*2 *2 1] → [*2 1 1 1] → [1 1 1 1 1]

Usage:
    # From Gutenberg pretrain checkpoint
    uv run python src/train_teacher_student_hybrid_folding.py \
        --resume models/experiments/teacher_student_hybrid/ckpt.pt \
        --reset-step --reset-optimizer \
        --steps 15000
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
from model_origami import OrigamiTransformer, parse_layout, apply_layout


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
#  Smooth fold warmup
# ═══════════════════════════════════════════════════════════════════════════════

def soft_fold_blend(model: OrigamiTransformer, target_segments: list,
                    step_in_phase: int, warmup_steps: int):
    """
    Gradually interpolate aliased blocks toward their anchor.
    alpha goes from 0 → 1 over `warmup_steps`.
    Called after each training step during the fold warmup.
    """
    if warmup_steps <= 0:
        return
    alpha = min(step_in_phase / warmup_steps, 1.0)
    if alpha <= 0:
        return

    h = model.transformer.h
    for phys_idx, loop_count in target_segments:
        if loop_count <= 1:
            continue
        anchor = h[phys_idx]
        for i in range(1, loop_count):
            alias_idx = phys_idx + i
            if alias_idx >= len(h):
                continue
            alias = h[alias_idx]
            with torch.no_grad():
                for (n_a, p_a), (n_b, p_b) in zip(
                    anchor.named_parameters(),
                    alias.named_parameters()
                ):
                    p_b.data.lerp_(p_a.data, alpha)


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
    parser.add_argument("--latent-weight", type=float, default=0.1)

    # Training
    parser.add_argument("--steps",      type=int, default=75_000,
                        help="Total steps across all phases (default 75k for 5 × 15k phases)")
    parser.add_argument("--warmup",     type=int, default=500)
    parser.add_argument("--max-lr",     type=float, default=5e-5)
    parser.add_argument("--min-lr",     type=float, default=1e-6)
    parser.add_argument("--tau",        type=float, default=0.99)
    parser.add_argument("--grad-clip",  type=float, default=1.0)
    parser.add_argument("--log-every",  type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=2000)

    # Folding curriculum
    parser.add_argument("--phases", type=str, nargs="+", default=None,
                        help="Phase specs: 'layout:steps' e.g. '1 1 1 1 1:3000' '*2 1 1 1:5000'")
    parser.add_argument("--fold-warmup-frac", type=float, default=0.33,
                        help="Fraction of each fold phase spent in soft-blend warmup")
    parser.add_argument("--fuse-on-fold", action="store_true", default=True,
                        help="Smart-fuse blocks before hard folding")

    # Resume
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-step", action="store_true")
    parser.add_argument("--reset-teacher", action="store_true")
    parser.add_argument("--fresh",      action="store_true")
    parser.add_argument("--out-dir",    type=str, default="models/experiments/teacher_student_hybrid_folding")

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

    mask_token_id = config.vocab_size

    # ── Default curriculum for 5 layers ──────────────────────────────────────
    if args.phases is None:
        # Cromulent noir-style curriculum for n=5
        args.phases = [
            f"1 1 1 1 1:{args.steps // 5}",      # all independent
            f"*2 1 1 1:{args.steps // 5}",       # fold bottom pair
            f"*2 *2 1:{args.steps // 5}",         # fold two pairs
            f"*2 1 1 1:{args.steps // 5}",       # partial unfold
            f"1 1 1 1 1:{args.steps // 5}",      # full unfold
        ]
    phases = []
    for p in args.phases:
        layout, steps = p.split(":")
        phases.append({"layout": layout.strip(), "steps": int(steps.strip())})
    total_steps = sum(p["steps"] for p in phases)

    print(f"Device: {device.upper()}")
    print(f"Model:  {config.n_layer} layers, {config.n_embd} dim, ffn_ratio={config.ffn_ratio}")
    print(f"Loss:   CE + {args.latent_weight} × cosine (temp={args.temp})")
    print(f"Mask:   span ({args.min_span}-{args.max_span}), {args.mask_prob*100:.0f}%")
    print(f"EMA:    tau={args.tau}")
    print(f"\nFolding curriculum ({total_steps:,} total steps)")
    for i, p in enumerate(phases):
        print(f"  Phase {i+1}: {p['layout']:>12s}  {p['steps']:>6,} steps")
    print(f"  Fold warmup: {args.fold_warmup_frac*100:.0f}% of each fold phase")

    # ── Models ───────────────────────────────────────────────────────────────
    student = OrigamiTransformer(config).to(device)
    teacher = OrigamiTransformer(config).to(device)

    # ── Resume / init ──────────────────────────────────────────────────────
    start_step = 0
    start_phase = 0
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.max_lr, betas=(0.9, 0.95),
    )

    resume_path = None
    if args.fresh:
        pass
    elif args.resume:
        resume_path = args.resume
    elif os.path.exists(ckpt_path):
        resume_path = ckpt_path

    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"], strict=False)
        teacher.load_state_dict(ckpt["teacher"], strict=False)
        if args.reset_teacher:
            teacher.load_state_dict(student.state_dict())
            print("  Teacher reset to student")
        if not args.reset_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  Optimizer restored")
        else:
            print("  Optimizer fresh")
        if args.reset_step:
            start_step = 0
            print("  Step counter reset")
        else:
            start_step = ckpt.get("step", 0)
        # Detect saved phase if present
        if "phase" in ckpt:
            start_phase = ckpt["phase"]
        print(f"  Starting at step {start_step}, phase {start_phase}")

    # Ensure valid layout
    for p in phases:
        parse_layout(p["layout"], config.n_layer)

    n_params = sum(p.numel() for p in student.parameters())
    print(f"\nParameters: {n_params / 1e6:.2f}M per network")
    print(f"Total:      {2 * n_params / 1e6:.2f}M")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = get_dataloader(args.data, args.batch_size, args.block_size, shuffle=True)
    val_loader = get_dataloader(args.val, args.batch_size, args.block_size, shuffle=False)

    # ── Training loop ────────────────────────────────────────────────────────
    student.train()
    teacher.train()

    batch_iter = iter(train_loader)
    ce_acc = 0.0
    lat_acc = 0.0
    t0 = time.perf_counter()

    # Build phase cumulative step map
    cum_steps = [0]
    for p in phases:
        cum_steps.append(cum_steps[-1] + p["steps"])

    # Find starting phase from start_step
    phase_idx = 0
    while phase_idx < len(phases) and cum_steps[phase_idx + 1] <= start_step:
        phase_idx += 1
    step_in_phase = start_step - cum_steps[phase_idx]

    for p_idx in range(phase_idx, len(phases)):
        phase = phases[p_idx]
        layout = phase["layout"]
        p_steps = phase["steps"]

        # Determine target segments for this phase
        target_spec = parse_layout(layout, config.n_layer)
        target_segments = []
        logical_pos = 0
        for loop_count in target_spec:
            target_segments.append((logical_pos, loop_count))
            logical_pos += loop_count

        # Check if this phase involves folding
        is_fold = any(loop > 1 for _, loop in target_segments)
        fold_warmup = max(1, int(p_steps * args.fold_warmup_frac)) if is_fold else 0

        print(f"\n{'='*60}")
        print(f"Phase {p_idx+1}: {layout}  ({p_steps:,} steps)")
        if is_fold:
            print(f"  Soft fold warmup: {fold_warmup:,} steps (old layout, gradual blending)")
            print(f"  Hard fold after warmup, then train {p_steps - fold_warmup:,} steps")
        print(f"{'='*60}")

        for s_in_phase in range(step_in_phase, p_steps):
            global_step = cum_steps[p_idx] + s_in_phase
            lr = get_lr(global_step, args.warmup, total_steps, args.max_lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            try:
                x = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                x = next(batch_iter)
            x = x.to(device)

            masked_x, mask = mask_spans(x, mask_token_id, args.mask_prob, args.min_span, args.max_span)

            # ── Soft fold warmup: blend blocks BEFORE hard fold ──────────────
            if fold_warmup > 0 and s_in_phase < fold_warmup:
                # Keep old layout, but blend target blocks toward each other
                soft_fold_blend(student, target_segments, s_in_phase, fold_warmup)
                soft_fold_blend(teacher, target_segments, s_in_phase, fold_warmup)
            elif fold_warmup > 0 and s_in_phase == fold_warmup:
                # Warmup just ended: apply hard fold NOW
                if student.layout != layout:
                    print(f"  Applying hard fold: {student.layout} → {layout}")
                    apply_layout(student, layout)
                    apply_layout(teacher, layout)
                    print(f"  Layout now: {student.layout}")

            # Student forward
            b, t = masked_x.size()
            s_emb = student.transformer.drop(student.transformer.wte(masked_x))
            s_h = s_emb
            for phys_idx, loop_count in student.segments:
                block = student.transformer.h[phys_idx]
                for loop_idx in range(loop_count):
                    s_h = block(s_h, start_pos=0, loop_idx=loop_idx)
            s_latent = student.transformer.ln_f(s_h)
            s_logits = student.lm_head(s_latent)

            # Teacher forward
            with torch.no_grad():
                t_emb = teacher.transformer.drop(teacher.transformer.wte(x))
                t_h = t_emb
                for phys_idx, loop_count in teacher.segments:
                    block = teacher.transformer.h[phys_idx]
                    for loop_idx in range(loop_count):
                        t_h = block(t_h, start_pos=0, loop_idx=loop_idx)
                t_latent = teacher.transformer.ln_f(t_h)

            # Losses
            targets = x[:, 1:].contiguous()
            s_logits_shift = s_logits[:, :-1, :].contiguous()
            mask_shift = mask[:, 1:].contiguous()

            ce_mask = ~mask_shift
            if ce_mask.any():
                ce_loss = F.cross_entropy(
                    s_logits_shift.view(-1, s_logits_shift.size(-1)),
                    targets.view(-1),
                    reduction='none',
                )
                ce_loss = ce_loss[ce_mask.view(-1)].mean()
            else:
                ce_loss = torch.tensor(0.0, device=device)

            s_norm = fixed_normalize(s_latent[:, 1:, :])
            t_norm = fixed_normalize(t_latent[:, 1:, :])
            if mask_shift.any():
                lat_loss = cosine_loss(s_norm, t_norm, mask_shift, temp=args.temp)
            else:
                lat_loss = torch.tensor(0.0, device=device)

            loss = ce_loss + args.latent_weight * lat_loss

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()

            # Teacher EMA
            update_teacher_ema(student, teacher, tau=args.tau)

            ce_acc += ce_loss.item()
            lat_acc += lat_loss.item()

            # ── Logging ──────────────────────────────────────────────────────
            if (global_step + 1) % args.log_every == 0:
                n = args.log_every
                dt = time.perf_counter() - t0
                t0 = time.perf_counter()
                print(f"Step {global_step + 1:>6}: ce {ce_acc/n:.4f}  lat {lat_acc/n:.4f}  "
                      f"lr {lr:.2e}  ({dt:.1f}s / {n})")
                ce_acc = 0.0
                lat_acc = 0.0

            # ── Eval ─────────────────────────────────────────────────────────
            if (global_step + 1) % args.eval_every == 0:
                val_loss = estimate_loss(student, val_loader, mask_token_id, args.mask_prob, device, eval_iters=100)
                print(f"  → val loss {val_loss:.4f}  (ppl {math.exp(val_loss):.1f})")

            # ── Checkpoint ─────────────────────────────────────────────────────
            if (global_step + 1) % args.ckpt_every == 0:
                torch.save({
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step + 1,
                    "phase": p_idx,
                    "config": config,
                }, ckpt_path)
                print(f"  → checkpoint saved to {ckpt_path}")

        step_in_phase = 0  # reset for next phase

    # ── Final save ─────────────────────────────────────────────────────────
    torch.save({
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete. Saved to {final_path}")


if __name__ == "__main__":
    main()
