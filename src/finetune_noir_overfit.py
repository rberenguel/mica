#!/usr/bin/env python3
"""
Aggressive noir fine-tune: high LR, short warmup, pure noir, moving teacher.

The v5b "brilliant" model was overfitting. This script embraces that.
No curriculum. No Gutenberg. Just raw noir with the same masked objective
as pretraining, and a teacher that follows the student via EMA.

Usage:
    uv run python src/finetune_noir_overfit.py \
        --ckpt models/experiments/teacher_student_hybrid_folding/ckpt.pt \
        --train data/train.bin --val data/val.bin \
        --steps 12000 --max-lr 2e-4 \
        --out-dir models/experiments/teacher_student_hybrid_folding/noir_finetune_v8
"""
import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def mask_spans(x, mask_token_id, mask_prob=0.15, min_span=3, max_span=10):
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


def fixed_normalize(x, eps=1e-6):
    return F.layer_norm(x, x.shape[-1:], weight=None, bias=None, eps=eps)


def cosine_loss(student_out, teacher_out, mask, temp=0.1):
    s = student_out[mask]
    t = teacher_out[mask]
    s = F.normalize(s, dim=-1, eps=1e-6)
    t = F.normalize(t, dim=-1, eps=1e-6)
    cos = (s * t).sum(dim=-1)
    return ((1.0 - cos) / temp).mean()


@torch.no_grad()
def update_teacher_ema(student, teacher, tau=0.99):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(tau).add_(s_param.data, alpha=1.0 - tau)


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_iters=200):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def load_models(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", MicaConfig())

    if "student" in ckpt:
        sd_student = ckpt["student"]
        sd_teacher = ckpt["teacher"]
    elif "model" in ckpt:
        sd_student = ckpt["model"]
        sd_teacher = ckpt["model"]
    else:
        sd_student = ckpt
        sd_teacher = ckpt

    wte_shape = sd_student.get("transformer.wte.weight")
    if wte_shape is not None:
        actual_vocab = wte_shape.shape[0]
        if actual_vocab != config.vocab_size:
            config.vocab_size = actual_vocab

    student = OrigamiTransformer(config).to(device)
    teacher = OrigamiTransformer(config).to(device)
    student.load_state_dict(sd_student, strict=False)
    teacher.load_state_dict(sd_teacher, strict=False)

    if "segments" in ckpt:
        restore_segments(student, ckpt["segments"])
        restore_segments(teacher, ckpt["segments"])

    student.lm_head.weight = student.transformer.wte.weight
    teacher.lm_head.weight = teacher.transformer.wte.weight

    print(f"  Layout: {student.layout}")
    print(f"  Params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")
    return student, teacher, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--train", type=str, default="data/train.bin")
    parser.add_argument("--val",   type=str, default="data/val.bin")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout",    type=float, default=0.05)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--steps",      type=int, default=12_000)
    parser.add_argument("--warmup",     type=int, default=500)
    parser.add_argument("--max-lr",     type=float, default=2e-4)
    parser.add_argument("--min-lr",     type=float, default=1e-6)
    parser.add_argument("--grad-clip",  type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--tau",        type=float, default=0.99)
    parser.add_argument("--temp",       type=float, default=0.1)
    parser.add_argument("--mask-prob",  type=float, default=0.15)
    parser.add_argument("--min-span",   type=int, default=3)
    parser.add_argument("--max-span",   type=int, default=10)
    parser.add_argument("--log-every",  type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=2_000)
    parser.add_argument("--out-dir", type=str,
                        default="models/experiments/teacher_student_hybrid_folding/noir_finetune_v8")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.upper()}")
    print(f"Pure noir fine-tune: {args.steps} steps @ max_lr={args.max_lr}")
    print(f"No curriculum. No Gutenberg. Just noir + moving teacher.")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    final_path = os.path.join(args.out_dir, "final.pt")

    student, teacher, config = load_models(args.ckpt, device)
    config.dropout = args.dropout
    mask_token_id = config.vocab_size

    # Reset teacher to student (co-evolve)
    print("Resetting teacher to student")
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    train_data = np.memmap(args.train, dtype=np.uint16, mode="r")
    val_data   = np.memmap(args.val,   dtype=np.uint16, mode="r")
    print(f"\nNoir: {len(train_data):,} train / {len(val_data):,} val")

    decay = [p for n, p in student.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in student.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.max_lr, betas=(0.9, 0.95),
    )

    student.train()
    t0 = time.perf_counter()
    ce_acc = 0.0
    lat_acc = 0.0

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        X, Y = get_batch(train_data, args.batch_size, args.block_size, device)
        masked_X, mask = mask_spans(X, mask_token_id, args.mask_prob,
                                    args.min_span, args.max_span)

        # Student forward (masked)
        s_emb = student.transformer.drop(student.transformer.wte(masked_X))
        s_h = s_emb
        for phys_idx, loop_count in student.segments:
            block = student.transformer.h[phys_idx]
            for loop_idx in range(loop_count):
                s_h = block(s_h, start_pos=0, loop_idx=loop_idx)
        s_latent = student.transformer.ln_f(s_h)
        s_logits = student.lm_head(s_latent)

        # Teacher forward (unmasked)
        with torch.no_grad():
            t_emb = teacher.transformer.drop(teacher.transformer.wte(X))
            t_h = t_emb
            for phys_idx, loop_count in teacher.segments:
                block = teacher.transformer.h[phys_idx]
                for loop_idx in range(loop_count):
                    t_h = block(t_h, start_pos=0, loop_idx=loop_idx)
            t_latent = teacher.transformer.ln_f(t_h)

        # Losses
        targets = Y[:, :-1].contiguous()
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        update_teacher_ema(student, teacher, tau=args.tau)

        ce_acc += ce_loss.item()
        lat_acc += lat_loss.item()

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f"Step {step + 1:>5}: ce {ce_acc/n:.4f}  lat {lat_acc/n:.4f}  "
                  f"lr {lr:.2e}  ({dt:.1f}s / {n})")
            ce_acc = 0.0
            lat_acc = 0.0

        if (step + 1) % args.eval_every == 0:
            val_loss = estimate_loss(student, val_data, args.batch_size, args.block_size, device, eval_iters=100)
            print(f"  → val {val_loss:.4f}  (ppl {math.exp(val_loss):.1f})")
            student.train()

        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "model": student.state_dict(),
                "config": config,
                "step": step + 1,
                "layout": student.layout,
                "segments": student.segments,
            }, ckpt_path)
            print(f"  → checkpoint saved")

    torch.save({
        "model": student.state_dict(),
        "config": config,
        "layout": student.layout,
        "segments": student.segments,
    }, final_path)
    print(f"\nFine-tuning complete. Saved to {final_path}")


if __name__ == "__main__":
    main()
