#!/usr/bin/env python3
"""
Fine-tune a Teacher-Student Hybrid pretrained model with LoRA and curriculum mixing.

Changes from v1:
  - Slow noir-ratio rampup (starts low, ramps to target over N steps)
  - Lower default LR for LoRA phase
  - Slightly higher dropout for the tiny corpus
  - Slightly longer head-only warmup

Usage:
    uv run python src/finetune_teacher_student_lora_v2.py \
        --ckpt models/experiments/teacher_student_hybrid_folding/ckpt.pt \
        --noir-train data/train.bin --noir-val data/val.bin \
        --gb-train data/wiki_train.bin --gb-val data/wiki_val.bin \
        --steps 8000 --head-steps 1500 --mix-ramp-steps 3000 \
        --lora-r 4 --lora-alpha 8 \
        --noir-ratio 0.90 --max-lr 5e-5 \
        --out-dir models/experiments/teacher_student_hybrid_folding/noir_finetune_v2
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
from model import MultiResolutionAttention


# ═══════════════════════════════════════════════════════════════════════════════
#  LoRA injection
# ═══════════════════════════════════════════════════════════════════════════════

def inject_lora_attn(attn: MultiResolutionAttention, r: int, alpha: float = 1.0):
    names = ['q_large', 'k_large', 'v_large',
             'q_med',   'k_med',   'v_med',
             'q_small', 'k_small', 'v_small']
    device = next(attn.parameters()).device
    for name in names:
        base = getattr(attn, f'W_{name}')
        lora_A = nn.Parameter(torch.zeros(base.in_features, r, device=device))
        lora_B = nn.Parameter(torch.zeros(r, base.out_features, device=device))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.zeros_(lora_B)
        attn.register_parameter(f'lora_A_{name}', lora_A)
        attn.register_parameter(f'lora_B_{name}', lora_B)

    attn.lora_r = r
    attn.lora_alpha = alpha
    attn.lora_scaling = alpha / r

    def lora_forward(self, x, start_pos=0):
        B, T, _ = x.size()
        dp = self.dropout if self.training else 0.0

        def lora_linear(x, base, a_name, b_name):
            out = base(x)
            if hasattr(self, a_name):
                A = getattr(self, a_name)
                B = getattr(self, b_name)
                out = out + self.lora_scaling * (x @ A @ B)
            return out

        q_l = lora_linear(x, self.W_q_large, 'lora_A_q_large', 'lora_B_q_large').view(B, T, 1, 64).transpose(1, 2)
        k_l = lora_linear(x, self.W_k_large, 'lora_A_k_large', 'lora_B_k_large').view(B, T, 1, 64).transpose(1, 2)
        v_l = lora_linear(x, self.W_v_large, 'lora_A_v_large', 'lora_B_v_large').view(B, T, 1, 64).transpose(1, 2)
        q_l, k_l = self.rope_large(q_l, k_l, start_pos)
        out_l = F.scaled_dot_product_attention(q_l, k_l, v_l, dropout_p=dp, is_causal=True)
        out_l = out_l.transpose(1, 2).contiguous().view(B, T, 64)

        q_m = lora_linear(x, self.W_q_med, 'lora_A_q_med', 'lora_B_q_med').view(B, T, 2, 32).transpose(1, 2)
        k_m = lora_linear(x, self.W_k_med, 'lora_A_k_med', 'lora_B_k_med').view(B, T, 2, 32).transpose(1, 2)
        v_m = lora_linear(x, self.W_v_med, 'lora_A_v_med', 'lora_B_v_med').view(B, T, 2, 32).transpose(1, 2)
        q_m, k_m = self.rope_med(q_m, k_m, start_pos)
        out_m = F.scaled_dot_product_attention(q_m, k_m, v_m, dropout_p=dp, is_causal=True)
        out_m = out_m.transpose(1, 2).contiguous().view(B, T, 64)

        q_s = lora_linear(x, self.W_q_small, 'lora_A_q_small', 'lora_B_q_small').view(B, T, 8, 8).transpose(1, 2)
        k_s = lora_linear(x, self.W_k_small, 'lora_A_k_small', 'lora_B_k_small').view(B, T, 8, 8).transpose(1, 2)
        v_s = lora_linear(x, self.W_v_small, 'lora_A_v_small', 'lora_B_v_small').view(B, T, 8, 8).transpose(1, 2)
        q_s, k_s = self.rope_small(q_s, k_s, start_pos)
        out_s = F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=dp, is_causal=True)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, 64)

        return self.resid_dropout(self.W_o(torch.cat([out_l, out_m, out_s], dim=-1)))

    attn.forward = lambda x, start_pos=0, self=attn: lora_forward(self, x, start_pos)


def add_lora_to_model(model, r, alpha):
    for block in model.transformer.h:
        inject_lora_attn(block.attn, r, alpha)


def get_lora_params(model):
    params = []
    for block in model.transformer.h:
        for name in ['q_large', 'k_large', 'v_large',
                     'q_med',   'k_med',   'v_med',
                     'q_small', 'k_small', 'v_small']:
            a_name = f'lora_A_{name}'
            b_name = f'lora_B_{name}'
            if hasattr(block.attn, a_name):
                params.append(getattr(block.attn, a_name))
                params.append(getattr(block.attn, b_name))
    return params


def count_lora_params(model):
    return sum(p.numel() for p in get_lora_params(model))


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_mixed_batch(noir_data, gb_data, batch_size, block_size, device, noir_ratio=0.85):
    if torch.rand(1).item() < noir_ratio:
        return get_batch(noir_data, batch_size, block_size, device)
    else:
        return get_batch(gb_data, batch_size, block_size, device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Masking
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fixed_normalize(x, eps=1e-6):
    return F.layer_norm(x, x.shape[-1:], weight=None, bias=None, eps=eps)


def cosine_loss(student_out, teacher_out, mask, temp=0.1):
    s = student_out[mask]
    t = teacher_out[mask]
    s = F.normalize(s, dim=-1, eps=1e-6)
    t = F.normalize(t, dim=-1, eps=1e-6)
    cos = (s * t).sum(dim=-1)
    return ((1.0 - cos) / temp).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval
# ═══════════════════════════════════════════════════════════════════════════════

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


@torch.no_grad()
def estimate_loss_mixed(model, noir_data, gb_data, batch_size, block_size, device,
                        noir_ratio=0.85, eval_iters=100):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(noir_data, gb_data, batch_size, block_size, device, noir_ratio)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
#  LR schedule
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════════

def _add_mask_token(model, nominal_vocab):
    device = next(model.parameters()).device
    old_wte = model.transformer.wte
    new_wte = nn.Embedding(nominal_vocab + 1, old_wte.embedding_dim).to(device)
    new_wte.weight.data[:nominal_vocab] = old_wte.weight.data
    new_wte.weight.data[nominal_vocab] = 0.0
    model.transformer.wte = new_wte

    new_lm_head = nn.Linear(model.lm_head.in_features, nominal_vocab + 1, bias=False).to(device)
    new_lm_head.weight = model.transformer.wte.weight
    model.lm_head = new_lm_head


def load_models(ckpt_path, device):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", MicaConfig())

    nominal_vocab = config.vocab_size
    if "student" in ckpt:
        wte = ckpt["student"]["transformer.wte.weight"]
    else:
        wte = ckpt["model"]["transformer.wte.weight"]
    actual_vocab = wte.shape[0]

    nominal_vocab = ckpt.get("nominal_vocab", nominal_vocab)
    if actual_vocab != nominal_vocab:
        print(f"  Checkpoint vocab {actual_vocab} ≠ nominal {nominal_vocab}; using checkpoint")
        config.vocab_size = actual_vocab
    else:
        print(f"  Adding mask token embedding at index {nominal_vocab}")
        config.vocab_size = nominal_vocab

    student = OrigamiTransformer(config).to(device)
    teacher = OrigamiTransformer(config).to(device)

    if "student" in ckpt and "teacher" in ckpt:
        student.load_state_dict(ckpt["student"], strict=False)
        teacher.load_state_dict(ckpt["teacher"], strict=False)
        if "segments" in ckpt:
            restore_segments(student, ckpt["segments"])
            restore_segments(teacher, ckpt["segments"])
    else:
        student.load_state_dict(ckpt["model"], strict=False)
        teacher.load_state_dict(ckpt["model"], strict=False)
        if "segments" in ckpt:
            restore_segments(student, ckpt["segments"])
            restore_segments(teacher, ckpt["segments"])

    student.lm_head.weight = student.transformer.wte.weight
    teacher.lm_head.weight = teacher.transformer.wte.weight

    if actual_vocab == nominal_vocab:
        _add_mask_token(student, nominal_vocab)
        _add_mask_token(teacher, nominal_vocab)
        config.vocab_size = nominal_vocab + 1

    print(f"  Layout: {student.layout}")
    print(f"  Params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")
    return student, teacher, config, nominal_vocab


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--noir-train", type=str, default="data/train.bin")
    parser.add_argument("--noir-val",   type=str, default="data/val.bin")
    parser.add_argument("--gb-train",   type=str, default="data/wiki_train.bin")
    parser.add_argument("--gb-val",     type=str, default="data/wiki_val.bin")
    parser.add_argument("--noir-ratio", type=float, default=0.90)
    parser.add_argument("--mix-ramp-steps", type=int, default=6_000,
                        help="Steps over which noir ratio ramps from start_ratio to target")
    parser.add_argument("--mix-start-ratio", type=float, default=0.10,
                        help="Starting noir ratio (ramp begins here)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--min-span",  type=int, default=3)
    parser.add_argument("--max-span",  type=int, default=10)
    parser.add_argument("--lora-r",     type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--steps",       type=int, default=16_000)
    parser.add_argument("--head-steps",  type=int, default=3_000)
    parser.add_argument("--warmup",      type=int, default=200)
    parser.add_argument("--max-lr",      type=float, default=5e-5)
    parser.add_argument("--min-lr",      type=float, default=1e-6)
    parser.add_argument("--grad-clip",   type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--temp",        type=float, default=0.1)
    parser.add_argument("--log-every",   type=int, default=100)
    parser.add_argument("--eval-every",  type=int, default=500)
    parser.add_argument("--ckpt-every",  type=int, default=2_000)
    parser.add_argument("--out-dir", type=str,
                        default="models/experiments/teacher_student_hybrid_folding/noir_finetune_v2")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.upper()}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    final_path = os.path.join(args.out_dir, "final.pt")

    # ── Load models ──────────────────────────────────────────────────────────
    student, teacher, config, nominal_vocab = load_models(args.ckpt, device)
    config.dropout = args.dropout
    mask_token_id = nominal_vocab

    # ── Freeze teacher ───────────────────────────────────────────────────
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher frozen")

    # ── Inject LoRA ───────────────────────────────────────────────────────
    add_lora_to_model(student, args.lora_r, args.lora_alpha)
    lora_count = count_lora_params(student)
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}  ({lora_count:,} params)")

    # ── Freeze / unfreeze strategy ───────────────────────────────────────
    for p in student.parameters():
        p.requires_grad = False
    student.lm_head.weight.requires_grad = True
    student.transformer.ln_f.weight.requires_grad = True
    if student.transformer.ln_f.bias is not None:
        student.transformer.ln_f.bias.requires_grad = True

    head_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Phase 1 trainable: {sum(p.numel() for p in head_params):,} params")

    # ── Data ───────────────────────────────────────────────────────────────
    noir_train = np.memmap(args.noir_train, dtype=np.uint16, mode="r")
    noir_val   = np.memmap(args.noir_val,   dtype=np.uint16, mode="r")
    gb_train   = np.memmap(args.gb_train,   dtype=np.uint16, mode="r")
    gb_val     = np.memmap(args.gb_val,     dtype=np.uint16, mode="r")

    print(f"\nNoir: {len(noir_train):,} / {len(noir_val):,}")
    print(f"Gutenberg: {len(gb_train):,} / {len(gb_val):,}")
    print(f"Mix: {args.mix_start_ratio*100:.0f}% → {args.noir_ratio*100:.0f}% noir over {args.mix_ramp_steps:,} steps")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(head_params, lr=args.max_lr, betas=(0.9, 0.95))

    # ── Training loop ─────────────────────────────────────────────────────
    student.train()
    t0 = time.perf_counter()
    ce_acc = 0.0
    lat_acc = 0.0

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Phase transition
        if step == args.head_steps:
            print(f"\n{'='*60}")
            print(f"Phase transition @ step {step}: adding LoRA")
            lora_params = get_lora_params(student)
            for p in lora_params:
                p.requires_grad = True
            all_trainable = list(head_params) + lora_params
            print(f"  Now training: {sum(p.numel() for p in all_trainable):,} params")
            optimizer = torch.optim.AdamW(all_trainable, lr=lr, betas=(0.9, 0.95))
            print(f"{'='*60}\n")

        # Curriculum mix ratio
        if step < args.mix_ramp_steps:
            ratio = args.mix_start_ratio + (args.noir_ratio - args.mix_start_ratio) * (step / args.mix_ramp_steps)
        else:
            ratio = args.noir_ratio

        # Mixed batch
        X, Y = get_mixed_batch(noir_train, gb_train, args.batch_size, args.block_size,
                               device, ratio)

        # Mask spans
        masked_X, mask = mask_spans(X, mask_token_id, args.mask_prob,
                                    args.min_span, args.max_span)

        # ── Student forward ───────────────────────────────────────────────
        s_h = student.transformer.drop(student.transformer.wte(masked_X))
        for phys_idx, loop_count in student.segments:
            block = student.transformer.h[phys_idx]
            for loop_idx in range(loop_count):
                s_h = block(s_h, start_pos=0, loop_idx=loop_idx)
        s_latent = student.transformer.ln_f(s_h)
        s_logits = student.lm_head(s_latent)

        # ── Teacher forward ───────────────────────────────────────────────
        with torch.no_grad():
            t_h = teacher.transformer.drop(teacher.transformer.wte(X))
            for phys_idx, loop_count in teacher.segments:
                block = teacher.transformer.h[phys_idx]
                for loop_idx in range(loop_count):
                    t_h = block(t_h, start_pos=0, loop_idx=loop_idx)
            t_latent = teacher.transformer.ln_f(t_h)

        # ── Losses ─────────────────────────────────────────────────────────
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

        # ── Backward ───────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        ce_acc += ce_loss.item()
        lat_acc += lat_loss.item()

        # ── Logging ────────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            phase = "HEAD" if step < args.head_steps else "LoRA"
            print(f"Step {step + 1:>5} [{phase}] (noir={ratio:.0%}): ce {ce_acc/n:.4f}  lat {lat_acc/n:.4f}  "
                  f"lr {lr:.2e}  ({dt:.1f}s / {n})")
            ce_acc = 0.0
            lat_acc = 0.0

        # ── Eval ───────────────────────────────────────────────────────────
        if (step + 1) % args.eval_every == 0:
            noir_loss = estimate_loss(student, noir_val, args.batch_size, args.block_size, device, eval_iters=100)
            gb_loss   = estimate_loss(student, gb_val,   args.batch_size, args.block_size, device, eval_iters=100)
            mix_loss  = estimate_loss_mixed(student, noir_val, gb_val, args.batch_size, args.block_size,
                                            device, args.noir_ratio, eval_iters=100)
            print(f"  → noir {noir_loss:.4f}  gb {gb_loss:.4f}  mix {mix_loss:.4f}  (ppl {math.exp(mix_loss):.1f})")
            student.train()

        # ── Checkpoint ─────────────────────────────────────────────────────
        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "model": student.state_dict(),
                "config": config,
                "step": step + 1,
                "layout": student.layout,
                "segments": student.segments,
                "nominal_vocab": nominal_vocab,
            }, ckpt_path)
            print(f"  → checkpoint saved")

    # ── Final save ─────────────────────────────────────────────────────────
    torch.save({
        "model": student.state_dict(),
        "config": config,
        "layout": student.layout,
        "segments": student.segments,
        "nominal_vocab": nominal_vocab,
    }, final_path)
    print(f"\nFine-tuning complete. Saved to {final_path}")


if __name__ == "__main__":
    main()
