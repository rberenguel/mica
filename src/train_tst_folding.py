#!/usr/bin/env python3
"""
Token Superposition & Attention Folding Training Pipeline
for the 2.5M parameter Mica small model.

Phase 1 — Mixed Superposition (optional):
  Output-only superposition with decaying bag size (default 4 → 2)
  on a configurable Gutenberg:Noir mix. Model fully unfolded.

Phase 2 — Structural Foundation (Syntax Drilling):
  Standard next-token prediction on 100% Gutenberg.
  Symmetric origami fold/unfold curriculum:
    5 → *2 3 → *2 *2 1 → *2 3 → 5
  Each layout trained for ~15k steps by default.

Phase 3 — Stylistic Unlocking (Noir Fine-Tuning):
  Standard next-token prediction on 100% Noir.
  Model fully unfolded. WSD learning-rate schedule.
  Optional Gutenberg blend that decays rapidly to pure Noir.

Usage:
  uv run python src/train_tst_folding.py
  # Resume from existing checkpoint as fresh start:
  uv run python src/train_tst_folding.py \
    --weights models/experiments/small/tst_folding/ckpt.pt \
    --skip-phase1
"""
import argparse
import math
import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from config import MicaConfig
from model_origami import (
    OrigamiTransformer, parse_layout, apply_layout, restore_segments, _fuse_on_fold
)

# ═══════════════════════════════════════════════════════════════════════════════
#  MCE Loss — Output-Only Superposition
# ═══════════════════════════════════════════════════════════════════════════════

def curriculum_bag_loss(logits: torch.Tensor, labels: torch.Tensor,
                        bag_size: int, ignore_index: int = -1) -> torch.Tensor:
    """
    Output-Only Superposition loss with dynamic bag sizes.

    logits : (batch, seq_len, vocab_size)  —  unshifted model outputs
    labels : (batch, seq_len)               —  unshifted target indices
    bag_size: s  (if s <= 1, falls back to standard cross-entropy)
    """
    if bag_size <= 1:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
        )

    batch, seq, vocab = logits.shape
    offset = bag_size - 1
    flat_logits = logits.reshape(-1, vocab)               # (batch*seq, vocab)

    # Pad the label sequence so the last position can form a full bag
    padded = F.pad(labels, (0, offset), value=ignore_index)  # (batch, seq+offset)
    # Overlapping sliding windows → (batch, seq, bag_size)
    bagged = padded.unfold(dimension=-1, size=bag_size, step=1)

    loss = torch.tensor(0.0, device=logits.device)
    w_total = 0.0
    for i in range(bag_size):
        # Power-law weighting stabilises large bags; uniform for small ones
        w = 1.0 if bag_size < 8 else (1.0 / (i + 1))
        target = bagged[..., i].reshape(-1)               # (batch*seq,)
        # Skip positions where the entire batch is padding (prevents NaN
        # when seq_len < bag_size or at the tail of a short sequence)
        if target.eq(ignore_index).all():
            continue
        loss = loss + w * F.cross_entropy(flat_logits, target, ignore_index=ignore_index)
        w_total += w

    if w_total == 0:
        return loss
    return loss / w_total


# ═══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch_simple(data: np.memmap, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)


def get_batch_mixed(gb_data, nr_data, batch_size, block_size, device, gutenberg_prob):
    data = gb_data if random.random() < gutenberg_prob else nr_data
    return get_batch_simple(data, batch_size, block_size, device)


# ═══════════════════════════════════════════════════════════════════════════════
#  LR schedules
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, phase_steps: int, max_lr: float, min_lr: float,
           warmup_steps: int, schedule: str, prev_end_lr: float | None = None,
           decay_frac: float = 0.25) -> float:
    """
    Unified LR schedule.

    * cosine  — warmup + cosine decay to min_lr
    * wsd     — Warmup-Stable-Decay (cosine tail)

    If prev_end_lr is given, the first *warmup_steps* linearly interpolate
    from prev_end_lr → max_lr for a smooth phase transition.
    """
    if step < warmup_steps and prev_end_lr is not None:
        t = (step + 1) / warmup_steps
        return prev_end_lr + t * (max_lr - prev_end_lr)

    if schedule == "cosine":
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        ratio = (step - warmup_steps) / max(1, phase_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

    if schedule == "wsd":
        decay_steps = int(phase_steps * decay_frac)
        stable_steps = phase_steps - warmup_steps - decay_steps
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step < warmup_steps + stable_steps:
            return max_lr
        d_step = step - (warmup_steps + stable_steps)
        ratio = d_step / max(1, decay_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

    raise ValueError(f"Unknown schedule: {schedule}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval helper
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, bag_size, get_batch_fn, eval_iters=200, device="mps"):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            # Forward with targets to obtain full-sequence logits
            logits, _ = model(X, Y)
            losses[k] = curriculum_bag_loss(logits, Y, bag_size).item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_phases(args):
    phases = []

    # ── Phase 1: Mixed Superposition ─────────────────────────────────────────
    if not args.skip_phase1:
        for bag_size in args.bag_sizes:
            phases.append({
                "name": f"superposition_s{bag_size}",
                "bag_size": bag_size,
                "data_source": "mixed",
                "layout": "5",
                "steps": args.superposition_steps,
                "max_lr": args.p1_lr,
                "min_lr": args.p1_min_lr,
                "warmup": args.p1_warmup,
                "schedule": "cosine",
                "gutenberg_ratio": args.gutenberg_ratio,
            })

    # ── Phase 2: Gutenberg + symmetric fold/unfold ───────────────────────────
    # 5 → *2 3 → *2 *2 1 → *2 3 → 5
    p2_layouts = ["5", "*2 3", "*2 *2 1", "*2 3", "5"]
    p2_steps = [
        args.p2_base_steps,
        args.p2_fold1_steps,
        args.p2_fold2_steps,
        args.p2_unfold1_steps,
        args.p2_unfold2_steps,
    ]
    p2_names = [
        "gutenberg_base",
        "gutenberg_fold_2_3",
        "gutenberg_fold_2_2_1",
        "gutenberg_unfold_2_3",
        "gutenberg_unfold_full",
    ]
    for layout, steps, name in zip(p2_layouts, p2_steps, p2_names):
        phases.append({
            "name": name,
            "bag_size": 1,
            "data_source": "gutenberg",
            "layout": layout,
            "steps": steps,
            "max_lr": args.p2_lr,
            "min_lr": args.p2_min_lr,
            "warmup": args.p2_warmup,
            "schedule": "cosine",
        })

    # ── Phase 3: Noir fine-tune ──────────────────────────────────────────────
    phases.append({
        "name": "noir_finetune",
        "bag_size": 1,
        "data_source": "noir",
        "layout": "5",
        "steps": args.phase3_steps,
        "max_lr": args.p3_lr,
        "min_lr": args.p3_min_lr,
        "warmup": args.p3_warmup,
        "schedule": "wsd",
        "decay_frac": args.p3_decay_frac,
        "blend_start": args.blend_start,
        "blend_decay_steps": args.blend_decay_steps,
        "midpoint_save": args.phase3_steps // 2,
    })

    return phases


# ═══════════════════════════════════════════════════════════════════════════════
#  Optimiser factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_optimizer(model, max_lr):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay = [p for n, p in param_dict.items() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=max_lr, betas=(0.9, 0.95),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--n-layer", type=int, default=5)
    parser.add_argument("--ffn-ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", default=None)

    # Phase 1 — Mixed Superposition
    parser.add_argument("--superposition-steps", type=int, default=5000,
                        help="Steps per bag-size sub-phase")
    parser.add_argument("--bag-sizes", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--p1-lr", type=float, default=3e-4)
    parser.add_argument("--p1-min-lr", type=float, default=3e-4,
                        help="Set equal to --p1-lr for fixed LR during superposition")
    parser.add_argument("--p1-warmup", type=int, default=100)
    parser.add_argument("--gutenberg-ratio", type=float, default=1.0,
                        help="Gutenberg:Noir mix ratio (1 = 1:1, 10 = 10:1)")

    # Phase 2 — Gutenberg + symmetric fold/unfold
    parser.add_argument("--p2-base-steps", type=int, default=15000)
    parser.add_argument("--p2-fold1-steps", type=int, default=15000)
    parser.add_argument("--p2-fold2-steps", type=int, default=15000)
    parser.add_argument("--p2-unfold1-steps", type=int, default=15000)
    parser.add_argument("--p2-unfold2-steps", type=int, default=15000)
    parser.add_argument("--p2-lr", type=float, default=1e-3)
    parser.add_argument("--p2-min-lr", type=float, default=1e-5)
    parser.add_argument("--p2-warmup", type=int, default=400)

    # Phase 3 — Noir fine-tune
    parser.add_argument("--phase3-steps", type=int, default=20000)
    parser.add_argument("--p3-lr", type=float, default=5e-5)
    parser.add_argument("--p3-min-lr", type=float, default=1e-7)
    parser.add_argument("--p3-warmup", type=int, default=500)
    parser.add_argument("--p3-decay-frac", type=float, default=0.25,
                        help="WSD decay fraction")
    parser.add_argument("--blend-start", type=float, default=0.1,
                        help="Initial Gutenberg probability in Phase 3 (0 = none)")
    parser.add_argument("--blend-decay-steps", type=int, default=2000,
                        help="Steps over which blend decays to 0")

    # Training mechanics
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--weights", type=str, default=None,
                        help="Load initial weights from file, start fresh (no resume)")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 superposition entirely")
    parser.add_argument("--experiment", type=str, default="tst_folding_v2")

    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────────
    EXPERIMENT = args.experiment
    OUT_DIR = f"models/experiments/small/{EXPERIMENT}"
    CKPT_PATH = f"{OUT_DIR}/ckpt.pt"
    MID_PATH = f"{OUT_DIR}/mid.pt"
    FINAL_PATH = f"{OUT_DIR}/final.pt"
    LOG_CSV = f"logs/small/train_log_{EXPERIMENT}.csv"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs/small", exist_ok=True)

    phases = build_phases(args)
    total_steps = sum(p["steps"] for p in phases)

    # ── Model ────────────────────────────────────────────────────────────────
    config = MicaConfig()
    config.n_layer = args.n_layer
    config.ffn_ratio = args.ffn_ratio
    config.dropout = args.dropout
    if args.device:
        config.device = args.device
    device = config.device

    print(f"Initialising OrigamiTransformer ({config.n_layer} layers, "
          f"ffn_ratio={config.ffn_ratio}, dropout={config.dropout}) on {device.upper()}...")
    model = OrigamiTransformer(config)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M  layout={model.layout}")

    # Validate all layouts up-front
    for p in phases:
        parse_layout(p["layout"], config.n_layer)

    # ── Data ─────────────────────────────────────────────────────────────────
    gb_train = np.memmap("data/wiki_train.bin", dtype=np.uint16, mode="r")
    gb_val = np.memmap("data/wiki_val.bin", dtype=np.uint16, mode="r")
    nr_train = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    nr_val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    print(f"Gutenberg: {len(gb_train):,} train / {len(gb_val):,} val")
    print(f"Noir:      {len(nr_train):,} train / {len(nr_val):,} val")

    # ── Resume / init ────────────────────────────────────────────────────────
    start_phase = 0
    global_step = 0
    prev_end_lr = None
    optimizer = make_optimizer(model, phases[0]["max_lr"])

    if args.weights:
        # Fresh start from external weights (e.g. aborted run checkpoint)
        print(f"\nLoading weights from {args.weights}...")
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "segments" in ckpt:
            restore_segments(model, ckpt["segments"])
        print(f"  Loaded layout: {model.layout}")

        # Force-unfold to the first target layout of the new curriculum
        target = phases[0]["layout"]
        if model.layout != target:
            print(f"  Unfolding {model.layout} → {target}...")
            apply_layout(model, target)
            print(f"  Layout now: {model.layout}")
        optimizer = make_optimizer(model, phases[0]["max_lr"])

        # Copy old log entries (if any) so the visualiser shows continuous history.
        # Convert 8-column old format → 6-column new format on the fly.
        OLD_LOG = "logs/small/train_log_tst_folding.csv"
        if os.path.exists(OLD_LOG):
            print(f"  Copying historical log from {OLD_LOG}...")
            with open(OLD_LOG, "r", newline="") as old_f, \
                 open(LOG_CSV, "w", newline="") as new_f:
                reader = csv.reader(old_f)
                writer = csv.writer(new_f)
                for row in reader:
                    if len(row) == 8:
                        # old: step, phase, name, layout, bag_size, train, val, lr
                        writer.writerow([row[0], row[1], row[3], row[5], row[6], row[7]])
                    elif len(row) == 6:
                        writer.writerow(row)
            print(f"  Historical data written to {LOG_CSV}")
        elif os.path.exists(LOG_CSV):
            os.remove(LOG_CSV)

    elif os.path.exists(CKPT_PATH):
        print(f"\nResuming from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "segments" in ckpt:
            restore_segments(model, ckpt["segments"])
        global_step = ckpt["step"]
        saved_phase = ckpt.get("phase", 0)
        print(f"  Resumed at step {global_step}, phase {saved_phase}, layout={model.layout}")

        # If the saved phase is complete, transition to the next one
        cum_steps = sum(phases[i]["steps"] for i in range(saved_phase + 1))
        if global_step >= cum_steps:
            if saved_phase + 1 < len(phases):
                print(f"  Phase {saved_phase} complete. Transitioning...")
                target = phases[saved_phase + 1]["layout"]
                if model.layout != target:
                    try:
                        _fuse_on_fold(model, target, optimizer)
                    except Exception as e:
                        print(f"  Fusion skipped: {e}")
                    apply_layout(model, target)
                optimizer = make_optimizer(model, phases[saved_phase + 1]["max_lr"])
                start_phase = saved_phase + 1
                print(f"  Layout now: {model.layout}")
            else:
                print("Training already complete.")
                return
        else:
            start_phase = saved_phase
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"  Optimizer state incompatible ({e}), starting fresh.")
                optimizer = make_optimizer(model, phases[start_phase]["max_lr"])
    else:
        if os.path.exists(LOG_CSV):
            os.remove(LOG_CSV)

    # ── Print curriculum ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Training Curriculum  ({total_steps:,} total steps)")
    print(f"{'=' * 70}")
    for i, p in enumerate(phases):
        lr_info = f"LR {p['max_lr']:.0e}→{p['min_lr']:.0e}"
        data_info = p['data_source']
        if p['data_source'] == 'mixed':
            data_info += f" ({p['gutenberg_ratio']:.0f}:1)"
        sched = p['schedule']
        print(f"  Phase {i + 1}: {p['name']:22s}  {p['layout']:>8s}  "
              f"bag={p['bag_size']:<2d}  {p['steps']:>6,} steps  "
              f"{sched:6s}  {lr_info}  [{data_info}]")
    print(f"{'=' * 70}\n")

    last_eval_time = time.perf_counter()
    torch.manual_seed(1337 + global_step)

    # ═══════════════════════════════════════════════════════════════════════
    #  Training loop
    # ═══════════════════════════════════════════════════════════════════════
    for phase_idx, phase in enumerate(phases):
        if phase_idx < start_phase:
            # Still need to track prev_end_lr for smooth transition
            prev_end_lr = get_lr(
                phase["steps"] - 1, phase["steps"], phase["max_lr"],
                phase["min_lr"], phase["warmup"], phase["schedule"],
                prev_end_lr, phase.get("decay_frac", 0.25),
            )
            continue

        target_layout = phase["layout"]
        if model.layout != target_layout:
            print(f"\n  Transition: {model.layout} → {target_layout}")
            try:
                _fuse_on_fold(model, target_layout, optimizer)
            except Exception as e:
                print(f"  Fusion skipped: {e}")
            apply_layout(model, target_layout)
            optimizer = make_optimizer(model, phase["max_lr"])
            print(f"  Layout now: {model.layout}")

        bag_size = phase["bag_size"]
        phase_steps = phase["steps"]

        print(f"\n{'=' * 70}")
        print(f"Phase {phase_idx + 1}: {phase['name']}  "
              f"layout={model.layout}  bag={bag_size}  ({phase_steps:,} steps)")
        print(f"{'=' * 70}")

        # Derive starting position within the phase so LR schedule resumes
        # correctly after a mid-phase checkpoint.
        cum_before = sum(phases[i]["steps"] for i in range(phase_idx))
        step_in_phase_start = (
            global_step - cum_before if phase_idx == start_phase else 0
        )

        for step_in_phase in range(step_in_phase_start, phase_steps):
            # ── Eval & log ──────────────────────────────────────────────────
            if (global_step % args.eval_interval == 0 or
                    step_in_phase == phase_steps - 1):

                # Build eval sampler (pure source, no blend)
                def eval_fn(split):
                    src = phase["data_source"]
                    if src == "gutenberg":
                        data = gb_train if split == "train" else gb_val
                        return get_batch_simple(data, args.batch_size, config.block_size, device)
                    if src == "noir":
                        data = nr_train if split == "train" else nr_val
                        return get_batch_simple(data, args.batch_size, config.block_size, device)
                    # mixed
                    gb = gb_train if split == "train" else gb_val
                    nr = nr_train if split == "train" else nr_val
                    prob = phase["gutenberg_ratio"] / (phase["gutenberg_ratio"] + 1.0)
                    return get_batch_mixed(gb, nr, args.batch_size, config.block_size, device, prob)

                losses = estimate_loss(model, bag_size, eval_fn, args.eval_iters, device)
                lr = optimizer.param_groups[0]["lr"]
                now = time.perf_counter()
                block_secs = round(now - last_eval_time)
                last_eval_time = now
                print(f"Step {global_step:>6} [P{phase_idx + 1} {phase['name'][:10]:>10s}]: "
                      f"train {losses['train']:.4f}  val {losses['val']:.4f}  "
                      f"lr {lr:.2e}  ({block_secs}s)")
                with open(LOG_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, phase_idx, model.layout,
                        f"{losses['train']:.6f}", f"{losses['val']:.6f}",
                        f"{lr:.6e}",
                    ])

            # ── LR ──────────────────────────────────────────────────────────
            lr = get_lr(
                step_in_phase, phase_steps, phase["max_lr"], phase["min_lr"],
                phase["warmup"], phase["schedule"], prev_end_lr,
                phase.get("decay_frac", 0.25),
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ── Train step ──────────────────────────────────────────────────
            src = phase["data_source"]
            if src == "gutenberg":
                X, Y = get_batch_simple(gb_train, args.batch_size, config.block_size, device)
            elif src == "noir":
                # Blend handling for Phase 3
                blend = phase.get("blend_start", 0.0)
                if blend > 0 and step_in_phase < phase["blend_decay_steps"]:
                    gutenberg_prob = blend * max(0.0, 1.0 - step_in_phase / phase["blend_decay_steps"])
                    if random.random() < gutenberg_prob:
                        X, Y = get_batch_simple(gb_train, args.batch_size, config.block_size, device)
                    else:
                        X, Y = get_batch_simple(nr_train, args.batch_size, config.block_size, device)
                else:
                    X, Y = get_batch_simple(nr_train, args.batch_size, config.block_size, device)
            else:  # mixed
                prob = phase["gutenberg_ratio"] / (phase["gutenberg_ratio"] + 1.0)
                X, Y = get_batch_mixed(gb_train, nr_train, args.batch_size,
                                         config.block_size, device, prob)

            logits, _ = model(X, Y)          # full-sequence logits; model CE ignored
            loss = curriculum_bag_loss(logits, Y, bag_size)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1

            # ── Checkpoint ──────────────────────────────────────────────────
            if global_step % args.ckpt_interval == 0 and global_step > 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "phase": phase_idx,
                    "layout": model.layout,
                    "segments": model.segments,
                }, CKPT_PATH)

            # Phase 3 midpoint save
            if phase.get("midpoint_save") and step_in_phase == phase["midpoint_save"]:
                torch.save({
                    "model": model.state_dict(),
                    "layout": model.layout,
                    "segments": model.segments,
                }, MID_PATH)
                print(f"  → Midpoint saved to {MID_PATH}")

        # ── End-of-phase bookkeeping ────────────────────────────────────────
        prev_end_lr = get_lr(
            phase_steps - 1, phase_steps, phase["max_lr"], phase["min_lr"],
            phase["warmup"], phase["schedule"], prev_end_lr,
            phase.get("decay_frac", 0.25),
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Final save
    # ═══════════════════════════════════════════════════════════════════════
    torch.save({
        "model": model.state_dict(),
        "layout": model.layout,
        "segments": model.segments,
    }, FINAL_PATH)
    print(f"\nTraining complete. Saved to {FINAL_PATH}  layout={model.layout}")


if __name__ == "__main__":
    main()
