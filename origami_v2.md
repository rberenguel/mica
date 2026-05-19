# Origami v2 — `[2+6+2]` with Loop-Specific LayerNorms

## What changed and why

The first origami experiment (`v1`) folded all the way to `[3+4+3]` (6 physical layers, 10 functional passes). Results were underwhelming — the uneven multi-resolution heads were clearly doing good work (visible in the trace UI), but the folding didn't improve quality.

**Diagnosis:** `[3+4+3]` is too aggressive for a 10-layer, 192-dim model. With only 4 independent bridge blocks, the outer shared layers had to behave as 3 completely different transformations (loop indices 0, 1, 2), differentiated only by a single 192-dim depth-embedding vector added to the residual stream. That's a very weak conditioning signal.

**Fix:** Two changes:

1. **Revert final layout to `[2+6+2]`** — only one fold. This keeps 6 independent bridge blocks and only shares the very outer pair (bottom loops 2×, top loops 2×). Physical layers: 8. Functional depth: 10.

2. **Loop-specific LayerNorm parameters** — each `OrigamiBlock` now owns a `ModuleList` of LayerNorms, one per possible loop index. The shared attention and MLP weights operate in different normalised spaces on each pass, giving the shared block dramatically more expressiveness at negligible cost (~6K extra parameters total).

   Before: `self.ln_1 = nn.LayerNorm(192)`  → one scale/shift for all loop indices.
   After:  `self.ln_1 = ModuleList([LN(192), LN(192), LN(192)])` → independent scale/shift per loop.

   In a pre-norm Transformer, these LN parameters dominate block behaviour. Letting loop 0 and loop 1 have their own LNs is the cheapest way to make the shared weights feel like different layers.

## Files

| File | State |
|---|---|
| `model_origami.py` | **Rewritten** — `OrigamiBlock` takes `max_loops`, stores `ln_1`/`ln_2` as `ModuleList`, uses `loop_idx` to index them. All blocks init with `max_loops=3` for headroom (bridge blocks only ever use index 0). |
| `train_origami.py` | **Updated** — curriculum stops after first hard fold (`[1+8+1]` → `[2+6+2]`). Total steps: 40,500. Checkpoint names changed to `mica_origami_v2_ckpt.pt` / `mica_origami_v2_phase1.pt` to avoid accidentally resuming from old v1 checkpoints (state-dict keys are incompatible because LN keys changed from `ln_1.weight` to `ln_1.0.weight`). |
| `train_origami2.py` | **Updated** — loads `mica_origami_v2_phase1.pt`, outputs `mica_origami_v2.pt`. |
| `generate_origami.py` | **Updated** — default weights now `mica_origami_v2.pt`. Added `--min-new-tokens` (default 5) to prevent immediate EOS termination on terminal prompts. |

## Architecture at a glance (final state)

```
Bottom:  h[0]  ──► h[0]        (same physical block, loop_idx 0 then 1)
Bridge:  h[2] → h[3] → h[4] → h[5] → h[6] → h[7]   (6 independent blocks)
Top:     h[9]  ──► h[9]        (same physical block, loop_idx 0 then 1)
```

Physical blocks: 8.  Functional depth: 10.  Shared blocks each have 2 sets of LN params.

## How to train

```bash
# Phase 1 — Gutenberg pre-training (folds to [2+6+2])
uv run python train_origami.py

# Phase 2 — Noir fine-tuning (standard: 10k steps @ 1e-4)
uv run python train_origami2.py

# Phase 2 — Extended fine-tune (20k steps @ 5e-5)
# Use this if the standard phase 2 feels under-cooked.
uv run python train_origami2.py
#   → outputs mica_origami_v2_long.pt

# Generate
uv run python generate_origami.py --weights mica_origami_v2_long.pt \
    "The fat man leaned back in his chair and"
```

## Curriculum

| Phase | Steps | Layout |
|---|---|---|
| Warmup (independent) | 0 – 20,000 | `[1+8+1]` |
| Soft-tie window | 20,000 – 20,500 | `[1+8+1]` (weights copied, grads still independent) |
| Post-fold | 20,500 – 50,000 | `[2+6+2]` |

Total: **50,000 steps** (~5.4M params on 188M tokens).

## What to watch

- **Post-fold loss behaviour** — after step 20,500 (hard fold), the `[2+6+2]` layout should settle without a spike. The loop-specific LNs for `loop_idx=1` start from init (weight=1, bias=0) and adapt during post-fold training.
- **Layout string** in logs should read `2+6+2` after the fold.
- **Parameter count** drops from ~5.4M (full 10-layer) to ~5.2M after fold — the saving is tiny because the model is already small. The goal here is regularisation / manifold quality, not compression.

## If this doesn't work

If `[2+6+2]` with loop-specific LNs still underperforms the baseline 10-layer model, the origami idea itself may not suit this geometry. At that point, the best move is to drop folding entirely and invest the compute budget in:
- Longer baseline training, or
- The DPO / narrow-specialisation ideas from `fine_tune_ideas.md`.
