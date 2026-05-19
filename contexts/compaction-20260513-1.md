# Session Compaction Summary — Asymmetric Hourglass v3

## User Intent

- Replace symmetric `[2+6+2]` hourglass notation with asymmetric `*N` notation where only early/left layers fold
- Run a fast n=6 experiment (`[6]` → `[*2 4]` → `[*2 *2 2]` → `[2 *2 2]` → `[6]`) to test whether partial unfolding works
- Understand why asymmetric unfolding struggles compared to symmetric unfolding
- Keep the code flexible enough to iterate quickly on curriculum design

## Contextual Work Summary

### Architecture Overhaul — Segment-Based Layout Engine
- Complete rewrite of `model_hourglass.py` around **segments**: list of `(physical_block_index, loop_count)` pairs
- `parse_layout()` parses `*N` notation (e.g. `*2 4` → 1 folded + 4 independent)
- `apply_layout()` handles arbitrary fold/unfold transitions: folds alias blocks, unfolds create fresh blocks from shared anchors
- `rebuild_aliases()` re-establishes parameter sharing after `load_state_dict`
- Backward-compat wrappers preserve old symmetric API (`apply_fold`, `apply_unfold`, `restore_folds`) so old checkpoints still load
- `HourglassTransformer.layout` renders in new notation; `segments` stored in checkpoints for resume

### Training Script Rewrite
- `train_hourglass_cyclic.py` now accepts `--phase "layout:steps[:lr]"` repeats for arbitrary curricula
- Default n=6 curriculum: 15k steps each phase, LR 1e-3 → 1e-5 with 1200-step warmup
- Saves `layout` + `segments` in checkpoints; resume works mid-phase or at boundaries
- Also handles old checkpoints (`bottom_loops`/`top_loops`) for backward compat

### Consumer Script Updates
- Updated `generate_hourglass.py`, `dpo_generate.py`, `dpo_train.py`, `train_hourglass.py`, `train_hourglass2.py`
- All now check for `segments` key first, fall back to `restore_folds()` for old checkpoints
- `config.py`: added `max_loops=8` to give headroom for deeper folds

### Muon Optimizer Research
- Researched Muon (Keller Jordan) — SGD-momentum + Newton-Schulz orthogonalization
- **Verdict: not suitable** for this project. Reasons: MPS `bfloat16` uncertainty; model too small (d=192); batch too small (8k tokens); cyclic transitions would be catastrophic with Muon's ~20× higher LR and lack of adaptive second moments; overhead ~12% at this scale vs <1% on A100s

### Fusion Strategy Discussion
- Discussed using AdamW momentum buffers (`exp_avg`, `exp_avg_sq`) to do better-than-anchor fusion when folding
- Considered: mean of weights + momentum, extrapolate-then-fuse, alignment-aware blending, per-parameter precision weighting
- **Decision: stick with anchor-only for this run.** Since AdamW restarts after every transition anyway, old momentum is irrelevant. The only thing that carries over is weight initialization.

### Experimental Results (n=6, in progress)
- Phase A `[6]`: val 3.40 — solid baseline
- Phase B `[*2 4]`: val 3.27 — **beat baseline**, fold was effective
- Phase C `[*2 *2 2]`: val 3.27 — flat, second fold added no value
- Phase D `[2 *2 2]` (partial unfold): **struggling at ~3.33**, flat for 6K+ steps

### Key Insight: The LayerNorm Discontinuity
- The reason asymmetric unfolding struggles: loop-specific LayerNorms break forward-pass continuity
- In `[*2 4]`, h[0]* uses `ln_1[0]` for loop 0 and `ln_1[1]` for loop 1
- When unfolded to `[2 4]`, h[1] inherits `ln_1[0]` but its input is h[0]'s output — **wrong statistics**
- Symmetric unfolds worked better because the expanded bridge (6→8 layers) had capacity to absorb the LN mismatch
- **Proposed fix for future runs:** LN-aware unfold — map `ln_1[loop_idx]` from the anchor to the new block's `ln_1[0]` based on which loop role it will play

## Files Touched

### Core Architecture
- **`model_hourglass.py`**: Complete rewrite — segment-based layout engine, `parse_layout()`, `apply_layout()`, `rebuild_aliases()`, backward-compat wrappers
- **`config.py`**: Added `max_loops=8`

### Training
- **`train_hourglass_cyclic.py`**: Rewritten for arbitrary phase-based layout curriculum with `--phase` CLI args
- **`train_hourglass.py`**: Updated checkpoint save/load for `segments` format
- **`train_hourglass2.py`**: Same checkpoint updates

### Generation & DPO
- **`generate_hourglass.py`**: Handles new `segments` checkpoints, falls back to old format
- **`dpo_generate.py`**: Same
- **`dpo_train.py`**: Same + saves `segments`/`layout` in output

### Checkpoints (not in git)
- `mica_hourglass_cyclic_n6_ckpt.pt` — current run, mid-Phase D
- `train_log_n6.csv` — live training log

## Critical Shared Understanding

1. **Notation:** `*N` means N logical layers collapsed into 1 physical block, looped N times. `[*2 4]` = h[0]* looped 2×, h[2..5] independent.
2. **Asymmetric folds work:** Phase B `[*2 4]` beat the `[6]` baseline (3.27 vs 3.40).
3. **Second fold is useless (at this scale):** Phase C `[*2 *2 2]` was flat — no improvement over `[*2 4]`.
4. **Asymmetric partial unfolding is broken:** Phase D `[2 *2 2]` is stuck ~0.06 above Phase B/C. Root cause identified as loop-specific LN mismatch on the newly unfolded block.
5. **LN-aware unfold is the likely fix:** When unfolding a block that played loop_idx=k in the folded state, give it the anchor's `ln_1[k]` (renamed to its `ln_1[0]`) instead of `ln_1[0]`.
6. **Data > architecture:** 6.6M tokens is still the ceiling. n=6 is a fast probe; n=10–12 would be the real test if the curriculum pattern works.

## Next Experiments (pending Phase E results)

1. **If Phase E `[6]` also struggles:** Implement LN-aware unfold and rerun the same curriculum
2. **If Phase E recovers:** The problem is specifically partial-unfold asymmetry; future curricula should avoid `2 *2 2` topology and unfold everything at once
3. **Skip second fold entirely:** Try `[6]` → `[*2 4]` → `[6]` — single fold, single unfold
4. **Scale to n=10:** `[10]` → `[*2 8]` → `[10]` with LN-aware unfold, if the simpler curriculum works
