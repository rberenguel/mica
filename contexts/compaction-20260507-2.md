# Session Compaction Summary — Hourglass Cyclic + Asymmetric Notation

## User Intent

- Understand whether cyclical fold/unfold training improves model quality on a tiny (5.4M param) noir-fiction language model
- Discover optimal folding topology: which layers to fold, how many, and in what order
- Develop a clearer notation for asymmetric folding (some layers loop, others never do)
- Move toward deeper models (n=10–12) where early/outer layers loop while inner/deeper layers remain independent forever

## Key Architectural Breakthrough

### New notation: `[*2 *2 8]` replaces `[1+8+3]`

**Old notation** (symmetric, confusing):
- `[1+8+1]` = bottom×1, bridge 8, top×1
- `[2+6+2]` = bottom×2, bridge 6, top×2
- Problem: symmetrical notation hides which side is folded. "Bottom" and "top" were ambiguous.

**New notation** (explicit, unambiguous):
- `[*2 *2 8]` for n=12: h[0] loops 2×, h[1] loops 2×, h[2]–h[9] independent, h[10]–h[11] independent
- `[*2 2 *2]` for n=6: h[0] loops 2×, h[1]–h[2] independent, h[5] loops 2×
- `[6]` for n=6: all independent
- `*` prefix marks a **looped layer** (left-to-right, early/outer/left side only)
- Everything **to the right** of starred layers is independent forever
- Inner/deeper layers (near final FFN) are NEVER touched by folding

**Why this matters:** The inner layers do the heavy lifting. Looping the outer layers acts like MQA (shared KV projections) — it forces the independent inner layers to learn richer, compressed representations. The early/left layers are "stupid" shared weights; the inner/right layers compensate.

### Layer roles (left-to-right in the stack)

| Position | Role | Fold resistance |
|---|---|---|
| h[0] (first) | Token embedding, local syntax | **Low** — easy to share |
| h[1] (second) | Phrase features | **Low** — easy to share |
| Middle h[2..k] | Rich abstract representations | **NEVER folded** — must stay independent |
| h[-2] (penultimate) | Context integration | **NEVER folded** — must stay independent |
| h[-1] (last) | Next-token prediction | **NEVER folded** — must stay independent |

**Empirical observation:** Folding the **inner/right/top** layers (near output) recovers significantly worse than folding the **outer/left/bottom** layers (near input). The output-side layers resist sharing because prediction preparation is a more distinct role than feature extraction.

## Cyclic Experiment Results (n=6)

### Curriculum tested

**First run** (ended folded `[2+2+2]`):
- A: `[1+4+1]` 15K steps — warm-up
- B: `[2+2+2]` 12K steps — fold
- C: `[1+4+1]` 12K steps — unfold
- D: `[2+2+2]` 9K steps — refold (too short)
- E: `[2+2+2]` 6K steps — cool-down (too short)

**Second run** (revised, currently ongoing):
- A: `[1+4+1]` 15K steps
- B: `[2+2+2]` 12K steps
- C: `[1+4+1]` 12K steps
- D1: `[2+3+1]` 12K steps — fold bottom only (outer/left)
- D2: `[2+2+2]` 12K steps — fold top (inner/right)
- E: `[1+4+1]` 12K steps — final unfold

### Key findings

1. **First fold (Phase B):** Clean spike, full recovery. Val returned to ~3.15.
2. **First unfold (Phase C):** Recovered **below** pre-fold levels. Val hit 3.09 — the unfold found a better basin than the initial warm-up. **Cyclic hypothesis partially validated.**
3. **Refold (Phase D):** Hung at ~3.22 — worse than pre-refold. The second fold degraded performance permanently.
4. **The "warm-up failure" pattern:** Every transition shows ~100 steps of nice improvement followed by a marked failure (~200–500 steps later), then eventual recovery. Root cause: AdamW momentum buffers immature during linear warmup. **Fix:** Longer per-phase warmup (600 → 1200 steps).
5. **Bottom/outer folds recover better than top/inner folds.** Output-side layers resist sharing.

### What works
- Single fold to `[2+2+2]` with high LR (1e-3): viable
- Unfold with high LR (1e-3): can beat pre-fold performance
- Longer phase durations (12K each): necessary for recovery

### What doesn't work
- Refolding after unfolding: permanently degrades performance
- Low LR during structural transitions: traps model in shared-optimum basin
- Symmetric folding of inner layers: worse recovery than outer-only folding

## Files Touched

### Architecture
- **`model_hourglass.py`**: Added `apply_fold_bottom()` and `apply_fold_top()` for asymmetric single-side folding. Fixed `apply_unfold()` to move new blocks to correct device (`new_block.to(device)`). Loop-specific LayerNorms remain the key innovation.

### Training
- **`train_hourglass_cyclic.py`**: Completely rewritten for configurable depth (`--n-layer`), base step scaling (`--base-steps`), soft-tie windows (`--soft-tie-steps`), phased refold (`--phased-refold`), and mid-phase resume support. CSV logging with per-block timing. Longer warmup (1200 steps). Final phase now always **unfolded**.

### Generation
- **`generate_hourglass.py`**: Added `--n-layer` override for non-standard depth checkpoints.

### Progress Tracking
- **`llm/progress.html`**: Added `?log=` URL parameter to switch CSV files. Fixed `ctx` scope bug.

### Docs
- **`hourglass_v2.md`**: Architecture rationale (slightly outdated now — refers to symmetric notation)
- **`contexts/compaction-20260507-1.md`**: Previous session summary

## Next Experiments (after current run finishes)

### 1. Asymmetric folding on n=10 or n=12

```
# n=12, 6.1M params
A: [12]       # all independent, 25K steps
B: [*2 10]    # h[0] loops 2x, h[1]-h[9] independent, h[10]-h[11] independent
C: [12]       # unfold, 20K steps
D: [*2 *2 8]  # h[0]×2, h[1]×2, h[2]-h[9] independent, h[10]-h[11] independent
E: [12]       # final unfold, 10K steps
```

No symmetric top folding. Only the left/early layers get looped. Inner layers (h[2] onward) never shared.

### 2. Deeper manifold with more independent bridge layers

- n=10: `[*2 8]` → 8 independent middle layers, 2 looped outer
- n=12: `[*2 *2 8]` → 8 independent middle layers, 2+2 looped outer
- n=14: `[*2 *2 10]` → 10 independent middle layers

More independent bridge = more capacity for the model to absorb the "stupid" shared outer layers.

### 3. Phase 2 (noir fine-tune)

After the best phase 1 checkpoint, run `train_hourglass2.py` on the expanded 6.6M-token noir corpus. Compare against `mica_hourglass_v2.pt` (previous best).

### 4. DPO round 2

Once a stable phase 2 model exists, regenerate candidates with the new base model and retrain DPO. The expanded corpus + better architecture should give stronger preference signal.

## Current Checkpoints

| File | State |
|---|---|
| `mica_hourglass_cyclic_n6_ckpt.pt` | Mid-run n=6 cyclic, Phase E (unfold), ~step 75K |
| `mica_hourglass_cyclic_n6.pt` | Previous n=6 run, ended folded `[2+2+2]` |
| `mica_hourglass_v2.pt` | Best model so far (10-layer `[2+6+2]`, noir phase 2) |
| `mica_dpo_v1.pt` | Best DPO checkpoint |

## Critical Shared Understanding

1. **Notation:** `[*2 *2 8]` means early/left/outer layers loop. Right side is independent forever. No more `[2+6+2]`.
2. **Never fold inner layers** (right side, near output). Only fold outer layers (left side, near input).
3. **Unfold at end.** The saved model should always be `[n]` (fully unfolded) for inference.
4. **High LR during transitions.** 1e-3 for folds/unfolds. Low LR only during cool-down.
5. **Long warmup.** 1200 steps minimum after any structural change.
6. **Data > architecture.** 6.6M tokens is still the bottleneck. n=12 with 8 independent bridge layers is worth testing, but don't expect miracles without more corpus.
