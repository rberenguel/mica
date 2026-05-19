# Plan: Origami v3.1 — Smart Fusion, LN-Aware Unfold, n=10 Scale-Up

## 1. Smart Fusion When Folding

**Problem:** When folding `h[0]` and `h[1]` into `h[0]*`, we currently keep `h[0]` as the anchor and discard `h[1]`'s learned weights. This wastes whatever `h[1]` discovered.

**Solution:** Precision-weighted fusion using AdamW's second-moment estimates (`exp_avg_sq`).

For each parameter tensor:
- Compute per-parameter precision: `p_i = 1 / (exp_avg_sq_i + ε)`
- Blend: `new_W = (W_0 · p_0 + W_1 · p_1) / (p_0 + p_1)`
- For scalar/vector params (biases, gains), fall back to simple mean

**When to apply:** Only at hard-fold transitions, after the soft-tie window (if any) and before creating aliases. The optimizer state is reset anyway, so we only need the weights themselves.

**Edge cases:**
- If both precisions are very low (high uncertainty), fall back to simple mean
- If one precision is >10× the other, trust the high-confidence block almost entirely

---

## 2. LN-Aware Unfolding

**Problem:** When unfolding `h[0]*` (which used `ln_1[0]` for loop_idx=0 and `ln_1[1]` for loop_idx=1) into `h[0]` + `h[1]`, both new blocks inherit `ln_1[0]`. But `h[1]`'s input is transformer output, not raw embeddings — its `ln_1[0]` has the wrong statistics. This creates a discontinuity that costs ~7–10K steps to recover from.

**Solution:** Map loop-specific LNs to new blocks based on their future role.

| Folded state | Unfolded state | h[0] inherits | h[1] inherits |
|---|---|---|---|
| `h[0]*` (2 loops) | `h[0]`, `h[1]` | anchor `ln_1[0]` | anchor `ln_1[1]` |
| `h[0]*` (3 loops) | `h[0]`, `h[1]`, `h[2]` | `ln_1[0]` | `ln_1[1]` | `ln_1[2]` |

Implementation: In `apply_layout()`, when creating a new block from an anchor that had `loop_count > 1`, copy the appropriate `ln_1[k]` and `ln_2[k]` into the new block's `ln_1[0]` and `ln_2[0]`. All other params (attention, MLP, depth_embed) still come from the anchor.

**Expected effect:** Unfold transitions should converge in ~2–3K steps instead of ~7–10K. The forward pass becomes continuous.

---

## 3. n=10 Curriculum Design

### Option A: Front-Only Folds, Inside-Out Unfold (Recommended)
Leave the last 4 layers (`h[6..9]`) completely untouched as a stable core. Fold progressively from the left, then unfold from the **inside out** — the key insight from the n=6 run.

**Why inside-out:** The front fold (h[0]) is the hardest to split because its two loops see completely different distributions (raw embeddings vs. transformer output). The near-core fold (h[4]) is the easiest because both loops operate on well-processed latent representations. By unfolding the easy splits first, the model builds capacity gradually. The hard front unfold happens last, when the deeper layers are already fully specialized.

```
Phase A: [10]           25K steps  all independent, baseline
Phase B: [*2 8]         20K steps  fold h[0],h[1]      (front)
Phase C: [*2 *2 6]      20K steps  fold h[2],h[3]      (middle)
Phase D: [*2 *2 *2 4]   20K steps  fold h[4],h[5]      (near-core)
Phase E: [*2 *2 6]      20K steps  unfold h[4],h[5]    (near-core first — easy)
Phase F: [*2 8]         20K steps  unfold h[2],h[3]    (middle — moderate)
Phase G: [10]           25K steps  unfold h[0],h[1]    (front last — hardest)
```

Total: **150K steps**. The untouched core `h[6..9]` (4 layers) trains continuously throughout.

**Key insight:** Phases E and F revisit layouts from Phases C and B, but with weights trained under deeper compression. The model decompresses back through known topologies, each time with a "wiser" initialization. The n=6 run showed that middle unfolds are almost spike-free; the LN fix should make the front unfold manageable too.

### Option B: Front-Then-Back (Experimental)
After front operations, also fold/unfold the back layers. This tests whether the LN fix makes back folds viable.

```
Phase A: [10]           20K steps
Phase B: [*2 8]         15K steps
Phase C: [*2 *2 6]      15K steps
Phase D: [2 *2 6]       15K steps  unfold front
Phase E: [2 *2 *2 4]    15K steps  fold back (h[8],h[9])
Phase F: [2 *2 2 *2 2]  15K steps  unfold back
Phase G: [10]           15K steps  full unfold
```

**Notation issue:** Our current `*N` syntax only supports left-to-right folding. Back folds need either:
- Right-side star: `4 *2` meaning "last 2 layers folded"
- Explicit position syntax: `*@8:2` meaning "fold 2 layers starting at position 8"

**Recommendation:** Run Option A first. Option B is only worth it if Option A shows clear benefit and we want to test whether back folds are now viable.

### Phase Duration Rationale
- Phase A (baseline): 25K steps. Gives n=10 time to find a stable basin before first fold
- Fold phases (B–D): 20K steps each. The n=6 run was still improving at step 15K; 20K gives headroom
- Partial unfold phases (E–F): 20K steps each. Even with LN fix, co-adaptation takes time
- Final unfold (G): 25K steps. Full unlock of all front layers — needs longest recovery
- Total 150K steps ≈ 6.5 epochs on 188M tokens. For ~4.8M average params, that's ~260 tokens/param — well into the underparameterized regime where more training helps
- Structural transitions act as hard regularisers; we can train longer without the usual overtraining risk because each fold/unfold forces forgetting

---

## 4. Control Run: No-Folding Baseline

**Purpose:** Establish whether the origami curriculum actually helps, or if a plain 10-layer model trained for the same total steps would reach the same (or better) loss.

**Setup:**
- Same model: 10 layers, same width, same data
- Same total steps as the origami run (150K for Option A)
- Single phase: `[10]` for all 150K steps
- Same LR schedule: 1e-3 → 1e-5 with 1200-step warmup

**Hypotheses:**
- **Worst case for origami:** Control reaches similar or better loss. This means folding provided no benefit — the model had enough capacity all along.
- **Best case for origami:** Control reaches similar loss but overtrains (train/val gap shrinks to near zero, or val stops improving while train keeps dropping). This means the origami's compression acts as a regularizer that prevents overfitting.
- **Expected:** Control might reach slightly better raw loss (more parameters active throughout), but with worse generalization on the small corpus.

**When to run:** After the origami run completes. Use the same random seed or at least compare final val losses and train/val gaps.

---

## 5. Implementation Order

1. **LN-aware unfold** — highest impact, simplest change. Update `apply_layout()` to remap LNs when unfolding.
2. **Smart fusion** — medium complexity. Add fusion helper called during fold transitions.
3. **Run n=6 verification** — quick 5-phase run with the fixes to confirm unfold speedup.
4. **Run n=10 Option A** — the main experiment.
5. **Run n=10 control** — baseline comparison.
6. **Option B** — only if results warrant it.

---

## Open Questions

1. Does smart fusion actually help, or is anchor-only sufficient because the loop-specific LNs do most of the separation work?
2. Does the LN fix make partial unfolds (like `[2 *2 6]`) converge as fast as full unfolds?
3. At n=10, is the untouched 4-layer core enough capacity to absorb 3 folded front blocks?
4. Does the control run overtrain, validating the origami as a regularizer?
