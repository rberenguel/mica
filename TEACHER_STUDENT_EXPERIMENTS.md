# Teacher-Student (data2vec-style) Experiments

## Goal
Investigate whether masked latent-prediction pretraining (student predicts teacher EMA latents) can learn useful language representations on Mica's 5-layer Origami blocks, as a potential replacement or complement to standard next-token prediction.

---

## Experiment v1 — MSE + Fixed LayerNorm + Individual Token Masking

**Date:** 2026-06-03  
**Checkpoint:** `models/experiments/teacher_student/ckpt_.pt` (step 35000)  
**Status:** ❌ **Failed** — representation collapse (loss → 0.09, model learned to output near-constant vectors)

### Architecture
- 5 layers × 192 dim, OrigamiBlocks with `max_loops=1` (no folding)
- Student + Teacher (identical), EMA update `τ = 0.999`
- Shared learnable `LayerNorm` before MSE
- Mask: 15% random individual tokens

### Problem
The learnable `LayerNorm` was optimized to drive MSE to zero by collapsing all representations to near-identical vectors. No asymmetry between student and teacher meant the student trivially copied the teacher.

---

## Experiment v2 — Fixed LayerNorm + Student Predictor MLP

**Date:** 2026-06-03  
**Status:** 🟡 **Partial** — no collapse, but representations are random at next-token prediction

### Fixes applied
- Replaced learnable `LayerNorm` with `F.layer_norm(..., weight=None, bias=None)` — fixed normalization, no parameters
- Added `StudentPredictor` (2-layer MLP, GELU, dropout) — only on student, BYOL/data2vec anti-collapse asymmetry
- Optimizer on student + predictor only

### Result
No collapse (loss ~1.9–2.0 stable). Loaded student into `OrigamiTransformer` with fresh LM head:
- Gutenberg val: **8.53 loss / 5066 ppl** (random init ≈ 8.52)
- Noir val: **8.53 loss / 5073 ppl** (random init ≈ 8.52)

**Interpretation:** The student learned to match teacher latents at masked positions, but the representations encode no language structure. The task was too easy — the student finds a degenerate "context average" solution.

---

## Experiment v3 — Cosine Similarity + Faster EMA + Span Masking

**Date:** 2026-06-03  
**Status:** ❌ **Failed** — still random at next-token prediction

### Fixes applied (over v2)
1. **Loss:** Replaced MSE with **cosine similarity loss** (`(1 - cos) / temp`, temp=0.1). Forces directional alignment; the model cannot collapse by shrinking magnitude.
2. **EMA:** Speed up from `τ = 0.999` to **`τ = 0.99`**. The teacher is now meaningfully older than the student, creating a harder prediction target.
3. **Masking:** Replaced random individual tokens with **span masking** — contiguous spans of 3–10 tokens, ~15% coverage. Forces the student to predict from broader context.
4. Kept the `StudentPredictor` MLP and fixed `F.layer_norm` from v2.

### Run command
```bash
uv run python src/train_teacher_student.py \
  --steps 30000 --warmup 2000 \
  --eval-every 1000 --ckpt-every 5000 --log-every 500 \
  --tau 0.99 --temp 0.1 --mask-prob 0.15 --min-span 3 --max-span 10 \
  --fresh
```

### Result
Gutenberg val: **8.57 loss / 5274 ppl** (random init ≈ 8.52 / 5066)

**Interpretation:** Even with cosine loss, faster EMA, and span masking, the student solves the latent prediction task without learning language structure. The representations are not degenerate (cosine loss decreases), but they are **task-orthogonal** — they encode whatever the teacher-EMA dynamics reward, not lexical or syntactic structure.

---

## Experiment v4 — Hybrid: Masked Next-Token + Latent Regulariser

**Date:** 2026-06-03  
**Status:** 🟢 **Working** — in progress, partial results promising

### Insight from v1–v3
Pure latent prediction is too indirect. The student can satisfy the loss (match teacher vectors) without ever needing to know which token is which. The next-token task is the only thing that forces actual language learning.

### Fix
**Combine both objectives:**
- **Primary:** Standard next-token cross-entropy on UNMASKED positions (like GPT).
- **Secondary:** Cosine latent-prediction loss on MASKED positions (teacher sees full sequence, student sees masked spans).

### Results

**Final (30k steps):**
```
CE loss:    3.6
Lat loss:   5.4  (stuck ~5.3 for a while, bounded oscillation)
Val loss:   2.89  →  ppl 18.0
```

Compare to pure-latent v3 after 35k steps: **ppl 5274** (random).  
The hybrid after 30k steps is **three orders of magnitude better**.

**Mid-run trajectory:**
```
Step   1000: ce 5.12  lat 4.55  →  val 4.46  (ppl 86.5)
Step   2000: ce 4.41  lat 4.98  →  val 3.82  (ppl 45.4)
```

The CE loss is the driver — it drops from 8.52 (random) to 2.89 in 30k steps. The latent loss settled into a bounded oscillation (~4.8–5.4), acting as a stable regulariser without dominating the gradient landscape. The val loss of **2.89** is competitive with the standard next-token baseline and suggests the model has learned genuine grammatical and lexical structure.

### Run command
```bash
rm -f models/experiments/teacher_student_hybrid/ckpt.pt

uv run python src/train_teacher_student_hybrid.py \
  --steps 30000 --warmup 2000 \
  --eval-every 1000 --ckpt-every 5000 --log-every 500 \
  --tau 0.99 --temp 0.1 --latent-weight 0.1 \
  --fresh
```

### Eval command (after training)
```bash
uv run python src/finetune_teacher_student.py \
  --student-ckpt models/experiments/teacher_student_hybrid/ckpt.pt \
  --eval-only
```

---

## Experiment v5 — Teacher-Student Post-Training on Noir

**Date:** 2026-06-03  
**Status:** ❌ **Failed** — domain mismatch prevents style transfer

### Idea
Keep the teacher-student system active during noir post-training. Hypothesis: the latent regulariser would prevent overfitting and improve stylistic transfer.

### Command
```bash
uv run python src/train_teacher_student_hybrid.py \
  --resume models/experiments/teacher_student_hybrid/ckpt.pt \
  --reset-optimizer --reset-step \
  --data data/train.bin --val data/val.bin \
  --out-dir models/experiments/teacher_student_hybrid_noir \
  --steps 15000 \
  --warmup 500 \
  --max-lr 5e-5 --min-lr 1e-6 \
  --eval-every 500 --ckpt-every 2000 --log-every 200 \
  --tau 0.99 --temp 0.1 --latent-weight 0.1 \
  --mask-prob 0.15 --min-span 3 --max-span 10
```

### Result
**Val loss stuck at ~4.5** — significantly worse than standard fine-tuning (3.61) and even worse than Gutenberg-only (2.89).

### Why it failed
**Domain mismatch.** The teacher is an EMA of the Gutenberg-pretrained student. During noir training, the teacher is still ~99% "Gutenberg" (τ=0.99 means a ~100-step lag). The latent regulariser forces the student to align with a teacher that hasn't adapted to noir, creating a **domain anchor** that actively resists the style shift. The student is being pulled back toward Gutenberg representations while trying to learn noir.

**Generation confirms this:** outputs are garbled, mixed-domain artifacts (e.g. "That is about one of the kinds of people who did not think it was the audience I can style above" — clearly Gutenberg syntax struggling with noir vocabulary).

### Comparison
| Model | Gutenberg val | Noir val | Style |
|---|---|---|---|
| v4 Gutenberg only | 2.89 | — | Grammatical, literary |
| v4 Standard noir fine-tune | — | 3.61 | Short, punchy, noir-ish |
| v5 Teacher-student noir post-train | — | 4.5 | Confused, mixed-domain |

### Conclusion
**The latent regulariser is a domain anchor.** When the target domain is the same as pretraining (Gutenberg → Gutenberg), it works. When the target domain shifts (Gutenberg → Noir), the teacher-student system actively resists adaptation.

### Recommended use
- **Use hybrid pretraining on the same corpus** (or a very similar domain) where the teacher and student should share the same latent manifold.
- **Do NOT use teacher-student during fine-tuning** across domains with the old teacher frozen. Standard fine-tuning is superior for domain shift.
- **Future test:** Reset teacher to student at start of fine-tuning, so both co-evolve on noir. The teacher becomes a smoothed, slightly older view of the noir-adapting student — a "noir anchor" rather than a "Gutenberg anchor."

### v5b — Teacher Reset for Noir Co-Evolution

**Date:** TBD  
**Status:** 🔄 **Next attempt**

**Fix:** At the start of noir fine-tuning, copy the student into the teacher. Both networks begin from the same Gutenberg foundation. The teacher then adapts to noir via EMA, creating a **noir-aligned latent target** instead of a Gutenberg anchor.

```bash
uv run python src/train_teacher_student_hybrid.py \
  --resume models/experiments/teacher_student_hybrid/ckpt.pt \
  --reset-teacher --reset-optimizer --reset-step \
  --data data/train.bin --val data/val.bin \
  --out-dir models/experiments/teacher_student_hybrid_noir_v2 \
  --steps 15000 \
  --warmup 500 \
  --max-lr 5e-5 --min-lr 1e-6 \
  --eval-every 500 --ckpt-every 2000 --log-every 200 \
  --tau 0.99 --temp 0.1 --latent-weight 0.1 \
  --mask-prob 0.15 --min-span 3 --max-span 10
```

**Hypothesis:** The teacher will now be a smoothed, slightly older version of the noir-adapting student. The latent regulariser aligns noir-with-noir, preventing overfitting without pulling back toward Gutenberg. This should outperform both standard fine-tuning and the frozen-teacher v5.  

---

## Cleanup

**2026-06-03:** Removed META_STEM (MoE/fat router) experiment files:
- `src/model_meta_stem.py`, `src/train_meta_stem.py`, `src/generate_meta_stem.py`
- `META_STEM.md`
- `logs/meta_stem/` (all CSV logs)
- `models/*meta_stem*` (16 checkpoint files, ~170MB)

The MoE approach did not produce competitive results. The project now focuses on the teacher-student hybrid architecture.

---

## Future Directions

- **Same-domain teacher-student throughout:** Pretrain + post-train on Gutenberg with teacher-student active, then standard fine-tune on noir. Test if the regulariser produces a better foundation.
- **Deeper EMA schedules:** Start with fast EMA (`τ=0.9`) and anneal to slow (`τ=0.999`) over training.
- **Target network momentum restart:** Periodically reset the teacher to the current student.
- **Fold the teacher-student model:** Apply origami folding to the hybrid architecture and observe if the latent regulariser enables better compression.
- **Lower latent weight on fine-tuning:** Test `--latent-weight 0.02` during noir to reduce the regulariser's grip while keeping structural coherence.
- **Scale up:** Try the hybrid on the full 10-layer Mica geometry with origami folding.
