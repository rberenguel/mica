# Session Compaction Summary

## User Intent
- Explore a teacher-student (data2vec-style) latent prediction objective on Mica's Origami blocks
- Use 5 layers with no folding initially, then investigate folding combined with the hybrid objective
- Investigate whether the student-teacher paradigm improves representation quality or generation coherence

## Contextual Work Summary

### Pure Latent Experiments (v1–v3) — Documented Failures
- v1: MSE with learnable LayerNorm → representation collapse (loss → 0.09)
- v2: Fixed LayerNorm + student predictor MLP → no collapse, but next-token ppl ~5066 (random)
- v3: Cosine similarity + faster EMA (τ=0.99) + span masking → still random at next-token (~5274 ppl)
- **Conclusion:** Pure latent prediction learns to match teacher vectors without encoding language structure

### Hybrid Objective (v4) — Breakthrough
- Combined next-token CE on unmasked positions + cosine latent loss on masked spans
- EMA teacher (τ=0.99), span masking (3–10 tokens), fixed LayerNorm, student predictor MLP
- **Result:** val 2.89 / ppl 18.0 on Gutenberg after 30k steps — genuine language learning
- **Key insight:** The CE loss anchors the model to language; the latent loss is a regulariser

### Noir Fine-Tuning (v4/v5)
- Standard fine-tuning (no teacher-student) on noir → val 3.61 / ppl 37.1, coherent generation
- Teacher-student noir post-training (frozen teacher) → val 4.5 (worse), domain anchor problem
- Teacher-student noir with reset teacher (v5b) → val ~4.5 but **best generation quality**, zero memorisation
- **Conclusion:** Teacher-student during fine-tuning across domains is harmful; standard fine-tuning is superior

### Web Integration
- Exported v5b noir model to ONNX (`mica_small_teacher_student.onnx`)
- Added third model button to generator and trace viewer
- Added example prompt buttons (10 noir staples)
- Wrote memorisation checker (`src/check_memorisation.py`)

### Folding + Hybrid (v6) — In Progress
- Created `src/train_teacher_student_hybrid_folding.py` with soft-fold warmup
- Key innovation: gradual weight interpolation before hard aliasing
- Default curriculum: 5 phases with 33% fold warmup
- Currently training from scratch on Gutenberg with 30k+15k×3+25k phase split
- Regenerated `data/wiki_train.bin` / `data/wiki_val.bin` after they went missing

### Planning
- Wrote `PLAN_STACKED_TEACHERS.md`: two-teacher hierarchy (Anchor τ=0.999 + Buffer τ=0.99, optional curriculum annealing)
- Wrote `TEACHER_STUDENT_EXPERIMENTS.md`: full experiment log with all results and lessons

## Files Touched

### Core Scripts
- **`src/train_teacher_student.py`**: Pure latent (v1–v3) — documented failure baseline
- **`src/train_teacher_student_hybrid.py`**: Working hybrid (v4/v5) — CE + cosine
- **`src/train_teacher_student_hybrid_folding.py`**: Hybrid + origami folding with soft fold warmup (v6)
- **`src/finetune_teacher_student.py`**: Load student → attach LM head → eval or fine-tune
- **`src/check_memorisation.py`**: Verbatim corpus matching detector
- **`src/generate_origami.py`**: Auto-detects ffn_ratio and teacher-student checkpoints

### Web / ONNX
- **`llm/mica_small_teacher_student.onnx`**: Inference model (v5b noir)
- **`llm/mica_small_teacher_student_trace.onnx`**: Trace viewer model
- **`llm/index.html`**: Added third model button + 10 example prompt buttons
- **`llm/trace.html`**: Added third model button
- **`llm/mica.js`**: Example prompt button handlers

### Documentation
- **`TEACHER_STUDENT_EXPERIMENTS.md`**: Canonical experiment log (v1–v6)
- **`PLAN_STACKED_TEACHERS.md`**: Two-teacher hierarchy design doc
- **`README.md`**: Minor updates

### Cleanup
- Deleted `src/model_meta_stem.py`, `src/train_meta_stem.py`, `src/generate_meta_stem.py`, `META_STEM.md`, `logs/meta_stem/`, `models/*meta_stem*` (16 files)
- Deleted failed experiment leftovers: `models/experiments/teacher_student/`, `models/experiments/teacher_student_hybrid_noir/`
