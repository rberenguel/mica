# Session Compaction Summary

## User Intent
- Fine-tune a pretrained teacher-student hybrid model (2.86 val loss on Gutenberg) for noir text generation
- Prevent catastrophic forgetting while absorbing the noir style
- Reproduce the "brilliant" prose quality of a previous model (v5b)

## Contextual Work Summary

### Pretraining Analysis
- The 2.86 val loss checkpoint is a **denoising autoencoder** (masked spans + cosine latent loss), not a standard causal LM
- Generation from the raw checkpoint is stilted — the model was never trained to predict next tokens on clean input
- The previous "brilliant" v5b model was also imperfect (~4.5 val loss, garbled output on inspection) but was overfitted, which gave it character

### Failed Fine-Tuning Strategies
- **v1/v2 (LoRA + frozen teacher):** Curriculum mixing too slow, model never learned noir
- **v3 (Pure next-token):** Dropped the teacher entirely; model lost all quality
- **v4/v5 (Frozen teacher + curriculum):** Teacher caused domain anchor problem — model bad at both domains
- **v6 (Moving teacher + next-token):** Teacher followed student but objective mismatch (unmasked input) caused confusion
- **v7 (Masked + moving teacher + curriculum):** Correct objective match but LR too low (5e-5), model underfit

### Key Insights
- The v5b "brilliant" model was **overfitting**, not well-generalized. The current approach is too conservative
- The correct objective is **masked input + next-token CE + cosine latent**, matching pretraining
- Pure noir with high LR (2e-4) and short training should force overfitting into coherence
- The frozen teacher causes domain anchoring; the moving teacher (reset to student) works better

### Final Script
- **v8 overfit script** (`src/finetune_noir_overfit.py`): High LR, pure noir, no curriculum, masked objective, moving teacher
- Ready to run from **v7 final.pt** (correct objective match, no distribution shift)

## Files Touched

### Fine-Tuning Scripts (all in `src/`)
- **`finetune_teacher_student_lora.py`**: v1 — LoRA + frozen teacher, failed
- **`finetune_teacher_student_lora_v2.py`**: v2 — doubled curriculum length, failed
- **`finetune_nexttoken_curriculum.py`**: v3 — pure next-token, failed
- **`finetune_nexttoken_teacher_curriculum.py`**: v4 — frozen teacher + next-token, failed
- **`finetune_nexttoken_teacher_phased.py`**: v5 — phased curriculum, failed
- **`finetune_nexttoken_moving_teacher.py`**: v6 — moving teacher, failed
- **`finetune_masked_hybrid_moving_teacher.py`**: v7 — masked objective, moving teacher, underfit
- **`finetune_noir_overfit.py`**: v8 — aggressive overfit script, NOT YET RUN

### Generation
- **`generate_origami.py`**: Added `nominal_vocab` handling for fine-tuned checkpoints
- Tested multiple checkpoints to diagnose generation quality

### Directories
- `models/experiments/teacher_student_hybrid_folding/noir_finetune_v{1,2,3,4,5,6,7}/` — all failed runs
- `models/experiments/teacher_student_hybrid_folding/ckpt.pt` — the original pretraining checkpoint

## Next Steps
- Run v8 from v7: `uv run python src/finetune_noir_overfit.py --ckpt models/experiments/teacher_student_hybrid_folding/noir_finetune_v7/final.pt ...`
- If v8 still fails, consider retraining the entire model from scratch with standard causal LM pretraining
- Or accept the model as a representation learner, not a text generator
