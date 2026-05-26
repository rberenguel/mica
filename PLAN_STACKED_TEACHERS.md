# Stacked Teachers: Anchor → Buffer → Student

## Motivation

The current teacher-student system uses a single EMA teacher (τ=0.99). The teacher is an exponentially-smoothed copy of the student, lagging by roughly 100 steps. This works well, but it has a structural limitation: the teacher is **too reactive**. When the student makes a rapid learning leap, the teacher quickly follows, reducing the prediction difficulty. The latent task becomes too easy, and the student can find degenerate solutions.

The stacked-teacher idea introduces a **temporal hierarchy**: a slow-moving anchor, a faster buffer, and the student. The student must predict the anchor, which is two steps removed from the student — it only sees the student through the buffer's smoothing. This creates a **deeper target manifold** that should be harder to collapse.

## Architecture

```
Teacher A (The Anchor)        Teacher B (The Buffer)        Student (The Learner)
    τ=0.999                       τ=0.99                        gradients
    ← EMA from B                  ← EMA from Student
    moderate (700-step lag)       fast (70-step lag)
```

### Update rules

- **Student** → backprop from `cosine_loss(student_latent, teacher_A_latent)` at masked positions
- **Teacher B** → `θ_B ← 0.99·θ_B + 0.01·θ_student` (after every student step)
- **Teacher A** → `θ_A ← 0.999·θ_A + 0.001·θ_B` (after every student step)

### Key property

The student has **no direct gradient path** to either teacher. The student only sees Teacher A's frozen latents. Teacher A is updated from Teacher B, which is updated from the student. This creates a **two-step smoothing chain**.

### Why τ=0.999 for the anchor (not 0.9999)

An early draft used τ=0.9999 for the anchor. The math shows this is a bad idea:

| τ | Update fraction per step | Half-life | After 1k steps |
|---|---|---|---|
| 0.99 | 1% | ~70 steps | ~100% new |
| 0.999 | 0.1% | ~700 steps | ~76% new |
| 0.9999 | 0.01% | ~7,000 steps | ~10% new |

With τ=0.9999, the anchor is essentially a **frozen init copy** for most of a 30k-step run. The student learns to predict a near-constant vector — the exact collapse mode we already saw in v1. The anchor must be **genuinely reachable** but still **older** than the buffer. τ=0.999 gives a ~700-step lag, which is about 10× the buffer's ~70-step lag. This creates a meaningful hierarchy without making the target impossible.

### Curriculum variant (recommended)

Even better: start with τ_A=0.99 (easy target, essentially one teacher) and **anneal to 0.999** over the first 5,000 steps.

- Early training (steps 0–1k): Anchor behaves like a second buffer. The system is essentially single-teacher with extra smoothing. The student learns quickly.
- Mid training (steps 1k–5k): Anchor slows down. The student must now predict a more stable target.
- Late training (steps 5k+): Anchor is a 700-step smoothed view. The student is forced to produce representations that are stable over longer timescales.

This is the same logic as a **learning rate warmup** but for target difficulty.

## Why this should help

### 1. Slower target drift

With a single teacher (τ=0.99), the teacher adapts to the student's current trajectory within ~70 steps. The anchor (τ=0.999) adapts 10× slower (~700 steps). The student must predict a target that changes on a longer timescale — a genuine "older self" that retains structural features from earlier in training.

### 2. Buffer as stabiliser

The buffer prevents the anchor from being either too static or too reactive. The buffer captures the student's recent trajectory and feeds it into the anchor. Without the buffer, the anchor would just be a frozen init copy (if τ_A were too low) or a direct copy of the student (if τ_A were too high). The buffer creates the necessary indirection.

### 3. Collapse resistance

In the current system, the student can learn to output a near-constant vector that the teacher also converges to. With two teachers, the student must align with Teacher A, which is **not a direct function of the student** — it is a function of Teacher B, which is a function of the student. This extra degree of separation should prevent the trivial collapse where student and teacher converge to the same degenerate point.

### 4. Natural curriculum (with τ annealing)

Early training (steps 0–1k): τ_A ≈ 0.99. The anchor behaves like a second buffer. The system is essentially single-teacher with extra smoothing. The student learns quickly.

Mid training (steps 1k–5k): τ_A anneals toward 0.999. The anchor slows down. The student must predict a more stable target.

Late training (steps 5k+): τ_A = 0.999. The anchor is a 700-step smoothed view. The student is forced to produce representations that are stable over longer timescales.

## Differences from single-teacher

| | Single Teacher (v4) | Stacked Teachers (planned) |
|---|---|---|
| Target | `EMA(student, τ=0.99)` | `EMA(EMA(student, τ=0.99), τ=0.9999)` |
| Adaptation timescale | ~100 steps | ~10,000 steps |
| Gradient path to student | 1 hop (direct) | 2 hops (indirect) |
| Memory | 2× model params | 3× model params |
| Compute per step | 2 forward passes | 3 forward passes |
| Collapse risk | Moderate (spikes then settles) | Lower (deeper target manifold) |

## Expected risks

### 1. Target is too old

If τ_A is too low (e.g., 0.9999), the anchor becomes a frozen init copy. The student predicts a constant and collapses. If τ_A is too high (e.g., 0.99), the anchor is just a second buffer and the system is redundant. The curriculum (annealing 0.99 → 0.999) mitigates this by letting the system "warm up" before the target becomes hard.

### 2. EMA chain breaks on domain shift

If the student is fine-tuned on a different domain (e.g., noir after Gutenberg), the buffer adapts quickly but the anchor remains stuck in the old domain. The student is forced to predict a Gutenberg target while learning noir — this is the same domain-mismatch problem we saw in v5, but worse because the anchor is more static.

**Mitigation:** Reset the anchor to the buffer at domain transitions. Or simply use stacked teachers only for same-domain pretraining.

### 3. Triple forward pass cost

Per step: student (masked) + Teacher B (full) + Teacher A (full). That's 3× the model's forward pass. On MPS with a 2.5M model, this is still fast (~1s per step), but it limits scaling to larger models.

### 4. Hyperparameter sensitivity

The τ values are coupled. If τ_B is too slow, the anchor sees nothing. If τ_B is too fast, the anchor is just a delayed copy of the student. The sweet spot (τ_B=0.99, τ_A=0.999 with annealing) is based on the half-life ratio (~10× separation). It needs empirical tuning.

## Implementation sketch (no code changes)

The implementation is a minimal extension of the existing hybrid training loop:

1. Instantiate `teacher_anchor` and `teacher_buffer` alongside `student`
2. At init: `teacher_anchor.load_state_dict(student.state_dict())`, `teacher_buffer.load_state_dict(student.state_dict())`
3. Training loop:
   - Student forward (masked input)
   - Teacher buffer forward (full input, `no_grad`)
   - Teacher anchor forward (full input, `no_grad`)
   - Loss: `cosine_loss(student_latent, teacher_anchor_latent)` + CE on unmasked
   - After optimizer step: `update_ema(teacher_buffer, student, τ=0.99)`, then `update_ema(teacher_anchor, teacher_buffer, τ=0.9999)`

The rest of the system (span masking, fixed LayerNorm, cosine loss, data loading) is unchanged.

## When to try this

### Recommended: if v6 (folding + teacher-student) reaches a plateau

If the current run with single teacher + folding gets stuck at val ~2.5–2.8 and cannot break below 2.5, the stacked teacher is a natural next experiment. The deeper target manifold might push the model through the plateau.

### Not recommended: if v6 collapses or fails

If the model collapses (latent → 0) or the fold breaks the teacher-student dynamic, fix the core architecture first before adding complexity.

### Alternative: try it on noir fine-tuning only

Use stacked teachers during the noir post-training phase, with both teachers reset to the student at the start. The anchor provides a "Gutenberg memory" that decays slowly, while the buffer adapts to noir. The student must align with the anchor — this might enforce structural coherence without pulling back to pure Gutenberg.

## Expected metrics

- **Latent loss:** Should be higher than single-teacher (not a bug — the target is genuinely harder). With τ_A=0.999, the anchor is a 700-step smoothed view. The student must align with a target that is genuinely different from its current trajectory. Expected range: 5.0–6.5 instead of 4.5–5.5.
- **CE loss:** Should drop at the same rate or slightly slower. The anchor is not competing with CE for gradient budget.
- **Val perplexity:** The hope is a 5–10% improvement over single-teacher at the same step count, due to better-structured representations.

## Summary

Stacked teachers (Anchor + Buffer) is a **depth-over-width** approach to the latent target. Instead of one smoothed copy, we use a cascade of two. The student predicts the most stable, oldest target. This should reduce collapse risk and improve representation quality. The cost is 3× forward passes per step and a slightly more complex EMA chain.

**Next step:** Implement if v6 val loss plateaus, or as a dedicated noir fine-tuning experiment.
