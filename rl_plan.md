# RL Plan for Mica — Local Judges, Custom Raters, and Online GRPO

## Why RL for a Small Model

A 3.69M-parameter model trained on ~194M tokens (Gutenberg + Noir) is already in the human-child data range, but passive next-token cross-entropy only teaches *local token plausibility*, not *whole-sentence coherence or voice*.

Earlier offline **DPO** was brittle at this scale (~100 static pairs → quick memorisation and collapse to immediate EOS). **Online RL (GRPO / RLOO)** avoids that because:
- The model generates fresh rollouts (`G = 4–8` completions per prompt) on every step.
- Each rollout is scored relative to the group mean (`advantage = (r_i - mean(r)) / (std(r) + eps)`), so **no critic/value network** is needed.
- A light **KL penalty** to the frozen Noir SFT checkpoint prevents drift and EOS collapse.
- On an M1 16GB, generating 8 short rollouts from a 3.69M looped model takes milliseconds.

---

## Chosen Direction — 100% From-Scratch "Mica-as-Critic" Rater

**Design principle:** No external pretrained LLMs, no distillation smell. Everything is trained from scratch on your own Gutenberg + Noir corpora and your own taste clicks.

### Architecture: Mica-as-Critic (~3.69M params)
- Clone your pretrained looped Mica backbone (`models/experiments/looped/best.pt` or `looped_noir/best.pt`).
- Replace `lm_head` with a scalar scoring head: `nn.Linear(256, 1)` over the mean-pooled (or final-token) hidden states.
- Runs at **thousands of ratings/sec** on M1 MPS — zero extra memory pressure alongside the policy and frozen reference.

### Training the Rater (Bradley-Terry Pairwise Loss)
Humans are much more consistent at comparing two phrases than assigning absolute 1–5 scores. Given pairs `(chosen, rejected)`:

```python
r_chosen   = critic(prompt + chosen)
r_rejected = critic(prompt + rejected)
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

At inference time, `critic(prompt + completion)` outputs a single real number `r` (higher = better) that plugs straight into GRPO.

### Pure From-Scratch Training Data (No External Models)
Mix two purely homegrown tiers:
1. **Self-generated & corpus negatives (thousands of free pairs):**
   - `chosen`: real continuations from `corpus/corpus.txt` (Hammett / *Black Mask*).
   - `rejected` (mix of failure modes):
     - **Mica's own raw generations** on the same prompt (teaches what still sounds "off").
     - **Gutenberg continuations** (teaches Noir voice vs. 19th-century literary prose).
     - **Corrupted Noir** — word-shuffled clauses, truncated mid-sentence, or n-gram-repeated phrases (teaches grammar, completeness, and anti-repetition).
2. **Your personal taste (`dpo_rankings.json` from `llm/rank.html`):**
   - Fine-tune (or upweight) the critic on your own rankings so it encodes *your* ear for what makes a line sing.

### Iterative Self-Play (Optional)
Once GRPO improves Mica's prose, the old negatives become too easy. You can run 2–3 short rounds:
1. Generate fresh rollouts from the updated Mica.
2. Refresh the critic briefly on `(real noir vs. new Mica rollouts)` + your `rank.html` pairs.
3. Run another short KL-anchored GRPO round.

*(This is essentially a stabilized, KL-regularized actor-critic / adversarial language game — 100% self-contained.)*

---

## Alternative (Parked — External Models)

Kept here only for reference if you ever want a comparison baseline; **not** the primary path:
- **Single-pass A/B logit judge:** 1.5B–3B local instruct model reading `sigmoid(logit("A") - logit("B"))` in one forward pass (~30–50 ms/pair).
- **Pretrained encoder / fluency rater:** `DeBERTa-v3-small` or `SmolLM2-360M` log-prob scoring.

---

## Minimal GRPO / RLOO Loop Sketch

Starting from `models/experiments/looped_noir/best.pt`:

1. Keep a frozen copy `ref_model` of the Noir SFT checkpoint.
2. Sample a batch of noir prompts (e.g., first 5–10 words of held-out noir sentences).
3. For each prompt, sample `G = 4` (or `8`) completions from the active `model`.
4. Compute scalar rewards `r_1..r_G` from the judge/rater + cheap deterministic guardrails:
   - Penalize n-gram repetition loops.
   - Penalize immediate EOS (< `min_tokens`) or unclosed quotes/mid-word cutoffs.
5. Group-normalize advantages: `A_i = (r_i - mean(r)) / (std(r) + 1e-6)`.
6. Policy gradient + KL anchor:
   ```python
   # Sum over completion tokens only (mask prompt tokens)
   loss_pg = -(A_i * log_pi(y_i | x)).mean()
   loss_kl = beta * kl_div(pi(y_i | x), ref(y_i | x))  # beta ~ 0.02 - 0.05
   loss = loss_pg + loss_kl
   ```
7. Low LR (`1e-5`–`3e-5`), monitor KL divergence and inspect live samples every 50 steps.

---

## Guardrails Against Reward Hacking
Small models find exploits fast. Always watch for:
- **Length collapse / verbosity gaming:** normalize or gently bound completion length.
- **Cliché spam:** if the rater loves "cigarette" and "rain", Mica will write *"I lit a cigarette in the rain"* forever → keep the KL penalty (`beta = 0.03–0.05`) and penalize repeated n-grams across the batch.
- **Judge blind spots:** periodically eyeball 10 raw generations and compare against the frozen Noir SFT baseline.

---

## Next Steps (After Pretrain + Noir Fine-Tune)
1. Finish Phase 1 Gutenberg pretrain (`models/experiments/looped/best.pt`).
2. Run Phase 2 Noir fine-tune (`models/experiments/looped_noir/best.pt`) and evaluate prose by ear.
3. Build `src/train_critic.py` — train the 3.69M **Mica-as-Critic** Bradley-Terry rater on `(real noir vs. Mica rollouts / Gutenberg / corrupted noir)` + your `dpo_rankings.json` clicks.
4. Build `src/rl_grpo.py` — run KL-anchored online GRPO against your own critic, with optional iterative self-play rounds.

---

## Idea to Explore Next — Live Interleaved Critic-RL During Training

Instead of strictly sequential `passive CE → passive CE → post-training RL`, mimic how humans/animals actually learn: **interleave passive listening (CE) with active speaking + live feedback (Critic + RL) throughout training.**

### How it would work (co-evolving teacher + learner)
After an initial grammar warmup (e.g. first 2K–5K steps so the learner produces real words), every `N` standard CE steps (e.g. `N = 20–50`):
1. **Speak:** Learner generates `G = 4` short continuations for a few real prompts.
2. **Critic co-evolves (1 step):** Update a live `MicaCritic` on `(real continuation vs. learner's current rollout)` + corrupted negatives. As the learner improves, the critic's standard automatically rises — an adaptive curriculum without manual tuning.
3. **Live RL nudge (1 step):** Score the learner's rollouts with the live critic and apply 1 light GRPO update.

### Why this could beat both EMA Teacher-Student and pure GANs
- **vs. EMA Teacher-Student:** Grounded in real-vs-generated text in *output space*, not drifting hidden vectors in latent space.
- **vs. SeqGAN / pure adversarial LM:** The continuous CE stream acts as a rock-solid anchor, preventing mode collapse while the live critic teaches whole-sentence coherence, rhythm, and anti-repetition *while* representations are still forming.
- **M1 cost:** At `1` RL/critic round per `25–50` CE steps, overhead is only ~10–15%.
