# Session Compaction Summary

## User Intent

- Explore parameter-efficient architectures for a tiny (5.4M param) noir-fiction language model, specifically a "hourglass" folding scheme where outer transformer layers share weights across multiple passes
- Compare the hourglass against a standard 10-layer baseline with multi-resolution attention (uneven heads: 1×64 + 2×32 + 8×8)
- Build a DPO (Direct Preference Optimisation) pipeline so the user can rank generated completions and encode personal taste into the model
- Expand the noir training corpus from ~4.4M to ~6.6M tokens by adding pre-1929 public-domain hardboiled fiction
- Investigate cyclical fold/unfold training as a dynamic regularisation strategy

## Contextual Work Summary

### Hourglass Architecture Evolution

**v1 (`[3+4+3]`)**: First attempt folded outer layers to 3 loops, leaving only 4 independent bridge layers. Results were underwhelming — too aggressive, shared layers couldn't serve 3 different functional roles.

**v2 (`[2+6+2]`)**: The breakthrough. Reverted to a gentler single fold. Added **loop-specific LayerNorms** — each HourglassBlock owns a `ModuleList` of LayerNorms (one per loop index), so shared attention/MLP weights operate in different normalised spaces per pass. This added ~6K parameters but dramatically increased expressiveness. The hourglass v2 was judged slightly better than the baseline by ear.

**Cyclic fold/unfold**: Explored treating folding as a *transient training regulariser* rather than permanent compression. Curriculum: `[1+8+1]` → `[2+6+2]` (fold) → `[1+8+1]` (unfold) → `[2+6+2]` (refold). Initial LR schedules were too conservative; revised to keep LR high (1e-3) during structural changes. Unfolding is more disruptive than folding — new blocks must learn from scratch even though they inherit shared weights. Currently running the revised cyclic curriculum with live progress tracking.

### DPO Pipeline

Built a full 3-stage DPO workflow:

1. **`dpo_generate.py`**: CLI script generates N completions per prompt at varied temperatures. Initially forced full-length output by masking EOS/EOT, which produced garbage. Fixed to allow natural stopping with a `min_tokens` threshold and sentence-boundary backtracking.

2. **Ranking UI (`llm/rank.html` + `rank.js`)**: Browser-based ranking with two modes:
   - **Tri-state**: Click each completion to cycle `○ → ★ good → ✗ bad → ○`. Only good×bad pairs are exported. Neutral items excluded.
   - **Full ranking**: Click completions from best (1) to worst (5), generating all C(5,2) pairwise combinations.
   
3. **`dpo_train.py`**: Loads active model + frozen reference copy. Supports three input formats (tri-state, full ranking, legacy pairs). `--rank-weight` weights loss by rank gap.

**Key lesson**: DPO at 5.4M params is extremely sensitive to hyperparameters. First run (14 pairs, 500 steps, β=0.1) collapsed — ratio exploded to 178, model memorised exact examples and degraded. Second run (98 pairs, 75 steps, β=0.05) was better — ratio peaked at 8.8 and regressed to 1.3, producing coherent output. DPO gives marginal, subtle improvements but is not transformative at this scale.

### Baseline Comparison

Trained a fresh baseline (10-layer, no folding) with conservative phase 2 settings (5K steps @ 5e-5, dropout 0.2) to compare head-to-head against hourglass v2. The baseline was actually **worse** — too much regularisation for a 4.4M-token corpus. Confirmed that the hourglass architecture is at least not harmful and potentially beneficial.

### Corpus Expansion

Researched additional public-domain noir sources (see `noir_data_research.md`). Key gaps identified:
- Carroll John Daly (inventor of hardboiled, Race Williams series in Black Mask 1922–1927)
- Frederick Nebel (Donahue stories, Black Mask 1926–1928)
- Missing Hammett Continental Op stories

User expanded corpus from ~4.4M to **~6.6M tokens** (47% increase). This triggered a full retrain: new tokenizer, re-encoded Gutenberg (`prepare_wiki.py`), fresh phase 1 and phase 2.

### Progress Tracking

Added CSV logging (`train_log.csv`) to `train_hourglass_cyclic.py` and a browser dashboard (`llm/progress.html`) using Chart.js. Shows live train/val curves with phase boundaries marked as green dashed lines. Auto-refreshes every 30 seconds. Served from project root with `python -m http.server`.

### Generation Improvements

Patched both `generate.py` (baseline) and `generate_hourglass.py` with:
- `--min-new-tokens`: Forces continuation until minimum length, preventing immediate EOS death
- Sentence-boundary backtracking: If max length is reached without natural stop, trims to last `.` `!` `?` (falls back to last space)
- One-shot BPE decoding: Tokens buffered and decoded together to avoid subword fragmentation

## Files Touched

### Core Architecture
- **`model_hourglass.py`**: HourglassBlock with loop-specific LayerNorms (`ModuleList` of LNs per loop index), `HourglassTransformer` with zone-aware forward, fold helpers (`soft_tie`, `apply_fold`), and `apply_unfold` (creates fresh independent blocks from shared anchors, moved to correct device)
- **`model.py`**: Unchanged — standard 10-layer multi-resolution attention baseline
- **`config.py`**: Unchanged

### Training Scripts
- **`train_hourglass.py`**: Phase 1 Gutenberg pre-training. Single fold to `[2+6+2]`, 50K steps. Saves `mica_hourglass_v2_phase1.pt`
- **`train_hourglass2.py`**: Phase 2 noir fine-tune. 10K steps @ 1e-4, dropout 0.1. Saves `mica_hourglass_v2.pt`
- **`train_hourglass_cyclic.py`**: Cyclic fold/unfold curriculum. 5 phases, 90K total steps. Logs to `train_log.csv`. Boundary-only resume support
- **`train_phase1.py`**: Standard baseline phase 1 (unchanged)
- **`train.py`**: Standard baseline phase 2. Updated to 5K steps @ 5e-5 with dropout 0.2, checkpoint resume

### DPO Pipeline
- **`dpo_generate.py`**: CLI generation for DPO candidates. Natural stopping, varied temperatures
- **`dpo_train.py`**: DPO training with frozen reference. Supports tri-state, ranking, and legacy pair formats. Rank-gap weighting
- **`dpo_extract_prompts.py`**: Samples corpus-derived prompts from `train.bin`
- **`dpo_prompts.txt`**: Hand-written noir seed prompts (24 prompts)
- **`llm/rank.html`** + **`llm/rank.js`**: Browser ranking UI with tri-state and full-ranking modes

### Generation
- **`generate.py`**: Baseline generation. Added force-continue, sentence-boundary trimming
- **`generate_hourglass.py`**: Hourglass generation. Same improvements, displays layout string

### Progress & Docs
- **`llm/progress.html`**: Live training dashboard with Chart.js. Phase boundaries, auto-refresh
- **`hourglass_v2.md`**: Architecture rationale, curriculum, and state documentation for the `[2+6+2]` design
- **`dpo.md`**: Full DPO pipeline guide
- **`noir_data_research.md`**: Public-domain noir source research
- **`rfolding.md`**: Original folding blueprint (superseded by v2)

### Data Prep
- **`prepare.py`**: Retrain BPE tokenizer on expanded corpus
- **`prepare_wiki.py`**: Re-encode Gutenberg with current tokenizer

### Checkpoints (not in git)
- `mica_hourglass_v2_phase1.pt` — Phase 1 weights (Gutenberg, `[2+6+2]`)
- `mica_hourglass_v2.pt` — Phase 2 weights (noir, `[2+6+2]`)
- `mica_dpo_v1.pt` — Best DPO checkpoint (β=0.05, 75 steps)
- `mica_phase1.pt` — Baseline phase 1
- `mica_weights_baseline.pt` — Baseline phase 2 (conservative)

## Key Observations & Decisions

1. **Loop-specific LayerNorms are essential** for hourglass viability. Without them, shared blocks collapse to a single behaviour.
2. **Folding is gentler than unfolding**. Fold spikes are ~+1.2 loss and recover. Unfold spikes are ~+0.9 but recovery plateaus unless LR is high.
3. **DPO at 5.4M params is surgical, not transformative**. Needs 50–100+ pairs, very few steps (50–200), low β (0.05). Ratio >50 = overfit.
4. **Data is the ceiling**. 6.6M tokens on 5.4M params is still small. Architecture tweaks give marginal gains; corpus expansion gives real gains.
5. **Cyclic training is unproven but promising**. The hypothesis: shared layers learn robust representations that specialisation layers can improve upon. Currently being tested with revised high-LR curriculum.
