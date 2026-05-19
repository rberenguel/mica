# DPO Pipeline for Mica

Direct Preference Optimisation — encode your taste into the weights using paired comparisons.

## Overview

1. **Generate** — CLI script produces N completions per prompt at varied temperatures
2. **Rank** — Browser UI lets you pick best (★) and worst (✗) per prompt
3. **Train** — DPO script pushes the model toward your preferences

## Step 1: Generate candidates

```bash
uv run python dpo_generate.py --prompts dpo_prompts.txt --output dpo_candidates.json
```

Options:
- `--n-per-prompt 5` — completions per prompt
- `--temperatures 0.6 0.7 0.8 0.9 1.0` — varied temps for diversity
- `--max-tokens 80` — completion length
- `--weights mica_origami_v2.pt` — base model (default)

Generates `dpo_candidates.json`:
```json
[
  {
    "prompt": "The fat man leaned back",
    "completions": [
      {"text": "...", "temperature": 0.7, "seed": 0},
      ...
    ]
  }
]
```

## Step 2: Rank in browser

```bash
cd llm && python -m http.server 8080
```

Open `http://localhost:8080/rank.html` and load `dpo_candidates.json`.

**Three ranking modes** (toggle in the UI):

1. **Tri-state GOOD/BAD** (recommended) — Click each card to cycle:
   `○` → `★ good` → `✗ bad` → `○`
   Mark **all** acceptable completions as good and **all** rubbish as bad.
   Generates **good×bad pairs** only — neutral items are excluded.
   Perfect when there's a sharp quality cliff (2 good, 3 bad → 6 clean pairs).

2. **Rank all 5** — Click from BEST (1) to WORST (5).
   Generates **10 pairwise pairs** (C(5,2)).
   Best when all completions are on a smooth quality spectrum.

3. **Skip** — If all 5 are garbage or all 5 are fine, hit Skip and move on.

Aim for **10–20 prompts with 4–10 pairs each** (= 50–150 total pairs).

Aim for **10–20 full rankings** (= 100–200 pairwise pairs). That's richer signal than 50 binary pairs.

## Step 3: DPO training

```bash
# From rankings (recommended)
uv run python dpo_train.py --pairs dpo_rankings.json --output mica_dpo.pt --rank-weight

# From legacy binary pairs
uv run python dpo_train.py --pairs dpo_pairs.json --output mica_dpo.pt
```

Hyperparameters (tune if needed):
- `--beta 0.1` — DPO temperature. Higher = stronger push. If ratio goes negative, lower it.
- `--steps 200` — Training steps. With 100+ pairs, 100–200 is plenty. Watch the ratio.
- `--lr 1e-5` — Very low. Gentle polish, not learning from scratch.
- `--rank-weight` — Weight pairs by rank gap (best-vs-worst counts more than 2nd-vs-3rd).

## Step 4: Generate and compare

```bash
uv run python generate_origami.py --weights mica_dpo.pt "The fat man leaned back" --max-new-tokens 80
uv run python generate_origami.py --weights mica_origami_v2.pt "The fat man leaned back" --max-new-tokens 80
```

Trust your ear over the numbers.

## What to watch

- **ratio** should trend positive — the model increasingly prefers chosen over rejected relative to the reference.
- If generations degrade (incoherent, repetitive), the model drifted too far from the reference. Lower `--beta` or reduce steps.
- DPO at 5.4M params on 50–100 examples is genuinely uncharted. Small changes should be audible.
