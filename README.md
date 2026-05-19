# Mica

> *De mica en mica s'omple la pica.*
> — Catalan proverb

*Mica* means crumb in Catalan. The proverb translates roughly as "drop by drop the sink fills up" — which is also a fair description of how you train a language model on very little data.

It is also the title of one of the first (and one of my favourite) "hard boiled" detective novels I read, and one of the first written in Catalan, _De mica en mica s'omple la pica_, by Jaume Fuster. Sadly, there is no English translation (that I know of).

Mica is a small, hand-crafted Transformer trained on hardboiled noir fiction. It is not meant to be useful. It is meant to sound like Dashiell Hammett.

---

## Architecture

Mica is a decoder-only Transformer in the GPT lineage, deliberately constrained to prevent memorisation and force generalisation on a small corpus.

| Hyperparameter | Value |
|---|---|
| Layers | 10 |
| Attention heads | 11 (non-uniform) |
| Embedding dimension | 192 |
| Context window | 512 tokens |
| Vocabulary | 5 000 (BPE, trained on corpus) |
| Positional encoding | RoPE (no learned position embeddings) |
| Parameters | ~5.4M |

The 192 embedding dimensions are split into three parallel attention blocks, each 64 dimensions wide:

| Block | Heads | Head size | Role |
|---|---|---|---|
| Large | 1 | 64 | Semantic tracking |
| Medium | 2 | 32 | Context integration |
| Small | 8 | 8 | Syntactic filtering |

Each block runs its own Q/K/V projections and a separate RoPE instance, then the outputs are concatenated and projected back. This gives heterogeneous attention resolution at no extra parameter cost.

### Origami folding

Mica uses an "origami" folding scheme where outer transformer layers can share weights across multiple passes during training, acting as a heavy compression regulariser. The inner layers (near the output) are never folded — they do the heavy lifting of next-token prediction. After training, the model is fully unfolded for inference.

Training uses a cyclic curriculum: the model is folded, trained, unfolded, and trained again. The shared outer layers learn robust features that the independent inner layers then specialise.

The tokenizer is trained from scratch on the target corpus and includes two boundary tokens: `<|eos|>` (sentence boundary) and `<|endoftext|>` (paragraph/document boundary).

The folding/unfolding process was a very slow, but seemingly "working" way of consistently reducing train and validation error without overfitting, although the improvements were very modest.

---

## Corpus

The training corpus is hardboiled detective fiction, assembled from public-domain sources.

**Dashiell Hammett**
- *The Continental Op* stories
- *Red Harvest*
- *The Dain Curse*
- *The Maltese Falcon*

**Pulp magazines** (pre-1928, via archive.org)
- *Black Mask* (where Hammett first published)
- *Detective Story Magazine*
- *Dime Detective*
- *True Detective Mysteries*
- *Hooded Detective*

Total corpus: ~6.6M tokens after cleaning. The `clean_corpus.py` script strips OCR artifacts, magazine headers, table-of-contents lines, and Gutenberg boilerplate from raw downloads.

---

## Setup

```bash
uv sync
```

Python 3.13+, PyTorch with MPS backend (Apple Silicon). CPU fallback works but is slow.

---

## Training

Training runs in two phases. All steps must be run in order.

**Step 1 — Build tokenizer and encode the noir corpus**

`prepare.py` trains the BPE tokenizer on the noir corpus and encodes it.

```bash
uv run python src/prepare.py
```

**Step 2 — Encode Gutenberg books**

Encodes the Project Gutenberg books using the tokenizer from step 1.

```bash
uv run python src/prepare_wiki.py
```

**Step 3 — Phase 1: grammar pre-training**

Pre-trains on the Gutenberg corpus (~150K steps, cyclic fold/unfold) to learn English syntax and general semantics. Saves to `models/experiments/origami/mica_origami_cyclic_n10.pt`.

```bash
uv run python src/train_origami_cyclic.py --n-layer 10
```

**Step 4 — Phase 2: style fine-tuning**

Fine-tunes on the noir corpus (~14K steps) at low learning rate, loading from the phase 1 checkpoint. The low learning rate shifts the style without destroying the grammatical foundation.

```bash
uv run python src/train_origami2_cyclic.py
```

The final weights are saved to `models/current/mica_origami_v2.pt`.

---

## Generation

```bash
uv run python src/generate_origami.py "The fat man leaned back"
uv run python src/generate_origami.py --temperature 0.9 --max-new-tokens 100
```

### Examples

| Prompt | Output |
|---|---|
| `The fat man leaned back` | The fat man leaned back and faced the man with a lean black face, and asked: "Maybe you'd better come back to town?" |
| `She walked into the room` | She walked into the room and froze the plump curtains. |
| `I lit a cigarette` | I lit a cigarette and drank another. |
| `A bullet whizzed past` | A bullet whizzed past him. Answered him. |

Generation stops naturally at sentence or document boundaries. Lower temperature is more rigid, higher is more chaotic.

---

## Browser demo

Export the trained model to ONNX and serve the `llm/` folder locally.

```bash
# Export to ONNX (inference + trace with attention weights)
uv run python src/export_onnx_origami.py

# Serve
python -m http.server 8080 --directory llm
```

Open `http://localhost:8080` for generation, `http://localhost:8080/trace.html` for the token trace viewer with attention head visualisation.

The trace viewer shows:
- **Token predictions** — what the model predicts at each layer (logit lens)
- **Attention maps** — per-head attention weights for every token
- **Residual stream** — how representations evolve through the network

---

## DPO — encoding taste into the weights

Mica includes a Direct Preference Optimisation pipeline for refining style. Generate candidate completions, rank them, and train the model to prefer your taste.

```bash
# Generate candidates
uv run python src/dpo_generate.py --prompts dpo_prompts.txt --output dpo_candidates.json

# Rank in browser (open llm/rank.html, load dpo_candidates.json)

# Train DPO
uv run python src/dpo_train.py --pairs dpo_rankings.json --output mica_dpo.pt
```

See `dpo.md` for the full guide.

---

## Project layout

```
src/                        # Python scripts (training, generation, export)
├── generate_origami.py     # CLI generation
├── train_origami2_cyclic.py
├── export_onnx_origami.py
└── ...

models/
├── current/                # Best model, tokenizer, checkpoint
│   ├── mica_origami_v2.pt
│   ├── mica_origami_v2_ckpt.pt
│   └── mica_tokenizer.json
├── experiments/            # All training runs
│   ├── origami/
│   ├── hourglass/
│   ├── baseline/
│   └── dpo/
└── backups/                # Pre-DPO, pre-extension checkpoints

data/                       # Encoded token binaries
├── train.bin               # Noir corpus (training)
├── val.bin                 # Noir corpus (validation)
├── wiki_train.bin          # Gutenberg corpus (training)
└── wiki_val.bin            # Gutenberg corpus (validation)

logs/                       # CSV training logs
├── origami/                # Main runs
├── failed/                 # Abandoned experiments
└── hourglass/

llm/                        # Browser demo (GitHub Pages)
├── index.html              # Generation UI
├── trace.html              # Attention visualiser
├── mica.onnx               # Inference model
├── mica_trace.onnx         # Trace model
└── ...

corpus/                     # Raw and cleaned text files
```

Convenience symlinks in the project root:
- `mica.pt` → `models/current/mica_origami_v2.pt`
- `mica_tokenizer.json` → `models/current/mica_tokenizer.json`

---

## Extending the corpus

`download_corpus.py` fetches plain-text OCR from archive.org magazine issues defined in the `ITEMS` list. Already-downloaded files are skipped.

```bash
uv run python src/download_corpus.py          # download everything in ITEMS
uv run python src/clean_corpus.py --rebuild   # clean all corpus/raw/*.txt → corpus/corpus.txt
uv run python src/prepare.py                  # retrain tokenizer and re-encode
uv run python src/prepare_wiki.py             # re-encode Wikipedia with the new tokenizer
```

---

## Publishing to GitHub Pages

The `llm/` folder is a self-contained static site. To publish:

```bash
# Ensure the ONNX model and tokenizer are in llm/
ls llm/mica.onnx llm/mica_trace.onnx llm/mica_tokenizer.json

# Push to GitHub Pages (using the gh-pages branch or GitHub Actions)
```

The generation UI (`index.html`) and trace viewer (`trace.html`) both load the ONNX model via `onnxruntime-web` and run entirely in the browser — no server required after the initial page load.
