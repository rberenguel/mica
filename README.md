# Mica

> *De mica en mica s'omple la pica.*
> — Catalan proverb, and the title of the first noir novel I ever read, by Jaume Fuster.

*Mica* means crumb in Catalan. The proverb translates roughly as "drop by drop the sink fills up" — which is also a fair description of how you train a language model on very little data.

Mica is a small, hand-crafted Transformer trained on hardboiled noir fiction. It is not meant to be useful. It is meant to write like Dashiell Hammett.

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
| Dropout | 0.1 |
| Optimizer | AdamW, β₂ = 0.95 |
| LR schedule | Linear warmup → cosine decay |

The 192 embedding dimensions are split into three parallel attention blocks, each 64 dimensions wide:

| Block | Heads | Head size | Role |
|---|---|---|---|
| Large | 1 | 64 | Semantic tracking |
| Medium | 2 | 32 | Context integration |
| Small | 8 | 8 | Syntactic filtering |

Each block runs its own Q/K/V projections and a separate RoPE instance, then the outputs are concatenated and projected back. This gives heterogeneous attention resolution at no extra parameter cost.

The tokenizer is trained from scratch on the target corpus and includes two boundary tokens: `<|eos|>` (sentence boundary) and `<|endoftext|>` (paragraph/document boundary). Generation stops naturally at either.

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

Total corpus: ~4.4M tokens after cleaning. The `clean_corpus.py` script strips OCR artifacts, magazine headers, table-of-contents lines, and Gutenberg boilerplate from raw downloads.

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

`prepare.py` trains the BPE tokenizer on the noir corpus and encodes it. This must run before anything else — `prepare_wiki.py` loads the tokenizer it produces.

```bash
uv run python prepare.py
```

**Step 2 — Encode Gutenberg books**

Encodes the Project Gutenberg books using the tokenizer from step 1.

```bash
uv run python prepare_wiki.py
```

**Step 3 — Phase 1: grammar pre-training**

Pre-trains on the Gutenberg corpus (~50 000 steps) to learn English syntax and general semantics. The Gutenberg text is encoded with the noir tokenizer, so out-of-vocabulary words become subword fragments — the model still learns grammar. Saves to `mica_phase1.pt`.

```bash
uv run python train_phase1.py
```

**Step 4 — Phase 2: style fine-tuning**

Fine-tunes on the noir corpus (~5 000 steps) at one tenth the Phase 1 learning rate, loading from `mica_phase1.pt`. The low learning rate shifts the style without destroying the grammatical foundation.

```bash
uv run python train.py
```

---

## Generation

```bash
uv run python generate.py "The fat man leaned back in his chair"
uv run python generate.py --temperature 0.9 --max-new-tokens 100
uv run python generate.py --weights mica_phase1.pt "It was raining"
```

Generation stops at the first sentence or document boundary the model produces. Lower temperature is more rigid, higher is more chaotic.

---

## Browser demo

Export the trained model to ONNX and serve the `llm/` folder locally.

```bash
# Standard export (inference only)
uv run python export_onnx.py
cp mica_tokenizer.json llm/

# Trace export (attention weights, residual stream, logit lens)
uv run python export_onnx_trace.py
cp mica_tokenizer.json llm/

# Serve
cd llm && python -m http.server 8080
```

Open `http://localhost:8080` for generation, `http://localhost:8080/trace.html` for the token trace viewer.

---

## Extending the corpus

`download_corpus.py` fetches plain-text OCR from archive.org magazine issues defined in the `ITEMS` list. Already-downloaded files are skipped.

```bash
uv run python download_corpus.py          # download everything in ITEMS
uv run python clean_corpus.py --rebuild   # clean all corpus/raw/*.txt → corpus/corpus.txt
uv run python prepare.py                  # retrain tokenizer and re-encode
uv run python prepare_wiki.py             # re-encode Wikipedia with the new tokenizer
```
