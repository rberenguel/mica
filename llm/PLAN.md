# Mica — Web Interface Plan

A self-contained browser demo. No build step, no framework, no server-side logic.
Serve the folder with `python -m http.server 8080` and open `localhost:8080`.

## Files

```
llm/
├── index.html        — UI shell
├── style.css         — noir-appropriate styling
├── tokenizer.js      — BPE encoder/decoder (reads mica_tokenizer.json)
├── mica.js           — model loading, generation loop, UI wiring
├── mica.onnx         — exported model  (copy from repo root after export_onnx.py)
├── mica_tokenizer.json — tokenizer     (copy from repo root after prepare.py)
└── PLAN.md           — this file
```

## Dependencies (CDN, no install)

| Library | Purpose | CDN |
|---|---|---|
| `onnxruntime-web` | Run ONNX model in browser (WASM / WebGPU) | jsDelivr |

No other JS dependencies. The tokenizer is implemented from scratch in `tokenizer.js`
by reading the HuggingFace tokenizer JSON format directly.

## Architecture

### Tokenizer (tokenizer.js)

HuggingFace BPE with ByteLevel pre-tokenization. Key facts:
- `encode(text) → int[]` — byte-level BPE, returns token IDs
- `decode(ids) → string` — reverse byte-level decoding back to UTF-8
- Byte-to-unicode table: byte 32 (space) → Ġ (U+0120), printable ASCII maps to itself
- Merge priority = index in `model.merges` array (0 = highest priority)

**Known limitation**: the simplified pre-tokenizer splits only on ASCII spaces.
Contractions ('re, 've, etc.) and punctuation adjacent to words may tokenize
differently from the Python training tokenizer. For a tinkering demo this is
acceptable; for production match the GPT-2 regex pattern exactly.

### Model (mica.js)

- `ort.InferenceSession.create('mica.onnx')` — loads model
- Input tensor: `int64` shape `(1, seq_len)` — current context IDs
- Output tensor: `float32` shape `(1, vocab_size)` — logits for next token
- Autoregressive loop: encode prompt → run model → sample → decode → append → repeat
- Context is cropped to `BLOCK_SIZE = 256` tokens if the sequence grows longer

### Sampling (mica.js → `sampleLogits`)

Temperature scaling + softmax + multinomial sampling.
Optional: top-k and top-p (nucleus) sampling are stubbed — straightforward to add.

## Generation Loop

```
prompt text
  → tokenizer.encode()
  → [id, id, id, ...]           (crop to 256)
  → ort session.run()
  → logits float32[5000]
  → sampleLogits(logits, temp)
  → next_id
  → tokenizer.decode([next_id])
  → append to output
  → repeat
```

## UI

### Generation mode (current)
- **Prompt textarea** — editable seed text
- **Generate button** — starts streaming generation
- **Stop button** — cancels in-flight generation
- **Temperature slider** — 0.5 (focused) to 1.2 (chaotic), default 0.8
- **Max tokens input** — default 200
- **Output area** — new tokens streamed in as they generate

### Rank mode (planned — for DPO dataset collection)
Toggle switches the UI from single-stream to a grid of N completed outputs.
Click ★ on the best, ✗ on the worst → pair saved to localStorage.
Export button downloads accumulated pairs as `dpo_pairs.json`.

New files needed:
- `rank.js` — parallel generation (Promise.all), grid rendering, click-to-rank logic
- `pairs.js` — localStorage accumulation, pair counter, JSON export

See `fine_tune_ideas.md` § Source C² for full description.

Styling: near-black background, amber text, monospace font. Minimal.

## Setup Sequence

```bash
# 1. Train and export
uv run python export_onnx.py          # produces llm/mica.onnx
cp mica_tokenizer.json llm/

# 2. Serve
cd llm
python -m http.server 8080

# 3. Open
open http://localhost:8080
```

## Status

- [x] export_onnx.py written
- [x] index.html written
- [x] style.css written
- [x] tokenizer.js written (simplified pre-tokenizer)
- [x] mica.js written
- [ ] Tested end-to-end with actual mica.onnx (pending Phase 1 training completing)
- [ ] Tokenizer edge cases validated against Python tokenizer output
- [ ] WebGPU backend tested (WASM fallback confirmed working)

## Potential Issues & Fixes

**Tokenizer mismatch**: If generation produces garbled output, the pre-tokenizer
split pattern is the likely cause. Fix: read `pre_tokenizer.pattern.Pattern` from
`mica_tokenizer.json` and replicate the regex in JS exactly.

**BigInt64Array**: `int64` tensors require `BigInt64Array`. Supported in all modern
browsers (Chrome 67+, Firefox 68+, Safari 14+). If targeting older browsers, add
a polyfill or lobby for `int32` input in the export wrapper.

**WASM MIME type**: Some servers don't serve `.wasm` files with
`application/wasm` MIME type. Python's `http.server` does it correctly.
Apache/Nginx may need config. The CDN-loaded `onnxruntime-web` fetches its own
WASM internally so this only matters if self-hosting the runtime.

**Model too slow**: For a 2.7M param model with context ≤ 256, WASM on a modern
laptop should generate 5-20 tokens/sec. If too slow, enable WebGPU backend
(change `executionProviders` in `mica.js`) and test in Chrome.
