# Session Compaction Summary

## User Intent
- Polish noir generation output (fix BPE spacing/capitalization artifacts)
- Organise project for clean GitHub Pages deployment
- Build interactive browser visualisation of training curves
- Experiment with a smaller model variant to test cyclic folding at reduced scale
- Add model switching to the browser generation and trace UIs

## Contextual Work Summary

### Generation polish
- Added post-processing to both Python CLI (`generate_origami.py`) and web JS (`llm/mica.js`) to clean BPE artifacts: `Theman` → `The man`, missing spaces after punctuation, capitalization after sentence boundaries, underscore/number stripping
- Changed default generation temperature from 0.8 → 0.65
- Fixed BPE decode bug in JS: was decoding each token individually (garbage); now decodes full accumulated sequence each step and yields only the newly added text
- Added EOS/EOT stop logic to JS generator to match Python behaviour

### DPO experiment (aborted)
- Ran DPO on 118 preference pairs (β=0.05, 150 steps); ratio spiked to 11.5
- Model collapsed — immediate EOS on all prompts. Restored from backup (`mica_origami_v2_pre_dpo.pt`)

### Project reorganisation
- Moved all Python scripts → `src/`, token binaries → `data/`, logs → `logs/`, model weights → `models/`
- Created `models/current/`, `models/experiments/{origami,hourglass,baseline,dpo,small}/`, `models/backups/`
- Added convenience symlinks: `mica.pt` → current weights, `mica_tokenizer.json` → current tokenizer
- Updated `.gitignore` to only commit `models/current/*.pt` and `llm/*.onnx` (ignore experiments/backups/binaries)

### ONNX export for browser
- Rewrote `export_onnx_origami.py` to support `--n-layer` and `--ffn-ratio` overrides
- Exported both models: `llm/mica.onnx` / `mica_trace.onnx` (5.4M) and `llm/mica_small.onnx` / `mica_small_trace.onnx` (2.5M)

### Web UI improvements
- Added model switcher buttons to both `index.html` and `trace.html`: **Mica (5.4M)** vs **Cromulent Noir (2.5M)**
- Added **Continue** button that feeds post-processed output back as the next prompt, with spurious-newline stripping for better continuation flow
- Added blurb to `index.html` explaining what Mica is and linking to the trace viewer
- Updated `style.css` with `.model-btn` and `.blurb` styling
- Changed default prompt from a complete sentence to a trailing fragment (`The rain came down hard on`) to encourage continuation

### Training visualisation
- Created `plots/training_n10.html`: interactive Chart.js plot of n=10 cyclic training
- Shows train/val loss across 7 phases with vertical boundary lines
- Hover magnifier snaps to phase interiors (skipping first 500 steps to avoid transition spikes), with tight y-axis framing
- Phase table includes end-of-phase train/val loss columns
- Copied `train_log_n10.csv` into `plots/` for self-contained deployment

### Trace viewer examples
- Added 6 pronoun-resolution prompt buttons to `trace.html` with honest hints about expected attention patterns
- Hints describe actual observed behaviour (e.g. "his → John in layers 4–8", "her → She but bleeds to the and in") rather than aspirational claims

### Small model experiment
- Created `train_small_cyclic.py`: n=5 layers, ffn_ratio=2, ~2.46M params
- Violent curriculum: 10K steps/phase, 4 fold/unfold cycles, only last layer independent
- Gutenberg pre-training to 90K steps (val loss 3.29)
- Noir fine-tune: 14K steps + 30K extended with cosine decay (5e-5 → 1e-7)
- Generates noir fragments with correct voice but weaker coherence than big model; the gap is smaller than the 2× parameter ratio suggests

### README updates
- Added origami folding architecture description
- Added generation example table, browser demo section, DPO section
- Added model organisation diagram (src/, data/, logs/, models/, llm/)
- Added training curves link, GitHub Pages publishing notes
- Embedded `mica.png` screenshot at top of README

## Files Touched

### Core logic
- **`src/generate_origami.py`**: Added `_postprocess()` for BPE artifact cleanup; updated default weights path
- **`src/export_onnx_origami.py`**: Complete rewrite supporting configurable depth and FFN ratio; exports inference + trace ONNX
- **`src/config.py`**: Added `ffn_ratio` field
- **`src/model.py`**: MLP uses `config.ffn_ratio` instead of hardcoded 4×

### Training scripts
- **`src/train_small_cyclic.py`**: New small-model cyclic trainer (n=5, ffn_ratio=2, violent 10K-step phases)
- **`src/train_small_noir.py`**: Noir fine-tune loader for small model
- **`src/train_small_noir_extend.py`**: Extended fine-tune with cosine LR decay
- **`src/train_origami_cyclic.py`**, **`src/train_origami2_cyclic.py`**: Updated log paths to `logs/origami/`
- **`src/dpo_generate.py`**, **`src/dpo_train.py`**: Updated default weights paths

### Data organisation
- **`data/`**: Moved `train.bin`, `val.bin`, `wiki_train.bin`, `wiki_val.bin` here; updated all `np.memmap` paths in scripts
- **`logs/`**: `logs/origami/train_log_*.csv`, `logs/failed/`, `logs/small/`
- **`models/`**: `models/current/`, `models/experiments/{origami,hourglass,baseline,dpo,small}/`, `models/backups/`
- **`plots/`**: `plots/training_n10.html`, `plots/train_log_n10.csv`

### Web UI
- **`llm/index.html`**: Model switcher, blurb, Continue button, trailing-fragment prompt
- **`llm/mica.js`**: Full rewrite with model switching, BPE-correct decode, post-processing, Continue logic
- **`llm/trace.html`**: Model switcher, 6 example prompt buttons with honest hints
- **`llm/trace.js`**: Model switching support, example button wiring
- **`llm/style.css`**: `.model-btn`, `.blurb`, `.trace-link` styles
- **`llm/mica_small.onnx`**, **`llm/mica_small_trace.onnx`**: New small model ONNX exports

### Documentation
- **`README.md`**: Architecture, examples, browser demo, model organisation, training curves link, screenshot
- **`.gitignore`**: Selective tracking of current model + ONNX files only
- **`mica.png`**: Browser generation screenshot embedded in README
