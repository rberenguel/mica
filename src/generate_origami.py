#!/usr/bin/env python3
"""
Generate text with Mica Origami.

Stopping behaviour:
  • Before --min-new-tokens: EOS/EOT are masked out so the model is forced
    to keep going (prevents 3-word outputs on terminal prompts).
  • After --min-new-tokens: the model may stop naturally on EOS/EOT.
  • If --max-new-tokens is reached without a natural stop, the output is
    back-tracked to the last sentence-ending punctuation so it never ends
    mid-word or mid-phrase.
"""
import argparse
import re
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments, restore_folds


def _postprocess(text: str) -> str:
    """Clean up BPE / generation artifacts for readable noir prose."""
    if not text:
        return text

    # 1. Capitalise first character
    text = text[0].upper() + text[1:]

    # 2. Capitalise after sentence boundaries
    text = re.sub(r'([.!?] +)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'(\n)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    # 3. Strip underscores (OCR / formatting artifacts from corpus)
    text = text.replace('_', ' ')

    # 4. Insert missing space after punctuation before next word
    text = re.sub(r'([.!?;,])([A-Za-z])', r'\1 \2', text)

    # 4. Fix common stuck-together BPE words (token "man" instead of " man")
    #    These are the ones actually seen in output.
    text = re.sub(r'\bThe(man|office|room|hall|door|boy|gun|car|street|house|hotel)\b',
                  r'The \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bA(man|boy|girl|gun|door|room|car|bullet|knife)\b',
                  r'A \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Man|Boy|Girl|Gun|Door)([a-z]+)\b',
                  r'\1 \2', text)

    # 5. Collapse multiple spaces
    text = re.sub(r' +', ' ', text)

    # 6. Strip isolated number garbage at line edges (e.g. "00.")
    text = re.sub(r'^\d+\.? *', '', text, flags=re.MULTILINE)
    text = re.sub(r' *\d+\.?$', '', text, flags=re.MULTILINE)

    return text

parser = argparse.ArgumentParser(description="Generate text with Mica Origami")
parser.add_argument("prompt", nargs="?", default="Suddenly there came a ", help="Prompt text")
parser.add_argument("--max-new-tokens", type=int,   default=None,                  help="Tokens to generate (default: block_size)")
parser.add_argument("--temperature",    type=float, default=0.8,                   help="Sampling temperature")
parser.add_argument("--weights",                    default="models/current/mica_origami_v2.pt", help="Path to weights")
parser.add_argument("--tokenizer",                  default="mica_tokenizer.json", help="Path to tokenizer")
parser.add_argument("--min-new-tokens", type=int,   default=5,                     help="Minimum tokens before natural stop is allowed")
parser.add_argument("--n-layer",        type=int,   default=None,                  help="Override config.n_layer (for non-standard depth checkpoints)")
parser.add_argument("--ffn-ratio",      type=int,   default=None,                  help="Override config.ffn_ratio (e.g. 2 for small models)")
args = parser.parse_args()

device    = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = Tokenizer.from_file(args.tokenizer)

print("Loading model...")
config = MicaConfig()
if args.n_layer is not None:
    config.n_layer = args.n_layer
if args.ffn_ratio is not None:
    config.ffn_ratio = args.ffn_ratio
model = OrigamiTransformer(config)

ckpt = torch.load(args.weights, map_location=device, weights_only=False)

# ── Read saved config if present ────────────────────────────────────────
if isinstance(ckpt, dict) and 'config' in ckpt:
    saved_cfg = ckpt['config']
    if hasattr(saved_cfg, 'n_layer'):
        config.n_layer = saved_cfg.n_layer
    if hasattr(saved_cfg, 'ffn_ratio'):
        config.ffn_ratio = saved_cfg.ffn_ratio
    if hasattr(saved_cfg, 'dropout'):
        config.dropout = saved_cfg.dropout
    # Rebuild model with correct architecture
    model = OrigamiTransformer(config)

# ── Detect checkpoint format ────────────────────────────────────────────
if isinstance(ckpt, dict) and 'student' in ckpt:
    sd = ckpt['student']
    print("  Detected teacher-student checkpoint, loading student encoder...")
    # If the student was a LatentOrigamiTransformer, wte has vocab_size+1 rows.
    if 'transformer.wte.weight' in sd and sd['transformer.wte.weight'].shape[0] == config.vocab_size + 1:
        print("  Slicing MASK token embedding (vocab+1 → vocab)")
        sd['transformer.wte.weight'] = sd['transformer.wte.weight'][:config.vocab_size, :]
    # Strip any predictor keys (from pure-latent runs)
    sd = {k: v for k, v in sd.items() if not k.startswith('predictor')}
    # Teacher-student checkpoints don't store segments separately;
    # the model's segments are baked into the state dict via physical block keys.
    # Default to [1 1 1 1 1] for n_layer=5 or derive from config.
    model.segments = [(i, 1) for i in range(config.n_layer)]
elif isinstance(ckpt, dict) and 'model' in ckpt:
    sd = ckpt['model']
else:
    sd = ckpt

# ── Auto-detect ffn_ratio from MLP weight shapes ──────────────────────────
if args.ffn_ratio is None:
    for k, v in sd.items():
        if 'mlp.c_fc.weight' in k:
            inferred_ffn = v.shape[0] // config.n_embd
            if inferred_ffn != config.ffn_ratio:
                print(f"  Auto-detected ffn_ratio={inferred_ffn} from checkpoint (config has {config.ffn_ratio})")
                config.ffn_ratio = inferred_ffn
                model = OrigamiTransformer(config)
            break

# ── Load weights ──────────────────────────────────────────────────────────
model.load_state_dict(sd, strict=False)

if isinstance(ckpt, dict) and 'segments' in ckpt:
    restore_segments(model, ckpt['segments'])
elif isinstance(ckpt, dict) and 'bottom_loops' in ckpt:
    restore_folds(model, ckpt.get('bottom_loops', 1), ckpt.get('top_loops', 1))

model.to(device)
model.eval()

eot_id         = tokenizer.token_to_id("<|endoftext|>")
eos_id         = tokenizer.token_to_id("<|eos|>")
max_new_tokens = args.max_new_tokens or config.block_size
idx            = torch.tensor(tokenizer.encode(args.prompt).ids, dtype=torch.long).unsqueeze(0).to(device)

print(f"\n--- Generating (max={max_new_tokens}, temp={args.temperature}, layout={model.layout}) ---")
print(args.prompt, end="", flush=True)

generated_ids = []
generated     = 0
stopped_early = False

with torch.no_grad():
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / args.temperature

        # Force continuation while we're below the minimum length
        if generated < args.min_new_tokens:
            logits[:, eos_id] = float('-inf')
            logits[:, eot_id] = float('-inf')

        probs    = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        tok      = idx_next.item()

        # Natural stop on a boundary token once we're past the minimum
        if (tok == eot_id or tok == eos_id) and generated >= args.min_new_tokens:
            stopped_early = True
            break

        generated_ids.append(tok)
        idx = torch.cat((idx, idx_next), dim=1)
        generated += 1

# Decode the full token sequence in one shot so BPE merging is correct
full_text = tokenizer.decode(generated_ids)

# If we hit the hard limit without a natural stop, trim to the last real
# sentence boundary so we don't end mid-word.  Fall back to the last space
# if no sentence-ending punctuation exists.
if not stopped_early and generated >= max_new_tokens:
    matches = list(re.finditer(r'[.!?]', full_text))
    if matches:
        cut = matches[-1].end()
        full_text = full_text[:cut]
    else:
        last_space = full_text.rfind(' ')
        if last_space > 0:
            full_text = full_text[:last_space]

full_text = _postprocess(full_text)
print(full_text, end="")
print("\n\n--- Done ---")
