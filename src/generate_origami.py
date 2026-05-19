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
args = parser.parse_args()

device    = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = Tokenizer.from_file(args.tokenizer)

print("Loading model...")
config = MicaConfig()
if args.n_layer is not None:
    config.n_layer = args.n_layer
model = OrigamiTransformer(config)

ckpt = torch.load(args.weights, map_location=device)
if isinstance(ckpt, dict) and 'model' in ckpt:
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        restore_segments(model, ckpt['segments'])
    else:
        restore_folds(model, ckpt.get('bottom_loops', 1), ckpt.get('top_loops', 1))
else:
    model.load_state_dict(ckpt, strict=False)

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
