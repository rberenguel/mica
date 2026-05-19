#!/usr/bin/env python3
"""
Generate text with Mica (baseline 10-layer model).

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
from model import MicaTransformer

parser = argparse.ArgumentParser(description="Generate text with Mica")
parser.add_argument("prompt", nargs="?", default="Suddenly there came a ", help="Prompt text")
parser.add_argument("--max-new-tokens", type=int,   default=None,                  help="Tokens to generate (default: block_size)")
parser.add_argument("--temperature",    type=float, default=0.8,                   help="Sampling temperature")
parser.add_argument("--weights",                    default="mica_weights.pt",     help="Path to weights")
parser.add_argument("--tokenizer",                  default="mica_tokenizer.json", help="Path to tokenizer")
parser.add_argument("--min-new-tokens", type=int,   default=5,                     help="Minimum tokens before natural stop is allowed")
args = parser.parse_args()

device    = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = Tokenizer.from_file(args.tokenizer)

print("Loading model...")
config = MicaConfig()
model  = MicaTransformer(config)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.to(device)
model.eval()

eot_id         = tokenizer.token_to_id("<|endoftext|>")
eos_id         = tokenizer.token_to_id("<|eos|>")
max_new_tokens = args.max_new_tokens or config.block_size
idx            = torch.tensor(tokenizer.encode(args.prompt).ids, dtype=torch.long).unsqueeze(0).to(device)

print(f"\n--- Generating (max={max_new_tokens}, temp={args.temperature}) ---")
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

print(full_text, end="")
print("\n\n--- Done ---")
