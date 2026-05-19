#!/usr/bin/env python3
"""
Generate DPO candidate completions from a list of prompts.

Usage:
  uv run python dpo_generate.py --prompts prompts.txt --output dpo_candidates.json

Outputs JSON:
  [
    {
      "prompt": "The fat man leaned back",
      "completions": [
        {"text": "...", "temperature": 0.7, "seed": 0},
        ...
      ]
    },
    ...
  ]
"""
import argparse
import json
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments, restore_folds

parser = argparse.ArgumentParser()
parser.add_argument("--prompts",       required=True, help="File with one prompt per line")
parser.add_argument("--output",        default="dpo_candidates.json")
parser.add_argument("--weights",       default="models/current/mica_origami_v2.pt")
parser.add_argument("--n-per-prompt",  type=int, default=5, help="Completions per prompt")
parser.add_argument("--max-tokens",    type=int, default=80)
parser.add_argument("--min-tokens",    type=int, default=20, help="Minimum tokens before EOS/EOT is allowed")
parser.add_argument("--temperatures",  nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 1.0])
parser.add_argument("--device",        default=None)
args = parser.parse_args()

device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = Tokenizer.from_file("mica_tokenizer.json")

config = MicaConfig()
model = OrigamiTransformer(config)

checkpoint = torch.load(args.weights, map_location=device)
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'], strict=False)
    if 'segments' in checkpoint:
        restore_segments(model, checkpoint['segments'])
    else:
        restore_folds(model, checkpoint.get('bottom_loops', 1), checkpoint.get('top_loops', 1))
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()

eot_id = tokenizer.token_to_id("<|endoftext|>")
eos_id = tokenizer.token_to_id("<|eos|>")

import re

def generate_one(prompt: str, temperature: float, seed: int) -> str:
    torch.manual_seed(seed)
    idx = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
    tokens = []
    stopped = False
    with torch.no_grad():
        for i in range(args.max_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            # force continuation until minimum length, then allow natural stop
            if i < args.min_tokens:
                logits[:, eos_id] = float('-inf')
                logits[:, eot_id] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tok = idx_next.item()
            if i >= args.min_tokens and (tok == eos_id or tok == eot_id):
                stopped = True
                break
            tokens.append(tok)
            idx = torch.cat((idx, idx_next), dim=1)

    text = tokenizer.decode(tokens)
    # backtrack to last sentence boundary if we hit the hard limit
    if not stopped:
        matches = list(re.finditer(r'[.!?]', text))
        if matches:
            text = text[:matches[-1].end()]
        else:
            last_space = text.rfind(' ')
            if last_space > 0:
                text = text[:last_space]
    return text

with open(args.prompts) as f:
    prompts = [line.strip() for line in f if line.strip()]

results = []
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")
    completions = []
    for j in range(args.n_per_prompt):
        temp = args.temperatures[j % len(args.temperatures)]
        text = generate_one(prompt, temp, seed=j)
        completions.append({"text": text, "temperature": temp, "seed": j})
        print(f"  t={temp:.1f} → {text[:80]}...")
    results.append({"prompt": prompt, "completions": completions})

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} prompts × {args.n_per_prompt} completions to {args.output}")
print("Next: open llm/rank.html in a browser and load the JSON.")
