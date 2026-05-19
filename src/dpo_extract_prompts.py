#!/usr/bin/env python3
"""
Extract good DPO prompts from the noir corpus.

Samples random sentence prefixes from train.bin that are:
  - 5–15 tokens long (enough context, not too much)
  - Don't end with EOS/EOT (so the model must continue, not stop)
  - Come from varied positions in the text

Usage:
  uv run python dpo_extract_prompts.py --output dpo_prompts_corpus.txt --n 30
"""
import argparse
import numpy as np
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="dpo_prompts_corpus.txt")
parser.add_argument("--n", type=int, default=30, help="Number of prompts to extract")
parser.add_argument("--min-len", type=int, default=5, help="Min prompt length in tokens")
parser.add_argument("--max-len", type=int, default=15, help="Max prompt length in tokens")
args = parser.parse_args()

tokenizer = Tokenizer.from_file("mica_tokenizer.json")
eos_id = tokenizer.token_to_id("<|eos|>")
eot_id = tokenizer.token_to_id("<|endoftext|>")

data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")

prompts = []
attempts = 0
while len(prompts) < args.n and attempts < args.n * 100:
    attempts += 1
    # pick a random position with room for prefix + continuation
    pos = np.random.randint(len(data) - 256)
    # find the next sentence boundary after pos
    end = pos
    while end < len(data) and data[end] not in (eos_id, eot_id):
        end += 1
    # walk back to find a good prefix length
    length = np.random.randint(args.min_len, args.max_len + 1)
    start = max(0, end - length)
    prefix_ids = data[start:end].tolist()
    prefix_text = tokenizer.decode(prefix_ids).strip()
    # filters
    if len(prefix_text) < 10:
        continue
    if prefix_text.endswith('.'):
        continue  # terminal prompt → immediate EOS
    if prefix_text.count('"') % 2 == 1:
        continue  # unclosed quote
    if prefix_text in prompts:
        continue
    prompts.append(prefix_text)

with open(args.output, 'w') as f:
    for p in prompts:
        f.write(p + '\n')

print(f"Extracted {len(prompts)} corpus-derived prompts to {args.output}")
for i, p in enumerate(prompts[:10], 1):
    print(f"  {i}. {p}")
