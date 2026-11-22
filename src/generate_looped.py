#!/usr/bin/env python3
"""
Generate text with Mica Looped Transformer.

Usage:
  uv run python src/generate_looped.py "The rain came down hard on"
  uv run python src/generate_looped.py --weights models/experiments/looped_noir/best.pt --temperature 0.7 "The fat man leaned back"
"""
from __future__ import annotations

import argparse
import re
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from model_looped import LoopedTransformer


def _postprocess(text: str) -> str:
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = re.sub(r'([.!?] +)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r'(\n)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = text.replace('_', ' ')
    text = re.sub(r'([.!?;,])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\bThe(man|office|room|hall|door|boy|gun|car|street|house|hotel)\b',
                  r'The \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bA(man|boy|girl|gun|door|room|car|bullet|knife)\b',
                  r'A \1', text, flags=re.IGNORECASE)
    text = re.sub(r' +', ' ', text)
    return text.strip()


@torch.no_grad()
def generate(
    model: LoopedTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    min_new_tokens: int = 10,
    temperature: float = 0.7,
    top_k: int = 40,
) -> str:
    model.eval()
    device = model.config.device
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    prompt_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    natural_stop = False

    for step in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if step < min_new_tokens:
            if eos_id is not None:
                logits[:, eos_id] = -float("inf")
            if eot_id is not None:
                logits[:, eot_id] = -float("inf")

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        tok = nxt.item()

        if tok in (eot_id, eos_id):
            natural_stop = True
            break

        generated.append(tok)
        idx = torch.cat([idx, nxt], dim=1)

    text = tokenizer.decode(prompt_ids + generated)
    if not natural_stop:
        # Backtrack to last sentence boundary if available
        m = list(re.finditer(r'[.!?](?=\s|$)', text))
        if m and m[-1].end() > len(prompt):
            text = text[:m[-1].end()]
    return _postprocess(text)


def main():
    p = argparse.ArgumentParser(description="Generate text with Mica Looped Transformer")
    p.add_argument("prompt", nargs="?", default="The rain came down hard on")
    p.add_argument("--weights", default="models/experiments/looped/best.pt")
    p.add_argument("--tokenizer", default="mica_tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--min-new-tokens", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--n", type=int, default=1, help="Number of samples")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = LoopedTransformer.from_checkpoint(args.weights, device=device)
    print(f"Loaded {args.weights} | layout={model.layout} | params={model.num_params()/1e6:.2f}M\n")

    for i in range(args.n):
        if args.n > 1:
            print(f"--- Sample {i+1} ---")
        print(generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ))
        print()


if __name__ == "__main__":
    main()
