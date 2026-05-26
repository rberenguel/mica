#!/usr/bin/env python3
"""
Check if generated text is memorised verbatim from the corpus.
Generates multiple completions and searches for exact or near-exact matches
in the raw corpus text.
"""
import argparse
import re
import torch
from tokenizers import Tokenizer
from config import MicaConfig
from model_origami import OrigamiTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--corpus", default="corpus/corpus.txt")
parser.add_argument("--prompts", nargs="+", default=["The fat man leaned back", "I lit a cigarette", "She walked into the room"])
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--max-new-tokens", type=int, default=60)
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = Tokenizer.from_file("mica_tokenizer.json")

print("Loading model...")
config = MicaConfig()
model = OrigamiTransformer(config)
ckpt = torch.load(args.weights, map_location=device, weights_only=False)

# Auto-detect ffn_ratio from checkpoint
sd = ckpt.get('student', ckpt.get('model', ckpt))
for k, v in sd.items():
    if 'mlp.c_fc.weight' in k:
        inferred_ffn = v.shape[0] // config.n_embd
        if inferred_ffn != config.ffn_ratio:
            print(f"  Auto-detected ffn_ratio={inferred_ffn}")
            config.ffn_ratio = inferred_ffn
            model = OrigamiTransformer(config)
        break

if isinstance(ckpt, dict) and 'student' in ckpt:
    sd = ckpt['student']
    if 'transformer.wte.weight' in sd and sd['transformer.wte.weight'].shape[0] == config.vocab_size + 1:
        sd['transformer.wte.weight'] = sd['transformer.wte.weight'][:config.vocab_size, :]
    sd = {k: v for k, v in sd.items() if not k.startswith('predictor')}
    model.load_state_dict(sd, strict=False)
    model.segments = [(i, 1) for i in range(config.n_layer)]
elif isinstance(ckpt, dict) and 'model' in ckpt:
    model.load_state_dict(ckpt['model'], strict=False)
    if 'segments' in ckpt:
        from model_origami import restore_segments
        restore_segments(model, ckpt['segments'])
else:
    model.load_state_dict(ckpt, strict=False)

model.to(device)
model.eval()

print(f"Loading corpus: {args.corpus}")
with open(args.corpus, 'r', encoding='utf-8') as f:
    corpus_text = f.read()

# Strip some normalisation for fair matching
def normalise(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

corpus_norm = normalise(corpus_text)

@torch.no_grad()
def generate(prompt, max_new=60, temp=0.8):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(x)
        logits = logits[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    text = tokenizer.decode(x[0].tolist())
    return text

print(f"\nChecking {len(args.prompts)} prompts × {args.n_samples} samples each")
print("=" * 60)

for prompt in args.prompts:
    print(f"\nPROMPT: {prompt}")
    matches = 0
    for i in range(args.n_samples):
        text = generate(prompt, args.max_new_tokens, args.temperature)
        # Extract the generated part (after the prompt)
        generated = text[len(prompt):].strip()
        # Check for exact match of any 10-word span
        words = generated.split()
        found = False
        for j in range(len(words) - 10):
            span = ' '.join(words[j:j+10])
            span_norm = normalise(span)
            if span_norm in corpus_norm:
                found = True
                matches += 1
                break
        status = "MATCH" if found else "new"
        print(f"  [{i+1}] {status}: {generated[:80]}...")
    print(f"  → {matches}/{args.n_samples} had 10-word matches in corpus")
