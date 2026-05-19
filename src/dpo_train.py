#!/usr/bin/env python3
"""
DPO fine-tuning for Mica Origami v2.

Usage:
  uv run python dpo_train.py --pairs dpo_pairs.json --output mica_dpo.pt

Loads the active model and a frozen reference copy, then runs DPO on the
preference pairs.  The reference model prevents over-optimisation — it acts
as a KL-regulariser keeping the model from drifting too far from coherent text.
"""
import argparse
import json
import math
import copy
import torch
from torch.nn import functional as F
from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments, restore_folds

parser = argparse.ArgumentParser()
parser.add_argument("--pairs",    required=True, help="Path to dpo_pairs.json")
parser.add_argument("--weights",  default="models/current/mica_origami_v2.pt", help="Base model weights")
parser.add_argument("--output",   default="mica_dpo.pt")
parser.add_argument("--beta",     type=float, default=0.1,  help="DPO temperature (higher = stronger preference push)")
parser.add_argument("--steps",    type=int,   default=500,  help="Training steps")
parser.add_argument("--lr",       type=float, default=1e-5, help="Max learning rate")
parser.add_argument("--batch-size",type=int,  default=1,    help="Pairs per step (gradient accumulation)")
parser.add_argument("--rank-weight", action="store_true", help="Weight loss by rank gap (larger gaps = stronger signal)")
parser.add_argument("--device",   default=None)
args = parser.parse_args()

device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')

# ── Load dataset ─────────────────────────────────────────────────────────────
with open(args.pairs) as f:
    raw = json.load(f)

# Support three formats:
# 1. Good/bad: [{"prompt": "...", "good": [...], "bad": [...]}, ...]
# 2. Rankings: [{"prompt": "...", "ranking": ["best", ..., "worst"]}, ...]
# 3. Legacy pairs: [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]
dataset = []
if isinstance(raw, list) and len(raw) > 0:
    if 'good' in raw[0] and 'bad' in raw[0]:
        # Tri-state good/bad format
        for item in raw:
            for g in item['good']:
                for b in item['bad']:
                    dataset.append({
                        'prompt': item['prompt'],
                        'chosen': g,
                        'rejected': b,
                        'rank_gap': 1
                    })
        print(f"Loaded {len(raw)} tri-state groups → expanded to {len(dataset)} pairwise pairs")
    elif 'ranking' in raw[0]:
        # Expand rankings into all pairwise combinations
        for item in raw:
            ranking = item['ranking']
            n = len(ranking)
            for i in range(n):
                for j in range(i + 1, n):
                    dataset.append({
                        'prompt': item['prompt'],
                        'chosen': ranking[i],
                        'rejected': ranking[j],
                        'rank_gap': j - i
                    })
        print(f"Loaded {len(raw)} rankings → expanded to {len(dataset)} pairwise pairs")
    else:
        dataset = raw
        print(f"Loaded {len(dataset)} preference pairs from {args.pairs}")
else:
    dataset = raw
    print(f"Loaded {len(dataset)} preference pairs from {args.pairs}")

if len(dataset) < 10:
    print("WARNING: DPO needs at least 20–50 pairs to work well.")

# ── Build models ─────────────────────────────────────────────────────────────
config = MicaConfig()

# Active model (gets gradients)
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

# Frozen reference model (no grads, ever)
ref_model = copy.deepcopy(model)
for p in ref_model.parameters():
    p.requires_grad_(False)
ref_model.eval()

print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  layout={model.layout}")

# Tokenizer — must match the model's vocab
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("mica_tokenizer.json")

eot_id = tokenizer.token_to_id("<|endoftext|>")
eos_id = tokenizer.token_to_id("<|eos|>")

def encode(text: str) -> list[int]:
    """Encode text to token IDs, strip trailing boundary tokens."""
    ids = tokenizer.encode(text).ids
    # strip trailing eos/eot so we don't compute loss on them
    while ids and ids[-1] in (eos_id, eot_id):
        ids.pop()
    return ids

# ── Log-probability helper ──────────────────────────────────────────────────
@torch.no_grad()
def log_prob_ref(prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """Sum of log P(token_t | token_<t) for completion tokens only."""
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    targets = ids.clone()
    targets[0, :len(prompt_ids)] = -1  # mask prompt
    logits, loss = ref_model(ids, targets)
    # loss is mean over non-masked positions; recover sum
    n_completion = len(completion_ids)
    return -loss * n_completion


def log_prob_active(prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """Same, but for the trainable model."""
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    targets = ids.clone()
    targets[0, :len(prompt_ids)] = -1
    logits, loss = model(ids, targets)
    n_completion = len(completion_ids)
    return -loss * n_completion


# ── Optimiser ────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

# ── Training loop ───────────────────────────────────────────────────────────
model.train()
for step in range(args.steps):
    # sample a batch of pairs
    losses = []
    ratios = []
    weights = []
    for b in range(args.batch_size):
        pair = dataset[(step * args.batch_size + b) % len(dataset)]
        prompt_ids   = encode(pair["prompt"])
        chosen_ids   = encode(pair["chosen"])
        rejected_ids = encode(pair["rejected"])

        log_pi_chosen   = log_prob_active(prompt_ids, chosen_ids)
        log_pi_rejected = log_prob_active(prompt_ids, rejected_ids)

        with torch.no_grad():
            log_ref_chosen   = log_prob_ref(prompt_ids, chosen_ids)
            log_ref_rejected = log_prob_ref(prompt_ids, rejected_ids)

        ratio = (log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected)
        loss = -F.logsigmoid(args.beta * ratio)
        losses.append(loss)
        ratios.append(ratio.item())
        # weight by rank gap if available and requested
        gap = pair.get('rank_gap', 1)
        weights.append(gap if args.rank_weight else 1.0)

    weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum() * len(weights_t)  # normalise so mean weight = 1
    batch_loss = (torch.stack(losses) * weights_t).mean()
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0 or step == args.steps - 1:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Step {step:>4}: loss {batch_loss.item():.4f}  ratio {avg_ratio:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")

# ── Save ─────────────────────────────────────────────────────────────────────
torch.save({
    'model':        model.state_dict(),
    'layout':       model.layout,
    'segments':     model.segments,
}, args.output)
print(f"\nDPO complete. Saved to {args.output}")
