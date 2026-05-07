# Progressive Hourglass Folding — Implementation Plan

## Experiment sequence

1. **Uneven heads baseline** (current state) — train and evaluate the
   192-dim multi-resolution attention model at depth 10. This is the
   comparison baseline.

2. **Depth-10 hourglass** — fold the outer layers of the existing 10-layer
   stack without changing `n_layer`. No config change, clean comparison.
   This is the first hourglass experiment.

3. **Depth-16 hourglass** (optional, later) — expand to 16 layers and run
   the full blueprint. Only worth attempting if depth-10 shows a clear win.

---

## What this is

A training curriculum that progressively ties layer weights together ("folds"),
producing a parameter-efficient "Sandwich Recurrent" architecture where outer
layers share weights across multiple passes and learn per-pass depth embeddings.

The rationale: starting with all layers independent lets the model build stable
weight manifolds before the folds force sharing, which acts as a strong
regulariser. Depth embeddings give the shared layer enough signal to behave
differently on each pass.

---

## Depth-10 hourglass (first experiment)

With `n_layer=10` the natural split is:

```
[ Bottom loop: 1 physical block × 2 iterations ]   (layers 0, 1)
[ Middle bridge: 6 independent blocks           ]   (layers 2–7)
[ Top loop:    1 physical block × 2 iterations ]   (layers 8, 9)
```

Physical layers: 8 (1 + 6 + 1).  Functional depth: 10 (2 + 6 + 2).

Alternatively, a 3-loop variant:

```
[ Bottom loop: 1 physical block × 3 iterations ]   (layers 0, 1, 2)
[ Middle bridge: 4 independent blocks           ]   (layers 3–6)
[ Top loop:    1 physical block × 3 iterations ]   (layers 7, 8, 9)
```

Physical layers: 6 (1 + 4 + 1).  Functional depth: 10 (3 + 4 + 3).

The 2-loop variant is less aggressive and safer for a first run. The 3-loop
variant compresses more and is closer to the full blueprint spirit. TBD which
to try first.

**No `config.py` change needed for either.** The folding happens at training
time by aliasing `ModuleList` entries.

---

## Depth-16 hourglass (later)

The full blueprint target:

```
[ Bottom loop: 1 physical block × 3 iterations ]   (layers 0–2)
[ Middle bridge: 10 independent blocks          ]   (layers 3–12)
[ Top loop:    1 physical block × 3 iterations ]   (layers 13–15)
```

Physical layers: 12.  Functional depth: 16.

Note: the current depth-10 middle bridge maps exactly to the 10-layer middle
bridge here. If the depth-10 hourglass trains well, its middle bridge weights
could seed the depth-16 run (with 3 new layers added at each end).

Requires `n_layer = 16` in `config.py` — a breaking change, existing
checkpoints are not reusable.

---

## File strategy — keep the standard model intact

The hourglass lives in its own files. The existing `model.py`, `config.py`,
`train.py`, and `train_phase1.py` are never touched, so switching back to the
standard model is just a matter of which script you run.

| New file | Purpose |
|---|---|
| `model_hourglass.py` | `HourglassBlock` (with depth embedding), `HourglassTransformer` (zone-aware forward + fold helpers) |
| `train_hourglass.py` | Full folding curriculum: warmup → soft-tie → hard-fold × N, saves to `mica_hourglass.pt` |
| `generate_hourglass.py` | Identical generation loop, but imports `HourglassTransformer` instead of `MicaTransformer` |

| Unchanged file | Reason |
|---|---|
| `config.py` / `model.py` | Standard model stays exactly as-is |
| `train.py` / `train_phase1.py` | Standard training pipeline unaffected |
| `generate.py` | Unchanged — still generates from the standard model |
| `prepare.py` / `prepare_wiki.py` | Same tokenised data, shared by both models |
| ONNX export scripts | Will need a hourglass-aware variant later, but not blocked |

`generate.py` hardcodes `from model import MicaTransformer`. The generation
loop itself is identical for both models (same external API: `model(idx)` →
`(logits, loss)`), so `generate_hourglass.py` is just `generate.py` with the
import and instantiation swapped. No logic duplication.

`HourglassTransformer` can import `MultiResolutionAttention` and `MLP` directly
from `model.py` — no duplication of those components.

For depth-16 only: override `n_layer = 16` inside `train_hourglass.py` by
constructing the config locally rather than touching `config.py`.

---

## Folding mechanics

### Epoch → step mapping

Training is step-based (random sampling). Map the curriculum:

| Phase | Suggested steps | Note |
|---|---|---|
| Warmup (independent) | 5 000 | Stable manifold before any fold |
| Soft-tie window | 300 | Increase to 500 if loss spikes |
| Post-fold training | 5 000 | Per fold pair |

### Soft-tie

Before each hard fold, copy weights to prevent the loss spike from suddenly
merging two diverged layers:

```python
def soft_tie(target_block, source_block):
    target_block.load_state_dict(source_block.state_dict())
```

Train for the soft-tie window with independent gradient updates.

### Hard-tie

Point the `ModuleList` entry to the same object:

```python
model.transformer.h[1] = model.transformer.h[0]
```

PyTorch's `named_parameters()` deduplicates by tensor identity — the shared
parameters appear once in the optimizer, no double updates.

**Pre-conditions (both already satisfied):**
- Pre-Norm (LayerNorm before attention/FFN) — ✓
- Gradient clipping `max_norm=1.0` — ✓

### Depth-10, 2-loop folding schedule

```
Warmup (5 000 steps): h[0..9] all independent

Soft-tie (300 steps):
  h[1].weights ← h[0].weights
  h[8].weights ← h[9].weights

Hard fold:
  h[1] = h[0]   → bottom loops twice
  h[8] = h[9]   → top loops twice
  (continue 5 000 steps)
```

### Depth-10, 3-loop folding schedule

```
Warmup (5 000 steps): h[0..9] all independent

Soft-tie (300 steps):
  h[1].weights ← h[0].weights
  h[8].weights ← h[9].weights

First fold:
  h[1] = h[0]   → bottom loops twice
  h[8] = h[9]   → top loops twice
  (continue 5 000 steps)

Soft-tie (300 steps):
  h[2].weights ← h[0].weights
  h[7].weights ← h[9].weights

Second fold:
  h[2] = h[0]   → bottom loops three times
  h[7] = h[9]   → top loops three times
  (continue 5 000 steps)
```

---

## Depth embeddings

### When to add them

At the first hard fold. A loop-index embedding on a block that only executes
once is meaningless and wastes parameters.

### Block modification

```python
class Block(nn.Module):
    def __init__(self, config, n_loops: int = 1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiResolutionAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)
        self.depth_embed = nn.Embedding(n_loops, config.n_embd) if n_loops > 1 else None
        if self.depth_embed is not None:
            nn.init.normal_(self.depth_embed.weight, std=0.02)  # match rest of model

    def forward(self, x, start_pos: int = 0, loop_idx: int = 0):
        if self.depth_embed is not None:
            x = x + self.depth_embed(torch.tensor(loop_idx, device=x.device))
        x = x + self.attn(self.ln_1(x), start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x
```

`n_loops=2` or `3` for the outer blocks after folding; `n_loops=1` (default,
no embedding) for the middle bridge.

### Forward pass after final fold (depth-10, 2-loop)

```python
for i in range(2):
    x = self.transformer.h[0](x, start_pos, loop_idx=i)   # bottom
for block in self.transformer.h[2:8]:
    x = block(x, start_pos)                                # bridge
for i in range(2):
    x = self.transformer.h[9](x, start_pos, loop_idx=i)   # top
```

---

## Risks and open questions

**Fold stability**: soft-tie prevents spikes but doesn't guarantee smooth loss.
If loss rises sharply on fold, extend the soft-tie window to 500–1000 steps.

**Depth embedding init**: default `nn.Embedding` is `N(0,1)`, which at fold
time adds a large perturbation to `x`. Initialise to `std=0.02` (already in
the code sketch above).

**ModuleList index fragility**: after folding, bridge indices depend on the
loop count. Using raw slices like `h[2:8]` is brittle if the schedule changes.
At fold time, cleanly rename to three separate lists (`bottom`, `bridge`,
`top`) in `MicaTransformer` to make the forward unambiguous.

**ONNX trace exporter**: `export_onnx_trace.py` iterates blocks 1:1 with
`attn_i` output indices. Once the forward is loop-aware, the exporter needs
updating — a looped block produces multiple attention matrices per "logical
layer". Plan: emit `attn_bottom_0`, `attn_bottom_1`, etc., and update
`trace.js` to handle the new names.

**Checkpoint resume mid-curriculum**: after a hard fold, `h[1]` and `h[0]`
share the same tensor. `state_dict()` stores only one copy. Resuming requires
the loading code to know about the tie and re-apply it after loading, otherwise
the two entries diverge silently. Document this in whatever training script
drives the folding.
