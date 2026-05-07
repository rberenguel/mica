import torch
import torch.nn as nn
from torch.nn import functional as F
from config import MicaConfig
from model import MultiResolutionAttention, MLP


class HourglassBlock(nn.Module):
    def __init__(self, config, max_loops: int = 3):
        super().__init__()
        self.max_loops = max_loops
        # Each loop iteration gets its own LayerNorm parameters.
        # This lets shared attention/MLP weights operate in different
        # normalised spaces per pass, dramatically increasing expressiveness
        # at negligible parameter cost (~2 × 192 per block per extra loop).
        self.ln_1 = nn.ModuleList([
            nn.LayerNorm(config.n_embd, bias=config.bias) for _ in range(max_loops)
        ])
        self.ln_2 = nn.ModuleList([
            nn.LayerNorm(config.n_embd, bias=config.bias) for _ in range(max_loops)
        ])
        self.attn = MultiResolutionAttention(config)
        self.mlp  = MLP(config)
        # Only needed when this block will be called with loop_idx > 0.
        self.depth_embed = nn.Embedding(max_loops - 1, config.n_embd) if max_loops > 1 else None
        if self.depth_embed is not None:
            nn.init.zeros_(self.depth_embed.weight)

    def forward(self, x, start_pos: int = 0, loop_idx: int = 0):
        assert 0 <= loop_idx < self.max_loops, (
            f"loop_idx {loop_idx} out of range [0, {self.max_loops})"
        )
        if loop_idx > 0 and self.depth_embed is not None:
            x = x + self.depth_embed(torch.tensor(loop_idx - 1, device=x.device))
        x = x + self.attn(self.ln_1[loop_idx](x), start_pos)
        x = x + self.mlp(self.ln_2[loop_idx](x))
        return x


class HourglassTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n = config.n_layer

        # max_loops=3 gives us headroom for [2+6+2] or [3+4+3] folds.
        # Bridge blocks never use loop_idx > 0, so the extra LNs sit idle
        # (no gradients, ~6K params total across all layers — negligible).
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([HourglassBlock(config, max_loops=3) for _ in range(n)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Zone state — updated by apply_fold()
        self.bottom_loops = 1
        self.top_loops    = 1
        self.bridge_start = 1
        self.bridge_end   = n - 1  # exclusive

        self.apply(self._init_weights)
        # Re-zero depth embeddings: _init_weights overwrites them with N(0, 0.02)
        for block in self.transformer.h:
            if block.depth_embed is not None:
                nn.init.zeros_(block.depth_embed.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos: int = 0):
        b, t = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        h = self.transformer.h

        for i in range(self.bottom_loops):
            x = h[0](x, start_pos, loop_idx=i)

        for block in h[self.bridge_start:self.bridge_end]:
            x = block(x, start_pos)

        for i in range(self.top_loops):
            x = h[-1](x, start_pos, loop_idx=i)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss

    @property
    def layout(self):
        bridge = self.bridge_end - self.bridge_start
        return f"{self.bottom_loops}+{bridge}+{self.top_loops}"


# ── Fold helpers ──────────────────────────────────────────────────────────────

def soft_tie(target: HourglassBlock, source: HourglassBlock):
    """Copy source weights into target before a hard fold to prevent loss spikes."""
    target.load_state_dict(source.state_dict())


def apply_fold(model: HourglassTransformer) -> int:
    """
    Advance one fold step: 1-loop → 2-loop, or 2-loop → 3-loop.
    Assumes soft_tie has already been called on the affected layers.
    Returns the new loop count, or 0 if already at max (3 loops).
    """
    bl = model.bottom_loops
    if bl >= 3:
        return 0
    n = model.config.n_layer
    h = model.transformer.h
    h[bl]           = h[0]      # bottom: next outer layer becomes the anchor
    h[n - 1 - bl]   = h[n - 1]  # top: symmetric
    model.bottom_loops += 1
    model.top_loops    += 1
    model.bridge_start  = model.bottom_loops
    model.bridge_end    = n - model.top_loops
    return model.bottom_loops


def restore_folds(model: HourglassTransformer, bottom_loops: int, top_loops: int):
    """Re-apply hard ties after loading a checkpoint. Call after load_state_dict()."""
    n = model.config.n_layer
    h = model.transformer.h
    for i in range(1, bottom_loops):
        h[i] = h[0]
    for i in range(1, top_loops):
        h[n - 1 - i] = h[n - 1]
    model.bottom_loops = bottom_loops
    model.top_loops    = top_loops
    model.bridge_start = bottom_loops
    model.bridge_end   = n - top_loops
