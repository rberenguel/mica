"""
Mica Looped Transformer — Prelude → Looped Core → Coda with Origami Kickstart.

Architecture:
  - Prelude:     1 independent block (lifts raw embeddings into latent space)
  - Looped Core: starts as K independent blocks (Origami kickstart), then soft-folds
                 and hard-folds permanently into 1 physical block looped K times,
                 preserving each unfolded block's RMSNorms as loop-specific norms.
  - Coda:        1 independent block (dedicated to next-token prediction)

Block upgrades over v1/v2:
  - Wider residual stream (d_model=256 default)
  - Fused QKV Multi-Resolution Attention (2x64 + 2x32 + 4x16) or Standard MHA
  - RMSNorm + QK-Norm + SwiGLU + RoPE
  - Gated input injection (x_prelude) + learned loop embedding per iteration
"""
from __future__ import annotations

import dataclasses
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclasses.dataclass
class LoopedConfig:
    vocab_size: int = 5000
    block_size: int = 256
    n_embd: int = 256
    ffn_hidden: int = 704      # SwiGLU hidden dim (~2.75x n_embd, multiple of 64)
    n_loops: int = 4           # Core loop count (effective depth = 2 + n_loops)
    attn_type: str = "multires"  # "multires" or "mha"
    n_head: int = 4            # Used when attn_type == "mha"
    dropout: float = 0.1
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        T = q.shape[2]
        cos = self.cos_cache[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin_cache[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0).to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, config: LoopedConfig):
        super().__init__()
        self.w12 = nn.Linear(config.n_embd, 2 * config.ffn_hidden, bias=False)
        self.w3 = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.dropout(self.w3(F.silu(a) * b))


class FusedMultiResolutionAttention(nn.Module):
    """
    Single fused QKV matmul split into three attention resolutions.
    For d_model=256:
      - Large:  2 heads x 64 dims (128) — semantic tracking
      - Medium: 2 heads x 32 dims (64)  — context integration
      - Small:  4 heads x 16 dims (64)  — syntactic filtering
    """
    def __init__(self, config: LoopedConfig):
        super().__init__()
        assert config.n_embd == 256, (
            f"FusedMultiResolutionAttention expects n_embd=256, got {config.n_embd}"
        )
        d = config.n_embd
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

        self.qn_l, self.kn_l = RMSNorm(64), RMSNorm(64)
        self.qn_m, self.kn_m = RMSNorm(32), RMSNorm(32)
        self.qn_s, self.kn_s = RMSNorm(16), RMSNorm(16)

        self.rope_l = RotaryEmbedding(64, config.block_size)
        self.rope_m = RotaryEmbedding(32, config.block_size)
        self.rope_s = RotaryEmbedding(16, config.block_size)

        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        dp = self.attn_dropout if self.training else 0.0
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        ql, qm, qs = q.split([128, 64, 64], dim=-1)
        kl, km, ks = k.split([128, 64, 64], dim=-1)
        vl, vm, vs = v.split([128, 64, 64], dim=-1)

        ql = self.qn_l(ql.view(B, T, 2, 64).transpose(1, 2))
        kl = self.kn_l(kl.view(B, T, 2, 64).transpose(1, 2))
        vl = vl.view(B, T, 2, 64).transpose(1, 2)
        ql, kl = self.rope_l(ql, kl, start_pos)
        ol = F.scaled_dot_product_attention(ql, kl, vl, dropout_p=dp, is_causal=True)
        ol = ol.transpose(1, 2).contiguous().view(B, T, 128)

        qm = self.qn_m(qm.view(B, T, 2, 32).transpose(1, 2))
        km = self.kn_m(km.view(B, T, 2, 32).transpose(1, 2))
        vm = vm.view(B, T, 2, 32).transpose(1, 2)
        qm, km = self.rope_m(qm, km, start_pos)
        om = F.scaled_dot_product_attention(qm, km, vm, dropout_p=dp, is_causal=True)
        om = om.transpose(1, 2).contiguous().view(B, T, 64)

        qs = self.qn_s(qs.view(B, T, 4, 16).transpose(1, 2))
        ks = self.kn_s(ks.view(B, T, 4, 16).transpose(1, 2))
        vs = vs.view(B, T, 4, 16).transpose(1, 2)
        qs, ks = self.rope_s(qs, ks, start_pos)
        os_ = F.scaled_dot_product_attention(qs, ks, vs, dropout_p=dp, is_causal=True)
        os_ = os_.transpose(1, 2).contiguous().view(B, T, 64)

        return self.resid_dropout(self.proj(torch.cat([ol, om, os_], dim=-1)))


class StandardMHA(nn.Module):
    def __init__(self, config: LoopedConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        dp = self.attn_dropout if self.training else 0.0
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, self.n_head, self.head_dim).transpose(1, 2))
        k = self.k_norm(k.view(B, T, self.n_head, self.head_dim).transpose(1, 2))
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, start_pos)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        return self.resid_dropout(self.proj(out.transpose(1, 2).contiguous().view(B, T, C)))


def _make_attn(config: LoopedConfig) -> nn.Module:
    if config.attn_type == "multires":
        return FusedMultiResolutionAttention(config)
    if config.attn_type == "mha":
        return StandardMHA(config)
    raise ValueError(f"Unknown attn_type: {config.attn_type}")


class LoopedBlock(nn.Module):
    def __init__(self, config: LoopedConfig, max_loops: int = 1):
        super().__init__()
        self.max_loops = max_loops
        self.ln_1 = nn.ModuleList([RMSNorm(config.n_embd) for _ in range(max_loops)])
        self.ln_2 = nn.ModuleList([RMSNorm(config.n_embd) for _ in range(max_loops)])
        self.attn = _make_attn(config)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, start_pos: int = 0, loop_idx: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln_1[loop_idx](x), start_pos)
        x = x + self.mlp(self.ln_2[loop_idx](x))
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, config: LoopedConfig, folded: bool = False):
        super().__init__()
        self.config = config
        self.folded = folded

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.prelude = LoopedBlock(config, max_loops=1)
        if folded:
            self.core = nn.ModuleList([LoopedBlock(config, max_loops=config.n_loops)])
        else:
            self.core = nn.ModuleList([
                LoopedBlock(config, max_loops=1) for _ in range(config.n_loops)
            ])
        self.coda = LoopedBlock(config, max_loops=1)

        self.loop_embed = nn.Embedding(config.n_loops, config.n_embd)
        self.inj_gate = nn.Parameter(torch.zeros(config.n_loops))

        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # Zero-init loop_embed and residual projections scaled by effective depth
        nn.init.zeros_(self.loop_embed.weight)
        eff_depth = 2 + config.n_loops
        resid_std = 0.02 / math.sqrt(2 * eff_depth)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w3.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def layout(self) -> str:
        k = self.config.n_loops
        return f"1 + *{k} + 1" if self.folded else f"1 + {k} + 1"

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
    ):
        x = self.drop(self.wte(idx))
        x = self.prelude(x, start_pos, loop_idx=0)
        x_pre = x

        k_loops = self.config.n_loops
        for k in range(k_loops):
            x = x + self.inj_gate[k] * x_pre + self.loop_embed.weight[k]
            if self.folded:
                x = self.core[0](x, start_pos, loop_idx=k)
            else:
                x = self.core[k](x, start_pos, loop_idx=0)

        x = self.coda(x, start_pos, loop_idx=0)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None) -> "LoopedTransformer":
        map_loc = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)
        cfg_dict = ckpt["config"]
        if isinstance(cfg_dict, LoopedConfig):
            cfg = cfg_dict
        else:
            cfg = LoopedConfig(**cfg_dict)
        if device is not None:
            cfg.device = device
        folded = ckpt.get("folded", True)
        model = cls(cfg, folded=folded).to(cfg.device)
        state = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
        model.load_state_dict(state)
        return model


# ── Origami Kickstart: Soft-Fold & Hard-Fold ──────────────────────────────────

def _shared_param_names(block: LoopedBlock) -> list[str]:
    """Parameters that will be shared across loops (everything except ln_1 / ln_2)."""
    return [
        name for name, _ in block.named_parameters()
        if not name.startswith("ln_1.") and not name.startswith("ln_2.")
    ]


def soft_tie_loss(model: LoopedTransformer) -> torch.Tensor:
    """
    Smooth L2 pull toward the mean across unfolded core blocks for shared weights
    (attn + mlp), leaving each block's RMSNorms (ln_1, ln_2) free to stay loop-specific.
    """
    if model.folded or len(model.core) <= 1:
        return torch.zeros((), device=model.wte.weight.device)

    param_names = _shared_param_names(model.core[0])
    core_dicts = [dict(blk.named_parameters()) for blk in model.core]

    total = torch.zeros((), device=model.wte.weight.device)
    count = 0
    for name in param_names:
        stacked = torch.stack([d[name] for d in core_dicts], dim=0)
        mean_w = stacked.mean(dim=0, keepdim=True).detach()
        total = total + F.mse_loss(stacked, mean_w.expand_as(stacked))
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def hard_fold_core(
    model: LoopedTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    eps: float = 1e-8,
    max_precision_ratio: float = 10.0,
) -> LoopedTransformer:
    """
    Collapse K independent core blocks into 1 physical LoopedBlock looped K times.

    - Shared weights (attn + mlp): precision-weighted fusion via AdamW exp_avg_sq
      when optimizer state is available, else simple mean.
    - Loop-specific RMSNorms: core[k].ln_1[0] -> folded.ln_1[k] and
      core[k].ln_2[0] -> folded.ln_2[k], preserving forward-pass continuity.
    """
    if model.folded:
        return model

    k_loops = len(model.core)
    device = model.wte.weight.device
    folded_blk = LoopedBlock(model.config, max_loops=k_loops).to(device)

    # 1. Copy each unfolded block's norms into its exact loop_idx slot
    for k in range(k_loops):
        folded_blk.ln_1[k].load_state_dict(model.core[k].ln_1[0].state_dict())
        folded_blk.ln_2[k].load_state_dict(model.core[k].ln_2[0].state_dict())

    # 2. Fuse shared attn + mlp parameters
    core_dicts = [dict(blk.named_parameters()) for blk in model.core]
    folded_params = dict(folded_blk.named_parameters())

    for name in _shared_param_names(model.core[0]):
        tensors = [d[name].data for d in core_dicts]
        use_precision = (
            optimizer is not None
            and tensors[0].ndim >= 2
            and all("exp_avg_sq" in optimizer.state.get(d[name], {}) for d in core_dicts)
        )
        if use_precision:
            precs = [
                1.0 / (optimizer.state[d[name]]["exp_avg_sq"].data + eps)
                for d in core_dicts
            ]
            # Clamp extreme precision ratios so one block doesn't dominate completely
            min_p = torch.stack(precs, dim=0).min(dim=0).values
            precs = [torch.clamp(p, max=min_p * max_precision_ratio) for p in precs]
            p_sum = sum(precs)
            fused = sum(w * p for w, p in zip(tensors, precs)) / p_sum
        else:
            fused = torch.stack(tensors, dim=0).mean(dim=0)

        folded_params[name].data.copy_(fused)

    model.core = nn.ModuleList([folded_blk])
    model.folded = True
    return model
