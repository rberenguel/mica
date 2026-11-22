"""
Mica-as-Critic — 100% from-scratch Bradley-Terry scalar reward model.

Clones the pretrained LoopedTransformer backbone (3.69M params) and replaces
the LM head with a lightweight scalar scoring head over masked mean-pooled
hidden states. Outputs a single real-valued quality score r(text) per sequence.
"""
from __future__ import annotations

import dataclasses
import torch
import torch.nn as nn

from model_looped import LoopedConfig, LoopedTransformer, RMSNorm


class MicaCritic(nn.Module):
    def __init__(self, config: LoopedConfig, folded: bool = True):
        super().__init__()
        self.config = config
        self.folded = folded
        self.backbone = LoopedTransformer(config, folded=folded)
        # Scalar head on top of mean-pooled final hidden states
        self.head_norm = RMSNorm(config.n_embd)
        self.score_head = nn.Linear(config.n_embd, 1, bias=True)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score_head.bias)

    def encode_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the LoopedTransformer backbone up to ln_f -> (B, T, C)."""
        b = self.backbone
        x = b.drop(b.wte(idx))
        x = b.prelude(x, start_pos=0, loop_idx=0)
        x_pre = x
        for k in range(b.config.n_loops):
            x = x + b.inj_gate[k] * x_pre + b.loop_embed.weight[k]
            if b.folded:
                x = b.core[0](x, start_pos=0, loop_idx=k)
            else:
                x = b.core[k](x, start_pos=0, loop_idx=0)
        x = b.coda(x, start_pos=0, loop_idx=0)
        return b.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        idx:  (B, T) token ids
        mask: (B, T) float/bool mask where 1 = tokens to pool over (e.g. non-pad
              or completion tokens). If None, mean-pools over all tokens.
        Returns: (B,) scalar reward scores.
        """
        h = self.encode_hidden(idx)  # (B, T, C)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.to(h.dtype).unsqueeze(-1)  # (B, T, 1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.score_head(self.head_norm(pooled)).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_lm_checkpoint(
        cls,
        lm_path: str,
        device: str | None = None,
        dropout: float = 0.1,
    ) -> "MicaCritic":
        """Initialize a fresh critic from a pretrained LoopedTransformer checkpoint."""
        map_loc = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        ckpt = torch.load(lm_path, map_location=map_loc, weights_only=False)
        cfg_dict = ckpt["config"]
        cfg = cfg_dict if isinstance(cfg_dict, LoopedConfig) else LoopedConfig(**cfg_dict)
        if device is not None:
            cfg.device = device
        cfg.dropout = dropout
        folded = ckpt.get("folded", True)

        critic = cls(cfg, folded=folded).to(cfg.device)
        state = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
        critic.backbone.load_state_dict(state)
        return critic

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None) -> "MicaCritic":
        """Load a trained MicaCritic checkpoint."""
        map_loc = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)
        cfg_dict = ckpt["config"]
        cfg = cfg_dict if isinstance(cfg_dict, LoopedConfig) else LoopedConfig(**cfg_dict)
        if device is not None:
            cfg.device = device
        folded = ckpt.get("folded", True)
        critic = cls(cfg, folded=folded).to(cfg.device)
        critic.load_state_dict(ckpt["critic"])
        return critic

    def save(self, path: str, **meta):
        payload = {
            "config": dataclasses.asdict(self.config),
            "folded": self.folded,
            "critic": self.state_dict(),
        }
        payload.update(meta)
        torch.save(payload, path)
