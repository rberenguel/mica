#!/usr/bin/env python3
"""
Export the Origami Mica model to ONNX for browser inference.

Produces:
  llm/mica.onnx       — inference only (last-position logits)
  llm/mica_trace.onnx — full trace with attention weights, residuals, logit lens

Usage:
    uv run python export_onnx_origami.py
    cp mica_tokenizer.json llm/
    cd llm && python -m http.server 8080
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import MicaConfig
from model_origami import OrigamiTransformer, restore_segments, restore_folds

WEIGHTS = Path("models/current/mica_origami_v2.pt")
OUT_DIR = Path("llm")
OPSET   = 17


def load_model(weights_path: Path) -> OrigamiTransformer:
    """Load origami model from checkpoint, restoring segments."""
    config = MicaConfig()
    config.device = "cpu"
    config.dropout = 0.0

    model = OrigamiTransformer(config)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    if "segments" in ckpt:
        restore_segments(model, ckpt["segments"])
    else:
        restore_folds(model, ckpt.get("bottom_loops", 1), ckpt.get("top_loops", 1))

    model.eval()
    print(f"Loaded {weights_path}  layout={model.layout}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


# ── Simple inference export ──────────────────────────────────────────────────

class OrigamiForONNX(nn.Module):
    """Thin wrapper: single input, single output (last-position logits)."""
    def __init__(self, model: OrigamiTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)  int64
        b, t = input_ids.size()
        x = self.model.transformer.drop(self.model.transformer.wte(input_ids))
        h = self.model.transformer.h

        for phys_idx, loop_count in self.model.segments:
            block = h[phys_idx]
            for loop_idx in range(loop_count):
                x = block(x, start_pos=0, loop_idx=loop_idx)

        x = self.model.transformer.ln_f(x)
        # Only return logits for the last position
        return self.model.lm_head(x[:, -1, :])


def export_inference(model: OrigamiTransformer):
    out = OUT_DIR / "mica.onnx"
    OUT_DIR.mkdir(exist_ok=True)

    wrapper = OrigamiForONNX(model)
    wrapper.eval()
    dummy = torch.zeros(1, 16, dtype=torch.long)

    print(f"\nExporting inference model to {out} (opset {OPSET})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
            external_data=False,
        )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  Exported: {out}  ({size_mb:.1f} MB)")
    return out


# ── Trace export (attention weights + residuals + logit lens) ─────────────────

class OrigamiTraceONNX(nn.Module):
    """Full forward pass surfacing all intermediate tensors."""
    def __init__(self, model: OrigamiTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> tuple:
        m = self.model.transformer
        x = m.wte(input_ids)          # (B, T, C)
        embedding = x

        residuals    = []
        attn_weights = []

        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)

        h = m.h
        for phys_idx, loop_count in self.model.segments:
            block = h[phys_idx]
            for loop_idx in range(loop_count):
                if loop_idx > 0 and block.depth_embed is not None:
                    x = x + block.depth_embed(torch.tensor(loop_idx - 1, device=x.device))

                normed = block.ln_1[loop_idx](x)
                B, T_local, C = normed.size()
                a = block.attn

                def _block_attn(W_q, W_k, W_v, rope, nh, hs):
                    q = W_q(normed).view(B, T_local, nh, hs).transpose(1, 2)
                    k = W_k(normed).view(B, T_local, nh, hs).transpose(1, 2)
                    v = W_v(normed).view(B, T_local, nh, hs).transpose(1, 2)
                    q, k = rope(q, k, 0)
                    w = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
                    w = w.masked_fill(causal_mask[:T_local, :T_local].unsqueeze(0).unsqueeze(0), float("-inf"))
                    w = F.softmax(w, dim=-1)
                    return w, (w @ v).transpose(1, 2).contiguous().view(B, T_local, nh * hs)

                w_l, out_l = _block_attn(a.W_q_large, a.W_k_large, a.W_v_large, a.rope_large, 1, 64)
                w_m, out_m = _block_attn(a.W_q_med,   a.W_k_med,   a.W_v_med,   a.rope_med,   2, 32)
                w_s, out_s = _block_attn(a.W_q_small, a.W_k_small, a.W_v_small, a.rope_small, 8,  8)

                attn_weights.append(torch.cat([w_l, w_m, w_s], dim=1))   # (B, 11, T, T)

                y = a.W_o(torch.cat([out_l, out_m, out_s], dim=-1))
                x = x + y
                x = x + block.mlp(block.ln_2[loop_idx](x))
                residuals.append(x)                                      # (B, T, C)

        final_norm = m.ln_f(x)                           # (B, T, C)
        logits     = self.model.lm_head(final_norm)      # (B, T, vocab_size)

        # Logit lens
        logit_lens = [self.model.lm_head(m.ln_f(r)) for r in residuals]

        return (
            (logits, embedding)
            + tuple(residuals)
            + tuple(attn_weights)
            + (final_norm,)
            + tuple(logit_lens)
        )


def export_trace(model: OrigamiTransformer):
    out = OUT_DIR / "mica_trace.onnx"
    OUT_DIR.mkdir(exist_ok=True)

    wrapper = OrigamiTraceONNX(model)
    wrapper.eval()

    # Use small dummy because trace model has many outputs
    dummy = torch.zeros(1, 8, dtype=torch.long)

    n = model.config.n_layer
    n_logical = sum(loop_count for _, loop_count in model.segments)
    # For unfolded model, n_logical == n

    output_names = (
        ["logits", "embedding"]
        + [f"residual_{i}"   for i in range(n_logical)]
        + [f"attn_{i}"       for i in range(n_logical)]
        + ["final"]
        + [f"logit_lens_{i}" for i in range(n_logical)]
    )

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits":    {0: "batch", 1: "seq_len"},
        "embedding": {0: "batch", 1: "seq_len"},
        "final":     {0: "batch", 1: "seq_len"},
        **{f"residual_{i}":   {0: "batch", 1: "seq_len"}               for i in range(n_logical)},
        **{f"attn_{i}":       {0: "batch", 2: "seq_len", 3: "seq_len"} for i in range(n_logical)},
        **{f"logit_lens_{i}": {0: "batch", 1: "seq_len"}               for i in range(n_logical)},
    }

    print(f"\nExporting trace model to {out} (opset {OPSET}, {len(output_names)} outputs)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out),
            input_names=["input_ids"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
            external_data=False,
        )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  Exported: {out}  ({size_mb:.1f} MB, {len(output_names)} outputs)")
    return out


def main():
    if not WEIGHTS.exists():
        print(f"Weights not found: {WEIGHTS}", file=sys.stderr)
        sys.exit(1)

    model = load_model(WEIGHTS)

    export_inference(model)
    export_trace(model)

    print("\nNext steps:")
    print("  cp mica_tokenizer.json llm/")
    print("  cd llm && python -m http.server 8080")
    print("  open http://localhost:8080")
    print("  open http://localhost:8080/trace.html")


if __name__ == "__main__":
    main()
