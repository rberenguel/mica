#!/usr/bin/env python3
"""
Export Mica Looped Transformer (1 + *4 + 1) to ONNX for browser inference & trace viewer.

Produces:
  llm/mica_looped.onnx       — inference only (last-position logits)
  llm/mica_looped_trace.onnx — full trace (logits, embedding, 6 residuals, 6x8 attn maps, final, 6 logit_lens)

Uses pure opset-17 primitives (manual RMSNorm + explicit causal attention) for 100%
compatibility with onnxruntime-web in the browser, and extends RoPE caches to 512
so BLOCK_SIZE=512 in llm/mica.js works seamlessly.

Usage:
  uv run python src/export_onnx_looped.py
  uv run python src/export_onnx_looped.py --weights models/experiments/interleaved_noir/best.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_looped import LoopedConfig, LoopedTransformer, RotaryEmbedding


OPSET = 17


def onnx_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _extend_rope(module: nn.Module, max_len: int = 512):
    for m in module.modules():
        if isinstance(m, RotaryEmbedding):
            head_dim = m.cos_cache.shape[-1]
            if m.cos_cache.shape[0] < max_len:
                inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
                t = torch.arange(max_len)
                freqs = torch.outer(t, inv_freq)
                m.cos_cache = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
                m.sin_cache = torch.cat([freqs.sin(), freqs.sin()], dim=-1)


def _onnx_attn(attn_mod, normed: torch.Tensor, causal_mask: torch.Tensor, return_weights: bool):
    B, T, C = normed.shape
    q, k, v = attn_mod.qkv(normed).chunk(3, dim=-1)

    if hasattr(attn_mod, "rope_l"):  # FusedMultiResolutionAttention (2x64 + 2x32 + 4x16)
        ql, qm, qs = q.split([128, 64, 64], dim=-1)
        kl, km, ks = k.split([128, 64, 64], dim=-1)
        vl, vm, vs = v.split([128, 64, 64], dim=-1)

        def _group(qg, kg, vg, qn, kn, rope, nh, hs):
            qg = onnx_rms_norm(qg.view(B, T, nh, hs).transpose(1, 2), qn.weight, qn.eps)
            kg = onnx_rms_norm(kg.view(B, T, nh, hs).transpose(1, 2), kn.weight, kn.eps)
            vg = vg.view(B, T, nh, hs).transpose(1, 2)
            qg, kg = rope(qg, kg, 0)
            w = (qg @ kg.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            w = w.masked_fill(causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
            w = F.softmax(w, dim=-1)
            out = (w @ vg).transpose(1, 2).contiguous().view(B, T, nh * hs)
            return w, out

        wl, ol = _group(ql, kl, vl, attn_mod.qn_l, attn_mod.kn_l, attn_mod.rope_l, 2, 64)
        wm, om = _group(qm, km, vm, attn_mod.qn_m, attn_mod.kn_m, attn_mod.rope_m, 2, 32)
        ws, os_ = _group(qs, ks, vs, attn_mod.qn_s, attn_mod.kn_s, attn_mod.rope_s, 4, 16)

        y = attn_mod.proj(torch.cat([ol, om, os_], dim=-1))
        if return_weights:
            return y, torch.cat([wl, wm, ws], dim=1)  # (B, 8, T, T)
        return y, None
    else:  # StandardMHA
        nh, hs = attn_mod.n_head, attn_mod.head_dim
        q = onnx_rms_norm(q.view(B, T, nh, hs).transpose(1, 2), attn_mod.q_norm.weight, attn_mod.q_norm.eps)
        k = onnx_rms_norm(k.view(B, T, nh, hs).transpose(1, 2), attn_mod.k_norm.weight, attn_mod.k_norm.eps)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        q, k = attn_mod.rope(q, k, 0)
        w = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        w = w.masked_fill(causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
        w = F.softmax(w, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = attn_mod.proj(out)
        return y, (w if return_weights else None)


def _onnx_block(block, x: torch.Tensor, causal_mask: torch.Tensor, loop_idx: int, return_weights: bool):
    ln1 = block.ln_1[loop_idx]
    ln2 = block.ln_2[loop_idx]
    y, w = _onnx_attn(block.attn, onnx_rms_norm(x, ln1.weight, ln1.eps), causal_mask, return_weights)
    x = x + y
    h = onnx_rms_norm(x, ln2.weight, ln2.eps)
    a, b = block.mlp.w12(h).chunk(2, dim=-1)
    x = x + block.mlp.w3(F.silu(a) * b)
    return x, w


class LoopedForONNX(nn.Module):
    def __init__(self, model: LoopedTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.model
        T = input_ids.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1)

        x = m.wte(input_ids)
        x, _ = _onnx_block(m.prelude, x, causal_mask, 0, False)
        x_pre = x
        for k in range(m.config.n_loops):
            x = x + m.inj_gate[k] * x_pre + m.loop_embed.weight[k]
            blk = m.core[0] if m.folded else m.core[k]
            l_idx = k if m.folded else 0
            x, _ = _onnx_block(blk, x, causal_mask, l_idx, False)
        x, _ = _onnx_block(m.coda, x, causal_mask, 0, False)
        x = onnx_rms_norm(x, m.ln_f.weight, m.ln_f.eps)
        return m.lm_head(x[:, -1, :])


class LoopedTraceONNX(nn.Module):
    """Full trace: prelude (L0) + K core loops (L1..LK) + coda (L{K+1})."""
    def __init__(self, model: LoopedTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        m = self.model
        T = input_ids.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1)

        x = m.wte(input_ids)
        embedding = x

        residuals = []
        attn_weights = []

        # L0: Prelude
        x, w = _onnx_block(m.prelude, x, causal_mask, 0, True)
        residuals.append(x)
        attn_weights.append(w)
        x_pre = x

        # L1..LK: Looped Core
        for k in range(m.config.n_loops):
            x = x + m.inj_gate[k] * x_pre + m.loop_embed.weight[k]
            blk = m.core[0] if m.folded else m.core[k]
            l_idx = k if m.folded else 0
            x, w = _onnx_block(blk, x, causal_mask, l_idx, True)
            residuals.append(x)
            attn_weights.append(w)

        # L{K+1}: Coda
        x, w = _onnx_block(m.coda, x, causal_mask, 0, True)
        residuals.append(x)
        attn_weights.append(w)

        final_norm = onnx_rms_norm(x, m.ln_f.weight, m.ln_f.eps)
        logits = m.lm_head(final_norm)
        logit_lens = [m.lm_head(onnx_rms_norm(r, m.ln_f.weight, m.ln_f.eps)) for r in residuals]

        return (
            (logits, embedding)
            + tuple(residuals)
            + tuple(attn_weights)
            + (final_norm,)
            + tuple(logit_lens)
        )


def main():
    p = argparse.ArgumentParser(description="Export Mica Looped Transformer to ONNX")
    p.add_argument("--weights", default="models/experiments/interleaved_noir/best.pt")
    p.add_argument("--output", default="llm/mica_looped.onnx")
    p.add_argument("--trace-output", default="llm/mica_looped_trace.onnx")
    p.add_argument("--max-seq-len", type=int, default=512)
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    model = LoopedTransformer.from_checkpoint(str(weights), device="cpu")
    model.eval()
    _extend_rope(model, max_len=args.max_seq_len)
    print(f"Loaded {weights} | layout={model.layout} | params={model.num_params()/1e6:.2f}M")

    # 1. Inference export
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = LoopedForONNX(model).eval()
    dummy = torch.zeros(1, 16, dtype=torch.long)

    # Sanity check vs PyTorch forward
    with torch.no_grad():
        ref_logits, _ = model(dummy)
        onnx_logits = wrapper(dummy)
        max_diff = (ref_logits[:, -1, :] - onnx_logits).abs().max().item()
        print(f"Sanity check max logit diff vs PyTorch: {max_diff:.2e}")

        torch.onnx.export(
            wrapper, (dummy,), str(out),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
            external_data=False,
        )
    print(f"Exported inference ONNX: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    # 2. Trace export
    trace_out = Path(args.trace_output)
    trace_wrapper = LoopedTraceONNX(model).eval()
    dummy_t = torch.zeros(1, 8, dtype=torch.long)
    n_logical = 2 + model.config.n_loops

    output_names = (
        ["logits", "embedding"]
        + [f"residual_{i}" for i in range(n_logical)]
        + [f"attn_{i}" for i in range(n_logical)]
        + ["final"]
        + [f"logit_lens_{i}" for i in range(n_logical)]
    )
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "embedding": {0: "batch", 1: "seq_len"},
        "final": {0: "batch", 1: "seq_len"},
        **{f"residual_{i}": {0: "batch", 1: "seq_len"} for i in range(n_logical)},
        **{f"attn_{i}": {0: "batch", 2: "seq_len", 3: "seq_len"} for i in range(n_logical)},
        **{f"logit_lens_{i}": {0: "batch", 1: "seq_len"} for i in range(n_logical)},
    }

    with torch.no_grad():
        torch.onnx.export(
            trace_wrapper, (dummy_t,), str(trace_out),
            input_names=["input_ids"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
            external_data=False,
        )
    print(f"Exported trace ONNX:     {trace_out} ({trace_out.stat().st_size / 1024 / 1024:.1f} MB, {len(output_names)} outputs)")


if __name__ == "__main__":
    main()
