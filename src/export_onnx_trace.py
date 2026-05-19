#!/usr/bin/env python3
"""
Export Mica to ONNX with intermediate layer outputs for browser-based tracing.
Produces llm/mica_trace.onnx.

Named outputs:
  logits        (1, T, vocab_size)   — next-token logits for every position
  embedding     (1, T, n_embd)       — token embeddings (pre-block)
  residual_0..N (1, T, n_embd)       — residual stream after each block
  attn_0..N     (1, n_head, T, T)    — softmaxed attention weights per block
  final         (1, T, n_embd)       — residual stream after final layer norm

Usage:
    uv run python export_onnx_trace.py
    cp mica_tokenizer.json llm/
    cd llm && python -m http.server 8080
    open http://localhost:8080/trace.html
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import MicaConfig
from model import MicaTransformer

WEIGHTS = Path("mica_weights.pt")
PHASE1  = Path("mica_phase1.pt")
OUT     = Path("llm/mica_trace.onnx")
OPSET   = 17


class MicaTraceONNX(nn.Module):
    """
    Full forward pass that surfaces all intermediate tensors as outputs.
    Attention dropout is disabled (config.dropout = 0.0 at export time).
    """
    def __init__(self, model: MicaTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> tuple:
        # input_ids: (B, T)  int64
        m = self.model.transformer
        x = m.wte(input_ids)               # (B, T, C)
        embedding = x

        residuals    = []
        attn_weights = []

        causal_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)

        for block in m.h:
            normed = block.ln_1(x)
            B, T, C = normed.size()
            a = block.attn

            def _block_attn(W_q, W_k, W_v, rope, nh, hs):
                q = W_q(normed).view(B, T, nh, hs).transpose(1, 2)
                k = W_k(normed).view(B, T, nh, hs).transpose(1, 2)
                v = W_v(normed).view(B, T, nh, hs).transpose(1, 2)
                q, k = rope(q, k, 0)
                w = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
                w = w.masked_fill(causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
                w = F.softmax(w, dim=-1)
                return w, (w @ v).transpose(1, 2).contiguous().view(B, T, nh * hs)

            w_l, out_l = _block_attn(a.W_q_large, a.W_k_large, a.W_v_large, a.rope_large, 1, 64)
            w_m, out_m = _block_attn(a.W_q_med,   a.W_k_med,   a.W_v_med,   a.rope_med,   2, 32)
            w_s, out_s = _block_attn(a.W_q_small, a.W_k_small, a.W_v_small, a.rope_small, 8,  8)

            attn_weights.append(torch.cat([w_l, w_m, w_s], dim=1))   # (B, 11, T, T)

            y = a.W_o(torch.cat([out_l, out_m, out_s], dim=-1))
            x = x + y
            x = x + block.mlp(block.ln_2(x))
            residuals.append(x)                                        # (B, T, C)

        final_norm = m.ln_f(x)                          # (B, T, C)
        logits     = self.model.lm_head(final_norm)     # (B, T, vocab_size)

        # Logit lens: project each intermediate residual through the same ln_f + lm_head.
        # Answers "what would the model predict if it stopped at layer i?"
        logit_lens = [self.model.lm_head(m.ln_f(r)) for r in residuals]  # list of (B, T, vocab)

        # Order must match output_names below
        return (
            (logits, embedding)
            + tuple(residuals)
            + tuple(attn_weights)
            + (final_norm,)
            + tuple(logit_lens)
        )


def load_weights() -> Path:
    if WEIGHTS.exists():
        return WEIGHTS
    if PHASE1.exists():
        print(f"mica_weights.pt not found; using {PHASE1}.", file=sys.stderr)
        return PHASE1
    print("No weights found. Train the model first.", file=sys.stderr)
    sys.exit(1)


def main():
    weights_path = load_weights()
    print(f"Loading weights from {weights_path}…")

    config         = MicaConfig()
    config.device  = "cpu"
    config.dropout = 0.0

    model = MicaTransformer(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    wrapper = MicaTraceONNX(model)
    wrapper.eval()

    dummy = torch.zeros(1, 8, dtype=torch.long)

    n = config.n_layer
    output_names = (
        ["logits", "embedding"]
        + [f"residual_{i}"   for i in range(n)]
        + [f"attn_{i}"       for i in range(n)]
        + ["final"]
        + [f"logit_lens_{i}" for i in range(n)]
    )

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits":    {0: "batch", 1: "seq_len"},
        "embedding": {0: "batch", 1: "seq_len"},
        "final":     {0: "batch", 1: "seq_len"},
        **{f"residual_{i}":   {0: "batch", 1: "seq_len"}               for i in range(n)},
        **{f"attn_{i}":       {0: "batch", 2: "seq_len", 3: "seq_len"} for i in range(n)},
        **{f"logit_lens_{i}": {0: "batch", 1: "seq_len"}               for i in range(n)},
    }

    print(f"Exporting to {OUT} (opset {OPSET}, {len(output_names)} outputs)…")
    OUT.parent.mkdir(exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(OUT),
            input_names=["input_ids"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
            external_data=False,
        )

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"Exported: {OUT}  ({size_mb:.1f} MB, {len(output_names)} outputs)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"input_ids": dummy.numpy()})
        assert out[0].shape == (1, 8, config.vocab_size), f"Unexpected logits shape: {out[0].shape}"
        print("Validation passed (onnxruntime).")
    except ImportError:
        print("onnxruntime not installed — skipping validation.")

    print("\nNext steps:")
    print("  cp mica_tokenizer.json llm/")
    print("  cd llm && python -m http.server 8080")
    print("  open http://localhost:8080/trace.html")


if __name__ == "__main__":
    main()
