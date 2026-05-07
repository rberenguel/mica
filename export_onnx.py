#!/usr/bin/env python3
"""
Export the trained Mica model to ONNX for browser inference via onnxruntime-web.

Output: llm/mica.onnx  (~11MB fp32)

After export, also copy mica_tokenizer.json:
    cp mica_tokenizer.json llm/

Then serve the llm/ folder locally:
    cd llm && python -m http.server 8080
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
from config import MicaConfig
from model import MicaTransformer

WEIGHTS   = Path("mica_weights.pt")
PHASE1    = Path("mica_phase1.pt")
OUT       = Path("llm/mica.onnx")
OPSET     = 17


class MicaForONNX(nn.Module):
    """Thin wrapper: single input, single output, no optional args."""
    def __init__(self, model: MicaTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)  int64
        b, t = input_ids.size()
        x = self.model.transformer.drop(self.model.transformer.wte(input_ids))
        for block in self.model.transformer.h:
            x = block(x)                          # start_pos=0, full context
        x = self.model.transformer.ln_f(x)
        # Only return logits for the last position → (batch, vocab_size)
        return self.model.lm_head(x[:, -1, :])


def load_weights() -> Path:
    if WEIGHTS.exists():
        return WEIGHTS
    if PHASE1.exists():
        print(f"mica_weights.pt not found; using {PHASE1} (phase 1 checkpoint).")
        return PHASE1
    print("No weights found. Train the model first.", file=sys.stderr)
    sys.exit(1)


def main():
    weights_path = load_weights()
    print(f"Loading weights from {weights_path}...")

    config = MicaConfig()
    config.device = "cpu"          # always export from CPU
    config.dropout = 0.0           # no dropout during export / inference

    model = MicaTransformer(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    wrapper = MicaForONNX(model)
    wrapper.eval()

    dummy = torch.zeros(1, 16, dtype=torch.long)

    print(f"Exporting to {OUT} (opset {OPSET})...")
    OUT.parent.mkdir(exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(OUT),
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

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"Exported: {OUT}  ({size_mb:.1f} MB)")

    # Optional validation with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input_ids": dummy.numpy()})
        assert out[0].shape == (1, config.vocab_size), f"Unexpected shape: {out[0].shape}"
        print("Validation passed (onnxruntime).")
    except ImportError:
        print("onnxruntime not installed — skipping validation. (uv add onnxruntime to validate)")

    print("\nNext steps:")
    print(f"  cp mica_tokenizer.json llm/")
    print(f"  cd llm && python -m http.server 8080")
    print(f"  open http://localhost:8080")


if __name__ == "__main__":
    main()
