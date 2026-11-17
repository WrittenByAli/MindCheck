"""ONNX / TorchScript export and verification (Task 1.7, Subtask 1.7.4).

Serializes a trained model to ONNX (with dynamic batch) and TorchScript for
low-latency CPU inference in the Streamlit app, and verifies the ONNX output
matches PyTorch within tolerance.

Subtask 1.7.4 is nominally owned by Muhammad Ali; implemented here so Task 1.7
is self-contained and testable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["export_to_onnx", "export_to_torchscript", "verify_onnx_matches"]


def _dummy_input(model: nn.Module, batch: int = 1) -> torch.Tensor:
    in_channels = getattr(model, "in_channels", 112)
    return torch.randn(batch, in_channels, 8, 8)


def export_to_onnx(model: nn.Module, output_path: str | Path, opset: int = 17) -> Path:
    """Export ``model`` to ONNX with a dynamic batch axis and run the checker."""
    import onnx

    model.eval()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        _dummy_input(model),
        str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        # Use the stable TorchScript-based exporter; the newer dynamo exporter
        # requires the optional `onnxscript` package.
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out)))
    return out


def export_to_torchscript(model: nn.Module, output_path: str | Path) -> Path:
    """Export ``model`` to a traced TorchScript module."""
    model.eval()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(model, _dummy_input(model))
    scripted.save(str(out))
    return out


def verify_onnx_matches(model: nn.Module, onnx_path: str | Path, atol: float = 1e-4) -> bool:
    """Return True iff the ONNX model's output matches PyTorch within ``atol``."""
    import onnxruntime

    model.eval()
    x = _dummy_input(model, batch=4)
    with torch.no_grad():
        reference = model(x).numpy()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"input": x.numpy()})[0]
    return np.allclose(reference, produced, atol=atol)
