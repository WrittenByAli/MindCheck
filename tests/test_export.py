"""Unit tests for ONNX / TorchScript export (Subtask 1.7.4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.residual_cnn import HistoryChessResNet

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from src.models.export import (  # noqa: E402
    export_to_onnx,
    export_to_torchscript,
    verify_onnx_matches,
)


def test_onnx_export_passes_checker(tmp_path: Path) -> None:
    model = HistoryChessResNet()
    path = export_to_onnx(model, tmp_path / "model.onnx")
    assert path.is_file()  # onnx.checker ran inside export_to_onnx without raising


def test_onnx_output_matches_pytorch(tmp_path: Path) -> None:
    model = HistoryChessResNet()
    path = export_to_onnx(model, tmp_path / "model.onnx")
    assert verify_onnx_matches(model, path, atol=1e-4)


def test_onnx_dynamic_batch(tmp_path: Path) -> None:
    import onnxruntime

    model = HistoryChessResNet().eval()
    path = export_to_onnx(model, tmp_path / "model.onnx")
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 2, 7):
        out = session.run(None, {"input": torch.randn(batch, 112, 8, 8).numpy()})[0]
        assert out.shape == (batch, 6)


def test_torchscript_export_and_match(tmp_path: Path) -> None:
    model = HistoryChessResNet().eval()
    path = export_to_torchscript(model, tmp_path / "model.ts.pt")
    assert path.is_file()
    loaded = torch.jit.load(str(path))
    x = torch.randn(3, 112, 8, 8)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_export_single_board_baseline(tmp_path: Path) -> None:
    from src.models.single_board_cnn import CurrentPositionResNet

    model = CurrentPositionResNet()
    path = export_to_onnx(model, tmp_path / "single.onnx")
    assert verify_onnx_matches(model, path, atol=1e-4)
