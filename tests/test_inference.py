"""Tests for the deployable model wrapper (random weights, no trained metrics)."""

from __future__ import annotations

import pytest
import torch

from src.models.export import export_to_onnx, export_to_torchscript, verify_onnx_matches
from src.models.inference import DeployableModel, load_deployable, save_deployable

BIAS = [-1.5, -0.5, 1.25, -0.25, -1.25, 2.25]


@pytest.fixture()
def model() -> DeployableModel:
    torch.manual_seed(0)
    return DeployableModel(keep=56, bias=BIAS).eval()


def test_accepts_spec_input_and_returns_six_logits(model):
    assert model(torch.randn(3, 112, 8, 8)).shape == (3, 6)


def test_uses_only_the_newest_channels(model):
    x = torch.randn(2, 112, 8, 8)
    changed_old = x.clone()
    changed_old[:, :56] = torch.randn(2, 56, 8, 8)
    changed_new = x.clone()
    changed_new[:, 56:] = torch.randn(2, 56, 8, 8)
    with torch.no_grad():
        assert torch.equal(model(x), model(changed_old))
        assert not torch.allclose(model(x), model(changed_new))
        assert torch.allclose(model(x), model.backbone(x[:, -56:]) + torch.tensor(BIAS))


def test_full_width_keeps_every_channel():
    full = DeployableModel(keep=112).eval()
    x = torch.randn(1, 112, 8, 8)
    with torch.no_grad():
        assert torch.allclose(full(x), full.backbone(x))


def test_predict_proba_sums_to_one_and_rejects_wrong_shape(model):
    probs = model.predict_proba(torch.randn(4, 112, 8, 8))
    assert torch.allclose(probs.sum(-1), torch.ones(4))
    with pytest.raises(ValueError):
        model.predict_proba(torch.randn(4, 56, 8, 8))


@pytest.mark.parametrize("kwargs", [{"keep": 50}, {"keep": 126}, {"bias": [0.0] * 5}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DeployableModel(**kwargs)


def test_save_load_round_trip(model, tmp_path):
    path = save_deployable(model, tmp_path / "m.pt", {"seed": 42})
    loaded = load_deployable(path)
    x = torch.randn(2, 112, 8, 8)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))
    assert not loaded.training


def test_load_rejects_other_checkpoints(tmp_path):
    torch.save({"model_state_dict": {}}, tmp_path / "other.pt")
    with pytest.raises(ValueError):
        load_deployable(tmp_path / "other.pt")


def test_onnx_and_torchscript_match(model, tmp_path):
    onnx_path = export_to_onnx(model, tmp_path / "m.onnx")
    assert verify_onnx_matches(model, onnx_path)
    ts = torch.jit.load(str(export_to_torchscript(model, tmp_path / "m.ts.pt")))
    x = torch.randn(5, 112, 8, 8)
    with torch.no_grad():
        assert torch.allclose(ts(x), model(x), atol=1e-5)
