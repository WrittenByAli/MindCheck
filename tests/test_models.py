"""Unit tests for the residual CNN and single-board baseline (Subtasks 1.7.1-1.7.3)."""

from __future__ import annotations

import pytest
import torch

from src.models.residual_cnn import HistoryChessResNet, ResidualBlock, count_parameters
from src.models.single_board_cnn import CurrentPositionResNet


# --------------------------------------------------------------------------- #
# 1.7.1 Backbone                                                               #
# --------------------------------------------------------------------------- #
def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(64)
    x = torch.randn(2, 64, 8, 8)
    assert block(x).shape == (2, 64, 8, 8)


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_history_forward_shape(batch: int) -> None:
    model = HistoryChessResNet().eval()
    out = model(torch.randn(batch, 112, 8, 8))
    assert out.shape == (batch, 6)
    assert torch.isfinite(out).all()


def test_parameter_count_in_budget() -> None:
    params = count_parameters(HistoryChessResNet())
    assert 200_000 <= params <= 350_000, params
    assert 250_000 <= params <= 300_000, f"{params} outside the ~250k-300k target"


def test_gradients_flow_through_all_blocks() -> None:
    model = HistoryChessResNet()
    out = model(torch.randn(3, 112, 8, 8))
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


# --------------------------------------------------------------------------- #
# 1.7.2 Head / probabilities                                                   #
# --------------------------------------------------------------------------- #
def test_predict_proba_sums_to_one() -> None:
    model = HistoryChessResNet()
    probs = model.predict_proba(torch.randn(5, 112, 8, 8))
    assert probs.shape == (5, 6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (probs >= 0).all()


def test_eval_is_deterministic_train_is_not() -> None:
    model = HistoryChessResNet()
    x = torch.randn(4, 112, 8, 8)
    model.eval()
    assert torch.allclose(model(x), model(x))
    model.train()
    # Dropout makes two training-mode passes differ.
    assert not torch.allclose(model(x), model(x))


# --------------------------------------------------------------------------- #
# 1.7.3 Single-board ablation baseline                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 8])
def test_single_board_forward_shape(batch: int) -> None:
    model = CurrentPositionResNet().eval()
    out = model(torch.randn(batch, 14, 8, 8))
    assert out.shape == (batch, 6)
    assert torch.isfinite(out).all()


def test_single_board_matches_history_except_input_channels() -> None:
    history = HistoryChessResNet()
    single = CurrentPositionResNet()
    # Same depth, same head, differ only in the stem's input channels.
    assert len(history.blocks) == len(single.blocks)
    assert history.head[-1].out_features == single.head[-1].out_features == 6
    assert history.stem[0].in_channels == 112
    assert single.stem[0].in_channels == 14
    # Single-board has fewer params only because of the smaller stem.
    assert count_parameters(single) < count_parameters(history)
