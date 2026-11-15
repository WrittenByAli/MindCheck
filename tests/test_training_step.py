"""Unit tests for the training engine (Subtasks 1.8.1 & 1.8.2)."""

from __future__ import annotations

import numpy as np
import torch

from src.data.chess_dataset import ChessPlanDataset
from src.models.logger import load_checkpoint, save_checkpoint
from src.models.residual_cnn import HistoryChessResNet
from src.models.train import (
    EarlyStopper,
    ModelTrainer,
    compute_class_weights,
    set_seed,
)


def synthetic_dataset(n: int = 32, seed: int = 0) -> ChessPlanDataset:
    rng = np.random.default_rng(seed)
    tensors = rng.standard_normal((n, 112, 8, 8)).astype(np.float32)
    masks = np.ones((n, 8), dtype=np.float32)
    # Give the label a learnable signal so the model can actually fit it.
    labels = (tensors[:, 0].sum(axis=(1, 2)) > 0).astype(int) * 3 + (
        tensors[:, 1].sum(axis=(1, 2)) > 0
    ).astype(int)
    return ChessPlanDataset(encoded_tensors=tensors, encoded_masks=masks, labels=labels.tolist())


def loader(ds, batch_size: int):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


# --------------------------------------------------------------------------- #
# 1.8.1 Loss, optimization step, class weights                                 #
# --------------------------------------------------------------------------- #
def test_training_step_reduces_loss_on_fixed_batch() -> None:
    set_seed(0)
    ds = synthetic_dataset(32)
    model = HistoryChessResNet()
    trainer = ModelTrainer(model, loader(ds, 32), loader(ds, 32), epochs=1)
    tensors, _masks, labels = next(iter(loader(ds, 32)))
    losses = [trainer.training_step(tensors, labels) for _ in range(15)]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_class_weights_penalize_rare_classes() -> None:
    weights = compute_class_weights([0, 0, 0, 0, 1, 2], num_classes=6)
    assert weights.shape == (6,)
    assert weights[1] > weights[0]  # rarer class -> larger weight
    assert torch.isfinite(weights).all()
    assert weights[3] == 1.0 and weights[5] == 1.0  # absent classes -> 1.0


# --------------------------------------------------------------------------- #
# 1.8.2 Early stopping, scheduler, seeds                                       #
# --------------------------------------------------------------------------- #
def test_early_stopper_triggers_on_plateau() -> None:
    stopper = EarlyStopper(patience=2)
    assert stopper.step(0.50) is False  # new best
    assert stopper.step(0.60) is False  # improved
    assert stopper.step(0.60) is False  # 1 bad
    assert stopper.step(0.59) is True  # 2 bad -> stop


def test_seeding_is_reproducible() -> None:
    set_seed(42)
    a = HistoryChessResNet()
    set_seed(42)
    b = HistoryChessResNet()
    pa = next(iter(a.parameters()))
    pb = next(iter(b.parameters()))
    assert torch.equal(pa, pb)


def test_evaluate_returns_metrics() -> None:
    set_seed(0)
    ds = synthetic_dataset(24)
    model = HistoryChessResNet()
    trainer = ModelTrainer(model, loader(ds, 8), loader(ds, 8), epochs=1)
    val_loss, acc, macro_f1, per_class = trainer.evaluate(loader(ds, 8))
    assert np.isfinite(val_loss)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    assert len(per_class) == 6


def test_fit_runs_and_restores_best(tmp_path) -> None:
    set_seed(0)
    ds = synthetic_dataset(32)
    model = HistoryChessResNet()
    trainer = ModelTrainer(
        model, loader(ds, 16), loader(ds, 16), epochs=3, early_stopping_patience=5
    )
    ckpt = tmp_path / "best_resnet.pt"
    result = trainer.fit(checkpoint_path=ckpt)
    assert result["best_epoch"] >= 1
    assert 0.0 <= result["best_f1"] <= 1.0
    assert len(trainer.logger.records) >= 1
    assert ckpt.is_file()  # best checkpoint persisted


# --------------------------------------------------------------------------- #
# Checkpoint round-trip (1.8.4)                                                 #
# --------------------------------------------------------------------------- #
def test_checkpoint_save_and_load(tmp_path) -> None:
    model = HistoryChessResNet()
    path = save_checkpoint(model, tmp_path / "ckpt.pt", epoch=7, best_f1=0.65)
    fresh = HistoryChessResNet()
    payload = load_checkpoint(path, fresh)
    assert payload["epoch"] == 7
    assert payload["best_f1"] == 0.65
    for p1, p2 in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p1, p2)
