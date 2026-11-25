"""Tests for the helpers in scripts/final_training.py (synthetic data only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import final_training as ft  # noqa: E402
import tune_logit_bias as tb  # noqa: E402


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self.labels = list(range(n))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return torch.zeros(1), torch.zeros(1), self.labels[i]


def test_report_perfect_predictions() -> None:
    ys = [0, 1, 2, 3, 4, 5, 5]
    rep = ft.report(ys, ys)
    assert rep["accuracy"] == 1.0
    assert rep["macro_f1"] == 1.0
    assert np.trace(np.array(rep["confusion_matrix"])) == len(ys)
    assert set(rep["per_class"]) == set(ft.CLASS_NAMES)


def test_report_counts_errors_and_support() -> None:
    rep = ft.report([0, 0, 1, 1], [0, 1, 1, 1])
    assert rep["accuracy"] == 0.75
    assert rep["per_class"]["K_STORM"]["support"] == 2
    assert rep["per_class"]["K_STORM"]["recall"] == 0.5
    assert rep["per_class"]["Q_EXP"]["precision"] == 2 / 3


def test_view_keeps_labels_aligned() -> None:
    view = ft._View(_ToyDataset(10), [7, 2, 5])
    assert len(view) == 3
    assert view.labels == [7, 2, 5]
    assert [view[i][2] for i in range(3)] == [7, 2, 5]


def test_checkpoint_path_tag_keeps_runs_separate() -> None:
    plain = ft._ckpt_path("C_4ply_mirror_reg", 42)
    tagged = ft._ckpt_path("C_4ply_mirror_reg", 42, "_80k")
    assert plain != tagged
    assert "_80k" in tagged.name


def test_fit_bias_recovers_a_shifted_class_prior() -> None:
    rng = np.random.RandomState(0)
    y = rng.randint(0, 6, size=3000)
    logits = rng.randn(3000, 6) * 0.3
    logits[np.arange(3000), y] += 2.0
    shifted = logits.copy()
    shifted[:, 5] += 3.0  # model wrongly over-predicts class 5
    before = tb.macro_f1(y, shifted.argmax(1))
    bias = tb.fit_bias(shifted, y)
    after = tb.macro_f1(y, (shifted + bias).argmax(1))
    assert after > before
    assert bias.shape == (6,)
    assert abs(bias.mean()) < 1e-9
