"""Tests for the pure-logic helpers in scripts/analyze_variant_c.py (synthetic data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_variant_c as av  # noqa: E402

NAMES = ["a", "b", "c"]


def test_confusion_summary_counts_and_row_normalization() -> None:
    y = [0, 0, 1, 1, 1, 2]
    pred = [0, 1, 1, 1, 2, 2]
    s = av.confusion_summary(y, pred, n=3)
    assert s["counts"] == [[1, 1, 0], [0, 2, 1], [0, 0, 1]]
    assert np.allclose(np.array(s["row_normalized"]).sum(1), 1.0)
    assert s["row_normalized"][1][1] == pytest.approx(2 / 3)


def test_confusion_summary_handles_empty_true_class() -> None:
    s = av.confusion_summary([0, 0], [0, 1], n=3)
    assert s["row_normalized"][2] == [0.0, 0.0, 0.0]


def test_top_confusions_ranks_both_directions() -> None:
    cm = [[8, 1, 3], [2, 5, 0], [4, 1, 6]]
    out = av.top_confusions(cm, 0, NAMES)
    assert out["false_negatives"][0] == ("c", 3, pytest.approx(3 / 12))
    assert out["false_positives"][0] == ("c", 4, pytest.approx(4 / 14))
    assert out["fn_total"] == 4
    assert out["fp_total"] == 6
    assert [n for n, _, _ in out["false_negatives"]] == ["c", "b"]


def test_error_share_counts_fn_and_fp() -> None:
    cm = [[8, 1, 3], [2, 5, 0], [4, 1, 6]]
    s = av.error_share(cm, 0)
    errors = 1 + 3 + 2 + 0 + 4 + 1
    assert s["fn_share"] == pytest.approx(4 / errors)
    assert s["fp_share"] == pytest.approx(6 / errors)
    assert s["involved_share"] == pytest.approx(10 / errors)


def test_error_share_no_errors() -> None:
    assert av.error_share([[3, 0], [0, 2]], 0)["involved_share"] == 0.0


def test_recall_gap_is_train_minus_val() -> None:
    assert av.recall_gap([0.9, 0.5], [0.7, 0.6]) == pytest.approx([0.2, -0.1])


def test_last_n_improvement_and_slope() -> None:
    v = [0.5, 0.6, 0.62, 0.64, 0.66]
    assert av.last_n_improvement(v, 2) == pytest.approx(0.04)
    assert av.last_n_improvement(v, 3) == pytest.approx(0.06)
    assert av.curve_slope(v, 3) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        av.last_n_improvement([0.1, 0.2], 2)


def test_metrics_block_perfect_and_split_guard() -> None:
    y = np.arange(6)
    blk = av.metrics_block(y, y)
    assert blk["accuracy"] == 1.0
    assert blk["macro_f1"] == 1.0
    with pytest.raises(ValueError):
        av.load_split("test")
