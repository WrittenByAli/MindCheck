"""Tests for the metric helpers of scripts/investigate_v2.py (synthetic inputs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import investigate_v2 as inv  # noqa: E402


def _random(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 6, n), rng.randint(0, 6, n)


def test_metrics_match_sklearn():
    y, p = _random()
    m = inv.evaluate(y, p)
    labels = list(range(6))
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    assert np.isclose(m["accuracy"], (y == p).mean())
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, p))
    assert np.isclose(m["macro_precision"], precision.mean())
    assert np.isclose(m["macro_recall"], recall.mean())
    assert np.isclose(m["macro_f1"], f1.mean())
    assert np.isclose(
        m["weighted_f1"], f1_score(y, p, labels=labels, average="weighted", zero_division=0)
    )


def test_subset_averages_only_over_requested_present_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 5, 1, 1])
    m = inv.evaluate(y, p, classes=range(5))
    assert np.isclose(m["balanced_accuracy"], (0.5 + 1.0) / 2)
    assert np.isclose(m["accuracy"], 0.75)


def test_stratified_random_expectation_matches_simulation():
    train = np.array([0.1, 0.1, 0.2, 0.05, 0.05, 0.5])
    truth = np.array([0.05, 0.15, 0.2, 0.05, 0.01, 0.54])
    rng = np.random.RandomState(1)
    y = rng.choice(6, 400000, p=truth / truth.sum())
    p = rng.choice(6, 400000, p=train)
    expected = inv.stratified_random(train, inv.share(y))
    simulated = inv.evaluate(y, p)
    assert abs(expected["accuracy"] - simulated["accuracy"]) < 0.005
    assert abs(expected["macro_f1"] - simulated["macro_f1"]) < 0.005
    assert abs(expected["balanced_accuracy"] - simulated["balanced_accuracy"]) < 0.005


def test_phase_bands():
    assert [inv.phase_of(p) for p in (12, 25, 26, 55, 56)] == [
        "opening",
        "opening",
        "middlegame",
        "middlegame",
        "late",
    ]


def test_game_bootstrap_detects_a_clearly_better_model():
    y, noise = _random(6000)
    games = np.repeat(np.arange(200), 30).astype(str)
    better = np.where(np.arange(6000) % 3 == 0, noise, y)
    out = inv.game_bootstrap(games, y, {"weak": noise, "strong": better}, n_boot=200)
    low, high = out["strong"]["accuracy_minus_reference"]
    assert out["games"] == 200 and 0 < low <= high
