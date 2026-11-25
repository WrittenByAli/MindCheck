"""Tests for the v2 gate rules and the fast macro-F1 (synthetic inputs only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_v2 as v2  # noqa: E402
from tune_logit_bias import macro_f1  # noqa: E402


def test_fast_macro_f1_equals_sklearn_including_absent_classes():
    rng = np.random.RandomState(0)
    for trial in range(50):
        y = rng.randint(0, 6 if trial % 2 else 4, size=200)  # classes 4, 5 sometimes absent
        pred = rng.randint(0, 6 if trial % 3 else 3, size=200)
        expected = f1_score(y, pred, labels=list(range(6)), average="macro", zero_division=0)
        assert abs(macro_f1(y, pred) - expected) < 1e-12


def test_winner_is_best_score_unless_within_tie_margin():
    assert v2.choose_winner({"F": 0.60, "G": 0.55}, {"F": 0.3, "G": 0.0}) == "F"
    assert v2.choose_winner({"F": 0.600, "G": 0.597}, {"F": 0.3, "G": 0.1}) == "G"


def _seed(score=0.60, raw=0.55, f1_gap=0.02, acc_gap=0.01, chain=0.05):
    return dict(score=score, raw_f1=raw, f1_gap=f1_gap, acc_gap=acc_gap, chain_gap=chain)


def test_validation_gates_pass_and_fail_as_specified():
    good = {42: _seed(), 123: _seed(raw=0.552), 999: _seed(raw=0.551)}
    gates = v2.validation_gates(good, b0_score=0.58)
    assert all(gates[g]["passed"] for g in "ABCD")
    bad = {42: _seed(f1_gap=0.031), 123: _seed(raw=0.57, chain=0.2), 999: _seed(acc_gap=0.021)}
    gates = v2.validation_gates(bad, b0_score=0.595)
    assert not gates["A"]["passed"]  # 0.60 is not 0.595 + 0.01
    assert gates["B"]["failed_seeds"] == [42, 999]
    assert not gates["C"]["passed"]
    assert gates["D"]["failed_seeds"] == [123] and not gates["D"]["blocks_promotion"]


def _hold(val=0.55, raw=0.55, biased=0.60):
    return dict(val_raw_f1=val, holdout_raw_f1=raw, holdout_biased_f1=biased)


def test_holdout_gates_require_beating_v1_with_the_chosen_seed():
    seeds = {42: _hold(), 123: _hold(), 999: _hold()}
    assert all(g["passed"] for g in v2.holdout_gates(seeds, 0.58, chosen=42).values())
    seeds[42] = _hold(biased=0.579)  # chosen seed does not beat v1
    assert not v2.holdout_gates(seeds, 0.58, chosen=42)["F"]["passed"]
    seeds = {42: _hold(raw=0.52), 123: _hold(), 999: _hold()}
    assert not v2.holdout_gates(seeds, 0.58, chosen=123)["E"]["passed"]


def test_promotion_ignores_target_gate_d_only():
    gates = {g: {"passed": True} for g in "ABCEF"}
    gates["D"] = {"passed": False}
    assert v2.promotable(gates)
    gates["F"] = {"passed": False}
    assert not v2.promotable(gates)


def test_per_game_sample_is_capped_uniform_and_deterministic():
    ids = ["a"] * 100 + ["b"] * 3
    first = v2.per_game_sample(ids, 25)
    assert len(first) == 28 and len(set(first.tolist())) == 28
    assert sum(i >= 100 for i in first) == 3
    assert np.array_equal(first, v2.per_game_sample(ids, 25))
