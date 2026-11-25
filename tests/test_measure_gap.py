"""Tests for the train-side sampling used by scripts/measure_gap.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from measure_gap import val_mix_rows  # noqa: E402


def _classes(sizes):
    start, out = 0, []
    for size in sizes:
        out.append(np.arange(start, start + size))
        start += size
    return out


def test_sample_matches_validation_mix_and_uses_only_trained_rows():
    trained = _classes([500, 500, 500])
    picked = val_mix_rows(trained, np.array([0.6, 0.3, 0.1]), 100, np.random.RandomState(0))
    counts = [int(np.isin(picked, rows).sum()) for rows in trained]
    assert counts == [60, 30, 10]
    assert len(set(picked.tolist())) == len(picked) == 100


def test_scales_down_instead_of_distorting_the_mix():
    trained = _classes([30, 500])  # class 0 is short: 100 * 0.6 = 60 wanted
    picked = val_mix_rows(trained, np.array([0.6, 0.4]), 100, np.random.RandomState(0))
    counts = [int(np.isin(picked, rows).sum()) for rows in trained]
    assert counts == [30, 20]


def test_absent_class_in_validation_is_not_sampled():
    trained = _classes([100, 100])
    picked = val_mix_rows(trained, np.array([1.0, 0.0]), 50, np.random.RandomState(0))
    assert len(picked) == 50 and picked.max() < 100
