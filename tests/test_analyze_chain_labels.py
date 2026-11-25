"""Tests for the pure helpers in scripts/analyze_chain_labels.py (synthetic inputs)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_chain_labels as acl  # noqa: E402


def test_first_push_offset_finds_earliest_lever() -> None:
    assert acl.first_push_offset([0.0, 0.0, 0.8, 1.0]) == 2
    assert acl.first_push_offset([1.0, 0.0]) == 0


def test_first_push_offset_none_when_no_lever_or_missing() -> None:
    assert acl.first_push_offset([0.0, 0.0, 0.0]) is None
    assert acl.first_push_offset([None, None, 0.0]) is None
    assert acl.first_push_offset([None, 0.8]) == 1


def test_first_push_offset_respects_level() -> None:
    assert acl.first_push_offset([0.0, 0.5, 1.0], level=0.5) == 1
    assert acl.first_push_offset([0.0, 0.5, 1.0]) == 2


def test_assign_group_visible_future_only_and_none() -> None:
    assert acl.assign_group(1.0, 0) == "visible"
    assert acl.assign_group(0.8, 0) == "visible"
    assert acl.assign_group(0.0, 3) == "future_only"
    assert acl.assign_group(0.0, None) == "no_push_evidence"


def test_assign_group_threshold_boundary() -> None:
    assert acl.assign_group(0.5, None) == "visible"
    assert acl.assign_group(0.49, 2) == "future_only"


def test_margin_stats_picks_runner_up_and_margin() -> None:
    agg = {"K_STORM": 0.1, "Q_EXP": 0.3, "OUTPOST": 0.0, "ROOK_FILE": 0.0, "CHAIN_UNDERMINE": 0.6}
    stats = acl.margin_stats(agg, "CHAIN_UNDERMINE")
    assert stats["runner_up"] == "Q_EXP"
    assert abs(stats["margin"] - 0.3) < 1e-9
    assert stats["near_threshold"] is False


def test_margin_stats_flags_near_threshold() -> None:
    agg = {"K_STORM": 0.0, "CHAIN_UNDERMINE": 0.52}
    assert acl.margin_stats(agg, "CHAIN_UNDERMINE")["near_threshold"] is True


def test_fraction_handles_zero_total() -> None:
    assert acl.fraction(0, 0) == 0.0
    assert acl.fraction(1, 4) == 0.25


def test_predicted_overlap_arithmetic_matches_lever_geometry() -> None:
    # A push at ply p is visible in windows ending at q in [p, p+6]. With the
    # label averaged over t..t+8 and needing >=5 windows at 1.0, the labelled
    # windows are t in [p-4, p+2]; the push is already visible for t >= p.
    p = 20
    labelled = [
        t for t in range(p - 10, p + 10) if sum(1 for k in range(9) if p <= t + k <= p + 6) >= 5
    ]
    visible = [t for t in labelled if p <= t <= p + 6]
    assert labelled == list(range(p - 4, p + 3))
    assert len(labelled) - len(visible) == 4 and len(labelled) == 7
