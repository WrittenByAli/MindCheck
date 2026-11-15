"""Unit tests for the game-level dataset splitter (Subtask 1.5.3)."""

from __future__ import annotations

import collections
from typing import List

import chess
import pytest

from src.data.dataset_splitter import GameLevelSplitter
from src.data.weak_labeler import LabeledPositionRecord


def make_records(game_id: str, n: int) -> List[LabeledPositionRecord]:
    return [
        LabeledPositionRecord(
            window_id=f"{game_id}#{i}",
            game_id=game_id,
            ply_index=i,
            turn="white",
            label_class="NO_PLAN",
            label_id=5,
            weak_confidence=0.9,
            fen=chess.STARTING_FEN,
        )
        for i in range(n)
    ]


def test_bad_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        GameLevelSplitter(train=0.6, val=0.3, test=0.3)


def test_assignment_is_deterministic_and_stable() -> None:
    splitter = GameLevelSplitter(seed=42)
    # Stable across calls and independent of interpreter hash randomization.
    assert splitter.assign_split("game_abc") == splitter.assign_split("game_abc")
    assert splitter.assign_split("game_abc") in {"train", "val", "test"}


def test_all_windows_of_a_game_share_one_split() -> None:
    splitter = GameLevelSplitter()
    records = make_records("gameX", 20)
    splitter.split(records)
    assert len({r.split for r in records}) == 1


def test_splits_are_game_disjoint() -> None:
    splitter = GameLevelSplitter()
    records: List[LabeledPositionRecord] = []
    for g in range(200):
        records.extend(make_records(f"game_{g}", 5))
    splitter.split(records)

    games = splitter.games_by_split(r.game_id for r in records)
    assert games["train"].isdisjoint(games["val"])
    assert games["train"].isdisjoint(games["test"])
    assert games["val"].isdisjoint(games["test"])
    assert splitter.verify_disjoint(records) is True


def test_split_proportions_are_roughly_70_15_15() -> None:
    splitter = GameLevelSplitter(seed=42)
    assignments = collections.Counter(splitter.assign_split(f"game_{g}") for g in range(2000))
    total = sum(assignments.values())
    assert abs(assignments["train"] / total - 0.70) < 0.05
    assert abs(assignments["val"] / total - 0.15) < 0.05
    assert abs(assignments["test"] / total - 0.15) < 0.05


def test_verify_disjoint_requires_split_assigned() -> None:
    splitter = GameLevelSplitter()
    with pytest.raises(ValueError):
        splitter.verify_disjoint(make_records("g", 3))  # split not assigned yet


def test_different_seed_can_change_assignment() -> None:
    # At least some games land differently under a different seed.
    a = GameLevelSplitter(seed=1)
    b = GameLevelSplitter(seed=2)
    changed = sum(a.assign_split(f"g{i}") != b.assign_split(f"g{i}") for i in range(200))
    assert changed > 0
