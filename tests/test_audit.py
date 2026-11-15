"""Unit tests for the dataset audit summary (Subtask 1.5.4)."""

from __future__ import annotations

import json
from typing import List

import chess

from src.data.audit import compute_summary, write_summary
from src.data.dataset_splitter import GameLevelSplitter
from src.data.weak_labeler import LabeledPositionRecord


def _records() -> List[LabeledPositionRecord]:
    recs: List[LabeledPositionRecord] = []
    for g in range(30):
        cls = "K_STORM" if g % 2 == 0 else "NO_PLAN"
        for p in range(4):
            recs.append(
                LabeledPositionRecord(
                    window_id=f"g{g}#{p}",
                    game_id=f"g{g}",
                    ply_index=12 + p,
                    turn="white",
                    label_class=cls,
                    label_id=0 if cls == "K_STORM" else 5,
                    weak_confidence=0.8,
                    fen=chess.STARTING_FEN,
                    white_elo=2400,
                    black_elo=2300,
                    result="1-0",
                )
            )
    GameLevelSplitter(seed=42).split(recs)
    return recs


def test_summary_totals_and_disjoint() -> None:
    summary = compute_summary(_records())
    assert summary["total_games"] == 30
    assert summary["total_windows"] == 120
    assert summary["game_overlap"]["disjoint"] is True
    assert set(summary["splits"]) == {"train", "val", "test"}
    # Split window counts sum to the total.
    assert sum(summary["splits"][s]["windows"] for s in ("train", "val", "test")) == 120


def test_summary_stats_populated() -> None:
    summary = compute_summary(_records())
    assert summary["mean_white_elo"] == 2400.0
    assert summary["mean_black_elo"] == 2300.0
    assert summary["result_breakdown"]["1-0"] == 120
    assert summary["overall_class_counts"]["K_STORM"] == 60
    assert summary["overall_class_counts"]["NO_PLAN"] == 60


def test_write_summary_creates_json(tmp_path) -> None:
    path = tmp_path / "dataset_summary.json"
    write_summary(_records(), path)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["total_windows"] == 120
    assert loaded["game_overlap"]["disjoint"] is True
