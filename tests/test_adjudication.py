"""Unit tests for Cohen's Kappa adjudication (Subtask 1.9.4)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.models.adjudicate import (
    build_ground_truth,
    compute_reliability,
    find_disagreements,
    kappa,
    load_annotations,
    observed_agreement,
)

_LABELS = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"]


def _sheet(labels: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [f"W{i:04d}" for i in range(len(labels))],
            "game_id": ["g"] * len(labels),
            "ply_index": list(range(len(labels))),
            "turn": ["white"] * len(labels),
            "fen_current": ["8/8/8/8/8/8/8/8 w - - 0 1"] * len(labels),
            "history_moves_san": ["e4 e5"] * len(labels),
            "primary_label": labels,
        }
    )


def test_identical_sheets_give_kappa_one() -> None:
    labels = [_LABELS[i % 6] for i in range(60)]
    m1 = _sheet(labels)
    m2 = _sheet(list(labels))
    rel = compute_reliability(m1, m2)
    assert rel["observed_agreement"] == 1.0
    assert rel["cohen_kappa"] == 1.0
    assert rel["n_disagreements"] == 0


def test_partial_disagreement() -> None:
    a = [_LABELS[i % 6] for i in range(60)]
    b = list(a)
    b[0] = "NO_PLAN" if b[0] != "NO_PLAN" else "K_STORM"
    b[1] = "NO_PLAN" if b[1] != "NO_PLAN" else "Q_EXP"
    rel = compute_reliability(_sheet(a), _sheet(b))
    assert rel["cohen_kappa"] < 1.0
    assert rel["n_disagreements"] == 2
    assert 0.0 <= rel["observed_agreement"] < 1.0


def test_observed_agreement_and_kappa_helpers() -> None:
    assert observed_agreement(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert observed_agreement(["A", "B"], ["A", "C"]) == 0.5
    assert kappa(["K_STORM", "Q_EXP"], ["K_STORM", "Q_EXP"]) == 1.0


def test_find_disagreements_lists_only_conflicts() -> None:
    a = ["K_STORM", "Q_EXP", "OUTPOST"]
    b = ["K_STORM", "NO_PLAN", "OUTPOST"]
    disagreements = find_disagreements(_sheet(a), _sheet(b))
    assert len(disagreements) == 1
    assert disagreements.iloc[0]["member1_label"] == "Q_EXP"
    assert disagreements.iloc[0]["member2_label"] == "NO_PLAN"


def test_build_ground_truth_requires_resolution() -> None:
    a = ["K_STORM", "Q_EXP"]
    b = ["K_STORM", "NO_PLAN"]  # W0001 disagrees
    with pytest.raises(ValueError):
        build_ground_truth(_sheet(a), _sheet(b))  # unresolved -> error
    truth = build_ground_truth(_sheet(a), _sheet(b), resolved={"W0001": "Q_EXP"})
    assert len(truth) == 2
    assert set(truth["label"]) == {"K_STORM", "Q_EXP"}


def test_build_ground_truth_all_agree() -> None:
    labels = ["OUTPOST", "ROOK_FILE", "NO_PLAN"]
    truth = build_ground_truth(_sheet(labels), _sheet(labels))
    assert list(truth["label"]) == labels


def test_load_annotations_rejects_bad_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _sheet(["K_STORM", "NOT_A_PLAN"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_annotations(path)


def test_load_annotations_valid(tmp_path: Path) -> None:
    path = tmp_path / "ok.csv"
    _sheet(["K_STORM", "NO_PLAN"]).to_csv(path, index=False)
    frame = load_annotations(path)
    assert len(frame) == 2
