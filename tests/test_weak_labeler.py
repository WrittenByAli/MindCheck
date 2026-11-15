"""Unit tests for the retrospective weak-labeler (Subtasks 1.5.1 & 1.5.2)."""

from __future__ import annotations

import collections
from typing import List

import chess
import chess.pgn

from src.data.weak_labeler import (
    LabeledPositionRecord,
    RetrospectiveWeakLabeler,
    filter_and_balance,
    material_balance,
)

_SHUFFLE = ["Nf3", "Nf6", "Ng1", "Ng8"]


def game_from_start(moves: List[str]) -> chess.pgn.Game:
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    board = chess.Board()
    for san in moves:
        move = board.push_san(san)
        node = node.add_main_variation(move)
    return game


def game_from_fen(fen: str, moves: List[str]) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.setup(chess.Board(fen))
    node: chess.pgn.GameNode = game
    board = chess.Board(fen)
    for san in moves:
        move = board.push_san(san)
        node = node.add_main_variation(move)
    return game


def record(
    label: str,
    conf: float,
    game_id: str = "g",
    ply: int = 12,
    material: int = 0,
) -> LabeledPositionRecord:
    return LabeledPositionRecord(
        window_id=f"{game_id}#{ply}",
        game_id=game_id,
        ply_index=ply,
        turn="white",
        label_class=label,
        label_id=0 if label != "NO_PLAN" else 5,
        weak_confidence=conf,
        fen=chess.STARTING_FEN,
        material_balance=material,
    )


# --------------------------------------------------------------------------- #
# 1.5.1 Label generation                                                       #
# --------------------------------------------------------------------------- #
def test_records_have_valid_structure() -> None:
    labeler = RetrospectiveWeakLabeler(start_ply=4)
    game = game_from_start(_SHUFFLE * 6)  # 24-ply quiet game
    records = labeler.label_game(game, game_id="quiet1")
    assert records
    for r in records:
        assert r.game_id == "quiet1"
        assert r.label_class in set(
            ["NO_PLAN"] + ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE"]
        )
        assert 0.0 <= r.weak_confidence <= 1.0
        assert chess.Board(r.fen)  # FEN parses


def test_quiet_game_is_all_no_plan() -> None:
    labeler = RetrospectiveWeakLabeler(start_ply=4)
    records = labeler.label_game(game_from_start(_SHUFFLE * 6), game_id="quiet2")
    assert all(r.label_class == "NO_PLAN" for r in records)


def test_sustained_outpost_is_labeled_outpost() -> None:
    # White Nd5 outpost sustained while both sides shuffle a rook on the a/b files.
    fen = "r5k1/8/3p4/3N4/4P3/8/8/R5K1 w - - 0 1"
    moves = ["Rb1", "Rb8", "Ra1", "Ra8"] * 3  # 12 quiet plies, knight stays on d5
    labeler = RetrospectiveWeakLabeler(start_ply=4)
    records = labeler.label_game(game_from_fen(fen, moves), game_id="outpost1")
    assert records
    assert any(r.label_class == "OUTPOST" for r in records)


def test_labeling_is_deterministic() -> None:
    labeler = RetrospectiveWeakLabeler(start_ply=4)
    game = game_from_start(_SHUFFLE * 6)
    first = labeler.label_game(game, game_id="d")
    second = labeler.label_game(game, game_id="d")
    assert [(r.label_class, r.weak_confidence) for r in first] == [
        (r.label_class, r.weak_confidence) for r in second
    ]


def test_short_game_yields_no_records() -> None:
    labeler = RetrospectiveWeakLabeler(start_ply=12)
    records = labeler.label_game(game_from_start(_SHUFFLE), game_id="short")  # only 4 plies
    assert records == []


def test_material_balance_helper() -> None:
    assert material_balance(chess.Board()) == 0
    # White up a full queen.
    assert material_balance(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) == 900


# --------------------------------------------------------------------------- #
# 1.5.2 Filtering & balancing                                                  #
# --------------------------------------------------------------------------- #
def test_low_confidence_records_dropped() -> None:
    recs = [record("K_STORM", 0.80), record("K_STORM", 0.40)]
    out = filter_and_balance(recs)
    assert all(r.weak_confidence >= 0.55 for r in out)
    assert len(out) == 1


def test_material_imbalance_dropped() -> None:
    recs = [record("K_STORM", 0.90, material=0), record("K_STORM", 0.90, material=900)]
    out = filter_and_balance(recs)
    assert len(out) == 1
    assert out[0].material_balance == 0


def test_per_game_cap_enforced() -> None:
    recs = [record("K_STORM", 0.60 + i * 0.001, game_id="G", ply=i) for i in range(30)]
    out = filter_and_balance(recs, max_per_game=25)
    assert len(out) == 25
    assert all(r.game_id == "G" for r in out)


def test_class_balancing_no_class_over_fraction() -> None:
    recs: List[LabeledPositionRecord] = []
    for i in range(50):  # over-represented NO_PLAN
        recs.append(record("NO_PLAN", 0.90, game_id=f"np{i}", ply=i))
    for cls in ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE"]:
        for i in range(10):
            recs.append(record(cls, 0.90, game_id=f"{cls}{i}", ply=i))
    out = filter_and_balance(recs, max_per_game=25, max_class_fraction=0.30)
    counts = collections.Counter(r.label_class for r in out)
    total = len(out)
    assert total > 0
    for cls, n in counts.items():
        assert n <= 0.30 * total + 1e-9, f"{cls} is {n}/{total}"


def test_class_balancing_does_not_collapse_to_zero() -> None:
    # Reproduces counts seen from a real generate_dataset.py run (11/11/3
    # across 3 classes, target fraction 0.30 -> effective 1/3). The per-pass
    # target here lands exactly on an integer (e.g. 7.0), but float division
    # computed it as 6.999999999999999: floor() then undercounted by one on
    # every pass, and since the under-shoot recurred each time a class was
    # revisited, repeated passes decayed every class to 0 instead of settling
    # at the balanced fixed point.
    recs: List[LabeledPositionRecord] = []
    for i in range(11):
        recs.append(record("OUTPOST", 0.90, game_id=f"o{i}", ply=i))
    for i in range(11):
        recs.append(record("Q_EXP", 0.90, game_id=f"q{i}", ply=i))
    for i in range(3):
        recs.append(record("NO_PLAN", 0.90, game_id=f"n{i}", ply=i))
    out = filter_and_balance(recs, max_per_game=1000, max_class_fraction=0.30)
    assert len(out) > 0
    counts = collections.Counter(r.label_class for r in out)
    total = len(out)
    for cls, n in counts.items():
        assert n <= (1.0 / 3) * total + 1e-6, f"{cls} is {n}/{total}"


def test_class_balancing_converges_with_several_comparable_classes() -> None:
    # Three near-equal large classes plus one tiny one: a single greedy pass
    # per class (the old `len(counts) + 1` iteration bound) stops before the
    # cap is actually satisfied, since fixing one class shrinks the total and
    # pushes the next-largest class back over the fraction.
    recs: List[LabeledPositionRecord] = []
    for cls in ["K_STORM", "Q_EXP", "OUTPOST"]:
        for i in range(100):
            recs.append(record(cls, 0.90, game_id=f"{cls}{i}", ply=i))
    recs.append(record("ROOK_FILE", 0.90, game_id="rf0", ply=0))
    out = filter_and_balance(recs, max_per_game=1000, max_class_fraction=0.30)
    counts = collections.Counter(r.label_class for r in out)
    total = len(out)
    assert total > 0
    for cls, n in counts.items():
        assert n <= 0.30 * total + 1e-9, f"{cls} is {n}/{total}"


def test_filter_is_deterministic() -> None:
    recs = [record("NO_PLAN", 0.9, game_id=f"g{i}", ply=i) for i in range(40)]
    recs += [record("K_STORM", 0.9, game_id=f"k{i}", ply=i) for i in range(20)]
    a = filter_and_balance(recs, seed=7)
    b = filter_and_balance(recs, seed=7)
    assert [r.window_id for r in a] == [r.window_id for r in b]
