"""Unit tests for the five-plan symbolic rule engine (Task 1.4)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess

from src.rules.fact_extractor import ChessFactExtractor, ExtractedBoardFacts
from src.rules.rule_engine import SymbolicRuleEngine

ENGINE = SymbolicRuleEngine()
EXTRACTOR = ChessFactExtractor()


def window(
    fen: Optional[str] = None, moves: Optional[List[str]] = None, history_length: int = 8
) -> Tuple[List[chess.Board], ExtractedBoardFacts]:
    """Build an (history, facts) pair by playing SAN moves from a start position.

    Uses ``stack=False`` to match the real pipeline (``extract_windows``).
    """
    board = chess.Board(fen) if fen else chess.Board()
    snapshots = [board.copy(stack=False)]
    for san in moves or []:
        board.push_san(san)
        snapshots.append(board.copy(stack=False))
    history = snapshots[-history_length:]
    facts = EXTRACTOR.extract_facts(history[-1])
    return history, facts


# --------------------------------------------------------------------------- #
# 1.4.1 Kingside storm                                                         #
# --------------------------------------------------------------------------- #
_STORM_START = "1n3rk1/pp3ppp/8/8/8/8/PP3PPP/1N3RK1 w - - 0 1"


def test_kingside_storm_detected() -> None:
    h, f = window(_STORM_START, ["g4", "Nc6", "h4", "Nb8", "g5", "Nc6"])
    assert ENGINE.evaluate_kingside_storm(h, f) > 0.70


def test_kingside_storm_quiet_is_low() -> None:
    # Both sides castled kingside but no pawns advanced at all.
    h, f = window("1n3rk1/pp3ppp/8/8/8/8/PP3PPP/1N3RK1 w - - 0 1", ["Nc3", "Nc6", "Nb1", "Nb8"])
    assert ENGINE.evaluate_kingside_storm(h, f) < 0.20


# --------------------------------------------------------------------------- #
# 1.4.2 Queenside expansion                                                    #
# --------------------------------------------------------------------------- #
_EXP_START = "1n3rk1/1p3ppp/8/8/8/8/PPP2PPP/1N3RK1 w - - 0 1"


def test_queenside_expansion_detected() -> None:
    h, f = window(_EXP_START, ["b4", "Nc6", "a4", "Nb8", "c4", "Nc6"])
    assert ENGINE.evaluate_queenside_expansion(h, f) > 0.70


def test_single_queenside_push_is_not_expansion() -> None:
    h, f = window(_EXP_START, ["b4", "Nc6"])
    assert ENGINE.evaluate_queenside_expansion(h, f) < 0.50


# --------------------------------------------------------------------------- #
# 1.4.3 Outpost occupation                                                     #
# --------------------------------------------------------------------------- #
def test_outpost_occupation_detected() -> None:
    h, f = window("7k/8/3p4/3N4/4P3/8/8/4K3 w - - 0 1")  # white Nd5 outpost
    assert ENGINE.evaluate_outpost_occupation(h, f) == 1.0


def test_no_outpost_in_starting_position() -> None:
    h, f = window()
    assert ENGINE.evaluate_outpost_occupation(h, f) == 0.0


# --------------------------------------------------------------------------- #
# 1.4.4 Rook-file buildup                                                      #
# --------------------------------------------------------------------------- #
def test_rook_battery_open_file_detected() -> None:
    h, f = window("2r1r1k1/5ppp/8/8/2R5/2R5/2Q5/6K1 w - - 0 1")  # Alekhine's Gun c-file
    assert ENGINE.evaluate_rook_file_buildup(h, f) == 1.0


def test_no_battery_on_closed_board() -> None:
    h, f = window()  # starting position, no open files
    assert ENGINE.evaluate_rook_file_buildup(h, f) == 0.0


# --------------------------------------------------------------------------- #
# 1.4.5 Pawn-chain undermining                                                 #
# --------------------------------------------------------------------------- #
def test_chain_undermining_french_c5_lever() -> None:
    # Black plays ...c5 striking the base (d4) of White's d4-e5 chain.
    h, f = window("4k3/2p5/4p3/3pP3/3P4/8/8/4K3 b - - 0 1", ["c5"])
    assert ENGINE.evaluate_chain_undermining(h, f) == 1.0


def test_no_undermining_without_lever() -> None:
    # White chain present, but black makes a quiet king move (no lever).
    h, f = window("4k3/2p5/4p3/3pP3/3P4/8/8/4K3 b - - 0 1", ["Kd7"])
    assert ENGINE.evaluate_chain_undermining(h, f) == 0.0


# --------------------------------------------------------------------------- #
# 1.4.6 Classifier & ambiguity resolution                                     #
# --------------------------------------------------------------------------- #
def test_classifier_selects_kingside_storm() -> None:
    h, f = window(_STORM_START, ["g4", "Nc6", "h4", "Nb8", "g5", "Nc6"])
    result = ENGINE.evaluate_window(h, f)
    assert result.predicted_class == "K_STORM"
    assert result.class_id == 0
    assert result.confidence > 0.70
    assert result.supporting_evidence


def test_classifier_selects_rook_file() -> None:
    h, f = window("2r1r1k1/5ppp/8/8/2R5/2R5/2Q5/6K1 w - - 0 1")
    result = ENGINE.evaluate_window(h, f)
    assert result.predicted_class == "ROOK_FILE"
    assert result.class_id == 3


def test_classifier_selects_chain_undermine() -> None:
    h, f = window("4k3/2p5/4p3/3pP3/3P4/8/8/4K3 b - - 0 1", ["c5"])
    result = ENGINE.evaluate_window(h, f)
    assert result.predicted_class == "CHAIN_UNDERMINE"
    assert result.class_id == 4


def test_quiet_position_is_no_plan() -> None:
    h, f = window(moves=["Nf3", "Nf6", "Ng1", "Ng8"])
    result = ENGINE.evaluate_window(h, f)
    assert result.predicted_class == "NO_PLAN"
    assert result.class_id == 5


def test_ambiguous_two_plans_tie_to_no_plan() -> None:
    # White has BOTH a c-file rook battery (open) and an occupied d5 outpost;
    # the two 1.0 scores are within the ambiguity margin -> NO_PLAN.
    h, f = window("6k1/8/3p4/3N4/4P3/8/2R5/2R3K1 w - - 0 1")
    result = ENGINE.evaluate_window(h, f)
    assert result.all_scores["ROOK_FILE"] == 1.0
    assert result.all_scores["OUTPOST"] == 1.0
    assert result.predicted_class == "NO_PLAN"


def test_evaluation_is_deterministic() -> None:
    h, f = window(_STORM_START, ["g4", "Nc6", "h4", "Nb8", "g5", "Nc6"])
    first = ENGINE.evaluate_window(h, f)
    second = ENGINE.evaluate_window(h, f)
    assert first.all_scores == second.all_scores
    assert first.predicted_class == second.predicted_class


def test_all_scores_cover_five_plans() -> None:
    h, f = window()
    result = ENGINE.evaluate_window(h, f)
    assert set(result.all_scores) == {"K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE"}
    assert all(0.0 <= v <= 1.0 for v in result.all_scores.values())


def test_empty_history_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        ENGINE.evaluate_window([])
