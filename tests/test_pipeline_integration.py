"""Cross-module integration tests: extract_windows -> fact_extractor -> rule_engine.

Exercises the real production path end-to-end so that contract
mismatches (like boards with empty move_stack) are caught.
"""

from __future__ import annotations

import io
from typing import List

import chess
import chess.pgn

from src.data.window_extractor import PositionWindow, extract_windows
from src.rules.fact_extractor import ChessFactExtractor
from src.rules.rule_engine import SymbolicRuleEngine

ENGINE = SymbolicRuleEngine()
EXTRACTOR = ChessFactExtractor()

_FRENCH_DEFENSE_PGN = """\
[Event "Test"]
[Site "integration"]
[White "A"]
[Black "B"]
[Result "*"]
[WhiteElo "2000"]
[BlackElo "2000"]

1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7
7. Qg4 O-O 8. Nf3 Nbc6 9. Bd3 f5 10. exf6 Rxf6 11. Bg5 Rf7
12. Bxe7 Nxe7 *
"""

_KINGSIDE_STORM_PGN = """\
[Event "Test"]
[Site "storm"]
[White "C"]
[Black "D"]
[Result "*"]
[WhiteElo "2100"]
[BlackElo "2100"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7
6. Re1 b5 7. Bb3 O-O 8. h3 d6 9. c3 Na5 10. Bc2 c5
11. d4 Nd7 12. d5 Nb6 13. Nbd2 f5 14. g4 fxg4 15. hxg4 *
"""


def _parse_pgn(pgn_text: str) -> chess.pgn.Game:
    return chess.pgn.read_game(io.StringIO(pgn_text))


def _windows_from_pgn(pgn_text: str, start_ply: int = 12) -> List[PositionWindow]:
    game = _parse_pgn(pgn_text)
    return list(extract_windows(game, start_ply=start_ply))


def test_pipeline_windows_have_empty_move_stack() -> None:
    windows = _windows_from_pgn(_FRENCH_DEFENSE_PGN)
    assert len(windows) > 0
    for w in windows:
        for board in w.board_history:
            assert board.move_stack == [], "Real pipeline boards must use copy(stack=False)"


def test_rule_engine_works_on_real_pipeline_windows() -> None:
    windows = _windows_from_pgn(_FRENCH_DEFENSE_PGN)
    for w in windows:
        facts = EXTRACTOR.extract_facts(w.current_board)
        result = ENGINE.evaluate_window(w.board_history, facts)
        assert result.predicted_class in {
            "K_STORM",
            "Q_EXP",
            "OUTPOST",
            "ROOK_FILE",
            "CHAIN_UNDERMINE",
            "NO_PLAN",
        }
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.all_scores) == 5


def test_chain_undermine_nonzero_on_real_french_lever() -> None:
    windows = _windows_from_pgn(_FRENCH_DEFENSE_PGN, start_ply=8)
    lever_window = None
    for w in windows:
        if w.last_move_san == "c5":
            lever_window = w
            break
    assert lever_window is not None, "Expected a ...c5 lever in the French PGN"
    facts = EXTRACTOR.extract_facts(lever_window.current_board)
    result = ENGINE.evaluate_window(lever_window.board_history, facts)
    assert (
        result.all_scores["CHAIN_UNDERMINE"] > 0.0
    ), "CHAIN_UNDERMINE must be non-zero when a real pipeline lever is played"


def test_kingside_storm_detected_on_real_pipeline() -> None:
    windows = _windows_from_pgn(_KINGSIDE_STORM_PGN, start_ply=12)
    last = windows[-1]
    facts = EXTRACTOR.extract_facts(last.current_board)
    result = ENGINE.evaluate_window(last.board_history, facts)
    assert (
        result.all_scores["K_STORM"] > 0.0
    ), "K_STORM must detect g4/hxg4 kingside activity in real pipeline"
