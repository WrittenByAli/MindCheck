"""Unit tests for the sliding history window extractor (Subtask 1.2.3)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess
import chess.pgn
import pytest

from src.data.window_extractor import PositionWindow, default_game_id, extract_windows

_SHUFFLE = ["Nf3", "Nf6", "Ng1", "Ng8"]


def build_game(
    num_plies: int = 24, headers: Optional[dict] = None
) -> Tuple[chess.pgn.Game, List[chess.Move]]:
    """Build a legal game and return it alongside its ordered move list."""
    game = chess.pgn.Game()
    game.headers["Result"] = "1-0"
    if headers:
        game.headers.update(headers)
    board = chess.Board()
    node: chess.pgn.GameNode = game
    moves: List[chess.Move] = []
    for i in range(num_plies):
        move = board.parse_san(_SHUFFLE[i % 4])
        moves.append(move)
        node = node.add_main_variation(move)
        board.push(move)
    return game, moves


def fen_after(moves: List[chess.Move], n: int) -> str:
    board = chess.Board()
    for move in moves[:n]:
        board.push(move)
    return board.fen()


# --------------------------------------------------------------------------- #
# Acceptance criteria                                                          #
# --------------------------------------------------------------------------- #
def test_window_count_and_length() -> None:
    game, _ = build_game(24)
    windows = list(extract_windows(game))  # start_ply=12, history=8
    assert len(windows) == 24 - 12 + 1  # plies 12..24 inclusive
    assert all(len(w.board_history) == 8 for w in windows)


def test_ply_indices_are_sequential() -> None:
    game, _ = build_game(24)
    windows = list(extract_windows(game))
    assert [w.ply_index for w in windows] == list(range(12, 25))


def test_final_board_matches_ply_index() -> None:
    game, moves = build_game(24)
    for w in extract_windows(game):
        assert w.current_board.fen() == fen_after(moves, w.ply_index)
        assert w.board_history[-1] is w.current_board


def test_boards_are_independent_clones() -> None:
    game, _ = build_game(24)
    windows = list(extract_windows(game))
    first, last = windows[0], windows[-1]
    assert first.current_board is not last.current_board
    assert first.current_board.fen() != last.current_board.fen()
    # Mutating one snapshot must not affect another.
    original = last.current_board.fen()
    first.current_board.push(chess.Move.from_uci("e2e4"))
    assert last.current_board.fen() == original


# --------------------------------------------------------------------------- #
# Early-game padding behaviour (shorter windows, handled by encoder later)     #
# --------------------------------------------------------------------------- #
def test_early_plies_have_up_to_eight_boards() -> None:
    game, _ = build_game(24)
    windows = list(extract_windows(game, start_ply=1))
    by_ply = {w.ply_index: w for w in windows}
    assert len(by_ply[1].board_history) == 2  # [B0, B1]
    assert len(by_ply[7].board_history) == 8  # [B0..B7]
    assert len(by_ply[8].board_history) == 8  # [B1..B8], capped
    assert all(len(w.board_history) <= 8 for w in windows)


# --------------------------------------------------------------------------- #
# Metadata                                                                     #
# --------------------------------------------------------------------------- #
def test_turn_alternates_with_ply_parity() -> None:
    game, _ = build_game(24)
    for w in extract_windows(game):
        expected = chess.WHITE if w.ply_index % 2 == 0 else chess.BLACK
        assert w.turn == expected


def test_metadata_fields_populated() -> None:
    game, _ = build_game(24, {"WhiteElo": "2500", "BlackElo": "2450"})
    w = next(extract_windows(game))
    assert w.last_move is not None
    assert isinstance(w.last_move_san, str) and w.last_move_san
    assert isinstance(w.is_check, bool)
    assert w.white_elo == 2500
    assert w.black_elo == 2450
    assert w.result == "1-0"


def test_start_ply_zero_yields_initial_position() -> None:
    game, _ = build_game(24)
    windows = list(extract_windows(game, start_ply=0))
    assert windows[0].ply_index == 0
    assert windows[0].last_move is None
    assert windows[0].last_move_san is None
    assert windows[0].board_history == [windows[0].current_board]
    assert windows[0].current_board.fen() == chess.STARTING_FEN
    assert [w.ply_index for w in windows] == list(range(0, 25))


def test_custom_history_length() -> None:
    game, _ = build_game(24)
    windows = list(extract_windows(game, history_length=4))
    assert all(len(w.board_history) == 4 for w in windows)


def test_default_game_id_prefers_site() -> None:
    game, _ = build_game(24, {"Site": "https://lichess.org/abcd1234"})
    assert default_game_id(game) == "https://lichess.org/abcd1234"


def test_default_game_id_falls_back_to_headers() -> None:
    game, _ = build_game(24, {"White": "Magnus", "Black": "Hikaru", "Date": "2026.01.01"})
    gid = default_game_id(game)
    assert "Magnus" in gid and "Hikaru" in gid


def test_explicit_game_id_used() -> None:
    game, _ = build_game(24)
    w = next(extract_windows(game, game_id="CUSTOM"))
    assert w.game_id == "CUSTOM"


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_invalid_history_length_raises() -> None:
    game, _ = build_game(24)
    with pytest.raises(ValueError):
        list(extract_windows(game, history_length=0))


def test_invalid_start_ply_raises() -> None:
    game, _ = build_game(24)
    with pytest.raises(ValueError):
        list(extract_windows(game, start_ply=-1))


def test_positionwindow_is_dataclass_instance() -> None:
    game, _ = build_game(24)
    w = next(extract_windows(game))
    assert isinstance(w, PositionWindow)
