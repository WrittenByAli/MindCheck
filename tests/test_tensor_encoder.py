"""Unit tests for the spatial-temporal tensor encoder (Subtasks 1.6.1-1.6.3)."""

from __future__ import annotations

from typing import List

import chess
import numpy as np
import pytest

from src.data.tensor_encoder import BoardPlaneEncoder, ChessTensorEncoder

ENC = ChessTensorEncoder()
PLANE = BoardPlaneEncoder()


def boards_from(moves: List[str]) -> List[chess.Board]:
    board = chess.Board()
    out = [board.copy(stack=False)]
    for san in moves:
        board.push_san(san)
        out.append(board.copy(stack=False))
    return out


# --------------------------------------------------------------------------- #
# 1.6.1 Single-board encoding                                                  #
# --------------------------------------------------------------------------- #
def test_single_board_shape_and_dtype() -> None:
    arr = PLANE.encode(chess.Board(), chess.WHITE)
    assert arr.shape == (14, 8, 8)
    assert arr.dtype == np.float32


def test_starting_position_piece_planes() -> None:
    arr = PLANE.encode(chess.Board(), chess.WHITE)  # white to move, attacker white
    # Friendly pawns (plane 0) on rank 2 -> row 6; opponent pawns (plane 6) row 1.
    assert arr[0, 6, :].sum() == 8
    assert arr[6, 1, :].sum() == 8
    # Kings: friendly e1 -> row7,col4 (plane 5); opponent e8 -> row0,col4 (plane 11).
    assert arr[5, 7, 4] == 1.0
    assert arr[11, 0, 4] == 1.0
    assert arr[5].sum() == 1.0 and arr[11].sum() == 1.0


def test_castling_quadrants_full_at_start() -> None:
    arr = PLANE.encode(chess.Board(), chess.WHITE)
    assert arr[12, 0:4, 0:4].all()  # friendly kingside
    assert arr[12, 0:4, 4:8].all()  # friendly queenside
    assert arr[12, 4:8, 0:4].all()  # enemy kingside
    assert arr[12, 4:8, 4:8].all()  # enemy queenside


def test_en_passant_plane() -> None:
    board = chess.Board()
    board.push_san("e4")  # sets ep target on e3
    arr = PLANE.encode(board, board.turn)  # black to move
    assert arr[13].sum() == 1.0  # exactly one ep square marked


# --------------------------------------------------------------------------- #
# 1.6.2 Attacker-perspective normalization                                     #
# --------------------------------------------------------------------------- #
def test_black_to_move_puts_black_in_friendly_planes() -> None:
    board = chess.Board()
    board.push_san("e4")  # now black to move; attacker = black
    arr = PLANE.encode(board, chess.BLACK)
    # Attacker (black) has one king in the friendly plane, opponent one in 11.
    assert arr[5].sum() == 1.0
    assert arr[11].sum() == 1.0
    # Friendly pawns count equals black's pawn count (8).
    assert arr[0].sum() == 8


def test_mirror_rotational_invariance() -> None:
    # A position and its colour-swapped vertical mirror must encode identically
    # once attacker-normalized.
    board = chess.Board("5rk1/5ppp/8/6P1/7P/8/8/5RK1 w - - 0 1")  # white kingside play
    white_tensor, _ = ENC.encode_window([board], chess.WHITE)
    black_tensor, _ = ENC.encode_window([board.mirror()], chess.BLACK)
    assert np.array_equal(white_tensor, black_tensor)


# --------------------------------------------------------------------------- #
# 1.6.3 Stacking, padding and validity mask                                    #
# --------------------------------------------------------------------------- #
def test_full_window_shape_and_mask() -> None:
    history = boards_from(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"])  # 8 boards
    tensor, mask = ENC.encode_window(history, chess.WHITE)
    assert tensor.shape == (112, 8, 8)
    assert mask.shape == (8,)
    assert mask.tolist() == [1.0] * 8
    assert not np.isnan(tensor).any()


def test_padding_for_short_history() -> None:
    history = boards_from(["e4", "e5"])  # 3 boards
    tensor, mask = ENC.encode_window(history, chess.WHITE)
    assert tensor.shape == (112, 8, 8)
    # Newest 3 slots valid, first 5 padded.
    assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert tensor[: 5 * 14].sum() == 0.0  # padded channels are all zero
    assert tensor[5 * 14 :].sum() > 0.0  # real channels populated


def test_newest_board_occupies_last_slot() -> None:
    history = boards_from(["e4", "e5", "Nf3"])  # 4 boards
    tensor, _ = ENC.encode_window(history, chess.WHITE)
    newest = PLANE.encode(history[-1], chess.WHITE)
    assert np.array_equal(tensor[-14:], newest)


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError):
        ENC.encode_window([], chess.WHITE)


def test_too_long_history_raises() -> None:
    with pytest.raises(ValueError):
        ENC.encode_window(
            boards_from(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"]), chess.WHITE
        )
