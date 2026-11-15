"""Spatial-temporal tensor encoding for MindCheck (Task 1.6, Milestone 5).

Turns an 8-ply history window into the canonical ``(112, 8, 8)`` float32 tensor
the residual CNN consumes, plus a ``(8,)`` validity mask for zero-padded early
plies.

Plane layout per ply (14 planes):
  * 0-5   friendly P, N, B, R, Q, K
  * 6-11  opponent p, n, b, r, q, k
  * 12    castling rights, 4 bits replicated over 8x8 quadrants
          (top-left friendly-K, top-right friendly-Q, bottom-left enemy-K,
           bottom-right enemy-Q)
  * 13    en-passant target square (1 on the target, else 0)

Attacker-perspective normalization: the network must always see the side to
move advancing toward row 0 (rank 8). When the attacker is Black we encode
``board.mirror()`` — python-chess mirrors vertically *and* swaps colors,
castling rights and the en-passant square, which is exactly the spec's
"flip rows + swap friendly/opponent planes" done correctly for every plane.

Rows map rank 8 -> row 0 (``row = 7 - rank``), files map a -> col 0.

Owner: Shahzad Ahmad (24I-3090). Subtasks 1.6.1, 1.6.2, 1.6.3.
"""

from __future__ import annotations

from typing import List, Tuple

import chess
import numpy as np

__all__ = ["BoardPlaneEncoder", "ChessTensorEncoder"]

PLANES_PER_PLY = 14
HISTORY_LENGTH = 8
BOARD_SIZE = 8
TOTAL_CHANNELS = PLANES_PER_PLY * HISTORY_LENGTH

# Friendly piece planes are indices 0-5 in this order; opponent planes 6-11.
_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_CASTLING_PLANE = 12
_EP_PLANE = 13


def _rc(square: chess.Square) -> Tuple[int, int]:
    """Map a square to (row, col) with rank 8 at row 0."""
    return 7 - chess.square_rank(square), chess.square_file(square)


class BoardPlaneEncoder:
    """Encode one board into 14 binary ``8x8`` planes from the attacker's view."""

    def encode(self, board: chess.Board, attacker: chess.Color) -> np.ndarray:
        """Return a ``(14, 8, 8)`` float32 array oriented for ``attacker``."""
        # Canonicalize so the attacker is always "White" advancing toward row 0.
        view = board if attacker == chess.WHITE else board.mirror()
        arr = np.zeros((PLANES_PER_PLY, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        for plane, piece_type in enumerate(_PIECE_ORDER):
            for sq in view.pieces(piece_type, chess.WHITE):  # friendly
                row, col = _rc(sq)
                arr[plane, row, col] = 1.0
            for sq in view.pieces(piece_type, chess.BLACK):  # opponent
                row, col = _rc(sq)
                arr[6 + plane, row, col] = 1.0

        # Castling rights: each of the 4 bits fills one 4x4 quadrant.
        if view.has_kingside_castling_rights(chess.WHITE):
            arr[_CASTLING_PLANE, 0:4, 0:4] = 1.0
        if view.has_queenside_castling_rights(chess.WHITE):
            arr[_CASTLING_PLANE, 0:4, 4:8] = 1.0
        if view.has_kingside_castling_rights(chess.BLACK):
            arr[_CASTLING_PLANE, 4:8, 0:4] = 1.0
        if view.has_queenside_castling_rights(chess.BLACK):
            arr[_CASTLING_PLANE, 4:8, 4:8] = 1.0

        # En-passant target square.
        if view.ep_square is not None:
            row, col = _rc(view.ep_square)
            arr[_EP_PLANE, row, col] = 1.0

        return arr


class ChessTensorEncoder:
    """Stack an 8-ply history into a ``(112, 8, 8)`` tensor with a validity mask."""

    def __init__(
        self, history_length: int = HISTORY_LENGTH, planes_per_ply: int = PLANES_PER_PLY
    ) -> None:
        self.history_length = history_length
        self.planes = planes_per_ply
        self.total_channels = history_length * planes_per_ply
        self.board_encoder = BoardPlaneEncoder()

    def encode_window(
        self, history: List[chess.Board], attacker: chess.Color
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode up to ``history_length`` boards (oldest->newest).

        Returns:
            tensor: ``(112, 8, 8)`` float32. The newest board always occupies
                the last 14-channel slot; missing early plies are zero-padded
                at the front.
            mask: ``(8,)`` float32 — 1.0 for a real ply, 0.0 for padding.
        """
        length = len(history)
        if length == 0:
            raise ValueError("history must contain at least one board.")
        if length > self.history_length:
            raise ValueError(f"history longer than {self.history_length}: {length}.")

        tensor = np.zeros((self.total_channels, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        mask = np.zeros(self.history_length, dtype=np.float32)

        offset = self.history_length - length  # pad the front, keep newest last
        for i, board in enumerate(history):
            slot = offset + i
            planes = self.board_encoder.encode(board, attacker)
            tensor[slot * self.planes : (slot + 1) * self.planes] = planes
            mask[slot] = 1.0

        return tensor, mask

    def encode_position_window(self, window) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a :class:`~src.data.window_extractor.PositionWindow`.

        The attacker is the side to move in the current position
        (``window.turn``), per the spatial-temporal spec.
        """
        return self.encode_window(window.board_history, window.turn)
