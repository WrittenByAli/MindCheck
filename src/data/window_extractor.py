"""8-ply sliding history window extraction (Subtask 1.2.3).

Given a validated :class:`chess.pgn.Game`, :func:`extract_windows` walks the
mainline ply-by-ply and, at each position, emits a :class:`PositionWindow`
holding the most recent ``history_length`` board snapshots. These windows are
the unit of input later encoded into ``(112, 8, 8)`` tensors.

The window at ply ``t`` contains the boards ``[B_{t-7}, ..., B_t]`` (up to
``history_length`` of them). For early plies fewer than ``history_length``
boards exist; the window is simply shorter, and the tensor encoder
(Subtask 1.6.3) is responsible for zero-padding and the validity mask. The
final board in ``board_history`` is always the position *after* the move at
``ply_index``.

Owner: Shahzad Ahmad (24I-3090). Milestone 2.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import chess
import chess.pgn

__all__ = ["PositionWindow", "extract_windows", "default_game_id"]

# Opening plies are dominated by book theory; strategic plans form later.
DEFAULT_START_PLY = 12
DEFAULT_HISTORY_LENGTH = 8


@dataclass
class PositionWindow:
    """A single training/inference sample: recent board history at one ply.

    Attributes:
        game_id: Stable identifier of the source game.
        ply_index: Number of half-moves played to reach the current board.
        turn: Side to move in the current position (``chess.WHITE``/``BLACK``).
        board_history: Up to ``history_length`` independent board clones,
            ordered oldest -> newest; the last element is the current board.
        last_move: The move that produced the current board (``None`` only if
            the window starts at ply 0).
        last_move_san: SAN of ``last_move`` for display/logging.
        is_check: Whether the side to move is in check in the current position.
        white_elo: White player rating, if present in the PGN headers.
        black_elo: Black player rating, if present.
        result: Game result string (e.g. ``"1-0"``).
    """

    game_id: str
    ply_index: int
    turn: chess.Color
    board_history: List[chess.Board]
    last_move: Optional[chess.Move]
    last_move_san: Optional[str] = None
    is_check: bool = False
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: str = "*"
    metadata: dict = field(default_factory=dict)

    @property
    def current_board(self) -> chess.Board:
        """The most recent board in the history (position at ``ply_index``)."""
        return self.board_history[-1]


def _parse_elo(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def default_game_id(game: chess.pgn.Game) -> str:
    """Derive a stable game id from PGN headers."""
    headers = game.headers
    site = headers.get("Site")
    if site and site not in {"?", ""}:
        return site
    parts = [
        headers.get("White", "?"),
        headers.get("Black", "?"),
        headers.get("Date", "?"),
        headers.get("Round", "?"),
    ]
    joined = "_".join(part.replace(" ", "") for part in parts)
    return joined or "game"


def extract_windows(
    game: chess.pgn.Game,
    game_id: Optional[str] = None,
    history_length: int = DEFAULT_HISTORY_LENGTH,
    start_ply: int = DEFAULT_START_PLY,
) -> Iterator[PositionWindow]:
    """Yield sliding history windows across a game's mainline.

    Args:
        game: A validated game (see :class:`StreamingPGNParser`).
        game_id: Explicit id; derived from headers when ``None``.
        history_length: Number of board snapshots per window (default 8).
        start_ply: First ply (half-move index) to emit a window for. Defaults
            to 12 to skip opening book moves.

    Yields:
        :class:`PositionWindow` objects, one per ply from ``start_ply`` onward.
    """
    if history_length < 1:
        raise ValueError(f"history_length must be >= 1, got {history_length}.")
    if start_ply < 0:
        raise ValueError(f"start_ply must be >= 0, got {start_ply}.")

    gid = game_id if game_id is not None else default_game_id(game)
    white_elo = _parse_elo(game.headers.get("WhiteElo"))
    black_elo = _parse_elo(game.headers.get("BlackElo"))
    result = game.headers.get("Result", "*")

    board = game.board()
    # Rolling buffer of the most recent `history_length` board snapshots.
    history: deque[chess.Board] = deque([board.copy()], maxlen=history_length)

    ply = 0
    for move in game.mainline_moves():
        # SAN must be computed before the move is pushed.
        san = board.san(move)
        board.push(move)
        ply += 1
        history.append(board.copy())

        if ply >= start_ply:
            yield PositionWindow(
                game_id=gid,
                ply_index=ply,
                turn=board.turn,
                board_history=list(history),
                last_move=move,
                last_move_san=san,
                is_check=board.is_check(),
                white_elo=white_elo,
                black_elo=black_elo,
                result=result,
            )
