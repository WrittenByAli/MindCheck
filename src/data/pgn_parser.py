"""Memory-safe streaming PGN parser with sanity filtering (Subtask 1.2.2).

The :class:`StreamingPGNParser` reads games one at a time with
``chess.pgn.read_game`` so that arbitrarily large archives never need to be
held in memory. Each game is passed through a set of sanity filters (variant,
ply count, player rating, move legality) and only clean standard-chess games
are yielded.

Supported inputs: plain ``.pgn`` as well as ``.pgn.zst`` (zstandard),
``.bz2`` and ``.gz`` compressed archives, streamed and decompressed on the fly.

Owner: Shahzad Ahmad (24I-3090). Milestone 2.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import logging
from pathlib import Path
from typing import Iterator, Optional, TextIO

import chess
import chess.pgn

logger = logging.getLogger(__name__)

__all__ = ["StreamingPGNParser"]

# Variant header values that denote ordinary standard chess.
_STANDARD_VARIANTS = {"", "standard", "chess", "from position"}


class StreamingPGNParser:
    """Stream and validate games from a PGN archive without loading it fully.

    Args:
        min_plies: Minimum number of half-moves for a game to be kept. Games
            shorter than this are too brief for strategic plan formation.
        min_elo: Minimum player rating. When an Elo header is present and below
            this value the game is rejected.
        require_elo: If ``True``, games missing a White/Black Elo header are
            rejected. If ``False`` (default), missing ratings are tolerated.
    """

    def __init__(
        self,
        min_plies: int = 20,
        min_elo: int = 2200,
        require_elo: bool = False,
    ) -> None:
        self.min_plies = min_plies
        self.min_elo = min_elo
        self.require_elo = require_elo

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def stream_games(
        self, pgn_path: str | Path, max_games: Optional[int] = None
    ) -> Iterator[chess.pgn.Game]:
        """Yield valid games from a PGN file path.

        Args:
            pgn_path: Path to a ``.pgn`` / ``.pgn.zst`` / ``.bz2`` / ``.gz`` file.
            max_games: If set, stop after yielding this many valid games.

        Yields:
            Validated :class:`chess.pgn.Game` objects, one at a time.
        """
        with self._open_text(pgn_path) as handle:
            yield from self.iter_games(handle, max_games=max_games)

    def iter_games(
        self, handle: TextIO, max_games: Optional[int] = None
    ) -> Iterator[chess.pgn.Game]:
        """Yield valid games from an already-open text stream.

        This separation lets tests feed an in-memory ``io.StringIO`` without
        touching the filesystem.
        """
        yielded = 0
        while True:
            try:
                game = chess.pgn.read_game(handle)
            except (ValueError, RuntimeError) as exc:  # pragma: no cover - rare
                logger.debug("Skipping game: reader error: %s", exc)
                continue
            if game is None:
                break
            if self.is_valid_game(game):
                yield game
                yielded += 1
                if max_games is not None and yielded >= max_games:
                    break

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    def is_valid_game(self, game: chess.pgn.Game) -> bool:
        """Return ``True`` if the game passes every sanity filter."""
        game_id = game.headers.get("Site", game.headers.get("Event", "<unknown>"))

        if game.errors:
            logger.debug("Skipping %s: parser reported %d error(s).", game_id, len(game.errors))
            return False

        if not self._is_standard_variant(game):
            logger.debug("Skipping %s: non-standard variant.", game_id)
            return False

        if not self._passes_elo(game):
            logger.debug("Skipping %s: rating below %d or missing.", game_id, self.min_elo)
            return False

        ply_count, legal = self._count_and_check_moves(game)
        if not legal:
            logger.debug("Skipping %s: illegal move encountered.", game_id)
            return False
        if ply_count < self.min_plies:
            logger.debug("Skipping %s: only %d plies (< %d).", game_id, ply_count, self.min_plies)
            return False

        return True

    def _is_standard_variant(self, game: chess.pgn.Game) -> bool:
        variant = game.headers.get("Variant", "Standard").strip().lower()
        return variant in _STANDARD_VARIANTS

    def _passes_elo(self, game: chess.pgn.Game) -> bool:
        white = self._parse_elo(game.headers.get("WhiteElo"))
        black = self._parse_elo(game.headers.get("BlackElo"))

        if self.require_elo and (white is None or black is None):
            return False
        if white is not None and white < self.min_elo:
            return False
        if black is not None and black < self.min_elo:
            return False
        return True

    @staticmethod
    def _parse_elo(raw: Optional[str]) -> Optional[int]:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _count_and_check_moves(self, game: chess.pgn.Game) -> tuple[int, bool]:
        """Replay the mainline, counting plies and verifying every move is legal."""
        try:
            board = game.board()
        except (ValueError, KeyError):  # pragma: no cover - malformed SetUp/FEN
            return 0, False

        ply = 0
        for move in game.mainline_moves():
            if not board.is_legal(move):
                return ply, False
            board.push(move)
            ply += 1
        return ply, True

    # ------------------------------------------------------------------ #
    # I/O helpers                                                        #
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def _open_text(self, pgn_path: str | Path) -> Iterator[TextIO]:
        """Open a (possibly compressed) PGN file as a UTF-8 text stream."""
        path = Path(pgn_path)
        if not path.is_file():
            raise FileNotFoundError(f"PGN file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".zst":
            try:
                import zstandard
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Reading .zst archives requires the 'zstandard' package. "
                    "Install it with `pip install zstandard`, or decompress the "
                    "file to a plain .pgn first."
                ) from exc
            import io

            raw = path.open("rb")
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            try:
                yield text
            finally:
                text.close()
                reader.close()
                raw.close()
        elif suffix == ".bz2":
            with bz2.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield handle
        elif suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield handle
        else:
            with path.open("rt", encoding="utf-8", errors="replace") as handle:
                yield handle
