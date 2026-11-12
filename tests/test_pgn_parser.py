"""Unit tests for the streaming PGN parser (Subtask 1.2.2)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import pytest

from src.data.pgn_parser import StreamingPGNParser

# A legal 4-ply cycle that returns the position to the start, so we can build a
# game of any even length out of guaranteed-legal moves.
_SHUFFLE = ["Nf3", "Nf6", "Ng1", "Ng8"]


def build_pgn(num_plies: int = 24, headers: Optional[dict] = None) -> str:
    """Return a PGN string for a legal game of ``num_plies`` half-moves."""
    game = chess.pgn.Game()
    game.headers["Result"] = "1-0"
    if headers:
        game.headers.update(headers)
    board = chess.Board()
    node: chess.pgn.GameNode = game
    for i in range(num_plies):
        move = board.parse_san(_SHUFFLE[i % 4])
        node = node.add_main_variation(move)
        board.push(move)
    return str(game)


def count_games(pgn_text: str, **kwargs) -> int:
    parser = StreamingPGNParser(**kwargs)
    return sum(1 for _ in parser.iter_games(io.StringIO(pgn_text)))


# --------------------------------------------------------------------------- #
# Acceptance: only valid standard games >= 20 plies are yielded               #
# --------------------------------------------------------------------------- #
def test_valid_game_is_yielded() -> None:
    pgn = build_pgn(24, {"WhiteElo": "2500", "BlackElo": "2450"})
    assert count_games(pgn) == 1


def test_short_game_is_rejected() -> None:
    pgn = build_pgn(10, {"WhiteElo": "2500", "BlackElo": "2500"})
    assert count_games(pgn) == 0


def test_low_elo_is_rejected() -> None:
    pgn = build_pgn(24, {"WhiteElo": "2000", "BlackElo": "2500"})
    assert count_games(pgn) == 0


def test_missing_elo_tolerated_by_default() -> None:
    pgn = build_pgn(24)  # no Elo headers
    assert count_games(pgn) == 1


def test_missing_elo_rejected_when_required() -> None:
    pgn = build_pgn(24)
    assert count_games(pgn, require_elo=True) == 0


def test_non_standard_variant_is_rejected() -> None:
    pgn = build_pgn(24, {"Variant": "Atomic", "WhiteElo": "2600", "BlackElo": "2600"})
    assert count_games(pgn) == 0


def test_illegal_move_game_is_rejected() -> None:
    # 2. Qh6 is illegal from the opening; python-chess records a parse error.
    illegal = (
        '[Event "x"]\n[Result "*"]\n[WhiteElo "2600"]\n[BlackElo "2600"]\n\n1. e4 e5 2. Qh6 Nf6 *\n'
    )
    assert count_games(illegal) == 0


# --------------------------------------------------------------------------- #
# Streaming behaviour                                                          #
# --------------------------------------------------------------------------- #
def test_mixed_stream_yields_only_valid() -> None:
    valid = build_pgn(24, {"WhiteElo": "2500", "BlackElo": "2500"})
    short = build_pgn(8, {"WhiteElo": "2500", "BlackElo": "2500"})
    low = build_pgn(24, {"WhiteElo": "1500", "BlackElo": "1500"})
    # Games must be separated by a blank line per the PGN spec.
    stream = "\n\n".join([valid, short, low, valid])
    assert count_games(stream) == 2


def test_max_games_limit() -> None:
    one = build_pgn(24, {"WhiteElo": "2500", "BlackElo": "2500"})
    stream = "\n\n".join([one] * 5)
    parser = StreamingPGNParser()
    got = list(parser.iter_games(io.StringIO(stream), max_games=2))
    assert len(got) == 2


def test_yields_game_objects() -> None:
    pgn = build_pgn(24, {"WhiteElo": "2500", "BlackElo": "2500"})
    parser = StreamingPGNParser()
    games = list(parser.iter_games(io.StringIO(pgn)))
    assert all(isinstance(g, chess.pgn.Game) for g in games)
    assert len(list(games[0].mainline_moves())) == 24


# --------------------------------------------------------------------------- #
# File I/O path                                                               #
# --------------------------------------------------------------------------- #
def test_stream_games_from_file(tmp_path: Path) -> None:
    pgn = build_pgn(24, {"WhiteElo": "2500", "BlackElo": "2500"})
    path = tmp_path / "sample.pgn"
    path.write_text(pgn, encoding="utf-8")
    parser = StreamingPGNParser()
    assert sum(1 for _ in parser.stream_games(path)) == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    parser = StreamingPGNParser()
    with pytest.raises(FileNotFoundError):
        list(parser.stream_games(tmp_path / "nope.pgn"))


# --------------------------------------------------------------------------- #
# Reader-error handling                                                       #
# --------------------------------------------------------------------------- #
def test_reader_exception_stops_stream_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read_game() exception must terminate the stream, not retry forever."""

    def _always_raises(_handle: object) -> None:
        raise ValueError("simulated corrupt stream offset")

    monkeypatch.setattr(chess.pgn, "read_game", _always_raises)
    parser = StreamingPGNParser()
    assert list(parser.iter_games(io.StringIO("anything"))) == []
