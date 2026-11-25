"""Tests for the fresh holdout builder on a tiny synthetic PGN archive."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List

import chess
import chess.pgn
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_fresh_holdout as fh  # noqa: E402
import build_scaled_dataset as bsd  # noqa: E402

SKIP = 3


def _game(seed: int, site: str, plies: int = 30) -> chess.pgn.Game:
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update(Site=site, WhiteElo="2500", BlackElo="2500", Result="*")
    node = game
    for _ in range(plies):
        quiet = [m for m in board.legal_moves if not board.is_capture(m)]
        move = rng.choice(sorted(quiet or list(board.legal_moves), key=str))
        board.push(move)
        node = node.add_variation(move)
        if board.is_game_over():
            break
    return game


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("holdout")
    # Raw order: 3 games "scanned by the 80k build", then: a fresh game, a copy of
    # an 80k movetext, fresh, a duplicate of the first fresh game, and two fresh.
    seeds = [1, 2, 3, 10, 2, 11, 10, 12, 13]
    games = [_game(s, f"g{i}") for i, s in enumerate(seeds)]
    source = root / "mini.pgn"
    source.write_text("\n\n".join(str(g) for g in games) + "\n\n", encoding="utf-8")

    exclude = root / "scaled_80k"
    exclude.mkdir()
    flt = bsd._default_cfg()
    (exclude / "meta.json").write_text(
        json.dumps({"source": str(source), "games_scanned": SKIP, "filter": flt}), encoding="utf-8"
    )
    hashes: List[str] = [bsd._move_hash(g) for g in games[:SKIP]]
    (exclude / "games.csv").write_text("move_hash\n" + "\n".join(hashes) + "\n", encoding="utf-8")

    outs = (root / "holdout_filtered", root / "holdout_full")
    metas = fh.build_fresh_holdout(
        exclude, *outs, source=source, max_games=3, workers=1, log=lambda _: None
    )
    return root, source, exclude, outs, metas, hashes, games


def test_only_unseen_unique_games_after_the_offset_are_used(setup):
    *_, outs, metas, old_hashes, games = setup
    rows = bsd._read_csv(outs[1] / "games.csv")
    assert [r["move_hash"] for r in rows] == [bsd._move_hash(games[i]) for i in (3, 5, 7)]
    assert not {r["move_hash"] for r in rows} & set(old_hashes)
    meta = metas["full"]
    assert meta["raw_games_skipped"] == SKIP
    assert meta["dropped_movetext_in_80k"] == 1
    assert meta["dropped_duplicates_within_holdout"] == 1
    assert meta["games_valid"] == 3


def test_all_integrity_checks_pass(setup):
    metas = setup[4]
    for meta in metas.values():
        assert meta["integrity"]["checks"] and all(meta["integrity"]["checks"].values())


def test_full_has_every_window_and_filtered_is_a_subset(setup):
    outs = setup[3]
    filtered = bsd._read_csv(outs[0] / "test_records.csv")
    full = bsd._read_csv(outs[1] / "test_records.csv")
    games = bsd._read_csv(outs[1] / "games.csv")
    assert len(full) == sum(int(g["n_plies"]) - 11 for g in games)
    assert {r["window_id"] for r in filtered} <= {r["window_id"] for r in full}
    assert 0 < len(filtered) <= len(full)


def test_loader_reads_both_dirs_with_test_split_only(setup):
    for out in setup[3]:
        data = bsd.load_scaled_dataset(out, balance_train=False, splits=("test",))
        tensor, mask, label = data["test"][0]
        assert tensor.shape == (112, 8, 8) and mask.shape == (8,) and 0 <= label < 6


def test_never_overwrites_and_rejects_a_different_archive(setup):
    root, source, exclude, outs = setup[:4]
    with pytest.raises(FileExistsError):
        fh.build_fresh_holdout(exclude, *outs, source=source, max_games=1, workers=1)
    other = root / "other.pgn"
    other.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="was built from"):
        fh.build_fresh_holdout(exclude, root / "a", root / "b", source=other, workers=1)


def test_second_holdout_skips_and_excludes_the_used_one(setup):
    root, source, exclude, outs, metas, _old, _games = setup
    used = {r["move_hash"] for r in bsd._read_csv(outs[1] / "games.csv")}
    second = fh.build_fresh_holdout(
        exclude,
        root / "second_filtered",
        root / "second_full",
        source=source,
        max_games=5,
        workers=1,
        log=lambda _: None,
        after_holdout=outs[1],
    )
    again = {r["move_hash"] for r in bsd._read_csv(root / "second_full" / "games.csv")}
    assert again and not again & used
    expected_skip = SKIP + metas["full"]["raw_games_scanned_after_skip"]
    assert second["full"]["raw_games_skipped"] == expected_skip
