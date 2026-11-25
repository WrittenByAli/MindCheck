"""Tests for scripts/build_allpos_dataset.py on seeded random games (no real data)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_allpos_dataset as bad  # noqa: E402
import build_scaled_dataset as bsd  # noqa: E402

from tests.test_scaled_dataset import BUILD_KWARGS, _game_texts  # noqa: E402

NO_BENCH = (set(), set())


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("allpos")
    pgn = root / "games.pgn"
    pgn.write_text("\n\n".join(_game_texts()) + "\n\n", encoding="utf-8")
    ref = root / "ref"
    bsd.build_scaled_dataset(pgn, ref, max_games=1000, **BUILD_KWARGS)
    games = bsd._read_csv(ref / "games.csv")
    return SimpleNamespace(root=root, pgn=pgn, ref=ref, games=games)


def _build(reference, split, out, bench=NO_BENCH, per_game=7):
    return bad.build_split(
        split,
        reference.ref,
        out,
        reference.pgn,
        workers=1,
        per_game=per_game,
        bench=bench,
        log=lambda _m: None,
    )


def test_select_plies_uniform_size_and_chain_enrichment():
    labels = {p: (4 if p in (20, 30, 40) else 5) for p in range(12, 60)}
    picked = bad.select_plies(labels, 5, True, random.Random(1))
    kinds = dict(picked)
    assert sum(1 for k in kinds.values() if k == "uniform") == 5
    assert all(labels[p] == 4 for p, k in kinds.items() if k == "enriched")
    assert {p for p in (20, 30, 40)} <= set(kinds)  # every CHAIN ply is kept
    assert len(picked) == len(kinds)  # no ply twice
    no_extra = bad.select_plies(labels, 5, False, random.Random(1))
    assert len(no_extra) == 5 and all(k == "uniform" for _, k in no_extra)


def test_select_plies_short_game_takes_everything():
    assert len(bad.select_plies({12: 5, 13: 2}, 30, False, random.Random(0))) == 2


def test_splits_match_the_reference_and_never_include_test(reference, tmp_path):
    for split in ("train", "val"):
        meta = _build(reference, split, tmp_path)
        assert all(meta["checks"].values()), meta["checks"]
        ids = {r["game_id"] for r in bsd._read_csv(tmp_path / f"{split}_records.csv")}
        assert ids == {g["game_id"] for g in reference.games if g["split"] == split}
        assert meta["games_written"] == meta["games_expected"]
    test_ids = {g["game_id"] for g in reference.games if g["split"] == "test"}
    for split in ("train", "val"):
        ids = {r["game_id"] for r in bsd._read_csv(tmp_path / f"{split}_records.csv")}
        assert not ids & test_ids
    assert not list(tmp_path.glob("test_*"))
    with pytest.raises(ValueError):
        _build(reference, "test", tmp_path / "x")


def test_no_confidence_filter_and_val_is_uniform_only(reference, tmp_path):
    meta = _build(reference, "val", tmp_path, per_game=1000)
    assert set(meta["windows_by_kind"]) == {"uniform"}
    assert meta["windows"] == meta["positions_available"]  # every position kept
    loaded = bsd.load_scaled_dataset(tmp_path, balance_train=False, splits=("val",))
    assert len(loaded["val"]) == meta["windows"]


def test_benchmark_games_and_inputs_are_removed(reference, tmp_path):
    _build(reference, "train", tmp_path / "clean")
    rows = bsd._read_csv(tmp_path / "clean" / "train_records.csv")
    tensors = np.load(tmp_path / "clean" / "train_tensors.npy")
    train_games = [g for g in reference.games if g["split"] == "train"]
    banned_game = train_games[0]
    other = next(i for i, r in enumerate(rows) if r["game_id"] != banned_game["game_id"])
    bench = ({banned_game["move_hash"]}, {bad.input_key(tensors[other])})

    meta = _build(reference, "train", tmp_path / "excl", bench=bench)

    kept = bsd._read_csv(tmp_path / "excl" / "train_records.csv")
    assert banned_game["game_id"] not in {r["game_id"] for r in kept}
    assert rows[other]["window_id"] not in {r["window_id"] for r in kept}
    assert meta["benchmark_games_dropped"] == 1
    assert meta["benchmark_input_windows_dropped"] >= 1
    assert meta["checks"]["no_annotation_game"] and meta["checks"]["no_annotation_model_input"]


def test_deterministic_and_refuses_overwrite(reference, tmp_path):
    _build(reference, "val", tmp_path / "a")
    _build(reference, "val", tmp_path / "b")
    a = bsd._read_csv(tmp_path / "a" / "val_records.csv")
    b = bsd._read_csv(tmp_path / "b" / "val_records.csv")
    assert a == b
    with pytest.raises(FileExistsError):
        _build(reference, "val", tmp_path / "a")


def test_loader_can_train_on_uniform_rows_only(reference, tmp_path):
    _build(reference, "train", tmp_path, per_game=3)
    rows = bsd._read_csv(tmp_path / "train_records.csv")
    n_uniform = sum(r["sample_kind"] == "uniform" for r in rows)
    everything = bsd.load_scaled_dataset(tmp_path, balance_train=False, splits=("train",))
    uniform = bsd.load_scaled_dataset(
        tmp_path, balance_train=False, splits=("train",), train_sample_kinds=("uniform",)
    )
    assert len(everything["train"]) == len(rows)
    assert len(uniform["train"]) == n_uniform
    expected = [int(r["label_id"]) for r in rows if r["sample_kind"] == "uniform"]
    assert uniform["train"].labels == expected
