"""Tests for scripts/build_scaled_dataset.py (scaled, leak-free dataset cache).

All data is generated from seeded random legal games, so the tests need neither
the real PGN subset nor the Lichess archive. Quality thresholds are disabled so
(almost) every window survives; the plumbing is what is under test.
"""

from __future__ import annotations

import csv
import io
import pickle
import random
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import chess
import chess.pgn
import numpy as np
import pytest

from scripts.build_scaled_dataset import (
    SPLITS,
    _RowView,
    build_scaled_dataset,
    integrity_report,
    iter_raw_games,
    load_scaled_dataset,
    pilot_throughput,
    verify_against_source,
)
from src.data.chess_dataset import create_dataloaders
from src.data.tensor_encoder import ChessTensorEncoder

BUILD_KWARGS = dict(
    workers=1,
    chunk_games=5,
    verify_games=0,
    min_confidence=0.0,
    max_material=100000,
    max_per_game=10,
    log=lambda _msg: None,
)
COLLIDING_ID = "White0_Black0_2025.11.01_-"


def _random_game(
    seed: int,
    plies: int = 70,
    white: str | None = None,
    black: str | None = None,
    elo: int = 2500,
) -> str:
    rng = random.Random(seed)
    board = chess.Board()
    for _attempt in range(100):
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if len(board.move_stack) == plies:
            break
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = white or f"White{seed}"
    game.headers["Black"] = black or f"Black{seed}"
    game.headers["Date"] = "2025.11.01"
    game.headers["Round"] = "-"
    game.headers["WhiteElo"] = str(elo)
    game.headers["BlackElo"] = str(elo)
    return str(game)


def _game_texts() -> List[str]:
    texts = [_random_game(i) for i in range(28)]
    # Two more games whose headers collide with game 0 (same default game id).
    texts.append(_random_game(100, white="White0", black="Black0"))
    texts.append(_random_game(101, white="White0", black="Black0"))
    # Identical movetext under different headers: must be dropped, never split-leaked.
    texts.append(_random_game(5, white="DupW", black="DupB"))
    # Invalid games the parser must drop: too short, and under-rated.
    texts.insert(3, _random_game(200, plies=10))
    texts.insert(7, _random_game(201, elo=1500))
    return texts


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def mini(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("scaled")
    texts = _game_texts()
    pgn = root / "games.pgn"
    pgn.write_text("\n\n".join(texts) + "\n\n", encoding="utf-8")
    out = root / "scaled"
    meta = build_scaled_dataset(pgn, out, max_games=1000, **BUILD_KWARGS)
    return SimpleNamespace(root=root, texts=texts, pgn=pgn, out=out, meta=meta)


def test_iter_raw_games_splits_on_event_headers():
    text = "".join(_random_game(i, plies=4) + "\n\n" for i in range(3))
    assert len(list(iter_raw_games(io.StringIO(text)))) == 3


def test_parser_filters_apply_and_all_valid_games_used(mini):
    assert mini.meta["games_scanned"] == 33
    assert mini.meta["games_valid"] == 30  # short + under-rated games dropped
    assert mini.meta["duplicate_games_dropped"] == 1
    games = _read_csv(mini.out / "games.csv")
    assert len(games) == 30
    assert "DupW_DupB_2025.11.01_-" not in {g["base_game_id"] for g in games}
    assert len({g["move_hash"] for g in games}) == 30


def test_splits_are_game_level_disjoint_and_nonempty(mini):
    report = integrity_report(mini.out)
    assert all(report["checks"].values()), report["checks"]
    assert sum(report["game_id_overlap"].values()) == 0
    assert sum(report["raw_game_id_overlap"].values()) == 0
    assert all(report["windows_per_split"][s] > 0 for s in SPLITS)


def test_colliding_game_ids_are_made_unique_and_share_a_split(mini):
    assert mini.meta["game_id_collisions"] == 2
    games = [g for g in _read_csv(mini.out / "games.csv") if g["base_game_id"] == COLLIDING_ID]
    assert sorted(g["game_id"] for g in games) == [
        COLLIDING_ID,
        f"{COLLIDING_ID}~2",
        f"{COLLIDING_ID}~3",
    ]
    assert len({g["split"] for g in games}) == 1


def test_window_ids_are_unique(mini):
    ids = [r["window_id"] for s in SPLITS for r in _read_csv(mini.out / f"{s}_records.csv")]
    assert len(ids) == len(set(ids)) > 0


def test_array_shapes_dtypes_and_binary_values(mini):
    for split in SPLITS:
        n = len(_read_csv(mini.out / f"{split}_records.csv"))
        tensors = np.load(mini.out / f"{split}_tensors.npy", mmap_mode="r")
        masks = np.load(mini.out / f"{split}_masks.npy")
        labels = np.load(mini.out / f"{split}_labels.npy")
        assert tensors.shape == (n, 112, 8, 8) and tensors.dtype == np.uint8
        assert masks.shape == (n, 8) and masks.dtype == np.uint8 and masks.all()
        assert labels.shape == (n,) and labels.dtype == np.int64
        assert set(np.unique(tensors)) <= {0, 1}
        assert labels.min() >= 0 and labels.max() <= 5


def test_cached_tensors_equal_dynamic_encoding_and_labels(mini):
    result = verify_against_source(mini.out, mini.pgn, n_games=1000)
    total = sum(mini.meta["windows_per_split"].values())
    assert result["windows_compared"] == total > 0
    assert result["mismatches"] == 0
    assert result["label_mismatches"] == 0


def test_input_tensor_is_reproducible_from_moves_up_to_ply_t_only(mini):
    rows = [r for r in _read_csv(mini.out / "train_records.csv") if "~" not in r["game_id"]]
    row = rows[len(rows) // 2]
    white = row["game_id"].split("_")[0]
    text = next(t for t in mini.texts if f'[White "{white}"]' in t)
    game = chess.pgn.read_game(io.StringIO(text))
    assert f"{game.headers['White']}_{game.headers['Black']}_2025.11.01_-" == row["game_id"]
    ply = int(row["ply_index"])
    board = game.board()
    history = [board.copy(stack=False)]
    for move in list(game.mainline_moves())[:ply]:  # nothing beyond ply t
        board.push(move)
        history.append(board.copy(stack=False))
    tensor, _mask = ChessTensorEncoder().encode_window(history[-8:], board.turn)
    index = [r["window_id"] for r in _read_csv(mini.out / "train_records.csv")].index(
        row["window_id"]
    )
    cached = np.load(mini.out / "train_tensors.npy", mmap_mode="r")[index]
    assert np.array_equal(cached.astype(np.float32), tensor)


def test_integrity_report_has_duplicate_and_distribution_sections(mini):
    report = integrity_report(mini.out)
    assert report["duplicate_positions"]["windows_total"] == mini.meta["windows"]
    assert set(report["class_distribution"]) == set(SPLITS)
    assert report["last_slot_check"]["windows_checked"] > 0
    assert report["last_slot_check"]["mismatches"] == 0


def test_balancing_touches_train_only(mini):
    raw = load_scaled_dataset(mini.out, balance_train=False)
    bal = load_scaled_dataset(mini.out, balance_train=True, seed=1, max_class_fraction=0.2)
    again = load_scaled_dataset(mini.out, balance_train=True, seed=1, max_class_fraction=0.2)

    for split in ("val", "test"):
        assert len(bal[split]) == len(raw[split]) > 0
        assert bal[split].labels == raw[split].labels
    assert len(raw["train"]) == len(_read_csv(mini.out / "train_records.csv"))
    assert len(bal["train"]) < len(raw["train"])
    assert bal["train"].labels == again["train"].labels  # deterministic

    raw_counts, bal_counts = Counter(raw["train"].labels), Counter(bal["train"].labels)
    assert all(bal_counts[c] <= raw_counts[c] for c in bal_counts)
    cap = max(0.2, 1.0 / len(raw_counts))
    assert max(bal_counts.values()) / len(bal["train"]) <= cap + 0.02


def test_loader_batches_have_expected_shapes_and_dtypes(mini):
    data = load_scaled_dataset(mini.out, balance_train=False)
    loader, _, _ = create_dataloaders(data["train"], data["val"], data["test"], batch_size=8)
    tensors, masks, labels = next(iter(loader))
    assert tensors.shape == (8, 112, 8, 8) and str(tensors.dtype) == "torch.float32"
    assert masks.shape == (8, 8) and str(masks.dtype) == "torch.float32"
    assert labels.shape == (8,) and str(labels.dtype) == "torch.int64"


def test_row_view_is_lazy_and_picklable(mini):
    view = _RowView(mini.out / "val_tensors.npy", np.array([1, 0]))
    first = np.array(view[0])
    clone = pickle.loads(pickle.dumps(view))
    assert clone._mm is None and len(clone) == 2
    assert np.array_equal(clone[0], first)
    assert np.array_equal(first, np.load(mini.out / "val_tensors.npy", mmap_mode="r")[1])


def test_zip_source_with_worker_pool_matches_serial_build(mini, tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("games.pgn", "\n\n".join(mini.texts) + "\n\n")
    out = tmp_path / "parallel"
    kwargs = dict(BUILD_KWARGS, workers=2)
    build_scaled_dataset(archive, out, max_games=1000, **kwargs)
    for split in SPLITS:
        for name in ("tensors", "masks", "labels"):
            a = np.load(mini.out / f"{split}_{name}.npy")
            b = np.load(out / f"{split}_{name}.npy")
            assert np.array_equal(a, b), (split, name)
        assert (mini.out / f"{split}_records.csv").read_bytes() == (
            out / f"{split}_records.csv"
        ).read_bytes()
    assert (mini.out / "games.csv").read_bytes() == (out / "games.csv").read_bytes()


def test_max_games_limits_valid_games(mini, tmp_path):
    out = tmp_path / "five"
    meta = build_scaled_dataset(mini.pgn, out, max_games=5, **BUILD_KWARGS)
    assert meta["games_valid"] == 5
    assert len(_read_csv(out / "games.csv")) == 5


def test_build_is_resume_safe(mini, tmp_path):
    with pytest.raises(FileExistsError):
        build_scaled_dataset(mini.pgn, mini.out, max_games=3, **BUILD_KWARGS)

    out = tmp_path / "scaled"
    with pytest.raises(FileNotFoundError):
        build_scaled_dataset(tmp_path / "missing.pgn", out, max_games=3, **BUILD_KWARGS)
    assert not out.exists()  # a failed build never publishes a final directory

    stale = tmp_path / "scaled.tmp"
    stale.mkdir(exist_ok=True)  # the failed build above may already have left one
    (stale / "junk.txt").write_text("stale", encoding="utf-8")
    build_scaled_dataset(mini.pgn, out, max_games=3, **BUILD_KWARGS)
    assert out.is_dir() and not stale.exists()
    assert (out / ".gitignore").read_text(encoding="utf-8").strip() == "*"


def test_max_games_must_be_positive(mini, tmp_path):
    with pytest.raises(ValueError):
        build_scaled_dataset(mini.pgn, tmp_path / "x", max_games=0, **BUILD_KWARGS)


def test_pilot_reports_throughput_and_projection(mini):
    report = pilot_throughput(mini.pgn, n_games=3, log=lambda _msg: None)
    assert report["games"] == 3
    assert report["sec_per_game_single_process"] > 0
    assert report["projected_games"] > 0


def test_restricted_splits_never_open_test_files(mini, monkeypatch):
    opened: List[str] = []
    real_load = np.load

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    data = load_scaled_dataset(mini.out, balance_train=False, splits=("train", "val"))
    for batch in (data["train"][0], data["val"][0]):  # touches the memmapped tensors too
        assert batch[0].shape == (112, 8, 8)
    assert set(data) == {"train", "val"}
    assert not [p for p in opened if "test_" in Path(p).name]
