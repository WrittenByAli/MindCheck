"""Scaled, leak-free weak-labeled dataset builder (anti-overfitting: more games).

Streams games straight out of the Lichess Elite ``.zip`` (never extracted, never
held in RAM), weak-labels and quality-filters each game in a worker process,
encodes the kept windows into ``(112, 8, 8)`` tensors with the real
``extract_windows`` + ``ChessTensorEncoder`` pipeline, and writes a compact
on-disk cache split at the *game* level (hash of the raw game id, identical to
``src.data.dataset_splitter.GameLevelSplitter`` so the original 3000-game
subset keeps the same split assignment).

Class balancing is NOT applied to the cache. :func:`load_scaled_dataset`
balances the TRAIN split only, at load time; val/test keep natural frequencies.

Output (per split ``train``/``val``/``test``)::

    {split}_tensors.npy   (N,112,8,8) uint8   binary planes; cast to float32 on read
    {split}_masks.npy     (N,8)       uint8   history validity mask
    {split}_labels.npy    (N,)        int64   weak plan label ids
    {split}_records.csv   one row per tensor row (same order)
    games.csv             one row per game (unique id, raw id, split, move hash)
    meta.json             arguments, counts, timings, integrity report

CLI::

    python scripts/build_scaled_dataset.py --pilot
    python scripts/build_scaled_dataset.py --max-games 15000 --workers 8

The build writes to ``<out>.tmp`` and renames to ``<out>`` only after every
integrity check passed, so a crashed run never leaves a half-built dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import shutil
import sys
import time
import zipfile
from collections import Counter, deque
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
import numpy as np  # noqa: E402

from src.data.dataset_splitter import GameLevelSplitter  # noqa: E402
from src.data.pgn_parser import StreamingPGNParser  # noqa: E402
from src.data.tensor_encoder import BoardPlaneEncoder, ChessTensorEncoder  # noqa: E402
from src.data.weak_labeler import (  # noqa: E402
    PLAN_IDS,
    LabeledPositionRecord,
    RetrospectiveWeakLabeler,
    balance_classes,
    default_game_id,
    filter_and_balance,
)
from src.data.window_extractor import extract_windows  # noqa: E402

__all__ = [
    "DEFAULT_ARCHIVE",
    "DEFAULT_OUT",
    "SPLITS",
    "build_scaled_dataset",
    "load_scaled_dataset",
    "integrity_report",
    "verify_against_source",
    "pilot_throughput",
    "open_pgn_text",
]

DEFAULT_ARCHIVE = _REPO_ROOT / "data" / "raw" / "lichess_elite_2025-11.zip"
DEFAULT_OUT = _REPO_ROOT / "data" / "processed" / "scaled"
SPLITS = ("train", "val", "test")
MAX_WORKERS = 14
CHANNELS = 112
NUM_CLASSES = 6
_ID_TO_NAME = {v: k for k, v in PLAN_IDS.items()}

RECORD_FIELDS = [
    "window_id",
    "game_id",
    "base_game_id",
    "ply_index",
    "turn",
    "label_class",
    "label_id",
    "weak_confidence",
    "material_balance",
    "split",
    "fen",
]
GAME_FIELDS = ["game_id", "base_game_id", "split", "move_hash", "n_plies", "n_raw", "n_kept"]


# --------------------------------------------------------------------------- #
# Source streaming                                                             #
# --------------------------------------------------------------------------- #
@contextmanager
def open_pgn_text(source: str | Path) -> Iterator[Any]:
    """Open a ``.zip`` (first ``.pgn`` member) or plain PGN as a UTF-8 text stream."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"PGN source not found: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".pgn")]
            if not members:
                raise ValueError(f"No .pgn member found inside {path}.")
            with archive.open(members[0]) as raw:
                yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    else:
        with path.open("rt", encoding="utf-8", errors="replace") as handle:
            yield handle


def iter_raw_games(handle: Any) -> Iterator[str]:
    """Yield the raw PGN text of each game, split on ``[Event `` header lines."""
    buf: List[str] = []
    for line in handle:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = []
        buf.append(line)
    if buf and any(line.strip() for line in buf):
        yield "".join(buf)


def _chunked(items: Iterator[str], size: int) -> Iterator[List[str]]:
    chunk: List[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
# Worker side                                                                  #
# --------------------------------------------------------------------------- #
_STATE: Dict[str, Any] = {}


def _make_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cfg": cfg,
        "parser": StreamingPGNParser(),
        "labeler": RetrospectiveWeakLabeler(
            start_ply=cfg["start_ply"], history_length=cfg["history_length"]
        ),
        "encoder": ChessTensorEncoder(history_length=cfg["history_length"]),
        "splitter": GameLevelSplitter(seed=cfg["seed"]),
    }


def _init_worker(cfg: Dict[str, Any]) -> None:
    _STATE.clear()
    _STATE.update(_make_state(cfg))


def _move_hash(game: chess.pgn.Game) -> str:
    moves = " ".join(m.uci() for m in game.mainline_moves())
    return hashlib.sha1(moves.encode("utf-8")).hexdigest()[:16]


def _process_text(text: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse, weak-label, filter and encode one raw game; ``None`` if invalid."""
    cfg = state["cfg"]
    game = next(state["parser"].iter_games(io.StringIO(text)), None)
    if game is None:
        return None

    base_id = default_game_id(game)
    raw_records = state["labeler"].label_game(game, game_id=base_id)
    # Steps 1-2 of filter_and_balance (confidence/material filter, per-game cap)
    # are per-record / per-game, so applying them here equals the global filter.
    kept = filter_and_balance(
        raw_records,
        min_confidence=cfg["min_confidence"],
        max_material=cfg["max_material"],
        max_per_game=cfg["max_per_game"],
        balance=False,
    )
    kept.sort(key=lambda r: r.ply_index)

    rows: List[Dict[str, Any]] = []
    tensors: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    if kept:
        wanted = {r.ply_index for r in kept}
        by_ply = {
            w.ply_index: w
            for w in extract_windows(
                game,
                game_id=base_id,
                history_length=cfg["history_length"],
                start_ply=cfg["start_ply"],
            )
            if w.ply_index in wanted
        }
        for r in kept:
            window = by_ply.get(r.ply_index)
            if window is None:  # cannot happen (same ply range); never fabricate
                continue
            tensor, mask = state["encoder"].encode_position_window(window)
            tensors.append(tensor.astype(np.uint8))
            masks.append(mask.astype(np.uint8))
            rows.append(
                {
                    "ply_index": r.ply_index,
                    "turn": r.turn,
                    "label_class": r.label_class,
                    "label_id": r.label_id,
                    "weak_confidence": r.weak_confidence,
                    "material_balance": r.material_balance,
                    "fen": r.fen,
                }
            )

    return {
        "base_id": base_id,
        "split": state["splitter"].assign_split(base_id),
        "move_hash": _move_hash(game),
        "n_plies": len(list(game.mainline_moves())),
        "n_raw": len(raw_records),
        "raw_class_counts": dict(Counter(r.label_class for r in raw_records)),
        "rows": rows,
        "tensors": np.stack(tensors) if tensors else np.zeros((0, CHANNELS, 8, 8), np.uint8),
        "masks": np.stack(masks) if masks else np.zeros((0, 8), np.uint8),
    }


def _process_chunk(texts: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    return [_process_text(t, _STATE) for t in texts]


# --------------------------------------------------------------------------- #
# Pilot                                                                        #
# --------------------------------------------------------------------------- #
def pilot_throughput(
    source: str | Path = DEFAULT_ARCHIVE,
    n_games: int = 200,
    workers: int = MAX_WORKERS,
    target_minutes: float = 35.0,
    parallel_efficiency: float = 0.75,
    cfg: Optional[Dict[str, Any]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, float]:
    """Time single-process labeling on ``n_games`` and project a game budget."""
    state = _make_state(cfg or _default_cfg())
    valid = windows = 0
    start = time.time()
    with open_pgn_text(source) as handle:
        for text in iter_raw_games(handle):
            result = _process_text(text, state)
            if result is None:
                continue
            valid += 1
            windows += len(result["rows"])
            if valid >= n_games:
                break
    elapsed = time.time() - start
    sec_per_game = elapsed / max(valid, 1)
    workers = max(1, min(workers, MAX_WORKERS))
    projected = int(target_minutes * 60 * workers * parallel_efficiency / sec_per_game)
    report = {
        "games": valid,
        "windows_kept": windows,
        "windows_per_game": round(windows / max(valid, 1), 2),
        "seconds": round(elapsed, 1),
        "sec_per_game_single_process": round(sec_per_game, 3),
        "workers": workers,
        "target_minutes": target_minutes,
        "parallel_efficiency_assumed": parallel_efficiency,
        "projected_games": projected,
    }
    log(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
def _default_cfg(**overrides: Any) -> Dict[str, Any]:
    cfg = {
        "seed": 42,
        "start_ply": 12,
        "history_length": 8,
        "min_confidence": 0.55,
        "max_material": 800,
        "max_per_game": 25,
    }
    cfg.update(overrides)
    return cfg


class _SplitWriter:
    """Append tensors/rows for one split while streaming."""

    def __init__(self, tmp_dir: Path, split: str) -> None:
        self.split = split
        self.raw_path = tmp_dir / f"{split}_tensors.u8"
        self._raw = self.raw_path.open("wb")
        self._csv_handle = (tmp_dir / f"{split}_records.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._csv = csv.DictWriter(self._csv_handle, fieldnames=RECORD_FIELDS)
        self._csv.writeheader()
        self.masks: List[np.ndarray] = []
        self.labels: List[int] = []
        self.n = 0

    def add(self, tensors: np.ndarray, masks: np.ndarray, rows: List[Dict[str, Any]]) -> None:
        if len(rows):
            self._raw.write(np.ascontiguousarray(tensors, dtype=np.uint8).tobytes())
            self.masks.append(masks)
            for row in rows:
                self.labels.append(int(row["label_id"]))
                self._csv.writerow(row)
            self.n += len(rows)

    def close(self) -> None:
        self._raw.close()
        self._csv_handle.close()


def _finalize_arrays(tmp_dir: Path, writer: _SplitWriter) -> None:
    n = writer.n
    shape = (n, CHANNELS, 8, 8)
    final = tmp_dir / f"{writer.split}_tensors.npy"
    if n == 0:
        np.save(final, np.zeros(shape, dtype=np.uint8))
    else:
        raw = np.memmap(writer.raw_path, dtype=np.uint8, mode="r", shape=shape)
        out = np.lib.format.open_memmap(final, mode="w+", dtype=np.uint8, shape=shape)
        block = 20000
        for lo in range(0, n, block):
            out[lo : lo + block] = raw[lo : lo + block]
        out.flush()
        del out, raw
    writer.raw_path.unlink()
    masks = np.concatenate(writer.masks) if writer.masks else np.zeros((0, 8), np.uint8)
    np.save(tmp_dir / f"{writer.split}_masks.npy", masks.astype(np.uint8))
    np.save(tmp_dir / f"{writer.split}_labels.npy", np.asarray(writer.labels, dtype=np.int64))


def _iter_results(
    source: str | Path, workers: int, chunk_games: int, cfg: Dict[str, Any]
) -> Iterator[Optional[Dict[str, Any]]]:
    """Yield per-game results (``None`` = invalid game) in deterministic source order.

    At most ``workers * 3`` chunks are in flight, so the archive is read lazily
    and closing the generator early terminates the pool.
    """
    pool = None
    try:
        with open_pgn_text(source) as handle:
            chunks = _chunked(iter_raw_games(handle), chunk_games)
            if workers == 1:
                _init_worker(cfg)
                for chunk in chunks:
                    yield from _process_chunk(chunk)
                return
            pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(cfg,))
            pending: deque = deque()
            for chunk in chunks:
                pending.append(pool.apply_async(_process_chunk, (chunk,)))
                if len(pending) >= workers * 3:
                    yield from pending.popleft().get()
            while pending:
                yield from pending.popleft().get()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


class _Sink:
    """Assign unique game ids and append each game's rows/tensors to its split."""

    def __init__(self, tmp_dir: Path) -> None:
        self.writers = {s: _SplitWriter(tmp_dir, s) for s in SPLITS}
        self._games_handle = (tmp_dir / "games.csv").open("w", newline="", encoding="utf-8")
        self._games = csv.DictWriter(self._games_handle, fieldnames=GAME_FIELDS)
        self._games.writeheader()
        self._base_seen: Counter = Counter()
        self.collisions = 0
        self.valid = 0
        self.windows = 0
        self.raw_counts: Counter = Counter()
        self._seen_hashes: set = set()
        self.duplicates_dropped = 0

    def add(self, result: Dict[str, Any]) -> bool:
        """Add one game; identical movetext seen before is dropped (returns False)."""
        if result["move_hash"] in self._seen_hashes:
            self.duplicates_dropped += 1
            return False
        self._seen_hashes.add(result["move_hash"])
        base = result["base_id"]
        seen_before = self._base_seen[base]
        self._base_seen[base] += 1
        if seen_before:
            self.collisions += 1
        unique = base if not seen_before else f"{base}~{seen_before + 1}"
        rows = result["rows"]
        for row in rows:
            row["game_id"] = unique
            row["base_game_id"] = base
            row["window_id"] = f"{unique}#{row['ply_index']}"
            row["split"] = result["split"]
        self.writers[result["split"]].add(result["tensors"], result["masks"], rows)
        self._games.writerow(
            {
                "game_id": unique,
                "base_game_id": base,
                "split": result["split"],
                "move_hash": result["move_hash"],
                "n_plies": result["n_plies"],
                "n_raw": result["n_raw"],
                "n_kept": len(rows),
            }
        )
        self.raw_counts.update(result["raw_class_counts"])
        self.valid += 1
        self.windows += len(rows)
        return True

    def close(self) -> None:
        self._games_handle.close()
        for writer in self.writers.values():
            writer.close()


def _prepare_tmp(out: Path, force: bool) -> Path:
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; pass force=True to rebuild.")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / ".gitignore").write_text("*\n", encoding="utf-8")
    return tmp


def _stream_into_sink(
    sink: _Sink,
    source: str | Path,
    max_games: int,
    workers: int,
    chunk_games: int,
    cfg: Dict[str, Any],
    log: Callable[[str], None],
) -> int:
    """Feed the first ``max_games`` valid games into ``sink``; return raw games scanned."""
    scanned = 0
    start = time.time()
    with closing(_iter_results(source, workers, chunk_games, cfg)) as results:
        for result in results:
            scanned += 1
            if result is None:
                continue
            sink.add(result)
            if sink.valid % 500 == 0:
                elapsed = time.time() - start
                rate = sink.valid / max(elapsed, 1e-9)
                eta = (max_games - sink.valid) / max(rate, 1e-9)
                log(
                    f"  {sink.valid}/{max_games} games, {sink.windows} windows, "
                    f"{elapsed / 60:.1f} min elapsed, {rate:.2f} games/s, "
                    f"ETA {eta / 60:.1f} min"
                )
            if sink.valid >= max_games:
                break
    return scanned


def _publish(tmp: Path, out: Path, meta: Dict[str, Any], verify_games: int) -> None:
    """Fail loudly (keeping ``tmp``) unless every check passed; else rename into place."""
    failures = [k for k, v in meta["integrity"]["checks"].items() if not v]
    if failures:
        raise AssertionError(f"Integrity checks FAILED: {failures} (kept in {tmp})")
    if verify_games > 0 and meta["source_verification"]["mismatches"]:
        raise AssertionError(
            f"Cache/source mismatch (kept in {tmp}): {meta['source_verification']}"
        )
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def build_scaled_dataset(
    source: str | Path = DEFAULT_ARCHIVE,
    out_dir: str | Path = DEFAULT_OUT,
    max_games: int = 15000,
    workers: int = MAX_WORKERS,
    chunk_games: int = 50,
    seed: int = 42,
    force: bool = False,
    verify_games: int = 50,
    min_confidence: float = 0.55,
    max_material: int = 800,
    max_per_game: int = 25,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Build the cache for the first ``max_games`` valid games of ``source``."""
    if max_games < 1:
        raise ValueError(f"max_games must be >= 1, got {max_games}.")
    out = Path(out_dir)
    tmp = _prepare_tmp(out, force)
    workers = max(1, min(int(workers), MAX_WORKERS))
    cfg = _default_cfg(
        seed=seed,
        min_confidence=min_confidence,
        max_material=max_material,
        max_per_game=max_per_game,
    )

    sink = _Sink(tmp)
    start = time.time()
    try:
        scanned = _stream_into_sink(sink, source, max_games, workers, chunk_games, cfg, log)
    finally:
        sink.close()
    build_seconds = time.time() - start
    log(f"Labeled {sink.valid} games ({scanned} raw scanned) in {build_seconds / 60:.1f} min.")
    for writer in sink.writers.values():
        _finalize_arrays(tmp, writer)

    integrity = integrity_report(tmp)
    meta: Dict[str, Any] = {
        "source": str(source),
        "max_games": max_games,
        "workers": workers,
        "seed": seed,
        "filter": {
            "min_confidence": min_confidence,
            "max_material": max_material,
            "max_per_game": max_per_game,
            "start_ply": cfg["start_ply"],
            "history_length": cfg["history_length"],
        },
        "games_scanned": scanned,
        "games_valid": sink.valid,
        "windows": sink.windows,
        "windows_per_split": {s: sink.writers[s].n for s in SPLITS},
        "games_per_split": dict(integrity["games_per_split"]),
        "raw_class_counts_before_filter": dict(sink.raw_counts),
        "game_id_collisions": sink.collisions,
        "duplicate_games_dropped": sink.duplicates_dropped,
        "build_seconds": round(build_seconds, 1),
        "integrity": integrity,
    }
    if verify_games > 0:
        meta["source_verification"] = verify_against_source(tmp, source, verify_games, cfg=cfg)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _publish(tmp, out, meta, verify_games)
    log(f"Wrote {out}")
    return meta


# --------------------------------------------------------------------------- #
# Reading helpers, integrity and verification                                  #
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def integrity_report(directory: str | Path) -> Dict[str, Any]:
    """Leak/uniqueness/distribution checks over a built cache (real data only)."""
    root = Path(directory)
    records = {s: _read_csv(root / f"{s}_records.csv") for s in SPLITS}
    games = _read_csv(root / "games.csv")

    ids_by_split = {s: {r["game_id"] for r in records[s]} for s in SPLITS}
    base_by_split = {s: {r["base_game_id"] for r in records[s]} for s in SPLITS}

    def overlaps(sets: Dict[str, set]) -> Dict[str, int]:
        return {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }

    game_overlap = overlaps(ids_by_split)
    base_overlap = overlaps(base_by_split)

    all_window_ids = [r["window_id"] for s in SPLITS for r in records[s]]
    duplicate_window_ids = len(all_window_ids) - len(set(all_window_ids))

    keys = {s: [_position_key(r["fen"]) for r in records[s]] for s in SPLITS}
    all_keys = [k for s in SPLITS for k in keys[s]]
    key_sets = {s: set(keys[s]) for s in SPLITS}
    position_overlap = overlaps(key_sets)

    hash_splits: Dict[str, set] = {}
    hash_count: Counter = Counter()
    for g in games:
        hash_count[g["move_hash"]] += 1
        hash_splits.setdefault(g["move_hash"], set()).add(g["split"])
    duplicate_game_hashes = sum(1 for c in hash_count.values() if c > 1)
    cross_split_duplicate_games = sum(1 for s in hash_splits.values() if len(s) > 1)

    arrays_ok = True
    last_slot_checked = last_slot_bad = 0
    encoder = BoardPlaneEncoder()
    for s in SPLITS:
        tensors = np.load(root / f"{s}_tensors.npy", mmap_mode="r")
        masks = np.load(root / f"{s}_masks.npy", mmap_mode="r")
        labels = np.load(root / f"{s}_labels.npy")
        n = len(records[s])
        arrays_ok &= tensors.shape == (n, CHANNELS, 8, 8) and tensors.dtype == np.uint8
        arrays_ok &= masks.shape == (n, 8) and labels.shape == (n,) and labels.dtype == np.int64
        if n:
            arrays_ok &= bool(labels.min() >= 0 and labels.max() < NUM_CLASSES)
            arrays_ok &= bool(tensors.max() <= 1)
        # The newest 14-plane slot must equal the encoding of the labeled ply's own
        # board (record FEN): the input stops at ply t, nothing from the look-ahead.
        # Planes 0-12 must match exactly. FEN drops an en-passant square when no legal
        # capture exists while the encoder uses the raw ``ep_square``, so for plane 13
        # the FEN-derived plane only has to be contained in the tensor's.
        step = max(1, n // 2000)
        for i in range(0, n, step):
            board = chess.Board(records[s][i]["fen"])
            expected = encoder.encode(board, board.turn).astype(np.uint8)
            got = np.asarray(tensors[i][-14:])
            last_slot_checked += 1
            if (
                not np.array_equal(got[:13], expected[:13])
                or bool((expected[13] > got[13]).any())
                or not masks[i].all()
            ):
                last_slot_bad += 1
        arrays_ok &= bool(
            all(_ID_TO_NAME[int(x)] == r["label_class"] for x, r in zip(labels, records[s]))
        )

    def dist(rows: List[Dict[str, str]]) -> Dict[str, Any]:
        counts = Counter(r["label_class"] for r in rows)
        total = max(len(rows), 1)
        return {
            "counts": {c: counts.get(c, 0) for c in PLAN_IDS},
            "proportions": {c: round(counts.get(c, 0) / total, 4) for c in PLAN_IDS},
        }

    checks = {
        "zero_game_id_overlap_across_splits": sum(game_overlap.values()) == 0,
        "zero_raw_game_id_overlap_across_splits": sum(base_overlap.values()) == 0,
        "window_ids_unique": duplicate_window_ids == 0,
        "no_duplicate_games_across_splits": cross_split_duplicate_games == 0,
        "arrays_consistent_with_records": bool(arrays_ok),
        "last_slot_equals_labeled_ply_board": last_slot_bad == 0 and last_slot_checked > 0,
    }
    return {
        "checks": checks,
        "game_id_overlap": game_overlap,
        "raw_game_id_overlap": base_overlap,
        "duplicate_window_ids": duplicate_window_ids,
        "games_per_split": {s: len(ids_by_split[s]) for s in SPLITS},
        "windows_per_split": {s: len(records[s]) for s in SPLITS},
        "class_distribution": {s: dist(records[s]) for s in SPLITS},
        "duplicate_positions": {
            "windows_total": len(all_keys),
            "unique_positions": len(set(all_keys)),
            "windows_sharing_a_position": len(all_keys) - len(set(all_keys)),
            "cross_split_shared_positions": position_overlap,
        },
        "duplicate_games_by_movetext": {
            "hashes_seen_more_than_once": duplicate_game_hashes,
            "hashes_spanning_multiple_splits": cross_split_duplicate_games,
        },
        "last_slot_check": {"windows_checked": last_slot_checked, "mismatches": last_slot_bad},
    }


def verify_against_source(
    directory: str | Path,
    source: str | Path,
    n_games: int = 50,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Re-parse the first ``n_games`` valid games and compare with the cache.

    Cached uint8 tensors must equal :class:`ChessPlanDataset` dynamic-mode
    tensors (float32) for the same windows, and re-labeling must reproduce the
    cached weak labels.
    """
    from src.data.chess_dataset import ChessPlanDataset

    cfg = cfg or _default_cfg()
    root = Path(directory)
    index: Dict[str, Tuple[str, int]] = {}
    rows_by_split: Dict[str, List[Dict[str, str]]] = {}
    for s in SPLITS:
        rows_by_split[s] = _read_csv(root / f"{s}_records.csv")
        for i, r in enumerate(rows_by_split[s]):
            index[r["window_id"]] = (s, i)
    tensors = {s: np.load(root / f"{s}_tensors.npy", mmap_mode="r") for s in SPLITS}
    masks = {s: np.load(root / f"{s}_masks.npy", mmap_mode="r") for s in SPLITS}

    parser = StreamingPGNParser()
    labeler = RetrospectiveWeakLabeler(
        start_ply=cfg["start_ply"], history_length=cfg["history_length"]
    )
    base_seen: Counter = Counter()
    seen_hashes: set = set()
    compared = mismatches = label_mismatches = games_done = 0
    with open_pgn_text(source) as handle:
        for text in iter_raw_games(handle):
            game = next(parser.iter_games(io.StringIO(text)), None)
            if game is None:
                continue
            if _move_hash(game) in seen_hashes:  # the build drops repeated movetext
                continue
            seen_hashes.add(_move_hash(game))
            base = default_game_id(game)
            seen_before = base_seen[base]
            base_seen[base] += 1
            unique = base if not seen_before else f"{base}~{seen_before + 1}"
            windows = [
                w
                for w in extract_windows(
                    game,
                    game_id=base,
                    history_length=cfg["history_length"],
                    start_ply=cfg["start_ply"],
                )
                if f"{unique}#{w.ply_index}" in index
            ]
            if windows:
                dynamic = ChessPlanDataset(windows, [0] * len(windows))
                fresh = {r.ply_index: r for r in labeler.label_game(game, game_id=base)}
                for k, w in enumerate(windows):
                    split, row = index[f"{unique}#{w.ply_index}"]
                    dyn_tensor, dyn_mask, _ = dynamic[k]
                    compared += 1
                    if not (
                        np.array_equal(
                            np.asarray(tensors[split][row], np.float32), dyn_tensor.numpy()
                        )
                        and np.array_equal(
                            np.asarray(masks[split][row], np.float32), dyn_mask.numpy()
                        )
                    ):
                        mismatches += 1
                    cached = rows_by_split[split][row]
                    ref = fresh[w.ply_index]
                    if (
                        cached["label_class"] != ref.label_class
                        or abs(float(cached["weak_confidence"]) - ref.weak_confidence) > 1e-9
                    ):
                        label_mismatches += 1
            games_done += 1
            if games_done >= n_games:
                break
    return {
        "games_checked": games_done,
        "windows_compared": compared,
        "mismatches": mismatches,
        "label_mismatches": label_mismatches,
    }


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
class _RowView:
    """Lazy, picklable row selection over an ``.npy`` memmap (no RAM copy)."""

    def __init__(self, path: str | Path, indices: np.ndarray) -> None:
        self.path = str(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._mm: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> np.ndarray:
        if self._mm is None:
            self._mm = np.load(self.path, mmap_mode="r")
        return self._mm[self.indices[i]]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_mm"] = None
        return state


def load_scaled_dataset(
    directory: str | Path = DEFAULT_OUT,
    balance_train: bool = True,
    seed: int = 42,
    max_class_fraction: float = 0.30,
    splits: Sequence[str] = SPLITS,
    balance_select: str = "confidence",
    train_sample_kinds: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load ``{'train','val','test'}`` as encoded-mode ``ChessPlanDataset`` objects.

    ``train_sample_kinds`` keeps only train rows with those ``sample_kind`` values
    (caches from ``build_allpos_dataset``), applied before balancing.

    Class balancing (``balance_classes``) is applied to the TRAIN split only;
    val/test always keep their natural class distribution. ``splits`` limits which
    splits are opened at all (e.g. ``('train', 'val')`` never touches test files). Tensors stay on disk
    (memmap) and are cast to float32 per sample.
    """
    from src.data.chess_dataset import ChessPlanDataset

    root = Path(directory)
    datasets: Dict[str, Any] = {}
    for split in splits:
        labels = np.load(root / f"{split}_labels.npy")
        indices = np.arange(len(labels), dtype=np.int64)
        if split == "train" and (balance_train or train_sample_kinds) and len(labels):
            rows = _read_csv(root / "train_records.csv")
            row_of = {r["window_id"]: i for i, r in enumerate(rows)}
            if train_sample_kinds:
                rows = [r for r in rows if r["sample_kind"] in set(train_sample_kinds)]
            records = [
                LabeledPositionRecord(
                    window_id=r["window_id"],
                    game_id=r["game_id"],
                    ply_index=int(r["ply_index"]),
                    turn=r["turn"],
                    label_class=r["label_class"],
                    label_id=int(r["label_id"]),
                    weak_confidence=float(r["weak_confidence"]),
                    fen=r["fen"],
                    split="train",
                )
                for r in rows
            ]
            kept = (
                balance_classes(
                    records,
                    max_class_fraction=max_class_fraction,
                    seed=seed,
                    select=balance_select,
                )
                if balance_train
                else records
            )
            indices = np.asarray(sorted(row_of[r.window_id] for r in kept), dtype=np.int64)
        datasets[split] = ChessPlanDataset(
            encoded_tensors=_RowView(root / f"{split}_tensors.npy", indices),
            encoded_masks=_RowView(root / f"{split}_masks.npy", indices),
            labels=labels[indices].tolist(),
        )
    return datasets


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the scaled weak-labeled dataset cache.")
    ap.add_argument("--pgn", default=str(DEFAULT_ARCHIVE), help="Input .zip or .pgn.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-games", type=int, default=15000, help="Valid games to use.")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="Worker processes (<=14).")
    ap.add_argument("--chunk-games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output dir.")
    ap.add_argument("--verify-games", type=int, default=50, help="0 disables source verification.")
    ap.add_argument("--pilot", action="store_true", help="Only measure throughput and exit.")
    ap.add_argument("--pilot-games", type=int, default=200)
    ap.add_argument("--target-minutes", type=float, default=35.0)
    args = ap.parse_args(argv)

    if args.pilot:
        pilot_throughput(
            args.pgn, args.pilot_games, args.workers, args.target_minutes, cfg=_default_cfg()
        )
        return 0

    meta = build_scaled_dataset(
        args.pgn,
        args.out_dir,
        max_games=args.max_games,
        workers=args.workers,
        chunk_games=args.chunk_games,
        seed=args.seed,
        force=args.force,
        verify_games=args.verify_games,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "integrity"}, indent=2))
    print(json.dumps(meta["integrity"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
