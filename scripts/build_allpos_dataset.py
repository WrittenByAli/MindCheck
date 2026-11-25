"""Build an UNFILTERED train/validation set from the 80k build's own games.

The 80k cache keeps only confident, balanced windows, so models trained on it
never see ambiguous positions. This reuses the exact games and game-level split
of that cache (matched by movetext hash, so no game can change split) and samples
positions from ply ``start_ply`` without any confidence or material filter:

* every split: ``K`` uniformly random windows per game (``sample_kind=uniform``);
* train only: every remaining CHAIN_UNDERMINE window (``sample_kind=enriched``).

Anything matching the human-annotation benchmark is removed: training games with
identical movetext, and windows with an identical 4-ply model input. Test-split
games are never labeled or written. One split is built per invocation.

Usage::

    python scripts/build_allpos_dataset.py --split val
    python scripts/build_allpos_dataset.py --split train
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, deque
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_scaled_dataset as bsd  # noqa: E402
import chess  # noqa: E402
import numpy as np  # noqa: E402

from src.data.pgn_parser import StreamingPGNParser  # noqa: E402
from src.data.tensor_encoder import BoardPlaneEncoder, ChessTensorEncoder  # noqa: E402
from src.data.window_extractor import default_game_id, extract_windows  # noqa: E402

DEFAULT_REFERENCE = _REPO_ROOT / "data" / "processed" / "scaled_80k"
DEFAULT_OUT = _REPO_ROOT / "data" / "processed" / "allpos"
SHEET = _REPO_ROOT / "data" / "annotations" / "raw_annotations_member1.csv"
SHEET_PGN = _REPO_ROOT / "data" / "raw" / "lichess_elite_subset.pgn"
PER_GAME = {"train": 30, "val": 25}
CHAIN_ID = 4
KEY_CHANNELS = 56  # the newest 4 plies: the deployed model's actual input
FIELDS = bsd.RECORD_FIELDS + ["sample_kind"]

_WORKER: Dict[str, Any] = {}


def input_key(tensor: np.ndarray) -> str:
    """Hash of the 4-ply model input of one ``(112, 8, 8)`` window."""
    return hashlib.sha1(
        np.ascontiguousarray(tensor[-KEY_CHANNELS:], np.uint8).tobytes()
    ).hexdigest()


def benchmark_keys(sheet: Path = SHEET, pgn: Path = SHEET_PGN) -> Tuple[Set[str], Set[str]]:
    """``(game movetext hashes, model-input keys)`` of every annotation window.

    Raises if any window cannot be rebuilt exactly, so the exclusion can never be
    silently incomplete.
    """
    wanted: Dict[str, List[Dict[str, str]]] = {}
    rows = bsd._read_csv(sheet)
    for row in rows:
        wanted.setdefault(row["game_id"], []).append(row)
    encoder = ChessTensorEncoder()
    hashes: Set[str] = set()
    keys: Dict[str, str] = {}
    for game in StreamingPGNParser().stream_games(pgn):
        gid = default_game_id(game)
        if gid not in wanted:
            continue
        windows = {w.ply_index: w for w in extract_windows(game, game_id=gid, start_ply=12)}
        for row in wanted[gid]:
            window = windows.get(int(row["ply_index"]))
            if window is not None and window.current_board.fen() == row["fen_current"]:
                tensor, _ = encoder.encode_position_window(window)
                keys[row["window_id"]] = input_key(tensor.astype(np.uint8))
                hashes.add(bsd._move_hash(game))
    missing = [r["window_id"] for r in rows if r["window_id"] not in keys]
    if missing:
        raise ValueError(f"cannot rebuild {len(missing)} annotation windows, e.g. {missing[:3]}")
    return hashes, set(keys.values())


def select_plies(
    labels_by_ply: Dict[int, int], per_game: int, enrich_chain: bool, rng: random.Random
) -> List[Tuple[int, str]]:
    """``[(ply, sample_kind)]`` sorted by ply: a uniform sample plus optional CHAIN extras."""
    plies = sorted(labels_by_ply)
    uniform = set(rng.sample(plies, min(per_game, len(plies))))
    picked = [(p, "uniform") for p in uniform]
    if enrich_chain:
        picked += [
            (p, "enriched") for p in plies if p not in uniform and labels_by_ply[p] == CHAIN_ID
        ]
    return sorted(picked)


def _init_worker(cfg: Dict[str, Any], job: Dict[str, Any]) -> None:
    _WORKER.clear()
    _WORKER.update(bsd._make_state(cfg), job=job)


def _process_game(text: str) -> Optional[Dict[str, Any]]:
    """``None`` unless the game is one of the reference games of the wanted split."""
    job, cfg = _WORKER["job"], _WORKER["cfg"]
    game = next(_WORKER["parser"].iter_games(io.StringIO(text)), None)
    if game is None:
        return None
    move_hash = bsd._move_hash(game)
    entry = job["games"].get(move_hash)
    if entry is None:
        return None
    if job["split"] == "train" and move_hash in job["bench_games"]:
        return {"move_hash": move_hash, "benchmark_game": True}
    game_id, base_id = entry
    if default_game_id(game) != base_id:
        return None  # same moves under other headers: the reference kept a different copy
    records = {r.ply_index: r for r in _WORKER["labeler"].label_game(game, game_id=base_id)}
    chosen = select_plies(
        {p: r.label_id for p, r in records.items()},
        job["per_game"],
        job["split"] == "train",
        random.Random(f"{cfg['seed']}:{move_hash}"),
    )
    kind_of = dict(chosen)
    rows, tensors, masks, dropped = [], [], [], 0
    for window in extract_windows(
        game, game_id=base_id, history_length=cfg["history_length"], start_ply=cfg["start_ply"]
    ):
        kind = kind_of.get(window.ply_index)
        if kind is None:
            continue
        tensor, mask = _WORKER["encoder"].encode_position_window(window)
        tensor = tensor.astype(np.uint8)
        if input_key(tensor) in job["bench_inputs"]:
            dropped += 1
            continue
        r = records[window.ply_index]
        tensors.append(tensor)
        masks.append(mask.astype(np.uint8))
        rows.append(
            {
                "window_id": f"{game_id}#{r.ply_index}",
                "game_id": game_id,
                "base_game_id": base_id,
                "ply_index": r.ply_index,
                "turn": r.turn,
                "label_class": r.label_class,
                "label_id": r.label_id,
                "weak_confidence": r.weak_confidence,
                "material_balance": r.material_balance,
                "split": job["split"],
                "fen": r.fen,
                "sample_kind": kind,
            }
        )
    return {
        "move_hash": move_hash,
        "benchmark_game": False,
        "game_id": game_id,
        "n_positions": len(records),
        "dropped_benchmark_inputs": dropped,
        "rows": rows,
        "tensors": np.stack(tensors) if tensors else np.zeros((0, 112, 8, 8), np.uint8),
        "masks": np.stack(masks) if masks else np.zeros((0, 8), np.uint8),
    }


def _process_chunk(texts: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    return [_process_game(t) for t in texts]


def _iter_results(
    source: Path, limit: int, workers: int, chunk_games: int, cfg: Dict, job: Dict
) -> Iterator[Optional[Dict[str, Any]]]:
    """Results in source order over the first ``limit`` raw games (what the reference read)."""
    pool = None
    try:
        with bsd.open_pgn_text(source) as handle:
            chunks = bsd._chunked(islice(bsd.iter_raw_games(handle), limit), chunk_games)
            if workers == 1:
                _init_worker(cfg, job)
                for chunk in chunks:
                    yield from _process_chunk(chunk)
                return
            pool = mp.get_context("spawn").Pool(
                workers, initializer=_init_worker, initargs=(cfg, job)
            )
            pending: deque = deque()
            for chunk in chunks:
                pending.append(pool.apply_async(_process_chunk, (chunk,)))
                if len(pending) >= workers * 2:
                    yield from pending.popleft().get()
            while pending:
                yield from pending.popleft().get()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


class _Writer(bsd._SplitWriter):
    """Split writer whose records CSV also carries ``sample_kind``."""

    def __init__(self, tmp_dir: Path, split: str) -> None:
        super().__init__(tmp_dir, split)
        self._csv_handle.seek(0)
        self._csv_handle.truncate()
        self._csv = bsd.csv.DictWriter(self._csv_handle, fieldnames=FIELDS)
        self._csv.writeheader()


def split_checks(
    directory: Path,
    split: str,
    reference: Dict[str, Tuple[str, str]],
    written: Dict[str, str],
    bench_games: Set[str],
    bench_inputs: Set[str],
) -> Dict[str, bool]:
    """Integrity checks recomputed from the files actually written."""
    records = bsd._read_csv(directory / f"{split}_records.csv")
    tensors = np.load(directory / f"{split}_tensors.npy", mmap_mode="r")
    labels = np.load(directory / f"{split}_labels.npy")
    masks = np.load(directory / f"{split}_masks.npy", mmap_mode="r")
    n = len(records)
    encoder = BoardPlaneEncoder()
    mismatches = 0
    for i in range(0, n, max(1, n // 2000)):
        board = chess.Board(records[i]["fen"])
        expected = encoder.encode(board, board.turn)[:12].astype(np.uint8)
        mismatches += int(not np.array_equal(tensors[i, 98:110], expected))
    leaked = 0
    for lo in range(0, n, 20000):
        block = np.asarray(tensors[lo : lo + 20000, -KEY_CHANNELS:])
        leaked += sum(hashlib.sha1(row.tobytes()).hexdigest() in bench_inputs for row in block)
    checks = {
        "only_reference_games_of_this_split": all(h in reference for h in written),
        "record_game_ids_match_games": {r["game_id"] for r in records} <= set(written.values()),
        "window_ids_unique": len({r["window_id"] for r in records}) == n,
        "arrays_consistent_with_records": bool(
            tensors.shape == (n, 112, 8, 8)
            and masks.shape == (n, 8)
            and labels.shape == (n,)
            and all(int(r["label_id"]) == int(v) for r, v in zip(records, labels))
        ),
        "last_slot_equals_labeled_ply_board": mismatches == 0,
        "no_annotation_model_input": leaked == 0,
        "not_empty": n > 0,
    }
    if split == "train":
        checks["no_annotation_game"] = not (set(written) & bench_games)
    else:
        checks["uniform_only"] = all(r["sample_kind"] == "uniform" for r in records)
    return checks


def build_split(
    split: str,
    reference_dir: Path = DEFAULT_REFERENCE,
    out_dir: Path = DEFAULT_OUT,
    source: Path = bsd.DEFAULT_ARCHIVE,
    workers: int = bsd.MAX_WORKERS,
    per_game: Optional[int] = None,
    bench: Optional[Tuple[Set[str], Set[str]]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Build one split into ``out_dir``; returns its metadata."""
    if split not in PER_GAME:
        raise ValueError(f"split must be train or val, got {split!r} (test is never built).")
    reference_dir, out_dir = Path(reference_dir), Path(out_dir)
    meta_path = out_dir / f"meta_{split}.json"
    if meta_path.exists():
        raise FileExistsError(f"{meta_path} exists; refusing to overwrite.")
    ref_meta = json.loads((reference_dir / "meta.json").read_text(encoding="utf-8"))
    reference = {
        g["move_hash"]: (g["game_id"], g["base_game_id"])
        for g in bsd._read_csv(reference_dir / "games.csv")
        if g["split"] == split
    }
    bench_games, bench_inputs = bench if bench is not None else benchmark_keys()
    cfg = bsd._default_cfg(
        start_ply=ref_meta["filter"]["start_ply"],
        history_length=ref_meta["filter"]["history_length"],
    )
    job = {
        "split": split,
        "games": reference,
        "per_game": per_game or PER_GAME[split],
        "bench_games": bench_games,
        "bench_inputs": bench_inputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".tmp_{split}"
    tmp.mkdir(exist_ok=True)
    writer = _Writer(tmp, split)
    written: Dict[str, str] = {}
    positions = dropped_inputs = dropped_games = 0
    start = time.time()
    results = _iter_results(Path(source), int(ref_meta["games_scanned"]), workers, 50, cfg, job)
    try:
        for result in results:
            if result is None:
                continue
            if result["benchmark_game"]:
                dropped_games += 1
                continue
            if result["move_hash"] in written:
                continue  # a repeat of the same movetext: the reference kept one copy
            written[result["move_hash"]] = result["game_id"]
            positions += result["n_positions"]
            dropped_inputs += result["dropped_benchmark_inputs"]
            writer.add(result["tensors"], result["masks"], result["rows"])
            if len(written) % 5000 == 0:
                log(
                    f"  {split}: {len(written)}/{len(reference)} games, {writer.n} windows, "
                    f"{(time.time() - start) / 60:.1f} min"
                )
    finally:
        results.close()
        writer.close()
    bsd._finalize_arrays(tmp, writer)
    checks = split_checks(tmp, split, reference, written, bench_games, bench_inputs)
    expected = set(reference) - (bench_games if split == "train" else set())
    checks["all_reference_games_found"] = len(expected - set(written)) <= 0.001 * len(expected)
    records = bsd._read_csv(tmp / f"{split}_records.csv")
    meta = {
        "split": split,
        "reference": str(reference_dir),
        "per_game_uniform": job["per_game"],
        "games_expected": len(expected),
        "games_written": len(written),
        "positions_available": positions,
        "windows": writer.n,
        "windows_by_kind": dict(Counter(r["sample_kind"] for r in records)),
        "uniform_class_counts": dict(
            Counter(r["label_class"] for r in records if r["sample_kind"] == "uniform")
        ),
        "class_counts": dict(Counter(r["label_class"] for r in records)),
        "benchmark_games_dropped": dropped_games,
        "benchmark_input_windows_dropped": dropped_inputs,
        "labels": "rule-engine weak labels (look-ahead derived)",
        "build_seconds": round(time.time() - start, 1),
        "checks": checks,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        (tmp / "meta_failed.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        raise AssertionError(f"integrity checks FAILED: {failed} (files kept in {tmp})")
    for path in sorted(tmp.iterdir()):
        path.replace(out_dir / path.name)
    tmp.rmdir()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {split}: {writer.n} windows from {len(written)} games")
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the unfiltered train/val sets.")
    ap.add_argument("--split", required=True, choices=sorted(PER_GAME))
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--source", default=str(bsd.DEFAULT_ARCHIVE))
    ap.add_argument("--workers", type=int, default=bsd.MAX_WORKERS)
    args = ap.parse_args(argv)
    meta = build_split(
        args.split,
        Path(args.reference),
        Path(args.out_dir),
        Path(args.source),
        workers=max(1, min(args.workers, bsd.MAX_WORKERS)),
        log=lambda s: print(s, flush=True),
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
