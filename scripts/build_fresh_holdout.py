"""Build a fresh, never-seen holdout from archive games the 80k build did not scan.

Two audit findings motivate this. The cached val/test splits keep only
confident, balanced-material windows with a top-25-by-confidence cap per game,
and that confidence is derived from look-ahead, so they are not natural
positions. The cached 80k test split was also already evaluated once.

The first ``games_scanned`` raw games of the archive (everything the 80k build
consumed) are skipped, games whose movetext hash appears in its ``games.csv``
are dropped, and two directories in the ``test_*`` cache layout are written:

* ``holdout_filtered``: the same confidence / material / per-game-cap rules;
* ``holdout_full``: every window from ``start_ply``, unfiltered and uncapped
  (closer to live use).

Labels are still rule-engine weak labels. Usage::

    python scripts/build_fresh_holdout.py --max-games 6000 --workers 14
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import shutil
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

from src.data.tensor_encoder import BoardPlaneEncoder  # noqa: E402
from src.data.weak_labeler import default_game_id, filter_and_balance  # noqa: E402
from src.data.window_extractor import extract_windows  # noqa: E402

DEFAULT_EXCLUDE = _REPO_ROOT / "data" / "processed" / "scaled_80k"
DEFAULT_FILTERED = _REPO_ROOT / "data" / "processed" / "holdout_filtered"
DEFAULT_FULL = _REPO_ROOT / "data" / "processed" / "holdout_full"
KINDS = ("filtered", "full")

_WORKER: Dict[str, Any] = {}


def _init_worker(cfg: Dict[str, Any], excluded: Set[str]) -> None:
    _WORKER.clear()
    _WORKER.update(bsd._make_state(cfg), excluded=excluded)


def _process_game(text: str) -> Optional[Dict[str, Any]]:
    """``None`` for games the parser rejects; excluded games are not labeled."""
    cfg = _WORKER["cfg"]
    game = next(_WORKER["parser"].iter_games(io.StringIO(text)), None)
    if game is None:
        return None
    move_hash = bsd._move_hash(game)
    if move_hash in _WORKER["excluded"]:
        return {"move_hash": move_hash, "excluded": True}
    base_id = default_game_id(game)
    records = _WORKER["labeler"].label_game(game, game_id=base_id)
    kept = filter_and_balance(
        records,
        min_confidence=cfg["min_confidence"],
        max_material=cfg["max_material"],
        max_per_game=cfg["max_per_game"],
        balance=False,
    )
    kept_plies = {r.ply_index for r in kept}
    windows = {
        w.ply_index: w
        for w in extract_windows(
            game, game_id=base_id, history_length=cfg["history_length"], start_ply=cfg["start_ply"]
        )
    }
    rows, tensors, masks = [], [], []
    for r in sorted(records, key=lambda rec: rec.ply_index):
        window = windows.get(r.ply_index)
        if window is None:  # same ply range as the labeler; never fabricate
            continue
        tensor, mask = _WORKER["encoder"].encode_position_window(window)
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
        "move_hash": move_hash,
        "excluded": False,
        "base_id": base_id,
        "n_plies": len(list(game.mainline_moves())),
        "rows": rows,
        "keep": [i for i, row in enumerate(rows) if row["ply_index"] in kept_plies],
        "tensors": np.stack(tensors) if tensors else np.zeros((0, 112, 8, 8), np.uint8),
        "masks": np.stack(masks) if masks else np.zeros((0, 8), np.uint8),
    }


def _process_chunk(texts: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    return [_process_game(t) for t in texts]


def _iter_results(
    source: Path, skip: int, workers: int, chunk_games: int, cfg: Dict[str, Any], excluded: Set[str]
) -> Iterator[Optional[Dict[str, Any]]]:
    """Results in source order, starting after the first ``skip`` raw games."""
    pool = None
    try:
        with bsd.open_pgn_text(source) as handle:
            fresh = islice(bsd.iter_raw_games(handle), skip, None)
            chunks = bsd._chunked(fresh, chunk_games)
            if workers == 1:
                _init_worker(cfg, excluded)
                for chunk in chunks:
                    yield from _process_chunk(chunk)
                return
            pool = mp.get_context("spawn").Pool(
                workers, initializer=_init_worker, initargs=(cfg, excluded)
            )
            pending: deque = deque()
            for chunk in chunks:
                pending.append(pool.apply_async(_process_chunk, (chunk,)))
                if len(pending) >= workers * 2:  # full-position results are large
                    yield from pending.popleft().get()
            while pending:
                yield from pending.popleft().get()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def holdout_checks(directory: Path, kind: str, excluded: Set[str], cfg: Dict[str, Any]) -> Dict:
    """Integrity checks recomputed from the files actually written."""
    records = bsd._read_csv(directory / "test_records.csv")
    games = bsd._read_csv(directory / "games.csv")
    tensors = np.load(directory / "test_tensors.npy", mmap_mode="r")
    masks = np.load(directory / "test_masks.npy", mmap_mode="r")
    labels = np.load(directory / "test_labels.npy")
    n = len(records)
    hashes = [g["move_hash"] for g in games]
    per_game = Counter(r["game_id"] for r in records)

    encoder = BoardPlaneEncoder()
    mismatches = checked = 0
    for i in range(0, n, max(1, n // 2000)):
        board = chess.Board(records[i]["fen"])
        expected = encoder.encode(board, board.turn)[:12]
        mismatches += int(not np.array_equal(tensors[i, 98:110], expected.astype(np.uint8)))
        checked += 1

    checks = {
        "zero_movetext_overlap_with_80k": not (set(hashes) & excluded),
        "no_duplicate_games_within_holdout": len(hashes) == len(set(hashes)),
        "window_ids_unique": len({r["window_id"] for r in records}) == n,
        "arrays_consistent_with_records": bool(
            tensors.shape == (n, 112, 8, 8)
            and masks.shape == (n, 8)
            and labels.shape == (n,)
            and all(int(r["label_id"]) == int(v) for r, v in zip(records, labels))
        ),
        "last_slot_equals_labeled_ply_board": mismatches == 0,
        "not_empty": n > 0,
    }
    if kind == "filtered":
        checks["filter_rules_hold"] = all(
            float(r["weak_confidence"]) >= cfg["min_confidence"]
            and abs(int(r["material_balance"])) <= cfg["max_material"]
            for r in records
        ) and all(c <= cfg["max_per_game"] for c in per_game.values())
    else:
        plies: Dict[str, List[int]] = {}
        for r in records:
            plies.setdefault(r["game_id"], []).append(int(r["ply_index"]))
        checks["every_window_from_start_ply"] = all(
            sorted(p) == list(range(cfg["start_ply"], cfg["start_ply"] + len(p)))
            for p in plies.values()
        ) and all(
            len(plies.get(g["game_id"], [])) == max(0, int(g["n_plies"]) - cfg["start_ply"] + 1)
            for g in games
        )
    counts = Counter(r["label_class"] for r in records)
    return {
        "checks": checks,
        "windows": n,
        "games": len(games),
        "class_counts": dict(counts),
        "class_proportions": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "last_slot_windows_checked": checked,
    }


def _collect(
    results: Iterator[Optional[Dict[str, Any]]],
    writers: Dict[str, Any],
    max_games: int,
    log: Callable[[str], None],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Stream the first ``max_games`` unseen unique games into both writers."""
    game_rows: List[Dict[str, Any]] = []
    stats = {"scanned": 0, "in_80k": 0, "within": 0}
    seen: Set[str] = set()
    base_seen: Counter = Counter()
    for result in results:
        stats["scanned"] += 1
        if result is None:
            continue
        if result["excluded"]:
            stats["in_80k"] += 1
            continue
        if result["move_hash"] in seen:
            stats["within"] += 1
            continue
        seen.add(result["move_hash"])
        base = result["base_id"]
        base_seen[base] += 1
        unique = f"holdout:{base}" + (f"~{base_seen[base]}" if base_seen[base] > 1 else "")
        rows = result["rows"]
        for row in rows:
            row.update(
                game_id=unique,
                base_game_id=base,
                window_id=f"{unique}#{row['ply_index']}",
                split="test",
            )
        keep = result["keep"]
        writers["full"].add(result["tensors"], result["masks"], rows)
        writers["filtered"].add(
            result["tensors"][keep], result["masks"][keep], [rows[i] for i in keep]
        )
        game_rows.append(
            {
                "game_id": unique,
                "base_game_id": base,
                "split": "test",
                "move_hash": result["move_hash"],
                "n_plies": result["n_plies"],
                "n_raw": len(rows),
                "n_kept": len(keep),
            }
        )
        if len(game_rows) % 500 == 0:
            log(f"  {len(game_rows)}/{max_games} holdout games, {writers['full'].n} full windows")
        if len(game_rows) >= max_games:
            break
    return game_rows, stats


def _skip_and_excluded(
    exclude_dir: Path, meta80: Dict[str, Any], after_holdout: Optional[str | Path]
) -> Tuple[int, Set[str]]:
    """Raw games to skip and movetext hashes that must not appear in the holdout."""
    skip = int(meta80["games_scanned"])
    excluded = {g["move_hash"] for g in bsd._read_csv(exclude_dir / "games.csv")}
    if after_holdout is not None:
        used = json.loads((Path(after_holdout) / "meta.json").read_text(encoding="utf-8"))
        if int(used["raw_games_skipped"]) != skip:
            raise ValueError(f"{after_holdout} was not built after {exclude_dir}.")
        skip += int(used["raw_games_scanned_after_skip"])
        excluded |= {g["move_hash"] for g in bsd._read_csv(Path(after_holdout) / "games.csv")}
    return skip, excluded


def build_fresh_holdout(
    exclude_dir: str | Path = DEFAULT_EXCLUDE,
    out_filtered: str | Path = DEFAULT_FILTERED,
    out_full: str | Path = DEFAULT_FULL,
    source: str | Path = bsd.DEFAULT_ARCHIVE,
    max_games: int = 6000,
    workers: int = bsd.MAX_WORKERS,
    chunk_games: int = 10,
    log: Callable[[str], None] = print,
    after_holdout: Optional[str | Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build both holdout directories; returns ``{kind: meta}``.

    ``after_holdout`` names an already-used holdout directory: its games are
    skipped and excluded too, so the new holdout has never been looked at.
    """
    exclude_dir, source = Path(exclude_dir), Path(source)
    if (
        after_holdout is None
        and exclude_dir.resolve() == DEFAULT_EXCLUDE.resolve()
        and (DEFAULT_FULL / "meta.json").exists()
        and Path(out_full).resolve() != DEFAULT_FULL.resolve()
    ):
        raise ValueError(
            "a used holdout already exists; pass after_holdout so its games are not reused."
        )
    outs = {"filtered": Path(out_filtered), "full": Path(out_full)}
    for out in outs.values():
        if out.exists():
            raise FileExistsError(f"{out} already exists; refusing to overwrite a holdout.")
    meta80 = json.loads((exclude_dir / "meta.json").read_text(encoding="utf-8"))
    if Path(str(meta80["source"]).replace("\\", "/")).name != source.name:
        raise ValueError(
            f"{exclude_dir} was built from {meta80['source']}, not {source.name}; "
            "skipping its scanned games would be meaningless."
        )
    skip, excluded = _skip_and_excluded(exclude_dir, meta80, after_holdout)
    flt = meta80["filter"]
    cfg = bsd._default_cfg(
        min_confidence=flt["min_confidence"],
        max_material=flt["max_material"],
        max_per_game=flt["max_per_game"],
        start_ply=flt["start_ply"],
        history_length=flt["history_length"],
    )
    workers = max(1, min(int(workers), bsd.MAX_WORKERS))

    tmps = {k: bsd._prepare_tmp(outs[k], force=False) for k in KINDS}
    writers = {k: bsd._SplitWriter(tmps[k], "test") for k in KINDS}
    start = time.time()
    results = _iter_results(source, skip, workers, chunk_games, cfg, excluded)
    try:
        game_rows, stats = _collect(results, writers, max_games, log)
    finally:
        results.close()
        for writer in writers.values():
            writer.close()
    valid = len(game_rows)
    if valid == 0:
        raise ValueError(f"No fresh valid games after skipping {skip} raw games.")

    metas: Dict[str, Dict[str, Any]] = {}
    for kind in KINDS:
        bsd._finalize_arrays(tmps[kind], writers[kind])
        with (tmps[kind] / "games.csv").open("w", newline="", encoding="utf-8") as handle:
            table = bsd.csv.DictWriter(handle, fieldnames=bsd.GAME_FIELDS)
            table.writeheader()
            for row in game_rows:
                table.writerow({**row, "n_kept": row["n_kept" if kind == "filtered" else "n_raw"]})
        integrity = holdout_checks(tmps[kind], kind, excluded, cfg)
        metas[kind] = {
            "kind": f"holdout_{kind}",
            "source": str(source),
            "excluded_dataset": str(exclude_dir),
            "after_holdout": None if after_holdout is None else str(after_holdout),
            "raw_games_skipped": skip,
            "raw_games_scanned_after_skip": stats["scanned"],
            "games_requested": max_games,
            "games_valid": valid,
            "dropped_movetext_in_80k": stats["in_80k"],
            "dropped_duplicates_within_holdout": stats["within"],
            "filter": flt if kind == "filtered" else {"start_ply": cfg["start_ply"]},
            "labels": "rule-engine weak labels (look-ahead derived)",
            "windows": writers[kind].n,
            "build_seconds": round(time.time() - start, 1),
            "integrity": integrity,
        }
        (tmps[kind] / "meta.json").write_text(json.dumps(metas[kind], indent=2), encoding="utf-8")

    failed = [
        f"{k}:{name}"
        for k, meta in metas.items()
        for name, ok in meta["integrity"]["checks"].items()
        if not ok
    ]
    if failed:
        raise AssertionError(f"Holdout integrity checks FAILED: {failed} (kept in .tmp dirs)")
    for kind in KINDS:  # "full" last: its meta.json marks the whole step complete
        if outs[kind].exists():
            shutil.rmtree(outs[kind])
        tmps[kind].rename(outs[kind])
        log(f"Wrote {outs[kind]} ({metas[kind]['windows']} windows, {valid} games)")
    return metas


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the fresh never-seen holdout.")
    ap.add_argument("--exclude-dir", default=str(DEFAULT_EXCLUDE))
    ap.add_argument("--out-filtered", default=str(DEFAULT_FILTERED))
    ap.add_argument("--out-full", default=str(DEFAULT_FULL))
    ap.add_argument("--source", default=str(bsd.DEFAULT_ARCHIVE))
    ap.add_argument("--max-games", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=bsd.MAX_WORKERS)
    ap.add_argument("--after-holdout", default=None, help="Used holdout dir to skip past.")
    args = ap.parse_args(argv)
    metas = build_fresh_holdout(
        args.exclude_dir,
        args.out_filtered,
        args.out_full,
        args.source,
        max_games=args.max_games,
        workers=args.workers,
        after_holdout=args.after_holdout,
    )
    print(json.dumps(metas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
