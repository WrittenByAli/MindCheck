"""Sample held-out test windows for human annotation (Task 1.9, Subtask 1.9.2).

Draws exactly ``n`` distinct position windows from the **test** game split only
(the same deterministic game-level split used everywhere else), stratified
across game phases, and writes two identical blank annotation sheets — one per
annotator. Weak labels and model predictions are deliberately absent so the
human ground truth is unbiased.

CLI::

    python -m src.data.sample_test_set --n 250

Subtask 1.9.2 is nominally owned by Muhammad Ali; implemented here so Task 1.9
is self-contained.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import chess
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset_splitter import GameLevelSplitter  # noqa: E402
from src.data.pgn_parser import StreamingPGNParser  # noqa: E402
from src.data.window_extractor import default_game_id  # noqa: E402

# Blank columns the annotators fill; the rest are read-only context.
SHEET_COLUMNS = [
    "window_id",
    "game_id",
    "ply_index",
    "turn",
    "fen_current",
    "history_moves_san",
    "primary_label",
    "confidence",
    "secondary_label",
    "notes",
]

# Phase bands by ply index (per the subtask spec).
_PHASES = {"opening": range(15, 26), "middlegame": range(26, 56)}  # "late" is 56+


def _phase(ply: int) -> Optional[str]:
    if ply in _PHASES["opening"]:
        return "opening"
    if ply in _PHASES["middlegame"]:
        return "middlegame"
    if ply >= 56:
        return "late"
    return None  # plies < 15 are excluded from annotation


def collect_test_candidates(
    pgn_path: str | Path,
    split_seed: int = 42,
    history_length: int = 8,
) -> List[Dict]:
    """Gather every annotation-ready window from every test-split game.

    All phases are collected (no per-game cap here); diversity and phase
    balance are handled in :func:`stratified_sample`. Window ids are made
    globally unique so header-derived game-id collisions never merge two
    distinct positions into one row.
    """
    parser = StreamingPGNParser()
    splitter = GameLevelSplitter(seed=split_seed)
    candidates: List[Dict] = []
    used_ids: set = set()

    for game in parser.stream_games(pgn_path):
        gid = default_game_id(game)
        if splitter.assign_split(gid) != "test":
            continue

        board = game.board()
        boards = [board.copy(stack=False)]
        sans: List[str] = []
        for move in game.mainline_moves():
            sans.append(board.san(move))
            board.push(move)
            boards.append(board.copy(stack=False))

        total_plies = len(boards) - 1
        for t in range(15, total_plies + 1):
            phase = _phase(t)
            if phase is None:
                continue
            window_id = f"{gid}#{t}"
            if window_id in used_ids:
                k = 2
                while f"{window_id}#{k}" in used_ids:
                    k += 1
                window_id = f"{window_id}#{k}"
            used_ids.add(window_id)

            current = boards[t]
            history = sans[max(0, t - history_length) : t]
            candidates.append(
                {
                    "window_id": window_id,
                    "game_id": gid,
                    "ply_index": t,
                    "turn": "white" if current.turn == chess.WHITE else "black",
                    "fen_current": current.fen(),
                    "history_moves_san": " ".join(history),
                    "phase": phase,
                    "primary_label": "",
                    "confidence": "",
                    "secondary_label": "",
                    "notes": "",
                }
            )
    return candidates


def stratified_sample(
    candidates: List[Dict], n: int, seed: int = 42, max_per_game: int = 3
) -> List[Dict]:
    """Sample ``n`` rows balanced across phases, capping windows per game."""
    rng = random.Random(seed)
    by_phase: Dict[str, List[Dict]] = {"opening": [], "middlegame": [], "late": []}
    for row in candidates:
        by_phase[row["phase"]].append(row)

    phases = [p for p in by_phase if by_phase[p]]
    if not phases:
        return []

    target = {p: n // len(phases) for p in phases}
    for i in range(n - sum(target.values())):  # distribute the remainder
        target[phases[i % len(phases)]] += 1

    per_game: Dict[str, int] = {}

    def _draw(pool: List[Dict], want: int) -> List[Dict]:
        picked = []
        for row in pool:
            if len(picked) >= want:
                break
            if per_game.get(row["game_id"], 0) < max_per_game:
                picked.append(row)
                per_game[row["game_id"]] = per_game.get(row["game_id"], 0) + 1
        return picked

    chosen: List[Dict] = []
    for p in phases:
        rng.shuffle(by_phase[p])
        chosen.extend(_draw(by_phase[p], target[p]))

    # Backfill any shortfall (phase ran dry or per-game cap bit) from leftovers.
    if len(chosen) < n:
        chosen_ids = {r["window_id"] for r in chosen}
        leftovers = [r for p in phases for r in by_phase[p] if r["window_id"] not in chosen_ids]
        rng.shuffle(leftovers)
        for row in leftovers:
            if len(chosen) >= n:
                break
            chosen.append(row)

    rng.shuffle(chosen)
    return chosen[:n]


def write_sheets(rows: List[Dict], out_dir: str | Path) -> List[Path]:
    """Write identical blank annotation sheets for member1 and member2."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)[SHEET_COLUMNS]
    paths = []
    for member in ("member1", "member2"):
        path = out / f"raw_annotations_{member}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sample held-out test windows for annotation.")
    ap.add_argument("--pgn", default=str(_REPO_ROOT / "data" / "raw" / "lichess_elite_subset.pgn"))
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "data" / "annotations"))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print("Scanning test-split games ...")
    candidates = collect_test_candidates(args.pgn, split_seed=args.seed)
    print(f"  {len(candidates)} candidate windows across test games.")
    rows = stratified_sample(candidates, args.n, seed=args.seed)
    if len(rows) < args.n:
        print(f"WARNING: only {len(rows)} windows available (< {args.n}).", file=sys.stderr)
    paths = write_sheets(rows, args.out_dir)
    print(f"Wrote {len(rows)} rows to:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
