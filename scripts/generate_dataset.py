"""End-to-end weak-labeled dataset builder (Task 1.5).

Streams the raw PGN, weak-labels every history window via look-ahead execution
confirmation, filters/balances the labels, performs a leak-free game-level
70/15/15 split, writes ``train/val/test_records.csv`` plus a
``dataset_summary.json`` audit, and (optionally) a class-distribution chart.

Example::

    python scripts/generate_dataset.py --pgn data/raw/lichess_elite_subset.pgn
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.audit import plot_class_distribution, write_summary  # noqa: E402
from src.data.dataset_splitter import GameLevelSplitter  # noqa: E402
from src.data.pgn_parser import StreamingPGNParser  # noqa: E402
from src.data.weak_labeler import (  # noqa: E402
    RetrospectiveWeakLabeler,
    default_game_id,
    filter_and_balance,
)

DEFAULT_PGN = _REPO_ROOT / "data" / "raw" / "lichess_elite_subset.pgn"
DEFAULT_OUT = _REPO_ROOT / "data" / "processed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the weak-labeled MindCheck dataset.")
    ap.add_argument("--pgn", default=str(DEFAULT_PGN), help="Input PGN (.pgn/.zip/.zst).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Output directory.")
    ap.add_argument("--max-games", type=int, default=None, help="Limit games processed.")
    ap.add_argument("--seed", type=int, default=42, help="Split/balancing seed.")
    ap.add_argument("--no-plot", action="store_true", help="Skip the distribution chart.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = StreamingPGNParser()
    labeler = RetrospectiveWeakLabeler()

    print(f"Labeling games from {args.pgn} ...")
    records = []
    for i, game in enumerate(parser.stream_games(args.pgn, max_games=args.max_games), start=1):
        records.extend(labeler.label_game(game, game_id=default_game_id(game)))
        if i % 200 == 0:
            print(f"  ... {i} games, {len(records)} raw records")
    print(f"Raw records: {len(records)}")

    balanced = filter_and_balance(records, seed=args.seed)
    print(f"After filter/balance: {len(balanced)}")
    if not balanced:
        print("No records survived filtering; aborting.", file=sys.stderr)
        return 1

    GameLevelSplitter(seed=args.seed).split(balanced)

    frame = pd.DataFrame(asdict(r) for r in balanced)
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        subset.to_csv(out_dir / f"{split}_records.csv", index=False)
        print(f"  {split}: {len(subset)} records -> {split}_records.csv")

    summary = write_summary(balanced, out_dir / "dataset_summary.json")
    print(f"Disjoint splits: {summary['game_overlap']['disjoint']}")
    print(
        f"Wrote dataset_summary.json ({summary['total_games']} games, {summary['total_windows']} windows)."
    )

    if not args.no_plot:
        try:
            plot_class_distribution(
                balanced, _REPO_ROOT / "experiments" / "plots" / "dataset_class_distribution.png"
            )
            print("Wrote experiments/plots/dataset_class_distribution.png")
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"(skipped chart: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
