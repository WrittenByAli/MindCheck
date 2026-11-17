"""Interactive adjudication helper for the consensus meeting (Task 1.9.4).

Steps through ``data/annotations/disagreements.csv`` one disputed window at a
time, shows both annotators' labels and the position context, and records the
agreed label into a resolved CSV. Feed that file back to
``src.models.adjudicate`` (``--resolved``) to freeze the ground truth.

Run interactively::

    python scripts/adjudicate_cli.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.adjudicate import VALID_LABELS  # noqa: E402

_ANN = _REPO_ROOT / "data" / "annotations"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step through disputed annotations.")
    ap.add_argument("--disagreements", default=str(_ANN / "disagreements.csv"))
    ap.add_argument("--out", default=str(_ANN / "resolved.csv"))
    args = ap.parse_args(argv)

    if not sys.stdin.isatty():
        print("This is an interactive tool; run it in a real terminal.", file=sys.stderr)
        return 1

    disputes = pd.read_csv(args.disagreements, dtype=str).fillna("")
    allowed = sorted(VALID_LABELS)
    print(f"{len(disputes)} disputes. Allowed labels: {', '.join(allowed)}\n")

    resolved = []
    for i, row in disputes.iterrows():
        print(
            f"[{i + 1}/{len(disputes)}] {row['window_id']}  (game {row['game_id']}, ply {row['ply_index']})"
        )
        print(f"  history: {row['history_moves_san']}")
        print(f"  FEN:     {row['fen_current']}")
        print(f"  member1={row['member1_label']}   member2={row['member2_label']}")
        while True:
            choice = input("  agreed label (or 's' to skip): ").strip()
            if choice.lower() == "s":
                break
            if choice in VALID_LABELS:
                resolved.append({"window_id": row["window_id"], "label": choice})
                break
            print(f"  '{choice}' is not a valid label.")
        print()

    pd.DataFrame(resolved).to_csv(args.out, index=False)
    print(f"Wrote {len(resolved)} resolved label(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
