"""Cohen's Kappa inter-rater reliability and adjudication (Task 1.9, Subtask 1.9.4).

Loads the two annotators' sheets, computes observed agreement and Cohen's Kappa,
lists disagreements for a joint adjudication session, and — once disputes are
resolved — freezes the consensus ``test_ground_truth.csv``.

CLI::

    python -m src.models.adjudicate --member1 ... --member2 ...

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import cohen_kappa_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VALID_LABELS = {"K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"}

__all__ = [
    "load_annotations",
    "observed_agreement",
    "kappa",
    "compute_reliability",
    "find_disagreements",
    "build_ground_truth",
    "VALID_LABELS",
]


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Load an annotation sheet, validating window ids and label strings."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "window_id" not in frame or "primary_label" not in frame:
        raise ValueError(f"{path}: missing required columns window_id/primary_label.")
    bad = set(frame["primary_label"]) - VALID_LABELS
    if bad:
        raise ValueError(f"{path}: invalid label(s) {sorted(bad)}; allowed {sorted(VALID_LABELS)}.")
    return frame


def _aligned_labels(m1: pd.DataFrame, m2: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Return (window_ids, labels1, labels2) aligned on common window ids."""
    l1 = dict(zip(m1["window_id"], m1["primary_label"]))
    l2 = dict(zip(m2["window_id"], m2["primary_label"]))
    common = [w for w in m1["window_id"] if w in l2]
    if not common:
        raise ValueError("no overlapping window_ids between the two sheets.")
    return common, [l1[w] for w in common], [l2[w] for w in common]


def observed_agreement(labels1: List[str], labels2: List[str]) -> float:
    """Fraction of items where the two annotators chose the same label."""
    if not labels1:
        return 0.0
    same = sum(1 for a, b in zip(labels1, labels2) if a == b)
    return same / len(labels1)


def kappa(labels1: List[str], labels2: List[str]) -> float:
    """Cohen's Kappa; 1.0 for identical labelings, by definition."""
    if labels1 == labels2:
        return 1.0
    return float(cohen_kappa_score(labels1, labels2, labels=sorted(VALID_LABELS)))


def compute_reliability(m1: pd.DataFrame, m2: pd.DataFrame) -> Dict:
    """Observed agreement, Kappa and basic counts for the two sheets."""
    ids, labels1, labels2 = _aligned_labels(m1, m2)
    p_o = observed_agreement(labels1, labels2)
    k = kappa(labels1, labels2)
    return {
        "n_items": len(ids),
        "observed_agreement": round(p_o, 4),
        "cohen_kappa": round(k, 4),
        "substantial_agreement": k >= 0.65,
        "n_disagreements": sum(1 for a, b in zip(labels1, labels2) if a != b),
    }


def find_disagreements(m1: pd.DataFrame, m2: pd.DataFrame) -> pd.DataFrame:
    """Rows where the annotators chose different primary labels."""
    ids, labels1, labels2 = _aligned_labels(m1, m2)
    context = m1.set_index("window_id")
    rows = []
    for w, a, b in zip(ids, labels1, labels2):
        if a != b:
            ctx = context.loc[w]
            rows.append(
                {
                    "window_id": w,
                    "game_id": ctx.get("game_id", ""),
                    "ply_index": ctx.get("ply_index", ""),
                    "fen_current": ctx.get("fen_current", ""),
                    "history_moves_san": ctx.get("history_moves_san", ""),
                    "member1_label": a,
                    "member2_label": b,
                    "adjudicated_label": "",  # filled during the consensus meeting
                }
            )
    return pd.DataFrame(rows)


def build_ground_truth(
    m1: pd.DataFrame,
    m2: pd.DataFrame,
    resolved: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Freeze consensus labels: agreements auto-fill, disputes use ``resolved``.

    Raises if a disagreement has no resolution, so the benchmark is never
    silently guessed.
    """
    ids, labels1, labels2 = _aligned_labels(m1, m2)
    resolved = resolved or {}
    context = m1.set_index("window_id")
    rows = []
    unresolved = []
    for w, a, b in zip(ids, labels1, labels2):
        if a == b:
            label = a
        elif w in resolved:
            label = resolved[w]
            if label not in VALID_LABELS:
                raise ValueError(f"resolved label for {w} is invalid: {label}.")
        else:
            unresolved.append(w)
            continue
        ctx = context.loc[w]
        rows.append(
            {
                "window_id": w,
                "game_id": ctx.get("game_id", ""),
                "ply_index": ctx.get("ply_index", ""),
                "turn": ctx.get("turn", ""),
                "fen_current": ctx.get("fen_current", ""),
                "label": label,
            }
        )
    if unresolved:
        raise ValueError(f"{len(unresolved)} disagreement(s) unresolved, e.g. {unresolved[:3]}.")
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ann = _REPO_ROOT / "data" / "annotations"
    ap = argparse.ArgumentParser(description="Compute Cohen's Kappa and list disagreements.")
    ap.add_argument("--member1", default=str(ann / "raw_annotations_member1.csv"))
    ap.add_argument("--member2", default=str(ann / "raw_annotations_member2.csv"))
    ap.add_argument("--resolved", default=None, help="CSV of adjudicated labels (window_id,label).")
    args = ap.parse_args(argv)

    m1 = load_annotations(args.member1)
    m2 = load_annotations(args.member2)

    reliability = compute_reliability(m1, m2)
    print(json.dumps(reliability, indent=2))
    (_REPO_ROOT / "experiments").mkdir(exist_ok=True)
    with (_REPO_ROOT / "experiments" / "inter_rater_reliability.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(reliability, fh, indent=2)

    disagreements = find_disagreements(m1, m2)
    disagreements.to_csv(ann / "disagreements.csv", index=False)
    print(f"Wrote {len(disagreements)} disagreement(s) to disagreements.csv")

    resolved = None
    if args.resolved and Path(args.resolved).is_file():
        rdf = pd.read_csv(args.resolved, dtype=str)
        resolved = dict(zip(rdf["window_id"], rdf["label"]))
    try:
        truth = build_ground_truth(m1, m2, resolved=resolved)
        truth.to_csv(ann / "test_ground_truth.csv", index=False)
        print(f"Froze test_ground_truth.csv with {len(truth)} labels.")
    except ValueError as exc:
        print(f"Ground truth not frozen yet: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
