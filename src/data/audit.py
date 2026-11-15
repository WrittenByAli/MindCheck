"""Dataset statistical audit and summary manifest (Task 1.5, Subtask 1.5.4).

Summarizes a split, weakly-labeled dataset: split sizes, per-class counts and
proportions, rating/ply distributions, result breakdown, and — crucially — a
verification that no game id leaks across splits. Writes
``data/processed/dataset_summary.json`` and (optionally) a class-distribution
bar chart.

Subtask 1.5.4 is nominally owned by Muhammad Ali; implemented here so the
end-to-end ``generate_dataset.py`` pipeline is self-contained.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.data.weak_labeler import PLAN_IDS, LabeledPositionRecord

__all__ = ["compute_summary", "write_summary", "plot_class_distribution"]

_ALL_CLASSES = list(PLAN_IDS.keys())  # 5 plans + NO_PLAN
_SPLITS = ["train", "val", "test"]


def _class_counts(records: Iterable[LabeledPositionRecord]) -> Dict[str, int]:
    counts = {cls: 0 for cls in _ALL_CLASSES}
    for r in records:
        counts[r.label_class] = counts.get(r.label_class, 0) + 1
    return counts


def _proportions(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {cls: 0.0 for cls in counts}
    return {cls: round(n / total, 4) for cls, n in counts.items()}


def compute_summary(records: List[LabeledPositionRecord]) -> dict:
    """Compute the full dataset summary dict from labeled, split records."""
    by_split: Dict[str, List[LabeledPositionRecord]] = {s: [] for s in _SPLITS}
    for r in records:
        by_split.setdefault(r.split, []).append(r)

    games_by_split = {s: {r.game_id for r in recs} for s, recs in by_split.items()}

    split_summary = {}
    for s in _SPLITS:
        recs = by_split.get(s, [])
        counts = _class_counts(recs)
        split_summary[s] = {
            "games": len(games_by_split.get(s, set())),
            "windows": len(recs),
            "class_counts": counts,
            "class_proportions": _proportions(counts),
        }

    elos_white = [r.white_elo for r in records if r.white_elo is not None]
    elos_black = [r.black_elo for r in records if r.black_elo is not None]
    plies = [r.ply_index for r in records]
    results = collections.Counter(r.result for r in records)

    train, val, test = (games_by_split.get(s, set()) for s in _SPLITS)
    overlap = {
        "train_val": len(train & val),
        "train_test": len(train & test),
        "val_test": len(val & test),
    }
    overlap["disjoint"] = sum(overlap.values()) == 0

    return {
        "total_games": len({r.game_id for r in records}),
        "total_windows": len(records),
        "splits": split_summary,
        "overall_class_counts": _class_counts(records),
        "mean_white_elo": round(sum(elos_white) / len(elos_white), 1) if elos_white else None,
        "mean_black_elo": round(sum(elos_black) / len(elos_black), 1) if elos_black else None,
        "mean_ply_index": round(sum(plies) / len(plies), 2) if plies else None,
        "result_breakdown": dict(results),
        "game_overlap": overlap,
    }


def write_summary(records: List[LabeledPositionRecord], path: str | Path) -> dict:
    """Compute the summary and write it to ``path`` as JSON; return the dict."""
    summary = compute_summary(records)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def plot_class_distribution(records: List[LabeledPositionRecord], path: str | Path) -> None:
    """Save a per-split class-distribution bar chart (optional; needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counts_by_split = {s: _class_counts([r for r in records if r.split == s]) for s in _SPLITS}
    x = np.arange(len(_ALL_CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(_SPLITS):
        heights = [counts_by_split[s][c] for c in _ALL_CLASSES]
        ax.bar(x + (i - 1) * width, heights, width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(_ALL_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("window count")
    ax.set_title("MindCheck weak-label class distribution by split")
    ax.legend()
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
