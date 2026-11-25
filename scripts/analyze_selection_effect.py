"""Does confidence-based class balancing inflate train recall (selection effect)?

``balance_classes`` keeps the MOST confident windows of any class it caps
(NO_PLAN, OUTPOST). If those kept windows are systematically easier than the
discarded ones, train recall for those classes overstates how well the model fits
typical windows, and the train/val recall gap is partly a selection effect, not
memorization. This compares model recall on KEPT vs DISCARDED train windows.

TRAIN split and checkpoint only; validation/test are never opened.

Usage::

    python scripts/analyze_selection_effect.py --seed 42 --tag _80k \
        --data-dir data/processed/scaled_80k
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from final_training import CLASS_NAMES, VARIANTS, _build_model, _ckpt_path  # noqa: E402

from src.data.weak_labeler import LabeledPositionRecord, balance_classes  # noqa: E402
from src.models.logger import load_checkpoint  # noqa: E402

KEEP_CHANNELS = 56


def recall_on_rows(
    model: torch.nn.Module, tensors: np.ndarray, rows: np.ndarray, label: int, batch: int = 512
) -> float:
    """Fraction of ``rows`` (all true class ``label``) the model predicts as ``label``."""
    hits = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = torch.from_numpy(np.asarray(tensors[np.sort(rows[i : i + batch])], np.float32))
            hits += int((model(x[:, -KEEP_CHANNELS:]).argmax(-1) == label).sum().item())
    return hits / max(len(rows), 1)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kept vs discarded train-window recall.")
    ap.add_argument("--variant", default="C_4ply_mirror_reg", choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--per-group", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.data_dir)
    with open(root / "train_records.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
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
    kept_ids = {r.window_id for r in balance_classes(records, max_class_fraction=0.30, seed=42)}
    row_of = {r["window_id"]: i for i, r in enumerate(rows)}

    model = _build_model(VARIANTS[args.variant])
    load_checkpoint(_ckpt_path(args.variant, args.seed, args.tag), model)
    model.eval()
    torch.set_num_threads(8)
    tensors = np.load(root / "train_tensors.npy", mmap_mode="r")
    rng = np.random.RandomState(0)

    result: Dict[str, Dict[str, float]] = {}
    for cls in ("NO_PLAN", "OUTPOST"):
        lab = CLASS_NAMES.index(cls)
        kept = np.array(
            [
                row_of[r.window_id]
                for r in records
                if r.label_class == cls and r.window_id in kept_ids
            ]
        )
        gone = np.array(
            [
                row_of[r.window_id]
                for r in records
                if r.label_class == cls and r.window_id not in kept_ids
            ]
        )
        conf = {r.window_id: r.weak_confidence for r in records}
        id_of = {i: w for w, i in row_of.items()}
        n = min(args.per_group, len(kept), len(gone))
        kept_s = rng.choice(kept, n, replace=False)
        gone_s = rng.choice(gone, n, replace=False)
        result[cls] = {
            "n_kept_total": int(len(kept)),
            "n_discarded_total": int(len(gone)),
            "sample_per_group": int(n),
            "mean_weak_confidence_kept": float(np.mean([conf[id_of[i]] for i in kept_s])),
            "mean_weak_confidence_discarded": float(np.mean([conf[id_of[i]] for i in gone_s])),
            "recall_kept": recall_on_rows(model, tensors, kept_s, lab),
            "recall_discarded": recall_on_rows(model, tensors, gone_s, lab),
        }
        print(cls, json.dumps(result[cls], indent=2), flush=True)

    out = Path(
        args.out
        or _REPO_ROOT / "experiments" / f"analysis_selection_effect{args.tag}_s{args.seed}.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
