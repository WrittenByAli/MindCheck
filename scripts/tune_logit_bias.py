"""Tune per-class logit biases on VALIDATION to undo double class-compensation.

Training data is class-balanced AND the loss is class-weighted, which pushes the
model to over-predict minority classes (high recall, low precision). A fixed
additive bias per class on the logits can correct this without retraining.

Biases are fit by coordinate ascent on validation macro-F1. To check they are
not just memorizing validation, they are fit on one half of the validation
GAMES and scored on the other half (and vice versa). The test split is never
loaded.

Usage::

    python scripts/tune_logit_bias.py --variant C_4ply_mirror_reg --seed 42 \
        --tag _80k --data-dir data/processed/scaled_80k
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
from build_scaled_dataset import load_scaled_dataset  # noqa: E402
from final_training import CLASS_NAMES, VARIANTS, _build_model, _ckpt_path, _slice  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.models.logger import load_checkpoint  # noqa: E402

NUM_CLASSES = len(CLASS_NAMES)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over all classes (a class never seen or predicted scores 0).

    Same value as sklearn's ``f1_score(..., zero_division=0)``, computed from a
    confusion matrix because the bias search calls this thousands of times.
    """
    cells = np.asarray(y, np.int64) * NUM_CLASSES + np.asarray(pred, np.int64)
    cm = np.bincount(cells, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(NUM_CLASSES), where=denom > 0)
    return float(f1.mean())


def fit_bias(logits: np.ndarray, y: np.ndarray, passes: int = 3) -> np.ndarray:
    """Coordinate ascent on per-class additive biases to maximize macro-F1."""
    bias = np.zeros(NUM_CLASSES)
    best = macro_f1(y, (logits + bias).argmax(1))
    grid = np.arange(-3.0, 3.01, 0.25)
    for _ in range(passes):
        for c in range(NUM_CLASSES):
            keep = bias[c]
            for g in grid:
                bias[c] = g
                score = macro_f1(y, (logits + bias).argmax(1))
                if score > best + 1e-9:
                    best, keep = score, g
            bias[c] = keep
    return bias - bias.mean()


def val_logits(variant: str, seed: int, tag: str, data_dir: str):
    datasets = load_scaled_dataset(data_dir, balance_train=False, seed=42, splits=("val",))
    cfg = VARIANTS[variant]
    model = _build_model(cfg)
    load_checkpoint(_ckpt_path(variant, seed, tag), model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    torch.set_num_threads(8)
    outs: List[np.ndarray] = []
    labels: List[int] = []
    with torch.no_grad():
        loader = DataLoader(_slice(datasets["val"], cfg["keep"]), batch_size=1024)
        for x, _m, y in loader:
            outs.append(model(x.to(device)).cpu().numpy())
            labels.extend(y.tolist())
    return np.concatenate(outs), np.asarray(labels)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tune logit biases on validation only.")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    logits, y = val_logits(args.variant, args.seed, args.tag, args.data_dir)
    with open(Path(args.data_dir) / "val_records.csv", newline="", encoding="utf-8") as fh:
        game_ids = [r["game_id"] for r in csv.DictReader(fh)]
    assert len(game_ids) == len(y), "val records do not match val arrays"

    games = sorted(set(game_ids))
    rng = np.random.RandomState(0)
    rng.shuffle(games)
    half_a = set(games[: len(games) // 2])
    in_a = np.array([g in half_a for g in game_ids])

    base = macro_f1(y, logits.argmax(1))
    folds: Dict[str, Dict[str, float]] = {}
    for name, fit_mask in (("fit_A_score_B", in_a), ("fit_B_score_A", ~in_a)):
        bias = fit_bias(logits[fit_mask], y[fit_mask])
        held = ~fit_mask
        folds[name] = {
            "heldout_f1_before": macro_f1(y[held], logits[held].argmax(1)),
            "heldout_f1_after": macro_f1(y[held], (logits[held] + bias).argmax(1)),
        }
    full_bias = fit_bias(logits, y)
    after = (logits + full_bias).argmax(1)
    per_before = f1_score(y, logits.argmax(1), labels=list(range(NUM_CLASSES)), average=None)
    per_after = f1_score(y, after, labels=list(range(NUM_CLASSES)), average=None)

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "tag": args.tag,
        "val_macro_f1_before": base,
        "val_macro_f1_after_full_fit": macro_f1(y, after),
        "val_acc_before": float((logits.argmax(1) == y).mean()),
        "val_acc_after": float((after == y).mean()),
        "bias": {n: float(b) for n, b in zip(CLASS_NAMES, full_bias)},
        "per_class_f1_before": {n: float(v) for n, v in zip(CLASS_NAMES, per_before)},
        "per_class_f1_after": {n: float(v) for n, v in zip(CLASS_NAMES, per_after)},
        "cross_fold_check": folds,
    }
    out = Path(
        args.out
        or _REPO_ROOT / "experiments" / f"logit_bias{args.tag}_{args.variant}_s{args.seed}.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
