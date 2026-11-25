"""Fair train-vs-validation gap for a trained checkpoint (no test data).

Accuracy measured on the class-balanced training set is not comparable to
validation accuracy (different class mix, and confidence-based balancing keeps
only easy windows). This reports two prior-independent gaps:

* per-class RECALL on windows the model actually trained on vs validation recall
  (recall does not depend on how many windows of each class there are);
* macro-F1 and accuracy on trained-on windows resampled to validation's class
  mix vs validation, i.e. the same class mix on both sides.

Usage::

    python scripts/measure_gap.py --variant C_4ply_mirror_reg --seed 42 \
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
from final_training import CLASS_NAMES, VARIANTS, _build_model, _ckpt_path  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

from src.data.weak_labeler import LabeledPositionRecord, balance_classes  # noqa: E402
from src.models.logger import load_checkpoint  # noqa: E402

LABELS = list(range(len(CLASS_NAMES)))


def predict(model, tensors: np.ndarray, rows: np.ndarray, keep: int, batch: int = 1024):
    order = np.sort(rows)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(order), batch):
            x = torch.from_numpy(np.asarray(tensors[order[i : i + batch]], np.float32))
            preds.append(model(x[:, -keep:]).argmax(-1).numpy())
    return order, np.concatenate(preds)


def val_mix_rows(
    trained_by_class: List[np.ndarray], val_share: np.ndarray, n: int, rng: np.random.RandomState
) -> np.ndarray:
    """Rows the model TRAINED on, resampled to validation's class mix.

    ``n`` is scaled down if some class has too few trained rows, so the mix is
    always preserved and nothing is drawn twice.
    """
    wanted = n * np.asarray(val_share, dtype=float)
    scale = min([1.0] + [len(rows) / w for rows, w in zip(trained_by_class, wanted) if w > 0])
    picked: List[int] = []
    for rows, w in zip(trained_by_class, wanted):
        take = min(len(rows), int(round(w * scale)))
        picked.extend(rng.choice(rows, take, replace=False).tolist() if take else [])
    return np.array(picked, dtype=np.int64)


def summarize(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    return {
        "n": int(len(y)),
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        "recall": {
            n: float(r)
            for n, r in zip(
                CLASS_NAMES, recall_score(y, p, labels=LABELS, average=None, zero_division=0)
            )
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prior-independent train/validation gap.")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--per-class", type=int, default=4000, help="Trained-on windows per class.")
    ap.add_argument("--natural", type=int, default=40000, help="Natural train / val sample size.")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = VARIANTS[args.variant]
    root = Path(args.data_dir)
    torch.set_num_threads(args.threads)
    model = _build_model(cfg)
    load_checkpoint(_ckpt_path(args.variant, args.seed, args.tag), model)
    model.eval()
    rng = np.random.RandomState(0)

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
    row_of = {r["window_id"]: i for i, r in enumerate(rows)}
    if cfg.get("rows"):  # the variant trained on these row kinds only
        records = [rec for rec, row in zip(records, rows) if row.get("sample_kind") in cfg["rows"]]
    trained = balance_classes(
        records, max_class_fraction=0.30, seed=42, select=cfg.get("balance", "confidence")
    )
    train_labels = np.load(root / "train_labels.npy")
    train_tensors = np.load(root / "train_tensors.npy", mmap_mode="r")

    trained_by_class = [
        np.array([row_of[r.window_id] for r in trained if r.label_class == name], dtype=np.int64)
        for name in CLASS_NAMES
    ]
    picked: List[int] = []
    for idx in trained_by_class:
        picked.extend(rng.choice(idx, min(args.per_class, len(idx)), replace=False).tolist())
    order, pred = predict(model, train_tensors, np.array(picked), cfg["keep"])
    trained_on = summarize(train_labels[order], pred)

    val_labels = np.load(root / "val_labels.npy")
    # Same class mix on both sides, using only rows the model actually trained on.
    # (Sampling all cached train rows would include rows that balancing discarded.)
    val_share = np.bincount(val_labels, minlength=len(CLASS_NAMES)) / len(val_labels)
    nat = val_mix_rows(trained_by_class, val_share, args.natural, rng)
    order, pred = predict(model, train_tensors, nat, cfg["keep"])
    train_natural = summarize(train_labels[order], pred)

    val_tensors = np.load(root / "val_tensors.npy", mmap_mode="r")
    vs = rng.choice(len(val_labels), min(args.natural, len(val_labels)), replace=False)
    order, pred = predict(model, val_tensors, vs, cfg["keep"])
    val = summarize(val_labels[order], pred)

    gaps = {n: trained_on["recall"][n] - val["recall"][n] for n in CLASS_NAMES}
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "trained_on_sample": trained_on,
        "train_natural_sample": train_natural,
        "val_sample": val,
        "recall_gap_trained_on_minus_val": gaps,
        "mean_recall_gap": float(np.mean(list(gaps.values()))),
        "macro_f1_gap_natural": train_natural["macro_f1"] - val["macro_f1"],
        "accuracy_gap_natural": train_natural["accuracy"] - val["accuracy"],
    }
    out = Path(
        args.out or _REPO_ROOT / "experiments" / f"gap{args.tag}_{args.variant}_s{args.seed}.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if "sample" not in k}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
