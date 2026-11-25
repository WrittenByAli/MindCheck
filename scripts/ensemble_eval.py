"""Validation-only check: does averaging several seeds' predictions help?

Averages log-softmax outputs of the given seeds' checkpoints on VALIDATION,
reports raw macro-F1, and fits per-class biases with the same game-level
two-fold check as ``tune_logit_bias.py``. The test split is never opened.

Usage::

    python scripts/ensemble_eval.py --seeds 42 123 999 --tag _80k \
        --data-dir data/processed/scaled_80k
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from tune_logit_bias import fit_bias, macro_f1, val_logits  # noqa: E402


def log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Seed-ensemble evaluation on validation only.")
    ap.add_argument("--variant", default="C_4ply_mirror_reg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    members, y = [], None
    for seed in args.seeds:
        logits, labels = val_logits(args.variant, seed, args.tag, args.data_dir)
        members.append(log_softmax(logits))
        y = labels if y is None else y
        assert np.array_equal(y, labels), "validation order differs between seeds"
    ens = np.mean(members, axis=0)

    with open(Path(args.data_dir) / "val_records.csv", newline="", encoding="utf-8") as fh:
        game_ids = [r["game_id"] for r in csv.DictReader(fh)]
    games = sorted(set(game_ids))
    rng = np.random.RandomState(0)
    rng.shuffle(games)
    half_a = set(games[: len(games) // 2])
    in_a = np.array([g in half_a for g in game_ids])

    folds = {}
    for name, fit_mask in (("fit_A_score_B", in_a), ("fit_B_score_A", ~in_a)):
        bias = fit_bias(ens[fit_mask], y[fit_mask])
        held = ~fit_mask
        folds[name] = macro_f1(y[held], (ens[held] + bias).argmax(1))
    full_bias = fit_bias(ens, y)

    result = {
        "seeds": args.seeds,
        "member_val_macro_f1_raw": [macro_f1(y, m.argmax(1)) for m in members],
        "ensemble_val_macro_f1_raw": macro_f1(y, ens.argmax(1)),
        "ensemble_val_acc_raw": float((ens.argmax(1) == y).mean()),
        "ensemble_val_macro_f1_bias_heldout_folds": folds,
        "ensemble_val_macro_f1_bias_full_fit": macro_f1(y, (ens + full_bias).argmax(1)),
        "ensemble_val_acc_bias_full_fit": float(((ens + full_bias).argmax(1) == y).mean()),
        "bias": full_bias.tolist(),
    }
    out = Path(args.out or _REPO_ROOT / "experiments" / f"ensemble{args.tag}_{args.variant}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
