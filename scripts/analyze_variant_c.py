"""Post-training analysis of Variant C (80k data, seeds 42/123/999).

Uses TRAIN and VALIDATION data only. The test split is never loaded, indexed or
predicted on: this script builds its own train/val datasets instead of calling
``load_scaled_dataset`` (which opens the test labels array as well). The only
test-related input is the already-written ``experiments/final_80k_test.json``,
which is read (not recomputed) for one recorded table.

Outputs (never overwritten; the script refuses to run if any exists):
  experiments/analysis_variant_c.json
  docs/figures/variant_c_confusion_seed42.png
  docs/figures/variant_c_learning_curves.png

Usage::

    python scripts/analyze_variant_c.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tune_logit_bias as tb  # noqa: E402
from build_scaled_dataset import _read_csv, _RowView  # noqa: E402
from final_training import (  # noqa: E402
    CLASS_NAMES,
    VARIANTS,
    _build_model,
    _ckpt_path,
    _predict_logits,
    _slice,
    _View,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from src.data.chess_dataset import ChessPlanDataset  # noqa: E402
from src.data.weak_labeler import LabeledPositionRecord, balance_classes  # noqa: E402
from src.models.logger import load_checkpoint  # noqa: E402

VARIANT = "C_4ply_mirror_reg"
TAG = "_80k"
SEEDS = (42, 123, 999)
DATA_DIR = _REPO_ROOT / "data" / "processed" / "scaled_80k"
EXP = _REPO_ROOT / "experiments"
OUT_JSON = EXP / "analysis_variant_c.json"
OUT_CM = _REPO_ROOT / "docs" / "figures" / "variant_c_confusion_seed42.png"
OUT_LC = _REPO_ROOT / "docs" / "figures" / "variant_c_learning_curves.png"
N = len(CLASS_NAMES)
CHAIN = CLASS_NAMES.index("CHAIN_UNDERMINE")


# --------------------------------------------------------------------------- #
# Pure-logic helpers (unit-tested with synthetic arrays)                        #
# --------------------------------------------------------------------------- #
def confusion_summary(y: Sequence[int], pred: Sequence[int], n: int = N) -> Dict[str, Any]:
    """Confusion matrix (rows = true) as counts and as row-normalized fractions."""
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    rows = cm.sum(1, keepdims=True).astype(float)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    return {"counts": cm.tolist(), "row_normalized": norm.tolist()}


def top_confusions(cm: Sequence[Sequence[int]], idx: int, names: Sequence[str]) -> Dict[str, Any]:
    """Rank off-diagonal confusions for class ``idx`` in both directions.

    ``false_negatives``: windows truly ``idx`` predicted as another class, as a
    fraction of all true-``idx`` windows. ``false_positives``: windows predicted
    ``idx`` whose true class is different, as a fraction of all predicted-``idx``.
    """
    m = np.asarray(cm)
    fn_total = int(m[idx].sum() - m[idx, idx])
    fp_total = int(m[:, idx].sum() - m[idx, idx])
    row_total, col_total = int(m[idx].sum()), int(m[:, idx].sum())
    fn = [
        (names[j], int(m[idx, j]), (m[idx, j] / row_total) if row_total else 0.0)
        for j in range(len(names))
        if j != idx
    ]
    fp = [
        (names[i], int(m[i, idx]), (m[i, idx] / col_total) if col_total else 0.0)
        for i in range(len(names))
        if i != idx
    ]
    return {
        "false_negatives": sorted(fn, key=lambda t: -t[1]),
        "false_positives": sorted(fp, key=lambda t: -t[1]),
        "fn_total": fn_total,
        "fp_total": fp_total,
    }


def error_share(cm: Sequence[Sequence[int]], idx: int) -> Dict[str, float]:
    """Fraction of all errors that are FN / FP / either for class ``idx``."""
    m = np.asarray(cm)
    errors = int(m.sum() - np.trace(m))
    fn = int(m[idx].sum() - m[idx, idx])
    fp = int(m[:, idx].sum() - m[idx, idx])
    if errors == 0:
        return {"fn_share": 0.0, "fp_share": 0.0, "involved_share": 0.0}
    involved = fn + fp  # a window cannot be both FN and FP for the same class
    return {"fn_share": fn / errors, "fp_share": fp / errors, "involved_share": involved / errors}


def recall_gap(train_recall: Sequence[float], val_recall: Sequence[float]) -> List[float]:
    """Per-class train recall minus validation recall."""
    return [float(t - v) for t, v in zip(train_recall, val_recall)]


def last_n_improvement(values: Sequence[float], n: int) -> float:
    """Change over the last ``n`` epoch-to-epoch steps (value[-1] - value[-1-n])."""
    if len(values) <= n:
        raise ValueError("not enough epochs")
    return float(values[-1] - values[-1 - n])


def curve_slope(values: Sequence[float], n: int) -> float:
    """Least-squares slope per epoch over the last ``n`` points."""
    y = np.asarray(values[-n:], dtype=float)
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def metrics_block(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    labels = list(range(N))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    summary = confusion_summary(y, pred)
    return {
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(np.mean(f)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in labels
        },
        "confusion_counts": summary["counts"],
        "confusion_row_normalized": summary["row_normalized"],
    }


# --------------------------------------------------------------------------- #
# Data: train + val only                                                        #
# --------------------------------------------------------------------------- #
def load_split(split: str, balance_train: bool = True) -> ChessPlanDataset:
    """Encoded-mode dataset for ``train`` or ``val`` (never ``test``)."""
    if split not in ("train", "val"):
        raise ValueError("this analysis may only load the train and val splits")
    labels = np.load(DATA_DIR / f"{split}_labels.npy")
    indices = np.arange(len(labels), dtype=np.int64)
    if split == "train" and balance_train:
        rows = _read_csv(DATA_DIR / "train_records.csv")
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
        kept = balance_classes(records, max_class_fraction=0.30, seed=42)
        row_of = {r["window_id"]: i for i, r in enumerate(rows)}
        indices = np.asarray(sorted(row_of[r.window_id] for r in kept), dtype=np.int64)
    return ChessPlanDataset(
        encoded_tensors=_RowView(DATA_DIR / f"{split}_tensors.npy", indices),
        encoded_masks=_RowView(DATA_DIR / f"{split}_masks.npy", indices),
        labels=labels[indices].tolist(),
    )


def load_model(seed: int):
    cfg = VARIANTS[VARIANT]
    model = _build_model(cfg)
    payload = load_checkpoint(_ckpt_path(VARIANT, seed, TAG), model)
    model.eval()
    return model, cfg, payload


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bias_vector(seed: int) -> np.ndarray:
    b = read_json(EXP / f"logit_bias{TAG}_{VARIANT}_s{seed}.json")["bias"]
    return np.array([b[n] for n in CLASS_NAMES], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Plots                                                                         #
# --------------------------------------------------------------------------- #
def plot_confusion(raw: Dict[str, Any], adj: Dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, blk, title in zip(axes, (raw, adj), ("Raw logits", "Validation-tuned biases")):
        m = np.array(blk["confusion_row_normalized"])
        ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(N):
            for j in range(N):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if m[i, j] > 0.5 else "black",
                    fontsize=8,
                )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title} (val, seed 42, row-normalized)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(curves: Dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for group, style in (("80k", "-"), ("22k", "--")):
        for seed, c in curves[group].items():
            ep = c["epochs"]
            axes[0].plot(ep, c["val_macro_f1"], style, label=f"{group} s{seed}")
            axes[1].plot(ep, c["train_acc"], style, color="tab:green", alpha=0.6)
            axes[1].plot(ep, c["val_acc"], style, color="tab:red", alpha=0.6)
    axes[0].set_title("Validation macro-F1 by epoch")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Train acc (green) vs val acc (red); solid=80k, dashed=22k")
    axes[1].set_xlabel("epoch")
    mean_pc = np.mean([c["per_class_f1"] for c in curves["80k"].values()], axis=0)
    for i, name in enumerate(CLASS_NAMES):
        axes[2].plot(range(1, mean_pc.shape[0] + 1), mean_pc[:, i], label=name)
    axes[2].set_title("Per-class val F1, 80k (mean of 3 seeds)")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis sections                                                             #
# --------------------------------------------------------------------------- #
def learning_curve_section() -> Dict[str, Any]:
    files = {
        "80k": {s: EXP / f"final_80k_C_s{s}.json" for s in SEEDS},
        "22k": {
            42: EXP / "final_C_4ply_mirror_reg.json",
            123: EXP / "final_C_seed123.json",
            999: EXP / "final_C_seed999.json",
        },
    }
    out: Dict[str, Any] = {"80k": {}, "22k": {}}
    summary: Dict[str, Any] = {}
    for group, per_seed in files.items():
        for seed, path in per_seed.items():
            row = read_json(path)[-1]
            h = row["history"]
            f1 = [e["val_macro_f1"] for e in h]
            diffs = np.diff(f1)
            out[group][str(seed)] = {
                "epochs": [e["epoch"] for e in h],
                "val_macro_f1": f1,
                "val_acc": [e["val_acc"] for e in h],
                "val_loss": [e["val_loss"] for e in h],
                "train_acc": [e["train_acc"] for e in h],
                "train_loss": [e["train_loss"] for e in h],
                "per_class_f1": [e["per_class_f1"] for e in h],
                "best_epoch": row["best_epoch"],
                "epochs_run": row["epochs_run"],
                "epoch_cap": row["config"]["epochs"],
            }
            summary[f"{group}_s{seed}"] = {
                "best_epoch": row["best_epoch"],
                "epochs_run": row["epochs_run"],
                "epoch_cap": row["config"]["epochs"],
                "best_val_macro_f1": max(f1),
                "improve_last2": last_n_improvement(f1, 2),
                "improve_last3": last_n_improvement(f1, 3),
                "slope_last3_per_epoch": curve_slope(f1, 3),
                "mean_abs_epoch_to_epoch_change": float(np.mean(np.abs(diffs))),
                "std_epoch_to_epoch_change": float(np.std(diffs)),
                "final_gap_train_minus_val_acc": h[-1]["train_acc"] - h[-1]["val_acc"],
                "val_loss_min": min(e["val_loss"] for e in h),
                "val_loss_min_epoch": int(np.argmin([e["val_loss"] for e in h]) + 1),
                "chain_f1_last3": [e["per_class_f1"][CHAIN] for e in h[-3:]],
            }
    return {"curves": out, "summary": summary}


def bias_verification(val_logits: Dict[int, np.ndarray], y: np.ndarray) -> Dict[str, Any]:
    fit_src = inspect.getsource(tb.fit_bias)
    tune_lines = [
        (i + 1, ln.strip())
        for i, ln in enumerate(Path(tb.__file__).read_text(encoding="utf-8").splitlines())
        if "test" in ln.lower()
    ]
    import final_training as ft

    train_src = inspect.getsource(ft.train_one)
    main_src = inspect.getsource(ft.main)
    load_src = inspect.getsource(sys.modules["build_scaled_dataset"].load_scaled_dataset)
    biases = {s: bias_vector(s) for s in SEEDS}
    shared = np.mean(list(biases.values()), axis=0)
    shared_eval = {}
    for s in SEEDS:
        lg = val_logits[s]
        shared_eval[str(s)] = {
            "raw": {k: metrics_block(y, lg.argmax(1))[k] for k in ("macro_f1", "accuracy")},
            "own_bias": {
                k: metrics_block(y, (lg + biases[s]).argmax(1))[k] for k in ("macro_f1", "accuracy")
            },
            "shared_mean_bias": {
                k: metrics_block(y, (lg + shared).argmax(1))[k] for k in ("macro_f1", "accuracy")
            },
        }
    tune_json = {s: read_json(EXP / f"logit_bias{TAG}_{VARIANT}_s{s}.json") for s in SEEDS}
    test_path = EXP / "final_80k_test.json"
    test_stat = os.stat(test_path)
    bias_times = {
        str(s): os.stat(EXP / f"logit_bias{TAG}_{VARIANT}_s{s}.json").st_mtime for s in SEEDS
    }
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))  # noqa: E731
    test_rows = read_json(test_path)
    return {
        "method": "coordinate ascent on per-class additive logit biases (tune_logit_bias.fit_bias)",
        "objective": "macro-F1 on VALIDATION labels/logits",
        "fit_bias_source": fit_src,
        "centering": "bias - bias.mean() after the search",
        "overfit_check_in_tuner": "validation games split into two halves (seed 0 shuffle); "
        "fit on one half, scored on the other; see cross_fold_check in each bias JSON",
        "bias_json_cross_fold": {s: tune_json[s]["cross_fold_check"] for s in tune_json},
        "tuner_lines_mentioning_test": tune_lines,
        "train_one_references_test_split": '"test"' in train_src,
        "final_training_main_loads_test_only_when_phase_test": (
            'balance_train=args.phase == "train"' in main_src
        ),
        "load_scaled_dataset_opens_every_split_incl_test_labels": (
            "for split in SPLITS" in load_src
        ),
        "bias_vectors_per_seed": {
            str(s): {n: float(v) for n, v in zip(CLASS_NAMES, biases[s])} for s in SEEDS
        },
        "shared_mean_bias": {n: float(v) for n, v in zip(CLASS_NAMES, shared)},
        "bias_vector_max_abs_diff_between_seeds": float(
            max(np.abs(biases[a] - biases[b]).max() for a in SEEDS for b in SEEDS)
        ),
        "validation_eval_shared_vs_own_vs_raw": shared_eval,
        "timestamps": {
            "bias_json_mtime": {s: fmt(t) for s, t in bias_times.items()},
            "final_80k_test_json_created(ctime)": fmt(test_stat.st_ctime),
            "final_80k_test_json_last_modified": fmt(test_stat.st_mtime),
            "all_bias_files_written_before_test_file_created": bool(
                max(bias_times.values()) < test_stat.st_ctime
            ),
            "all_bias_files_written_before_test_file_last_modified": bool(
                max(bias_times.values()) < test_stat.st_mtime
            ),
        },
        "same_per_seed_bias_applied_to_each_test_entry": {
            str(r["seed"]): {n: float(v) for n, v in r["bias"].items()}
            == {n: float(v) for n, v in tune_json[r["seed"]]["bias"].items()}
            for r in test_rows
        },
        "recorded_test_numbers_read_only": {
            str(r["seed"]): {
                "raw_macro_f1": r["test"]["macro_f1"],
                "raw_accuracy": r["test"]["accuracy"],
                "adjusted_macro_f1": r["test_bias_adjusted"]["macro_f1"],
                "adjusted_accuracy": r["test_bias_adjusted"]["accuracy"],
            }
            for r in test_rows
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Variant C post-training analysis (train/val only).")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train-sample-per-class", type=int, default=4000)
    args = ap.parse_args(argv)
    for out in (OUT_JSON, OUT_CM, OUT_LC):
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing {out}")
    torch.set_num_threads(args.threads)
    t_start = time.time()
    timing: Dict[str, float] = {}
    result: Dict[str, Any] = {
        "environment": {
            "variant": VARIANT,
            "tag": TAG,
            "seeds": list(SEEDS),
            "torch": torch.__version__,
            "threads": args.threads,
            "splits_loaded": ["train", "val"],
            "test_split_loaded": False,
        }
    }

    val_ds = load_split("val")
    y = np.asarray(val_ds.labels)
    result["environment"]["val_windows"] = int(len(y))

    # 1 + 3: reproduction and validation error analysis, one inference pass per seed
    val_logits: Dict[int, np.ndarray] = {}
    reproduction: Dict[str, Any] = {}
    validation: Dict[str, Any] = {}
    max_diff = 0.0
    for seed in SEEDS:
        t0 = time.time()
        model, cfg, payload = load_model(seed)
        ys, lg = _predict_logits(model, _slice(val_ds, cfg["keep"]))
        assert list(ys) == y.tolist(), "validation label order changed"
        val_logits[seed] = lg
        stored = read_json(EXP / f"final_80k_C_s{seed}.json")[-1]["val"]
        raw = metrics_block(y, lg.argmax(1))
        d_f1, d_acc = abs(raw["macro_f1"] - stored["macro_f1"]), abs(
            raw["accuracy"] - stored["accuracy"]
        )
        max_diff = max(max_diff, d_f1, d_acc)
        reproduction[str(seed)] = {
            "checkpoint_epoch": payload["epoch"],
            "checkpoint_best_f1": payload["best_f1"],
            "checkpoint_config": payload["metadata"],
            "recomputed_val_macro_f1": raw["macro_f1"],
            "stored_val_macro_f1": stored["macro_f1"],
            "recomputed_val_accuracy": raw["accuracy"],
            "stored_val_accuracy": stored["accuracy"],
            "abs_diff_macro_f1": d_f1,
            "abs_diff_accuracy": d_acc,
        }
        adj = metrics_block(y, (lg + bias_vector(seed)).argmax(1))
        validation[str(seed)] = {
            "raw": raw,
            "bias_adjusted": adj,
            "chain_undermine": {
                "raw": {
                    **top_confusions(raw["confusion_counts"], CHAIN, CLASS_NAMES),
                    **error_share(raw["confusion_counts"], CHAIN),
                },
                "bias_adjusted": {
                    **top_confusions(adj["confusion_counts"], CHAIN, CLASS_NAMES),
                    **error_share(adj["confusion_counts"], CHAIN),
                },
            },
        }
        timing[f"val_inference_seed{seed}"] = round(time.time() - t0, 1)
        print(
            f"seed {seed}: val F1 {raw['macro_f1']:.4f} (stored {stored['macro_f1']:.4f})",
            flush=True,
        )
    reproduction["max_abs_diff_any_metric"] = max_diff
    result["baseline_reproduction"] = reproduction
    result["validation_analysis"] = validation

    # 2: class distributions
    meta = read_json(DATA_DIR / "meta.json")
    cd = meta["integrity"]["class_distribution"]
    train_bal = load_split("train", balance_train=True)
    bal_counts = np.bincount(train_bal.labels, minlength=N)
    val_counts = np.bincount(y, minlength=N)
    result["class_distributions"] = {
        "train_natural_counts": cd["train"]["counts"],
        "train_natural_proportions": cd["train"]["proportions"],
        "val_natural_counts": cd["val"]["counts"],
        "val_natural_proportions": cd["val"]["proportions"],
        "train_balanced_used_for_training": {
            "windows": int(len(train_bal)),
            "counts": {n: int(c) for n, c in zip(CLASS_NAMES, bal_counts)},
            "proportions": {
                n: float(c / bal_counts.sum()) for n, c in zip(CLASS_NAMES, bal_counts)
            },
        },
        "val_recomputed": {n: int(c) for n, c in zip(CLASS_NAMES, val_counts)},
        "note": "training used the class-balanced train set (max class fraction 0.30) AND "
        "class-weighted loss; validation keeps natural frequencies, so train and val "
        "accuracy/precision are measured under different class priors.",
    }

    # 4: learning curves
    lc = learning_curve_section()
    result["learning_curves"] = lc
    plot_curves(lc["curves"], OUT_LC)

    # 5: per-class train recall vs val recall
    gap: Dict[str, Any] = {}
    rng = np.random.RandomState(0)
    train_labels = np.asarray(train_bal.labels)
    pick = np.concatenate(
        [
            rng.choice(
                np.where(train_labels == c)[0],
                size=min(args.train_sample_per_class, int((train_labels == c).sum())),
                replace=False,
            )
            for c in range(N)
        ]
    )
    pick.sort()
    for seed in (42, 123):
        t0 = time.time()
        model, cfg, _ = load_model(seed)
        ys, lg = _predict_logits(model, _slice(_View(train_bal, pick.tolist()), cfg["keep"]))
        tr_pred = lg.argmax(1)
        ys = np.asarray(ys)
        tr_recall = [float((tr_pred[ys == c] == c).mean()) for c in range(N)]
        va_recall = [validation[str(seed)]["raw"]["per_class"][n]["recall"] for n in CLASS_NAMES]
        gap[str(seed)] = {
            "train_sample_size": int(len(pick)),
            "train_recall": dict(zip(CLASS_NAMES, tr_recall)),
            "val_recall": dict(zip(CLASS_NAMES, va_recall)),
            "train_minus_val_recall": dict(zip(CLASS_NAMES, recall_gap(tr_recall, va_recall))),
        }
        timing[f"train_sample_seed{seed}"] = round(time.time() - t0, 1)
    result["train_vs_val_recall_gap"] = gap

    # 6: bias verification (validation only)
    result["bias_verification"] = bias_verification(val_logits, y)

    plot_confusion(validation["42"]["raw"], validation["42"]["bias_adjusted"], OUT_CM)
    timing["total_seconds"] = round(time.time() - t_start, 1)
    result["timing_seconds"] = timing
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
