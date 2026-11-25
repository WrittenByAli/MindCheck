"""Final training on the scaled dataset: mirror augmentation + regularization.

Variants are screened on the VALIDATION split only. The test split is touched
exactly once per chosen model, by ``--phase test``, after a variant has been
selected from validation results.

Typical usage::

    python scripts/final_training.py --phase train --variant B_8ply_mirror_reg --seeds 42
    python scripts/final_training.py --phase test  --variant B_8ply_mirror_reg --seeds 42 123 999
"""

from __future__ import annotations

import argparse
import json
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
from build_scaled_dataset import load_scaled_dataset  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from src.data.fast_loader import BatchLoader  # noqa: E402
from src.models.logger import MetricsLogger, load_checkpoint, save_checkpoint  # noqa: E402
from src.models.residual_cnn import HistoryChessResNet, count_parameters  # noqa: E402
from src.models.train import ModelTrainer, compute_class_weights, set_seed  # noqa: E402
from src.models.train_ablation import ChannelSliceDataset  # noqa: E402

CLASS_NAMES = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"]
DEFAULT_RESULTS = _REPO_ROOT / "experiments" / "final_training.json"
CKPT_DIR = _REPO_ROOT / "models" / "checkpoints"

VARIANTS: Dict[str, Dict[str, Any]] = {
    "A_8ply_plain": dict(keep=112, mirror=False, dropout=0.2, weight_decay=1e-4),
    "B_8ply_mirror_reg": dict(keep=112, mirror=True, dropout=0.3, weight_decay=1e-3),
    "C_4ply_mirror_reg": dict(keep=56, mirror=True, dropout=0.3, weight_decay=1e-3),
    # Controlled follow-ups to C: identical except balancing selection / loss weights.
    "D_random_balance": dict(
        keep=56, mirror=True, dropout=0.3, weight_decay=1e-3, balance="random"
    ),
    "E_random_balance_sqrtw": dict(
        keep=56, mirror=True, dropout=0.3, weight_decay=1e-3, balance="random", weights="sqrt"
    ),
    # Round-3 fallback for D, used only if a confirmation seed fails gate G1: one step
    # more regularization (dropout 0.3 -> 0.4), everything else identical.
    "D_random_balance_strongreg": dict(
        keep=56, mirror=True, dropout=0.4, weight_decay=1e-3, balance="random"
    ),
    # v2 (docs/V2_PROTOCOL.md): D's hyperparameters on the unfiltered cache from
    # build_allpos_dataset; they differ only in which train rows are used.
    "F_allpos": dict(keep=56, mirror=True, dropout=0.3, weight_decay=1e-3, balance="random"),
    "G_allpos_uniform": dict(
        keep=56, mirror=True, dropout=0.3, weight_decay=1e-3, balance="random", rows=("uniform",)
    ),
    # Amendment 2: F with stronger regularization (dropout 0.4, weight decay 3e-3).
    "H_allpos_strongreg": dict(
        keep=56, mirror=True, dropout=0.4, weight_decay=3e-3, balance="random"
    ),
}


class _PrintingLogger(MetricsLogger):
    """MetricsLogger that also prints each epoch so long runs are observable."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, val_macro_f1, *a, **k):
        rec = super().log_epoch(
            epoch, train_loss, val_loss, train_acc, val_acc, val_macro_f1, *a, **k
        )
        print(
            f"[{self.tag}] epoch {epoch}: train_acc {train_acc:.3f} val_acc {val_acc:.3f} "
            f"train_loss {train_loss:.3f} val_loss {val_loss:.3f} val_macro_f1 {val_macro_f1:.4f}",
            flush=True,
        )
        return rec


class _View(Dataset):
    """Index-subset view of a dataset that keeps a ``.labels`` list."""

    def __init__(self, base: Any, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)
        self.labels = [base.labels[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


def _predict_logits(model: torch.nn.Module, loader: BatchLoader):
    """Labels and logits for every sample of ``loader``, in loader order."""
    model.eval()
    ys: List[torch.Tensor] = []
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for x, _mask, y in loader:
            outs.append(model(x).cpu())
            ys.append(y.cpu())
    return torch.cat(ys).tolist(), torch.cat(outs).numpy()


def _predict(model: torch.nn.Module, loader: BatchLoader):
    ys, logits = _predict_logits(model, loader)
    return ys, logits.argmax(1).tolist()


def _loader(ds: Any, keep: int, positions: Optional[np.ndarray] = None, **kwargs) -> BatchLoader:
    """BatchLoader over a cached dataset (``positions`` selects a subset of it)."""
    if positions is None:
        positions = np.arange(len(ds))
    labels = np.asarray(ds.labels)[positions]
    return BatchLoader(
        ds.tensors.path, ds.tensors.indices[positions], labels, keep_channels=keep, **kwargs
    )


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available.")
    return name


def report(ys: Sequence[int], ps: Sequence[int]) -> Dict[str, Any]:
    """Accuracy, macro-F1, per-class P/R/F1/support and the confusion matrix."""
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(ys, ps, labels=labels, zero_division=0)
    return {
        "n": len(ys),
        "accuracy": float(np.mean(np.asarray(ys) == np.asarray(ps))) if len(ys) else 0.0,
        "macro_f1": float(np.mean(f)),
        "per_class": {
            name: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(ys, ps, labels=labels).tolist(),
    }


def _slice(ds: Any, keep: int) -> Any:
    return ds if keep >= 112 else ChannelSliceDataset(ds, keep)


def _build_model(cfg: Dict[str, Any]) -> HistoryChessResNet:
    return HistoryChessResNet(in_channels=cfg["keep"], dropout=cfg["dropout"])


def _ckpt_path(variant: str, seed: int, tag: str = "") -> Path:
    return CKPT_DIR / f"scaled{tag}_{variant}_s{seed}.pt"


def _append_result(path: Path, row: Dict[str, Any]) -> None:
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def train_one(variant: str, seed: int, datasets: Dict[str, Any], args: argparse.Namespace) -> Dict:
    cfg = VARIANTS[variant]
    set_seed(seed)
    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    full_train = datasets["train"]
    n_train = min(len(full_train), args.max_train or len(full_train))
    positions = np.arange(n_train)
    train = _loader(
        full_train,
        cfg["keep"],
        positions,
        batch_size=args.batch_size,
        shuffle=True,
        mirror_p=0.5 if cfg["mirror"] else 0.0,
        seed=seed,
        device=device,
    )
    val = _loader(datasets["val"], cfg["keep"], batch_size=1024, device=device)
    print(f"device {device}; train storage {train.storage}; val storage {val.storage}", flush=True)

    model = _build_model(cfg)
    train_labels = [full_train.labels[i] for i in positions]
    weights = compute_class_weights(train_labels)
    if cfg.get("weights") == "sqrt":
        weights = weights.sqrt()
    elif cfg.get("weights") == "none":
        weights = None
    trainer = ModelTrainer(
        model,
        train,
        val,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=cfg["weight_decay"],
        label_smoothing=0.05,
        early_stopping_patience=args.patience,
        class_weights=weights,
        device=device,
        logger=_PrintingLogger(f"{variant} s{seed}"),
    )
    start = time.time()
    # Save the best epoch as training goes, so an interrupted run keeps its best model.
    result = trainer.fit(checkpoint_path=_ckpt_path(variant, seed, args.tag))
    seconds = time.time() - start

    val_rep = report(*_predict(model, val))
    probe = _loader(
        full_train,
        cfg["keep"],
        positions[:: max(1, n_train // 20000)],
        batch_size=1024,
        device=device,
        storage="memmap",
    )
    train_rep = report(*_predict(model, probe))

    config = {"variant": variant, "seed": seed, **cfg, "epochs": args.epochs}
    config.update(patience=args.patience, batch_size=args.batch_size, max_train=args.max_train)
    config["device"] = device
    save_checkpoint(
        model,
        _ckpt_path(variant, seed, args.tag),
        optimizer=trainer.optimizer,
        epoch=result["best_epoch"],
        best_f1=result["best_f1"],
        metadata=config,
    )
    return {
        "phase": "train",
        "variant": variant,
        "seed": seed,
        "params": count_parameters(model),
        "train_windows": n_train,
        "best_epoch": result["best_epoch"],
        "epochs_run": len(result["history"]),
        "val": val_rep,
        "train_probe": {"accuracy": train_rep["accuracy"], "macro_f1": train_rep["macro_f1"]},
        "acc_gap": train_rep["accuracy"] - val_rep["accuracy"],
        "f1_gap": train_rep["macro_f1"] - val_rep["macro_f1"],
        "history": result["history"],
        "config": config,
        "seconds": round(seconds, 1),
    }


def test_one(
    variant: str,
    seed: int,
    datasets: Dict[str, Any],
    tag: str = "",
    bias: Optional[Dict[str, float]] = None,
    device: str = "cpu",
) -> Dict:
    cfg = VARIANTS[variant]
    model = _build_model(cfg)
    payload = load_checkpoint(_ckpt_path(variant, seed, tag), model)
    model.to(device)
    loader = _loader(datasets["test"], cfg["keep"], batch_size=1024, device=device)
    ys, logits = _predict_logits(model, loader)
    row: Dict[str, Any] = {
        "phase": "test",
        "variant": variant,
        "seed": seed,
        "params": count_parameters(model),
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_val_macro_f1": payload["best_f1"],
        "test": report(ys, logits.argmax(1)),
    }
    if bias is not None:
        vec = np.array([bias[name] for name in CLASS_NAMES], dtype=np.float32)
        row["test_bias_adjusted"] = report(ys, (logits + vec).argmax(1))
        row["bias"] = bias
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train/evaluate on the scaled dataset.")
    ap.add_argument("--phase", choices=["train", "test"], required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--bias-file", default=None, help="JSON from tune_logit_bias.py (test phase).")
    ap.add_argument("--tag", default="", help="Checkpoint name suffix, e.g. _80k.")
    ap.add_argument("--max-train", type=int, default=None, help="Smoke-test cap on train windows.")
    ap.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "processed" / "scaled"))
    ap.add_argument("--out", default=str(DEFAULT_RESULTS))
    args = ap.parse_args(argv)

    splits = ("train", "val") if args.phase == "train" else ("test",)
    datasets = load_scaled_dataset(
        args.data_dir,
        balance_train=args.phase == "train",
        seed=42,
        splits=splits,
        balance_select=VARIANTS[args.variant].get("balance", "confidence"),
        train_sample_kinds=VARIANTS[args.variant].get("rows"),
    )
    bias = None
    if args.bias_file:
        bias = json.loads(Path(args.bias_file).read_text(encoding="utf-8"))["bias"]
    print({k: len(v) for k, v in datasets.items()}, flush=True)
    for seed in args.seeds:
        row = (
            train_one(args.variant, seed, datasets, args)
            if args.phase == "train"
            else test_one(
                args.variant, seed, datasets, args.tag, bias, _resolve_device(args.device)
            )
        )
        _append_result(Path(args.out), row)
        key = "val" if args.phase == "train" else "test"
        print(
            f"{args.phase} {args.variant} seed {seed}: {key} macro-F1 "
            f"{row[key]['macro_f1']:.4f}, acc {row[key]['accuracy']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
