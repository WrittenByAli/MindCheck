"""Regularization sweep: which settings shrink the train/val gap on the 3000-game data.

Selection uses the VALIDATION split only; the test split is never encoded or
evaluated. The game-level split seed is fixed (42) for every config so configs
are directly comparable; only the training/init seed varies.

Typical usage::

    python scripts/regularization_sweep.py --cache-dir <dir> --prepare
    python scripts/regularization_sweep.py --cache-dir <dir> --run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.chess_dataset import ChessPlanDataset, create_dataloaders  # noqa: E402
from src.data.tensor_encoder import ChessTensorEncoder  # noqa: E402
from src.models.residual_cnn import HistoryChessResNet, count_parameters  # noqa: E402
from src.models.train import (  # noqa: E402
    ModelTrainer,
    compute_class_weights,
    prepare_datasets,
    set_seed,
)

SPLIT_SEED = 42
TRAIN_SEEDS = [42, 123, 999]
PGN_PATH = _REPO_ROOT / "data" / "raw" / "lichess_elite_subset.pgn"
DEFAULT_OUT = _REPO_ROOT / "experiments" / "regularization_sweep.json"
PLANES_PER_PLY = 14


@dataclass(frozen=True)
class SweepConfig:
    name: str
    weight_decay: float = 1e-4
    dropout: float = 0.20
    label_smoothing: float = 0.05
    channels: int = 64
    plies: int = 8


_COMBO = dict(weight_decay=1e-2, dropout=0.5, label_smoothing=0.1)
REGISTRY: Dict[str, SweepConfig] = {
    c.name: c
    for c in [
        SweepConfig("baseline"),
        SweepConfig("wd1e-3", weight_decay=1e-3),
        SweepConfig("wd1e-2", weight_decay=1e-2),
        SweepConfig("do0.4", dropout=0.4),
        SweepConfig("do0.5", dropout=0.5),
        SweepConfig("ls0.1", label_smoothing=0.1),
        SweepConfig("ch32", channels=32),
        SweepConfig("combo", **_COMBO),
        SweepConfig("combo_ch32", channels=32, **_COMBO),
        SweepConfig("4ply_baseline", plies=4),
        SweepConfig("4ply_combo", plies=4, **_COMBO),
    ]
}


# --------------------------------------------------------------------------- #
# Data cache (train + val only; the test split is never encoded)               #
# --------------------------------------------------------------------------- #
def _encode(ds: ChessPlanDataset) -> Dict[str, np.ndarray]:
    enc = ChessTensorEncoder()
    tensors, masks = [], []
    for w in ds.windows:
        t, m = enc.encode_position_window(w)
        tensors.append(t.astype(np.uint8))  # planes are binary; widened to float32 on read
        masks.append(m)
    return {
        "x": np.stack(tensors),
        "m": np.stack(masks).astype(np.float32),
        "y": np.asarray(ds.labels, dtype=np.int64),
    }


def prepare_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    datasets = prepare_datasets(PGN_PATH, seed=SPLIT_SEED)
    print(f"prepare_datasets: {time.time() - t0:.0f}s", flush=True)
    for split in ("train", "val"):  # test deliberately omitted
        arrays = _encode(datasets[split])
        np.savez(cache_dir / f"{split}.npz", **arrays)
        counts = np.bincount(arrays["y"], minlength=6).tolist()
        print(f"  {split}: {len(arrays['y'])} windows, class counts {counts}", flush=True)


def load_cache(cache_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for split in ("train", "val"):
        with np.load(cache_dir / f"{split}.npz") as z:
            out[split] = {k: z[k] for k in ("x", "m", "y")}
    return out


def _make_ds(arrays: Dict[str, np.ndarray], plies: int) -> ChessPlanDataset:
    keep = plies * PLANES_PER_PLY
    return ChessPlanDataset(
        encoded_tensors=np.ascontiguousarray(arrays["x"][:, -keep:]),
        encoded_masks=arrays["m"],
        labels=arrays["y"].tolist(),
    )


# --------------------------------------------------------------------------- #
# One run                                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _val_nll(model: torch.nn.Module, loader: DataLoader) -> float:
    """Unweighted, unsmoothed val cross-entropy (comparable across configs)."""
    model.eval()
    total, n = 0.0, 0
    for x, _m, y in loader:
        total += float(F.cross_entropy(model(x), y, reduction="sum").item())
        n += len(y)
    return total / max(n, 1)


def run_one(cfg: SweepConfig, seed: int, data, epochs: int, patience: int) -> Dict:
    set_seed(seed)
    train_ds = _make_ds(data["train"], cfg.plies)
    val_ds = _make_ds(data["val"], cfg.plies)
    # val is passed in the test slot too: the real test split is never loaded here.
    train_loader, val_loader, _ = create_dataloaders(train_ds, val_ds, val_ds, batch_size=64)
    model = HistoryChessResNet(
        in_channels=cfg.plies * PLANES_PER_PLY, channels=cfg.channels, dropout=cfg.dropout
    )
    trainer = ModelTrainer(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=1e-3,
        weight_decay=cfg.weight_decay,
        label_smoothing=cfg.label_smoothing,
        early_stopping_patience=patience,
        class_weights=compute_class_weights(train_ds.labels),
        device="cpu",
    )
    t0 = time.time()
    result = trainer.fit()
    secs = time.time() - t0

    history = result["history"]
    best_epoch = result["best_epoch"]
    best = history[best_epoch - 1]
    min_loss_rec = min(history, key=lambda r: r["val_loss"])

    # Best weights are restored by fit(); measure train in eval mode (dropout off).
    train_eval = DataLoader(train_ds, batch_size=256, shuffle=False)
    _, tr_acc, tr_f1, _ = trainer.evaluate(train_eval)
    _, val_acc, val_f1, per_class = trainer.evaluate(val_loader)
    return {
        "config": cfg.name,
        "seed": seed,
        "params": count_parameters(model),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(val_f1, 4),
        "val_acc": round(val_acc, 4),
        "train_acc_eval": round(tr_acc, 4),
        "train_macro_f1_eval": round(tr_f1, 4),
        "train_acc_trainmode": best["train_acc"],
        "acc_gap": round(tr_acc - val_acc, 4),
        "f1_gap": round(tr_f1 - val_f1, 4),
        "min_val_loss": min_loss_rec["val_loss"],
        "min_val_loss_epoch": min_loss_rec["epoch"],
        "val_nll_best_state": round(_val_nll(model, val_loader), 4),
        "per_class_f1": [round(x, 4) for x in per_class],
        "seconds": round(secs, 1),
    }


# --------------------------------------------------------------------------- #
# Aggregation + driver                                                         #
# --------------------------------------------------------------------------- #
def _ms(values: List[float]) -> Dict[str, float]:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": round(statistics.fmean(values), 4), "std": round(sd, 4)}


def aggregate(rows: List[Dict]) -> List[Dict]:
    by_cfg: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    summary = []
    for name, rs in by_cfg.items():
        summary.append(
            {
                "config": name,
                "settings": asdict(REGISTRY[name]),
                "params": rs[0]["params"],
                "n_seeds": len(rs),
                "val_macro_f1": _ms([r["best_val_macro_f1"] for r in rs]),
                "val_acc": _ms([r["val_acc"] for r in rs]),
                "acc_gap": _ms([r["acc_gap"] for r in rs]),
                "f1_gap": _ms([r["f1_gap"] for r in rs]),
                "best_epoch": _ms([float(r["best_epoch"]) for r in rs]),
                "val_nll_best_state": _ms([r["val_nll_best_state"] for r in rs]),
                "min_val_loss": _ms([r["min_val_loss"] for r in rs]),
                "per_class_f1": [
                    round(statistics.fmean(r["per_class_f1"][c] for r in rs), 4) for c in range(6)
                ],
            }
        )
    summary.sort(key=lambda s: -s["val_macro_f1"]["mean"])
    base = next((s for s in summary if s["config"] == "baseline"), None)
    for s in summary:
        if base is not None:
            s["acc_gap_shrank_vs_baseline"] = s["acc_gap"]["mean"] < base["acc_gap"]["mean"]
            s["f1_gap_shrank_vs_baseline"] = s["f1_gap"]["mean"] < base["f1_gap"]["mean"]
    return summary


def read_results(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_sweep(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    data = load_cache(Path(args.cache_dir))
    results_path = Path(args.cache_dir) / "results.jsonl"
    rows = read_results(results_path)
    done = {(r["config"], r["seed"]) for r in rows}

    names = args.configs or list(REGISTRY)
    for seed in args.seeds:  # seed-major: partial runs still cover every config once
        for name in names:
            if (name, seed) in done:
                continue
            row = run_one(REGISTRY[name], seed, data, args.epochs, args.patience)
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"[seed {seed}] {name:14s} F1={row['best_val_macro_f1']:.4f} "
                f"ep={row['best_epoch']} acc_gap={row['acc_gap']:+.3f} "
                f"f1_gap={row['f1_gap']:+.3f} ({row['seconds']:.0f}s)",
                flush=True,
            )
    write_output(rows, Path(args.out), args)


def write_output(rows: List[Dict], out: Path, args: Optional[argparse.Namespace] = None) -> None:
    payload = {
        "protocol": {
            "split_seed": SPLIT_SEED,
            "train_seeds": sorted({r["seed"] for r in rows}),
            "max_epochs": getattr(args, "epochs", None),
            "early_stopping_patience": getattr(args, "patience", None),
            "selection_split": "validation only (test split never loaded)",
            "gap_note": (
                "gaps use eval-mode train metrics on the best-epoch weights; train is "
                "class-balanced, val keeps natural class frequencies, so accuracy gaps "
                "are partly a distribution effect (prefer f1_gap)"
            ),
        },
        "ranked_summary": aggregate(rows),
        "runs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir", default=str(Path(tempfile.gettempdir()) / "mindcheck_sweep_cache")
    )
    ap.add_argument("--prepare", action="store_true", help="build the encoded train/val cache")
    ap.add_argument("--run", action="store_true", help="run the sweep from the cache")
    ap.add_argument(
        "--aggregate", action="store_true", help="rebuild the output JSON from results.jsonl"
    )
    ap.add_argument("--configs", nargs="+", choices=list(REGISTRY), default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    if args.prepare:
        prepare_cache(Path(args.cache_dir))
    if args.run:
        run_sweep(args)
    if args.aggregate:
        rows = read_results(Path(args.cache_dir) / "results.jsonl")
        write_output(rows, Path(args.out), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
