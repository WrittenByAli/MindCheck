"""History-length ablation: 1-ply vs 4-ply vs 8-ply (Task 1.8, Subtask 1.8.3).

Trains three models under matched settings (same splits, seed, optimizer) that
differ only in how much history they see, by slicing the last ``K*14`` channels
of the shared ``(112, 8, 8)`` tensor. This directly answers MindCheck's primary
research question: does recent move history improve plan recognition?

CLI::

    python -m src.models.train_ablation --seeds 42 123 999

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.chess_dataset import create_dataloaders  # noqa: E402
from src.models.residual_cnn import HistoryChessResNet, count_parameters  # noqa: E402
from src.models.train import (  # noqa: E402
    ModelTrainer,
    compute_class_weights,
    prepare_datasets,
    set_seed,
)

PLANES_PER_PLY = 14
# (name, history plies, input channels)
VARIANTS = [("1-ply", 1, 14), ("4-ply", 4, 56), ("8-ply", 8, 112)]


class ChannelSliceDataset(Dataset):
    """Wrap a dataset to expose only the most recent ``keep_channels`` planes."""

    def __init__(self, base: Dataset, keep_channels: int) -> None:
        self.base = base
        self.keep = keep_channels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        tensor, mask, label = self.base[index]
        return tensor[-self.keep :], mask, label


def _train_variant(name, channels, datasets, labels, epochs, batch_size, seed, device) -> Dict:
    set_seed(seed)
    train_ds = ChannelSliceDataset(datasets["train"], channels)
    val_ds = ChannelSliceDataset(datasets["val"], channels)
    train_loader, val_loader, _ = create_dataloaders(
        train_ds, val_ds, val_ds, batch_size=batch_size
    )

    model = HistoryChessResNet(in_channels=channels)
    weights = compute_class_weights(labels)
    trainer = ModelTrainer(
        model, train_loader, val_loader, epochs=epochs, class_weights=weights, device=device
    )
    result = trainer.fit()
    _, _, macro_f1, per_class = trainer.evaluate(val_loader)
    return {
        "variant": name,
        "in_channels": channels,
        "parameters": count_parameters(model),
        "best_val_macro_f1": round(result["best_f1"], 4),
        "final_val_macro_f1": round(macro_f1, 4),
        "per_class_f1": [round(x, 4) for x in per_class],
    }


def run_ablation(
    pgn_path: str | Path,
    seeds: List[int],
    epochs: int = 15,
    batch_size: int = 64,
    max_games: Optional[int] = None,
    device: str = "cpu",
) -> List[Dict]:
    """Train every variant for every seed; return a flat list of result rows."""
    rows: List[Dict] = []
    for seed in seeds:
        datasets = prepare_datasets(pgn_path, seed=seed, max_games=max_games)
        labels = datasets["train"].labels
        for name, _plies, channels in VARIANTS:
            row = _train_variant(name, channels, datasets, labels, epochs, batch_size, seed, device)
            row["seed"] = seed
            rows.append(row)
            print(f"  seed {seed} {name}: val macro-F1 {row['best_val_macro_f1']:.4f}")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the 1/4/8-ply history ablation.")
    ap.add_argument("--pgn", default=str(_REPO_ROOT / "data" / "raw" / "lichess_elite_subset.pgn"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = run_ablation(
        args.pgn,
        args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_games=args.max_games,
        device=device,
    )
    out = _REPO_ROOT / "experiments" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
