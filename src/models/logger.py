"""Training metrics logging and checkpoint management (Task 1.8, Subtask 1.8.4).

:class:`MetricsLogger` accumulates per-epoch metrics, writes them to JSON, and
plots learning curves. It also provides checkpoint save/load helpers that store
model + optimizer state alongside the metadata needed to resume or audit a run.

Subtask 1.8.4 is nominally owned by Muhammad Ali; implemented here so the
training pipeline is self-contained and testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["MetricsLogger", "save_checkpoint", "load_checkpoint"]


class MetricsLogger:
    """Collect per-epoch training/validation metrics and export them."""

    def __init__(self) -> None:
        self.records: List[Dict] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        val_macro_f1: float,
        per_class_f1: List[float],
        lr: Optional[float] = None,
    ) -> Dict:
        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_acc": round(float(val_acc), 6),
            "val_macro_f1": round(float(val_macro_f1), 6),
            "per_class_f1": [round(float(x), 6) for x in per_class_f1],
        }
        if lr is not None:
            record["lr"] = float(lr)
        self.records.append(record)
        return record

    def save_history(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2)
        return out

    def plot_curves(self, path: str | Path) -> None:
        """Plot train/val loss and validation macro-F1 (needs matplotlib)."""
        if not self.records:
            raise ValueError("no epochs logged yet.")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in self.records]
        fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
        ax_loss.plot(epochs, [r["train_loss"] for r in self.records], label="train")
        ax_loss.plot(epochs, [r["val_loss"] for r in self.records], label="val")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_f1.plot(epochs, [r["val_macro_f1"] for r in self.records], color="tab:green")
        ax_f1.set_xlabel("epoch")
        ax_f1.set_ylabel("macro-F1")
        ax_f1.set_title("Validation macro-F1")
        fig.tight_layout()

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_f1: float = 0.0,
    metadata: Optional[Dict] = None,
) -> Path:
    """Save a checkpoint dict with weights, optimizer state and metadata."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_f1": float(best_f1),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load weights (and optionally optimizer state) from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
