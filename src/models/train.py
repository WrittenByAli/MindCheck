"""Training and optimization engine for MindCheck (Task 1.8, Subtasks 1.8.1-1.8.2).

``ModelTrainer`` runs class-weighted, label-smoothed cross-entropy training with
AdamW, gradient clipping and a cosine-annealing schedule, selecting the best
epoch by **validation macro-F1** with early stopping. All randomness is seeded
for reproducibility.

CLI::

    python -m src.models.train --config config/config.yaml --epochs 30 --batch_size 64
    python -m src.models.train --config config/config.yaml --ablation_single_board

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.logger import MetricsLogger, save_checkpoint  # noqa: E402
from src.models.residual_cnn import HistoryChessResNet  # noqa: E402
from src.models.single_board_cnn import CurrentPositionResNet  # noqa: E402

__all__ = ["set_seed", "compute_class_weights", "EarlyStopper", "ModelTrainer"]

NUM_CLASSES = 6


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(labels: List[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights ``w_c = N / (K * N_c)`` (absent -> 1.0)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopper:
    """Stop after ``patience`` epochs without macro-F1 improvement."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = -float("inf")
        self.num_bad = 0

    def step(self, score: float) -> bool:
        """Record ``score``; return True if training should stop."""
        if score > self.best + self.min_delta:
            self.best = score
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience


class ModelTrainer:
    """Train a model with weighted CE, AdamW, cosine annealing and early stop."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        early_stopping_patience: int = 5,
        max_grad_norm: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cpu",
        num_classes: int = NUM_CLASSES,
        logger: Optional[MetricsLogger] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.num_classes = num_classes
        self.logger = logger or MetricsLogger()

        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1), eta_min=1e-5
        )
        self.early_stopper = EarlyStopper(patience=early_stopping_patience)
        self.best_f1 = -float("inf")
        self.best_epoch = 0

    def training_step(self, tensors: torch.Tensor, labels: torch.Tensor) -> float:
        """One forward/backward/update step; returns the scalar loss."""
        self.model.train()
        tensors = tensors.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(tensors)
        loss = self.criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, n, correct = 0.0, 0, 0
        for tensors, _masks, labels in self.train_loader:
            loss = self.training_step(tensors, labels)
            with torch.no_grad():
                logits = self.model(tensors.to(self.device))
                correct += int((logits.argmax(-1).cpu() == labels).sum().item())
            total_loss += loss * len(labels)
            n += len(labels)
        return total_loss / max(n, 1), correct / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, List[float]]:
        self.model.eval()
        total_loss, n = 0.0, 0
        y_true: List[int] = []
        y_pred: List[int] = []
        for tensors, _masks, labels in loader:
            tensors = tensors.to(self.device)
            logits = self.model(tensors)
            loss = self.criterion(logits, labels.to(self.device))
            total_loss += float(loss.item()) * len(labels)
            n += len(labels)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(-1).cpu().tolist())

        labels_range = list(range(self.num_classes))
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        macro_f1 = f1_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)
        per_class = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
        return total_loss / max(n, 1), acc, float(macro_f1), [float(x) for x in per_class]

    def fit(self, checkpoint_path: Optional[str | Path] = None) -> Dict:
        """Full training loop with early stopping; restores the best weights."""
        best_state = copy.deepcopy(self.model.state_dict())
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1, per_class = self.evaluate(self.val_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            self.logger.log_epoch(
                epoch, train_loss, val_loss, train_acc, val_acc, val_f1, per_class, lr=lr
            )

            if val_f1 > self.best_f1 + 1e-6:
                self.best_f1 = val_f1
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    save_checkpoint(
                        self.model,
                        checkpoint_path,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_f1=val_f1,
                        metadata={"best_epoch": epoch, "per_class_f1": per_class},
                    )

            if self.early_stopper.step(val_f1):
                break

        self.model.load_state_dict(best_state)  # restore best epoch
        return {
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "history": self.logger.records,
        }


# --------------------------------------------------------------------------- #
# Real-data wiring + CLI                                                        #
# --------------------------------------------------------------------------- #
def prepare_datasets(
    pgn_path: str | Path,
    seed: int = 42,
    max_games: Optional[int] = None,
    start_ply: int = 12,
    history_length: int = 8,
):
    """Build train/val/test datasets from a PGN (label -> filter/balance -> split)."""
    from src.data.chess_dataset import ChessPlanDataset
    from src.data.dataset_splitter import GameLevelSplitter
    from src.data.pgn_parser import StreamingPGNParser
    from src.data.weak_labeler import (
        RetrospectiveWeakLabeler,
        balance_classes,
        default_game_id,
        filter_and_balance,
    )
    from src.data.window_extractor import extract_windows

    parser = StreamingPGNParser()
    labeler = RetrospectiveWeakLabeler(start_ply=start_ply, history_length=history_length)

    windows_by_id = {}
    records = []
    for game in parser.stream_games(pgn_path, max_games=max_games):
        gid = default_game_id(game)
        windows = {
            w.ply_index: w
            for w in extract_windows(
                game, game_id=gid, start_ply=start_ply, history_length=history_length
            )
        }
        for record in labeler.label_game(game, game_id=gid):
            window = windows.get(record.ply_index)
            if window is not None:
                windows_by_id[record.window_id] = window
                records.append(record)

    # Quality-filter every record, but DO NOT class-balance yet: balancing must
    # run after the game-level split and only on the train split, so val/test
    # keep their natural class distribution and stay honest evaluation sets.
    kept = filter_and_balance(records, seed=seed, balance=False)
    GameLevelSplitter(seed=seed).split(kept)

    train_records = balance_classes([r for r in kept if r.split == "train"], seed=seed)
    split_records = {
        "train": train_records,
        "val": [r for r in kept if r.split == "val"],
        "test": [r for r in kept if r.split == "test"],
    }

    datasets = {}
    for split, recs in split_records.items():
        windows = [windows_by_id[r.window_id] for r in recs]
        labels = [r.label_id for r in recs]
        datasets[split] = ChessPlanDataset(windows, labels)
    return datasets


def main(argv: Optional[List[str]] = None) -> int:
    from src.data.chess_dataset import create_dataloaders
    from src.utils.config import load_config

    ap = argparse.ArgumentParser(description="Train the MindCheck plan-recognition model.")
    ap.add_argument("--config", default=str(_REPO_ROOT / "config" / "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--ablation_single_board", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    epochs = args.epochs or cfg.training.max_epochs
    batch_size = args.batch_size or cfg.training.batch_size
    device = "cuda" if (cfg.training.device != "cpu" and torch.cuda.is_available()) else "cpu"
    set_seed(cfg.training.seed)

    print("Preparing datasets from PGN (this labels + encodes windows) ...")
    datasets = prepare_datasets(
        cfg.dataset.raw_pgn_path, seed=cfg.training.seed, max_games=args.max_games
    )
    for split, ds in datasets.items():
        print(f"  {split}: {len(ds)} windows")

    train_loader, val_loader, test_loader = create_dataloaders(
        datasets["train"],
        datasets["val"],
        datasets["test"],
        batch_size=batch_size,
        num_workers=cfg.training.num_workers,
    )

    if args.ablation_single_board:
        model: nn.Module = CurrentPositionResNet(
            num_classes=cfg.model.num_classes, num_blocks=cfg.model.num_blocks
        )
    else:
        model = HistoryChessResNet(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            channels=cfg.model.channels,
            num_blocks=cfg.model.num_blocks,
            dropout=cfg.model.dropout,
        )

    weights = compute_class_weights(datasets["train"].labels, cfg.model.num_classes)
    trainer = ModelTrainer(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        label_smoothing=cfg.training.label_smoothing,
        early_stopping_patience=cfg.training.early_stopping_patience,
        class_weights=weights,
        device=device,
        num_classes=cfg.model.num_classes,
    )
    result = trainer.fit(checkpoint_path=cfg.model.best_checkpoint_path)
    trainer.logger.save_history(_REPO_ROOT / "experiments" / "training_history.json")
    print(f"Best val macro-F1 {result['best_f1']:.4f} at epoch {result['best_epoch']}.")
    print(f"Saved checkpoint to {cfg.model.best_checkpoint_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
