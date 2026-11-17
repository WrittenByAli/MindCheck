"""PyTorch dataset and dataloaders for MindCheck (Task 1.6, Subtask 1.6.4).

:class:`ChessPlanDataset` yields ``(tensor, mask, label_id)`` samples where the
tensor is the ``(112, 8, 8)`` spatial-temporal encoding. It supports two modes:

* **dynamic** — hold :class:`~src.data.window_extractor.PositionWindow` objects
  and encode each on access (flexible, low memory);
* **encoded** — hold pre-encoded ``(N, 112, 8, 8)`` / ``(N, 8)`` arrays
  (``np.ndarray`` or ``np.memmap``) for maximum throughput.

:func:`create_dataloaders` wraps the three split datasets into PyTorch
``DataLoader``s.

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.tensor_encoder import ChessTensorEncoder

__all__ = ["ChessPlanDataset", "create_dataloaders"]


class ChessPlanDataset(Dataset):
    """A dataset of encoded 8-ply windows and their weak plan labels."""

    def __init__(
        self,
        windows: Optional[Sequence] = None,
        labels: Optional[Sequence[int]] = None,
        encoded_tensors: Optional[np.ndarray] = None,
        encoded_masks: Optional[np.ndarray] = None,
        encoder: Optional[ChessTensorEncoder] = None,
    ) -> None:
        self.encoder = encoder or ChessTensorEncoder()

        if windows is not None:
            if labels is None or len(windows) != len(labels):
                raise ValueError("windows and labels must be provided and equal length.")
            self._mode = "dynamic"
            self.windows = list(windows)
            self.labels = [int(x) for x in labels]
        elif encoded_tensors is not None:
            if encoded_masks is None or labels is None:
                raise ValueError("encoded mode needs encoded_tensors, encoded_masks and labels.")
            if not (len(encoded_tensors) == len(encoded_masks) == len(labels)):
                raise ValueError("encoded_tensors, encoded_masks and labels must be equal length.")
            self._mode = "encoded"
            self.tensors = encoded_tensors
            self.masks = encoded_masks
            self.labels = [int(x) for x in labels]
        else:
            raise ValueError("provide either windows+labels or encoded_tensors+masks+labels.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self._mode == "dynamic":
            tensor, mask = self.encoder.encode_position_window(self.windows[index])
        else:
            tensor = np.asarray(self.tensors[index], dtype=np.float32)
            mask = np.asarray(self.masks[index], dtype=np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(tensor)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            self.labels[index],
        )


def create_dataloaders(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders (train shuffled; val/test deterministic)."""
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    train_loader = DataLoader(train, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader
