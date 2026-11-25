"""Batch-level loader for cached window tensors, built for GPU training.

The per-sample ``Dataset``/``DataLoader`` path spends most of its time in Python
indexing a memmap one window at a time. :class:`BatchLoader` instead loads only
the channels a model needs, once, as ``uint8`` (the planes are binary) into VRAM
when it fits, else into RAM, else leaves them memmapped, and then indexes whole
batches. Mirror augmentation is vectorized with the same semantics as
:class:`~src.data.augment.MirrorAugmentedDataset`: only eligible labels, and never
a window with a castling right in any ply the (sliced) tensor contains.

Batches are ``(x_float32, None, labels_int64)`` so ``ModelTrainer`` consumes them
unchanged (it ignores the mask slot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.data.augment import DEFAULT_FLIP_LABELS, MIRROR_LABEL_MAP

__all__ = ["BatchLoader"]

VRAM_HEADROOM_BYTES = int(1.5 * 1024**3)
MAX_RAM_BYTES = int(4.5 * 1024**3)
_READ_CHUNK = 20_000

Batch = Tuple[torch.Tensor, None, torch.Tensor]


class BatchLoader:
    """Iterate whole batches of cached ``(N, C, 8, 8)`` windows.

    Args:
        tensors: Path to an ``.npy`` cache (or an array) of shape ``(M, C, 8, 8)``.
        indices: Row numbers to serve, in the order they are served when
            ``shuffle`` is False.
        labels: Class id per entry of ``indices`` (same length).
        keep_channels: Serve only the last ``keep_channels`` planes (newest plies).
        batch_size: Samples per batch; the final batch may be smaller.
        shuffle: Reshuffle every epoch with a seeded generator.
        mirror_p: Probability of mirroring an eligible sample (training only).
        mirror_labels: Label ids eligible for mirroring.
        seed: Seed for shuffling and augmentation.
        device: Device the batches are returned on.
        storage: ``auto`` | ``device`` | ``ram`` | ``memmap``.
    """

    def __init__(
        self,
        tensors: Union[str, Path, np.ndarray],
        indices: Sequence[int],
        labels: Sequence[int],
        keep_channels: int = 112,
        batch_size: int = 128,
        shuffle: bool = False,
        mirror_p: float = 0.0,
        mirror_labels: Iterable[int] = DEFAULT_FLIP_LABELS,
        seed: int = 42,
        device: Union[str, torch.device] = "cpu",
        storage: str = "auto",
        max_ram_bytes: int = MAX_RAM_BYTES,
        vram_headroom_bytes: int = VRAM_HEADROOM_BYTES,
    ) -> None:
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have equal length.")
        if not 0.0 <= mirror_p <= 1.0:
            raise ValueError(f"mirror_p must be in [0, 1], got {mirror_p}.")
        if storage not in ("auto", "device", "ram", "memmap"):
            raise ValueError(f"unknown storage mode {storage!r}.")
        self._mm = (
            tensors if isinstance(tensors, np.ndarray) else np.load(str(tensors), mmap_mode="r")
        )
        total_channels = self._mm.shape[1]
        if not 0 < keep_channels <= total_channels:
            raise ValueError(f"keep_channels must be in 1..{total_channels}.")
        self.first_channel = total_channels - keep_channels
        self.keep_channels = keep_channels
        # Castling planes that survive the slice (plane 12 of each 14-plane ply).
        self._castling_channels = [
            c for c in range(keep_channels) if (self.first_channel + c) % 14 == 12
        ]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = [int(v) for v in labels]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mirror_p = mirror_p
        self.device = torch.device(device)

        self._y = torch.tensor(self.labels, dtype=torch.long, device=self.device)
        eligible = torch.zeros(len(MIRROR_LABEL_MAP), dtype=torch.bool)
        eligible[list(mirror_labels)] = True
        self._eligible = eligible.to(self.device)
        self._label_map = torch.tensor(
            [MIRROR_LABEL_MAP[i] for i in range(len(MIRROR_LABEL_MAP))],
            dtype=torch.long,
            device=self.device,
        )
        self._order_rng = torch.Generator().manual_seed(seed)
        self._flip_rng = torch.Generator(device=self.device).manual_seed(seed + 1)

        self.storage = self._choose_storage(storage, max_ram_bytes, vram_headroom_bytes)
        self._data: Optional[torch.Tensor] = None
        if self.storage != "memmap":
            self._data = self._materialize()

    # ------------------------------------------------------------------ setup
    @property
    def nbytes(self) -> int:
        return len(self.indices) * self.keep_channels * 64

    def _choose_storage(self, requested: str, max_ram: int, headroom: int) -> str:
        if requested != "auto":
            return requested
        if self.device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self.device)
            if self.nbytes + headroom <= free:
                return "device"
        return "ram" if self.nbytes <= max_ram else "memmap"

    def _read(self, rows: np.ndarray) -> np.ndarray:
        """Read ``rows`` (any order) of the kept channels, hitting disk in order."""
        order = np.argsort(rows, kind="stable")
        block = np.asarray(self._mm[rows[order], self.first_channel :], dtype=np.uint8)
        out = np.empty_like(block)
        out[order] = block
        return out

    def _materialize(self) -> torch.Tensor:
        target = self.device if self.storage == "device" else torch.device("cpu")
        data = torch.empty(
            (len(self.indices), self.keep_channels, 8, 8), dtype=torch.uint8, device=target
        )
        for start in range(0, len(self.indices), _READ_CHUNK):
            rows = self.indices[start : start + _READ_CHUNK]
            data[start : start + len(rows)] = torch.from_numpy(self._read(rows)).to(target)
        return data

    # -------------------------------------------------------------- iteration
    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        n = len(self.indices)
        order = torch.randperm(n, generator=self._order_rng) if self.shuffle else torch.arange(n)
        for start in range(0, n, self.batch_size):
            pos = order[start : start + self.batch_size]
            if self._data is None:
                raw = torch.from_numpy(self._read(self.indices[pos.numpy()]))
            else:
                raw = self._data[pos.to(self._data.device)]
            x = raw.to(self.device, non_blocking=True).float()
            y = self._y[pos.to(self.device)]
            if self.mirror_p > 0.0:
                x, y = self._mirror(x, y)
            yield x, None, y

    def _mirror(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        draw = torch.rand(len(y), generator=self._flip_rng, device=self.device)
        castling = x[:, self._castling_channels].flatten(1).any(dim=1)
        flip = (draw < self.mirror_p) & self._eligible[y] & ~castling
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return x, torch.where(flip, self._label_map[y], y)
