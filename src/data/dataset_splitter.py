"""Leak-free game-level train/val/test splitter (Task 1.5, Subtask 1.5.3).

Consecutive plies of one game are highly correlated, so splitting at the
*position* level would leak near-duplicate windows across train/val/test. This
module therefore partitions at the **game** level: every window of a given game
is assigned to exactly one split, chosen by a deterministic hash of the game id.

``hashlib`` is used rather than the builtin ``hash()`` because ``hash()`` on
strings is randomized per interpreter run (PYTHONHASHSEED), which would make
splits non-reproducible.

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Set

from src.data.weak_labeler import LabeledPositionRecord

__all__ = ["GameLevelSplitter"]


class GameLevelSplitter:
    """Assign whole games to train / val / test by a deterministic hash."""

    def __init__(
        self,
        train: float = 0.70,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
    ) -> None:
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}.")
        self.train = train
        self.val = val
        self.test = test
        self.seed = seed
        self._train_cut = train * 100.0
        self._val_cut = (train + val) * 100.0

    def assign_split(self, game_id: str) -> str:
        """Return 'train' / 'val' / 'test' for a game id (stable across runs)."""
        digest = hashlib.sha256(f"{game_id}:{self.seed}".encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % 100
        if bucket < self._train_cut:
            return "train"
        if bucket < self._val_cut:
            return "val"
        return "test"

    def split(self, records: List[LabeledPositionRecord]) -> List[LabeledPositionRecord]:
        """Set ``record.split`` for every record in place and return the list."""
        cache: Dict[str, str] = {}
        for record in records:
            split = cache.get(record.game_id)
            if split is None:
                split = self.assign_split(record.game_id)
                cache[record.game_id] = split
            record.split = split
        return records

    def games_by_split(self, game_ids: Iterable[str]) -> Dict[str, Set[str]]:
        """Group unique game ids into their assigned splits."""
        out: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        for gid in set(game_ids):
            out[self.assign_split(gid)].add(gid)
        return out

    def verify_disjoint(self, records: Iterable[LabeledPositionRecord]) -> bool:
        """Return True iff no game id appears in more than one split."""
        seen: Dict[str, str] = {}
        for record in records:
            if not record.split:
                raise ValueError("records must be split before verification.")
            prior = seen.setdefault(record.game_id, record.split)
            if prior != record.split:
                return False
        return True
