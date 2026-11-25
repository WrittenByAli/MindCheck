"""Left-right mirror augmentation for the training split only.

Mirroring a board across the vertical axis (a-file <-> h-file) maps a kingside
plan onto a queenside plan and vice versa, so a mirrored window is a plausible
extra training sample whose label is ``MIRROR_LABEL_MAP[label]``. Because the
tensor already uses attacker-perspective orientation (a vertical flip), the
horizontal flip commutes with it and is a plain reversal of the last axis. The
castling plane stores K/Q rights in left/right quadrants, so the same reversal
swaps them consistently.

The weak labels come from a rule engine that is *not* exactly mirror-equivariant
(the storm rule has a king-exposure term, the expansion rule a piece-support
term), so the label swap is only applied to ``only_flip_labels``. The default
set was chosen from :func:`measure_label_equivariance` on train-split games.

A window with any castling right in any ply it contains is never mirrored: the
flipped position would put a king on the d-file while still holding castling
rights, which cannot occur in a real game. Since only some classes are eligible,
a network could otherwise learn that impossible pattern as a label shortcut.

Only ever wrap the **training** dataset in :class:`MirrorAugmentedDataset`;
validation and test splits must stay untouched so their metrics are honest.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

__all__ = [
    "CLASS_NAMES",
    "MIRROR_LABEL_MAP",
    "DEFAULT_FLIP_LABELS",
    "mirror_tensor",
    "mirror_board",
    "has_castling_rights",
    "MirrorAugmentedDataset",
    "measure_label_equivariance",
]

CLASS_NAMES = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"]
MIRROR_LABEL_MAP: Dict[int, int] = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5}

# Measured on 1200 train-split windows (seed 0): K_STORM->Q_EXP agreement 96.5%,
# but Q_EXP->K_STORM only 64.5% (33% of mirrored Q_EXP windows become NO_PLAN),
# so labels 0/1 are excluded by default. Others: OUTPOST 98.0%, ROOK_FILE 98.1%,
# CHAIN_UNDERMINE 93.0%, NO_PLAN 88.8%. Pass range(6) to mirror every class.
DEFAULT_FLIP_LABELS: FrozenSet[int] = frozenset({2, 3, 4, 5})

_CASTLING_PLANE = 12
_PLANES_PER_PLY = 14

ArrayLike = Union[torch.Tensor, np.ndarray]


def mirror_tensor(x: ArrayLike) -> ArrayLike:
    """Flip an encoded window along the file axis (a-file <-> h-file)."""
    if x.shape[-2:] != (8, 8):
        raise ValueError(f"expected trailing (8, 8) board dims, got {tuple(x.shape)}.")
    if isinstance(x, torch.Tensor):
        return torch.flip(x, dims=[-1])
    return np.ascontiguousarray(x[..., ::-1])


def has_castling_rights(x: ArrayLike) -> bool:
    """True if any castling plane of a ``(C, 8, 8)`` window (whole plies) is set."""
    if x.shape[0] % _PLANES_PER_PLY:
        raise ValueError(f"expected whole 14-plane plies, got {x.shape[0]} channels.")
    return bool(x[_CASTLING_PLANE::_PLANES_PER_PLY].any())


def mirror_board(board: chess.Board) -> chess.Board:
    """Horizontally mirrored copy of ``board`` with castling rights cleared.

    Mirrored kings no longer sit on their castling squares, so keeping the
    transformed rights would describe an inconsistent position.
    """
    mirrored = board.transform(chess.flip_horizontal)
    mirrored.castling_rights = 0
    return mirrored


class MirrorAugmentedDataset(Dataset):
    """Wrap a dataset so samples are randomly mirrored (training split only).

    Args:
        base: A dataset yielding ``(tensor, mask, label)`` with a ``.labels``
            list (e.g. :class:`~src.data.chess_dataset.ChessPlanDataset`).
        p: Probability of mirroring an eligible sample.
        seed: Seed of the augmentation RNG. In DataLoader workers the RNG is
            re-derived from the worker's torch seed, which changes every epoch.
        only_flip_labels: Label ids eligible for mirroring; others pass through.
            Windows with any castling right always pass through.
    """

    def __init__(
        self,
        base: Dataset,
        p: float = 0.5,
        seed: int = 42,
        only_flip_labels: Iterable[int] = DEFAULT_FLIP_LABELS,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}.")
        self.base = base
        self.p = p
        self.seed = seed
        self.only_flip_labels = frozenset(int(x) for x in only_flip_labels)
        self.labels: List[int] = list(base.labels)
        self._main_rng: Optional[np.random.Generator] = None
        self._worker_key: Optional[int] = None
        self._worker_rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self.base)

    def _rng(self) -> np.random.Generator:
        if get_worker_info() is None:
            if self._main_rng is None:
                self._main_rng = np.random.default_rng(self.seed)
            return self._main_rng
        key = torch.initial_seed()
        if self._worker_key != key:
            self._worker_key = key
            self._worker_rng = np.random.default_rng([self.seed, key])
        return self._worker_rng

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tensor, mask, label = self.base[index]
        # One draw per call, eligible or not, keeps the RNG stream stable.
        draw = self._rng().random()
        if label in self.only_flip_labels and draw < self.p and not has_castling_rights(tensor):
            return mirror_tensor(tensor), mask, MIRROR_LABEL_MAP[label]
        return tensor, mask, label


def measure_label_equivariance(
    pgn_path: Union[str, Path],
    n_windows: int = 1200,
    split_seed: int = 42,
    seed: int = 0,
    max_per_game: int = 3,
) -> Dict:
    """Check whether mirroring a window mirrors its rule-engine class.

    Samples windows from train-split games only, evaluates the rule engine on
    each original and its mirrored history, and reports how often the mirrored
    class equals ``MIRROR_LABEL_MAP[original class]``. Also counts piece/en-passant
    plane mismatches between ``mirror_tensor(encode(w))`` and ``encode(mirror(w))``.
    """
    from src.data.dataset_splitter import GameLevelSplitter
    from src.data.pgn_parser import StreamingPGNParser
    from src.data.tensor_encoder import ChessTensorEncoder
    from src.data.window_extractor import default_game_id, extract_windows
    from src.rules.rule_engine import SymbolicRuleEngine

    rng = random.Random(seed)
    splitter = GameLevelSplitter(seed=split_seed)
    pool = []
    for game in StreamingPGNParser().stream_games(pgn_path):
        gid = default_game_id(game)
        if splitter.assign_split(gid) != "train":
            continue
        windows = list(extract_windows(game, game_id=gid))
        pool.extend(rng.sample(windows, min(max_per_game, len(windows))))
    sampled = rng.sample(pool, min(n_windows, len(pool)))

    engine = SymbolicRuleEngine()
    encoder = ChessTensorEncoder()
    confusion: Counter = Counter()
    plane_mismatches = 0
    keep = np.ones(encoder.total_channels, dtype=bool)
    for slot in range(encoder.history_length):
        keep[slot * _PLANES_PER_PLY + _CASTLING_PLANE] = False
    for w in sampled:
        mirrored_history = [mirror_board(b) for b in w.board_history]
        orig = engine.evaluate_window(w.board_history).class_id
        mirr = engine.evaluate_window(mirrored_history).class_id
        confusion[(orig, mirr)] += 1

        tensor, _ = encoder.encode_position_window(w)
        mirrored_tensor, _ = encoder.encode_window(mirrored_history, w.turn)
        if not np.array_equal(mirror_tensor(tensor)[keep], mirrored_tensor[keep]):
            plane_mismatches += 1

    per_class: Dict[str, Dict] = {}
    for cid, name in enumerate(CLASS_NAMES):
        n = sum(c for (o, _), c in confusion.items() if o == cid)
        agree = confusion[(cid, MIRROR_LABEL_MAP[cid])]
        per_class[name] = {
            "n": n,
            "agreement": round(agree / n, 4) if n else None,
            "mirrored_as": {
                CLASS_NAMES[m]: c for (o, m), c in sorted(confusion.items()) if o == cid
            },
        }
    total = sum(confusion.values())
    agree_all = sum(confusion[(c, MIRROR_LABEL_MAP[c])] for c in range(len(CLASS_NAMES)))
    return {
        "n_windows": total,
        "n_train_pool": len(pool),
        "overall_agreement": round(agree_all / total, 4) if total else None,
        "per_class": per_class,
        "plane_mismatches": plane_mismatches,
    }
