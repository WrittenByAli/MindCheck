"""Unit tests for ChessPlanDataset and create_dataloaders (Subtask 1.6.4)."""

from __future__ import annotations

from typing import List

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from src.data.chess_dataset import ChessPlanDataset, create_dataloaders
from src.data.window_extractor import PositionWindow, extract_windows

_SHUFFLE = ["Nf3", "Nf6", "Ng1", "Ng8"]


def make_windows(n_moves: int = 24) -> List[PositionWindow]:
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    board = chess.Board()
    for i in range(n_moves):
        move = board.push_san(_SHUFFLE[i % 4])
        node = node.add_main_variation(move)
    return list(extract_windows(game, game_id="g", start_ply=12))


# --------------------------------------------------------------------------- #
# Dynamic mode                                                                 #
# --------------------------------------------------------------------------- #
def test_dynamic_getitem_shapes() -> None:
    windows = make_windows()
    labels = [i % 6 for i in range(len(windows))]
    ds = ChessPlanDataset(windows, labels)
    tensor, mask, label = ds[0]
    assert tensor.shape == (112, 8, 8)
    assert tensor.dtype == torch.float32
    assert mask.shape == (8,)
    assert isinstance(label, int)
    assert len(ds) == len(windows)


def test_dynamic_batch_shapes() -> None:
    windows = make_windows()
    labels = [i % 6 for i in range(len(windows))]
    ds = ChessPlanDataset(windows, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    tensors, masks, label_batch = next(iter(loader))
    assert tensors.shape == (4, 112, 8, 8)
    assert masks.shape == (4, 8)
    assert label_batch.shape == (4,)


# --------------------------------------------------------------------------- #
# Encoded mode                                                                 #
# --------------------------------------------------------------------------- #
def test_encoded_mode_getitem() -> None:
    tensors = np.zeros((5, 112, 8, 8), dtype=np.float32)
    masks = np.ones((5, 8), dtype=np.float32)
    labels = [0, 1, 2, 3, 4]
    ds = ChessPlanDataset(encoded_tensors=tensors, encoded_masks=masks, labels=labels)
    tensor, mask, label = ds[2]
    assert tensor.shape == (112, 8, 8)
    assert mask.shape == (8,)
    assert label == 2


def test_bad_construction_raises() -> None:
    with pytest.raises(ValueError):
        ChessPlanDataset(windows=[1, 2], labels=[0])  # length mismatch
    with pytest.raises(ValueError):
        ChessPlanDataset()  # nothing provided


# --------------------------------------------------------------------------- #
# create_dataloaders                                                           #
# --------------------------------------------------------------------------- #
def test_create_dataloaders_batches() -> None:
    windows = make_windows()
    labels = [i % 6 for i in range(len(windows))]
    ds = ChessPlanDataset(windows, labels)
    train, val, test = create_dataloaders(ds, ds, ds, batch_size=8)
    for loader in (train, val, test):
        tensors, masks, label_batch = next(iter(loader))
        assert tensors.shape[1:] == (112, 8, 8)
        assert masks.shape[1] == 8
        assert label_batch.ndim == 1


def test_multi_epoch_iteration_is_stable() -> None:
    windows = make_windows()
    labels = [i % 6 for i in range(len(windows))]
    ds = ChessPlanDataset(windows, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    counts = []
    for _ in range(3):
        counts.append(sum(t.shape[0] for t, _, _ in loader))
    assert counts[0] == counts[1] == counts[2] == len(windows)
