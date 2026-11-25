"""Unit tests for left-right mirror augmentation."""

from __future__ import annotations

from typing import List

import chess
import chess.pgn
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.augment import (
    DEFAULT_FLIP_LABELS,
    MIRROR_LABEL_MAP,
    MirrorAugmentedDataset,
    has_castling_rights,
    mirror_board,
    mirror_tensor,
)
from src.data.chess_dataset import ChessPlanDataset
from src.data.tensor_encoder import ChessTensorEncoder
from src.data.window_extractor import PositionWindow, extract_windows

_ALL = range(6)
# Ends with a pending en-passant target (d6) and has uncastled kings.
_OPENING = ["e4", "e6", "Nf3", "d5", "e5", "Nc6", "Bc4", "Nge7", "O-O", "Ng6", "d4", "f5"]


def make_game(moves: List[str]) -> chess.pgn.Game:
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    board = chess.Board()
    for san in moves:
        node = node.add_main_variation(board.push_san(san))
    return game


def make_windows() -> List[PositionWindow]:
    moves = ["e4", "e6", "Nf3", "d5", "e5", "c5", "Bc4", "Nc6", "O-O", "Nge7", "d4", "b5"]
    moves += ["Bb3", "a5", "a4", "b4", "c3", "bxc3", "bxc3", "Ba6"]
    return list(extract_windows(make_game(moves), game_id="g", start_ply=12))


def encoded_dataset(n: int = 24) -> ChessPlanDataset:
    rng = np.random.default_rng(0)
    tensors = rng.random((n, 112, 8, 8)).astype(np.float32)
    tensors[:, 12::14] = 0.0  # no castling rights, so every window may be mirrored
    masks = np.ones((n, 8), dtype=np.float32)
    return ChessPlanDataset(
        encoded_tensors=tensors, encoded_masks=masks, labels=[i % 6 for i in range(n)]
    )


def test_mirror_tensor_is_involution_and_preserves_type() -> None:
    x = torch.rand(112, 8, 8)
    y = mirror_tensor(x)
    assert y.shape == x.shape and y.dtype == torch.float32
    assert not torch.equal(x, y)
    assert torch.equal(mirror_tensor(y), x)

    arr = np.random.default_rng(1).random((112, 8, 8)).astype(np.float32)
    out = mirror_tensor(arr)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    assert np.array_equal(mirror_tensor(out), arr)


def test_mirror_tensor_rejects_non_board_shape() -> None:
    with pytest.raises(ValueError):
        mirror_tensor(torch.rand(112, 7, 8))


def test_label_map_swaps_storm_and_expansion_only() -> None:
    assert MIRROR_LABEL_MAP[0] == 1 and MIRROR_LABEL_MAP[1] == 0
    assert all(MIRROR_LABEL_MAP[i] == i for i in (2, 3, 4, 5))
    assert all(MIRROR_LABEL_MAP[MIRROR_LABEL_MAP[i]] == i for i in _ALL)


def test_default_excludes_storm_and_expansion() -> None:
    assert DEFAULT_FLIP_LABELS == frozenset({2, 3, 4, 5})


@pytest.mark.parametrize("attacker", [chess.WHITE, chess.BLACK])
def test_mirror_tensor_matches_encoding_of_mirrored_boards(attacker: bool) -> None:
    game = make_game(_OPENING)
    boards, board = [game.board()], game.board()
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy(stack=False))
    history = boards[-8:]
    assert history[-1].ep_square is not None  # exercises the en-passant plane

    encoder = ChessTensorEncoder()
    original, _ = encoder.encode_window(history, attacker)
    mirrored, _ = encoder.encode_window([mirror_board(b) for b in history], attacker)
    flipped = mirror_tensor(original)

    keep = np.ones(112, dtype=bool)
    keep[12::14] = False  # castling planes: mirrored boards have their rights cleared
    assert np.array_equal(flipped[keep], mirrored[keep])
    assert flipped[12::14].sum() > 0  # the flip itself preserves castling information


def test_p_zero_is_identity() -> None:
    base = encoded_dataset()
    ds = MirrorAugmentedDataset(base, p=0.0, only_flip_labels=_ALL)
    for i in range(len(base)):
        t0, m0, l0 = base[i]
        t1, m1, l1 = ds[i]
        assert torch.equal(t0, t1) and torch.equal(m0, m1) and l0 == l1


def test_p_one_always_flips_eligible_samples() -> None:
    base = encoded_dataset()
    ds = MirrorAugmentedDataset(base, p=1.0, only_flip_labels=_ALL)
    for i in range(len(base)):
        t0, m0, l0 = base[i]
        t1, m1, l1 = ds[i]
        assert torch.equal(t1, mirror_tensor(t0))
        assert torch.equal(m1, m0)  # validity mask is orientation-independent
        assert l1 == MIRROR_LABEL_MAP[l0]


def test_only_flip_labels_restricts_augmentation() -> None:
    base = encoded_dataset()
    ds = MirrorAugmentedDataset(base, p=1.0)  # default set
    for i in range(len(base)):
        t0, _, l0 = base[i]
        t1, _, l1 = ds[i]
        if l0 in DEFAULT_FLIP_LABELS:
            assert torch.equal(t1, mirror_tensor(t0)) and l1 == l0
        else:
            assert torch.equal(t1, t0) and l1 == l0


def test_labels_attribute_keeps_original_labels() -> None:
    base = encoded_dataset()
    assert MirrorAugmentedDataset(base, p=1.0, only_flip_labels=_ALL).labels == base.labels


def test_deterministic_for_fixed_seed() -> None:
    base = encoded_dataset()
    a = MirrorAugmentedDataset(base, p=0.5, seed=7, only_flip_labels=_ALL)
    b = MirrorAugmentedDataset(base, p=0.5, seed=7, only_flip_labels=_ALL)
    c = MirrorAugmentedDataset(base, p=0.5, seed=8, only_flip_labels=_ALL)
    out_a = [a[i % len(base)] for i in range(100)]
    out_b = [b[i % len(base)] for i in range(100)]
    out_c = [c[i % len(base)] for i in range(100)]
    assert all(torch.equal(x[0], y[0]) and x[2] == y[2] for x, y in zip(out_a, out_b))
    assert any(not torch.equal(x[0], y[0]) for x, y in zip(out_a, out_c))


def test_flip_rate_tracks_p() -> None:
    base = encoded_dataset()
    ds = MirrorAugmentedDataset(base, p=0.5, seed=3, only_flip_labels=_ALL)
    flips = sum(not torch.equal(ds[i % len(base)][0], base[i % len(base)][0]) for i in range(1000))
    assert 420 <= flips <= 580


def test_wraps_dynamic_mode_dataset() -> None:
    windows = make_windows()
    labels = [i % 6 for i in range(len(windows))]
    base = ChessPlanDataset(windows, labels)
    ds = MirrorAugmentedDataset(base, p=1.0, only_flip_labels=_ALL)
    tensor, mask, label = ds[0]
    assert tensor.shape == (112, 8, 8) and tensor.dtype == torch.float32
    assert mask.shape == (8,)
    # Black can still castle in this game, so the real window passes through.
    assert has_castling_rights(base[0][0])
    assert torch.equal(tensor, base[0][0]) and label == labels[0]


@pytest.mark.parametrize("slot", [0, 3, 7])
def test_window_with_castling_rights_in_any_ply_is_never_flipped(slot: int) -> None:
    tensors = np.zeros((6, 112, 8, 8), dtype=np.float32)
    tensors[:, 0, 6, 0] = 1.0  # an asymmetric piece so a flip would be visible
    tensors[:, slot * 14 + 12, 0:4, 0:4] = 1.0
    base = ChessPlanDataset(
        encoded_tensors=tensors, encoded_masks=np.ones((6, 8)), labels=list(range(6))
    )
    ds = MirrorAugmentedDataset(base, p=1.0, only_flip_labels=_ALL)
    for i in range(6):
        assert torch.equal(ds[i][0], base[i][0]) and ds[i][2] == i


def test_has_castling_rights_handles_sliced_windows() -> None:
    x = torch.zeros(56, 8, 8)
    assert not has_castling_rights(x)
    x[2 * 14 + 12, 5, 5] = 1.0
    assert has_castling_rights(x)
    with pytest.raises(ValueError):
        has_castling_rights(torch.zeros(20, 8, 8))


def test_invalid_probability_rejected() -> None:
    with pytest.raises(ValueError):
        MirrorAugmentedDataset(encoded_dataset(), p=1.5)


def test_dataloader_workers_produce_valid_mixed_batches() -> None:
    base = encoded_dataset(32)
    ds = MirrorAugmentedDataset(base, p=0.5, seed=1, only_flip_labels=_ALL)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    flipped = 0
    for batch_idx, (tensors, masks, labels) in enumerate(loader):
        assert tensors.shape == (8, 112, 8, 8) and masks.shape == (8, 8)
        for j in range(8):
            t0, _, l0 = base[batch_idx * 8 + j]
            if torch.equal(tensors[j], t0):
                assert int(labels[j]) == l0
            else:
                assert torch.equal(tensors[j], mirror_tensor(t0))
                assert int(labels[j]) == MIRROR_LABEL_MAP[l0]
                flipped += 1
    assert 0 < flipped < 32
