"""BatchLoader must be a faster, semantically identical stand-in for the Dataset path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.augment import MIRROR_LABEL_MAP, mirror_tensor
from src.data.chess_dataset import ChessPlanDataset
from src.data.fast_loader import BatchLoader
from src.models.train_ablation import ChannelSliceDataset

N = 53
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    tensors = (rng.random((N, 112, 8, 8)) < 0.1).astype(np.uint8)
    tensors[:, 12::14] = 0  # no castling rights unless a test adds them
    labels = rng.integers(0, 6, size=N)
    path = tmp_path_factory.mktemp("cache") / "tensors.npy"
    np.save(path, tensors)
    return path, tensors, labels


def _collect(loader):
    xs, ys = [], []
    for x, mask, y in loader:
        assert mask is None
        assert x.dtype == torch.float32 and y.dtype == torch.int64
        xs.append(x.cpu())
        ys.append(y.cpu())
    return torch.cat(xs), torch.cat(ys)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("storage", ["auto", "ram", "memmap"])
@pytest.mark.parametrize("keep", [14, 56, 112])
def test_matches_dataset_path_for_selected_rows(cache, keep, storage, device) -> None:
    path, tensors, labels = cache
    rows = np.array([40, 3, 17, 17, 52, 0, 9, 31, 2])  # unsorted, with a repeat
    dataset = ChessPlanDataset(
        encoded_tensors=tensors[rows], encoded_masks=np.ones((len(rows), 8)), labels=labels[rows]
    )
    reference = ChannelSliceDataset(dataset, keep)
    x, y = _collect(
        BatchLoader(path, rows, labels[rows], keep, batch_size=4, storage=storage, device=device)
    )
    assert x.shape == (len(rows), keep, 8, 8)
    for i in range(len(rows)):
        ref_x, _mask, ref_y = reference[i]
        assert torch.equal(x[i], ref_x)
        assert int(y[i]) == ref_y


@pytest.mark.parametrize("device", DEVICES)
def test_mirror_matches_mirror_tensor_and_respects_eligibility(cache, device) -> None:
    path, tensors, labels = cache
    idx = np.arange(N)
    x, y = _collect(BatchLoader(path, idx, labels, 56, batch_size=8, mirror_p=1.0, device=device))
    for i in range(N):
        original = torch.from_numpy(tensors[i, -56:].astype(np.float32))
        eligible = int(labels[i]) in {2, 3, 4, 5}
        expected = mirror_tensor(original) if eligible else original
        assert torch.equal(x[i], expected)
        assert int(y[i]) == (MIRROR_LABEL_MAP[int(labels[i])] if eligible else int(labels[i]))


@pytest.mark.parametrize("device", DEVICES)
def test_windows_with_castling_rights_are_never_mirrored(cache, device) -> None:
    _path, tensors, _labels = cache
    arr = tensors.copy()
    arr[0, 7 * 14 + 12, 0:4, 0:4] = 1  # newest ply: inside the 4-ply slice
    arr[1, 4 * 14 + 12, 4:8, 4:8] = 1  # oldest ply of the 4-ply slice
    arr[2, 0 * 14 + 12, 0:4, 0:4] = 1  # outside the 4-ply slice
    labels = np.full(N, 5)
    for keep, blocked in ((56, {0, 1}), (112, {0, 1, 2})):
        x, _ = _collect(BatchLoader(arr, np.arange(N), labels, keep, mirror_p=1.0, device=device))
        for i in range(4):
            original = torch.from_numpy(arr[i, -keep:].astype(np.float32))
            expected = original if i in blocked else mirror_tensor(original)
            assert torch.equal(x[i], expected), (keep, i)


def test_mirror_label_map_applies_when_all_labels_eligible(cache) -> None:
    path, _tensors, labels = cache
    _x, y = _collect(BatchLoader(path, np.arange(N), labels, mirror_p=1.0, mirror_labels=range(6)))
    assert y.tolist() == [MIRROR_LABEL_MAP[int(v)] for v in labels]


def test_mirror_probability_is_roughly_half_and_zero_is_identity(cache) -> None:
    path, tensors, _labels = cache
    idx = np.tile(np.arange(N), 40)
    labels = np.full(len(idx), 5)
    x, _ = _collect(BatchLoader(path, idx, labels, 14, batch_size=64, mirror_p=0.5, seed=1))
    plain = torch.from_numpy(tensors[idx, -14:].astype(np.float32))
    changed = (x != plain).flatten(1).any(1).float().mean().item()
    assert 0.44 < changed < 0.56
    x0, _ = _collect(BatchLoader(path, idx, labels, 14, batch_size=64, mirror_p=0.0))
    assert torch.equal(x0, plain)


@pytest.mark.parametrize("device", DEVICES)
def test_shuffled_epoch_covers_every_sample_once_and_reshuffles(cache, device) -> None:
    path, _tensors, _labels = cache
    ids = np.arange(N)  # use the label slot as a sample id (no mirroring)
    loader = BatchLoader(path, ids, ids, 14, batch_size=10, shuffle=True, seed=3, device=device)
    assert len(loader) == 6
    first = _collect(loader)[1].tolist()
    second = _collect(loader)[1].tolist()
    assert sorted(first) == sorted(second) == list(range(N))
    assert first != second
    again = BatchLoader(path, ids, ids, 14, batch_size=10, shuffle=True, seed=3, device=device)
    assert _collect(again)[1].tolist() == first


def test_cpu_auto_storage_falls_back_to_memmap_over_ram_budget(cache) -> None:
    path, _tensors, labels = cache
    idx = np.arange(N)
    assert BatchLoader(path, idx, labels).storage == "ram"
    assert BatchLoader(path, idx, labels, max_ram_bytes=10).storage == "memmap"


def test_invalid_arguments_raise(cache) -> None:
    path, _tensors, labels = cache
    with pytest.raises(ValueError):
        BatchLoader(path, np.arange(N), labels[:-1])
    with pytest.raises(ValueError):
        BatchLoader(path, np.arange(N), labels, keep_channels=113)
    with pytest.raises(ValueError):
        BatchLoader(path, np.arange(N), labels, storage="gpu")


def test_model_trainer_consumes_batch_loader(cache) -> None:
    from src.models.residual_cnn import HistoryChessResNet
    from src.models.train import ModelTrainer

    path, _tensors, labels = cache
    for device in DEVICES:
        loader = BatchLoader(path, np.arange(N), labels, 56, batch_size=16, device=device)
        trainer = ModelTrainer(
            HistoryChessResNet(in_channels=56), loader, loader, epochs=1, device=device
        )
        loss, acc = trainer.train_epoch()
        assert np.isfinite(loss) and 0.0 <= acc <= 1.0
        assert 0.0 <= trainer.evaluate(loader)[1] <= 1.0
