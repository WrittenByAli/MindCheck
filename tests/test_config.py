"""Unit tests for the type-safe configuration loader (Subtask 1.1.3)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import AppConfig, SplitConfig, load_config

# Resolve the real project config relative to this test file, so the tests pass
# regardless of the working directory pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="module")
def raw_config() -> dict:
    """The real config.yaml parsed into a plain dict (for mutation in tests)."""
    with REAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(tmp_path: Path, data: dict) -> Path:
    """Serialize a config dict to a temp YAML file and return its path."""
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_load_real_config_returns_appconfig() -> None:
    cfg = load_config(REAL_CONFIG_PATH)
    assert isinstance(cfg, AppConfig)


def test_loaded_values_and_types() -> None:
    cfg = load_config(REAL_CONFIG_PATH)

    # Nested typed access with correct runtime types.
    assert cfg.project.name == "MindCheck"
    assert isinstance(cfg.dataset.min_player_elo, int)
    assert cfg.dataset.min_player_elo == 2200
    assert isinstance(cfg.training.learning_rate, float)
    assert cfg.model.in_channels == 112
    assert cfg.model.in_channels == cfg.model.history_length * cfg.model.planes_per_ply
    assert isinstance(cfg.fusion.alpha, float)
    assert cfg.ui.theme in {"dark", "light"}


def test_splits_sum_to_one() -> None:
    cfg = load_config(REAL_CONFIG_PATH)
    total = cfg.dataset.splits.train + cfg.dataset.splits.val + cfg.dataset.splits.test
    assert total == pytest.approx(1.0)


def test_num_classes_matches_taxonomy() -> None:
    cfg = load_config(REAL_CONFIG_PATH)
    assert cfg.model.num_classes == len(cfg.classes.taxonomy)
    assert cfg.model.num_classes == 6


def test_default_path_loads_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # Calling load_config() with no argument uses config/config.yaml relative to cwd.
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_config_is_immutable() -> None:
    cfg = load_config(REAL_CONFIG_PATH)
    with pytest.raises(ValidationError):
        cfg.model.dropout = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Error handling / validation                                                  #
# --------------------------------------------------------------------------- #
def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_non_mapping_root_raises_valueerror(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_bad_type_triggers_validation_error(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["training"]["learning_rate"] = "not-a-number"
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_missing_key_triggers_validation_error(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    del data["model"]["in_channels"]
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_unknown_key_is_forbidden(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["training"]["mystery_flag"] = True
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_splits_not_summing_to_one_rejected(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["dataset"]["splits"] = {"train": 0.5, "val": 0.2, "test": 0.2}  # sums to 0.9
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_in_channels_mismatch_rejected(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["model"]["in_channels"] = 100  # != 8 * 14
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_num_classes_mismatch_rejected(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["model"]["num_classes"] = 7  # taxonomy has 6
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_out_of_range_dropout_rejected(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["model"]["dropout"] = 1.5
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_invalid_device_literal_rejected(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["training"]["device"] = "tpu"
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_non_contiguous_taxonomy_ids_rejected(tmp_path: Path, raw_config: dict) -> None:
    data = copy.deepcopy(raw_config)
    data["classes"]["taxonomy"][0]["id"] = 99
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


# --------------------------------------------------------------------------- #
# Direct model unit tests                                                      #
# --------------------------------------------------------------------------- #
def test_splitconfig_valid() -> None:
    split = SplitConfig(train=0.7, val=0.15, test=0.15)
    assert split.train == 0.7


def test_splitconfig_invalid_sum() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(train=0.6, val=0.6, test=0.6)
