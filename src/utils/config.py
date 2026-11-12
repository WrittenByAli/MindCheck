"""Type-safe runtime configuration for MindCheck.

This module loads ``config/config.yaml`` into a strongly-typed Pydantic v2
model tree (:class:`AppConfig`). Every section of the YAML is validated on
load, so misspelled keys, out-of-range values, or structurally inconsistent
settings fail fast with a clear error instead of surfacing as silent bugs deep
inside the training or inference pipelines.

Typical usage::

    from src.utils.config import load_config

    cfg = load_config()                 # reads config/config.yaml
    print(cfg.model.in_channels)        # 112
    print(cfg.dataset.splits.train)     # 0.70

Subtask: 1.1.3 (Milestone 1). Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "TeamMember",
    "ProjectConfig",
    "SplitConfig",
    "DatasetConfig",
    "PlanClass",
    "ClassesConfig",
    "ModelConfig",
    "TrainingConfig",
    "TrackingConfig",
    "FusionConfig",
    "UIConfig",
    "AppConfig",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]

DEFAULT_CONFIG_PATH = "config/config.yaml"

# A small tolerance for float comparisons (e.g. train/val/test split sums).
_SPLIT_SUM_TOLERANCE = 1e-6


class _StrictModel(BaseModel):
    """Base model that forbids unknown keys so typos are caught immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TeamMember(_StrictModel):
    """A single Group 29 team member entry."""

    name: str
    roll_no: str
    role: str


class ProjectConfig(_StrictModel):
    """Top-level project metadata."""

    name: str
    version: str
    path: str
    team: List[TeamMember] = Field(min_length=1)


class SplitConfig(_StrictModel):
    """Train/validation/test fractions. Must sum to 1.0."""

    train: float = Field(gt=0.0, lt=1.0)
    val: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _splits_sum_to_one(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > _SPLIT_SUM_TOLERANCE:
            raise ValueError(
                f"Dataset splits must sum to 1.0, got {total:.6f} "
                f"(train={self.train}, val={self.val}, test={self.test})."
            )
        return self


class DatasetConfig(_StrictModel):
    """Dataset acquisition, filtering, and windowing parameters."""

    raw_pgn_path: str
    processed_dir: str
    manifest_path: str
    annotations_path: str
    min_game_plies: int = Field(ge=1)
    min_player_elo: int = Field(ge=0)
    max_positions_per_game: int = Field(ge=1)
    history_plies: int = Field(ge=1)
    lookahead_plies: int = Field(ge=1)
    splits: SplitConfig


class PlanClass(_StrictModel):
    """A single strategic plan class in the taxonomy."""

    id: int = Field(ge=0)
    name: str
    description: str


class ClassesConfig(_StrictModel):
    """The strategic plan taxonomy (5 plans + NO_PLAN)."""

    taxonomy: List[PlanClass] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_contiguous_and_unique(self) -> "ClassesConfig":
        ids = [plan.id for plan in self.taxonomy]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Taxonomy class ids must be unique, got {ids}.")
        expected = list(range(len(self.taxonomy)))
        if sorted(ids) != expected:
            raise ValueError(
                f"Taxonomy class ids must be contiguous from 0, got {sorted(ids)} "
                f"(expected {expected})."
            )
        return self


class ModelConfig(_StrictModel):
    """Residual CNN architecture and checkpoint paths."""

    name: str
    history_length: int = Field(ge=1)
    planes_per_ply: int = Field(ge=1)
    in_channels: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    channels: int = Field(ge=1)
    num_blocks: int = Field(ge=1)
    dropout: float = Field(ge=0.0, le=1.0)
    checkpoint_dir: str
    best_checkpoint_path: str
    onnx_export_path: str

    @model_validator(mode="after")
    def _in_channels_matches_history(self) -> "ModelConfig":
        expected = self.history_length * self.planes_per_ply
        if self.in_channels != expected:
            raise ValueError(
                f"in_channels ({self.in_channels}) must equal "
                f"history_length * planes_per_ply "
                f"({self.history_length} * {self.planes_per_ply} = {expected})."
            )
        return self


class TrainingConfig(_StrictModel):
    """Optimization and training loop hyperparameters."""

    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    max_epochs: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=1)
    seed: int = Field(ge=0)
    device: Literal["cuda", "cpu", "auto"]
    label_smoothing: float = Field(ge=0.0, le=1.0)
    num_workers: int = Field(ge=0)


class TrackingConfig(_StrictModel):
    """Temporal plan state-machine hysteresis parameters."""

    hysteresis_activation_count: int = Field(ge=1)
    decay_count: int = Field(ge=1)
    min_plan_confidence: float = Field(ge=0.0, le=1.0)


class FusionConfig(_StrictModel):
    """Neural-rule score fusion and abstention gating parameters."""

    alpha: float = Field(ge=0.0, le=1.0)
    abstention_threshold: float = Field(ge=0.0, le=1.0)
    margin_threshold: float = Field(ge=0.0, le=1.0)


class UIConfig(_StrictModel):
    """Streamlit UI and Stockfish defaults."""

    theme: Literal["dark", "light"]
    default_game_index: int = Field(ge=0)
    stockfish_enabled_default: bool
    stockfish_depth: int = Field(ge=1)
    stockfish_time_limit: float = Field(gt=0.0)


class AppConfig(_StrictModel):
    """Aggregate root configuration for the entire MindCheck system."""

    project: ProjectConfig
    dataset: DatasetConfig
    classes: ClassesConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    fusion: FusionConfig
    ui: UIConfig

    @model_validator(mode="after")
    def _num_classes_matches_taxonomy(self) -> "AppConfig":
        n_taxonomy = len(self.classes.taxonomy)
        if self.model.num_classes != n_taxonomy:
            raise ValueError(
                f"model.num_classes ({self.model.num_classes}) must equal the "
                f"number of taxonomy classes ({n_taxonomy})."
            )
        return self

    @field_validator("project", mode="before")
    @classmethod
    def _reject_null_section(cls, value: object) -> object:
        # Only catches an explicit `project: null` in the YAML. A fully
        # absent 'project' key is already rejected by Pydantic's own
        # "field required" error before this validator would run.
        if value is None:
            raise ValueError("'project' section in config must not be null.")
        return value


def load_config(yaml_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate a MindCheck configuration file.

    Args:
        yaml_path: Path to the YAML configuration file. Defaults to
            ``config/config.yaml`` (resolved relative to the current working
            directory).

    Returns:
        A fully validated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        ValueError: If the file does not contain a YAML mapping at the top
            level, or if the YAML cannot be parsed.
        pydantic.ValidationError: If any field is missing, mistyped, or fails a
            validation constraint.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough wrapping
        raise ValueError(f"Failed to parse YAML config at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Configuration root must be a mapping/object, got {type(raw).__name__} " f"in {path}."
        )

    return AppConfig(**raw)
