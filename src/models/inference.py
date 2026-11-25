"""Deployable MindCheck classifier with the spec input shape.

The final network may use only the newest plies (e.g. the last 56 channels =
4 plies, since the tensor is ordered oldest -> newest) and a validation-tuned
per-class logit bias. :class:`DeployableModel` hides both: callers always pass
the spec tensor ``(B, 112, 8, 8)`` float32 and get biased logits back.

The bias calibrates outputs to the class frequencies of the rule-engine weak
labels on validation data; it does not make the labels any more correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.models.residual_cnn import HistoryChessResNet

__all__ = ["DeployableModel", "save_deployable", "load_deployable", "FORMAT"]

FORMAT = "mindcheck_deployable_v1"
SPEC_CHANNELS = 112


class DeployableModel(nn.Module):
    """``(B, 112, 8, 8)`` -> slice newest ``keep`` channels -> logits + bias."""

    def __init__(
        self,
        keep: int = SPEC_CHANNELS,
        bias: Optional[Sequence[float]] = None,
        num_classes: int = 6,
        channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if not 0 < keep <= SPEC_CHANNELS or keep % 14:
            raise ValueError(f"keep must be a multiple of 14 in (0, 112], got {keep}.")
        values = [0.0] * num_classes if bias is None else [float(b) for b in bias]
        if len(values) != num_classes:
            raise ValueError(f"bias needs {num_classes} values, got {len(values)}.")
        self.in_channels = SPEC_CHANNELS  # the exporters build dummy inputs from this
        self.config: Dict[str, Any] = dict(
            keep=keep,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.keep = keep
        self.backbone = HistoryChessResNet(
            in_channels=keep,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.register_buffer("logit_bias", torch.tensor(values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x[:, SPEC_CHANNELS - self.keep :]) + self.logit_bias

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax of the biased logits, always computed in eval mode."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (SPEC_CHANNELS, 8, 8):
            raise ValueError(f"expected (B, 112, 8, 8), got {tuple(x.shape)}.")
        was_training = self.training
        self.eval()
        probs = torch.softmax(self.forward(x), dim=-1)
        if was_training:
            self.train()
        return probs


def save_deployable(
    model: DeployableModel, path: str | Path, metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Write a checkpoint of plain tensors/types (loadable with ``weights_only``)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "config": model.config,
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        out,
    )
    return out


def load_deployable(path: str | Path) -> DeployableModel:
    """Rebuild a :class:`DeployableModel` in eval mode from :func:`save_deployable`."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint.")
    model = DeployableModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()
