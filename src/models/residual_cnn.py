"""History-aware residual CNN for chess plan recognition (Task 1.7, Milestone 6).

``HistoryChessResNet`` is the primary classifier: a compact ResNet over the
``(B, 112, 8, 8)`` spatial-temporal tensor producing 6-class logits (5 plans +
``NO_PLAN``). The board is never spatially downsampled — padding keeps the full
``8x8`` geometry through every block, and a global average pool collapses it
before the head.

Owner: Shahzad Ahmad (24I-3090). Subtasks 1.7.1 and 1.7.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ResidualBlock", "HistoryChessResNet", "count_parameters"]


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a GELU-activated identity skip connection."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class HistoryChessResNet(nn.Module):
    """Residual CNN over an 8-ply history tensor -> 6-class plan logits."""

    def __init__(
        self,
        in_channels: int = 112,
        num_classes: int = 6,
        channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Dropout(p=dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits ``(B, num_classes)``."""
        out = self.stem(x)
        out = self.blocks(out)
        out = self.gap(out)
        return self.head(out)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities ``(B, num_classes)`` (eval-mode)."""
        was_training = self.training
        self.eval()
        probs = torch.softmax(self.forward(x), dim=-1)
        if was_training:
            self.train()
        return probs
