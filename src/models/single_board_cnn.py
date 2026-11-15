"""Single-board (current-position) ablation baseline (Task 1.7, Subtask 1.7.3).

``CurrentPositionResNet`` is architecturally identical to
:class:`~src.models.residual_cnn.HistoryChessResNet` in every respect except its
input: only the current board's 14 planes, with no move history. It exists to
answer MindCheck's primary research question — does recent history actually
improve plan recognition over evaluating the static current position? Keeping
the backbone/head matched makes that a controlled comparison.

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

from src.models.residual_cnn import HistoryChessResNet

__all__ = ["CurrentPositionResNet"]


class CurrentPositionResNet(HistoryChessResNet):
    """History-ablation baseline: same ResNet but 14 input channels (one board)."""

    def __init__(
        self,
        num_classes: int = 6,
        channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__(
            in_channels=14,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )
