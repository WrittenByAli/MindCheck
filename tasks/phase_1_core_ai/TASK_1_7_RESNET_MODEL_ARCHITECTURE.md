# Task 1.7: Deep Residual CNN Architecture Implementation

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 6  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Implement the core Deep Residual Convolutional Neural Network (ResNet) in PyTorch for 6-class chess strategic plan recognition. In addition, implement the matched Single-Board Baseline CNN (14 input channels) to perform a direct history-ablation experiment as promised in the proposal. Provide automated ONNX and TorchScript export pipelines.

---

## 2. Subtask Breakdown

### Subtask 1.7.1: Residual Convolutional Backbone (`src/models/residual_cnn.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Class `ResidualBlock(nn.Module)`:
    - `Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)`
    - `BatchNorm2d(channels)`
    - `GELU()`
    - `Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)`
    - `BatchNorm2d(channels)`
    - Skip connection: $y = \text{GELU}(x + \mathcal{F}(x))$
  - Class `HistoryChessResNet(nn.Module)`:
    - Stem Conv: `Conv2d(in_channels=112, out_channels=64, kernel_size=3, padding=1)` + `BatchNorm2d(64)` + `GELU()`
    - 4 Sequential `ResidualBlock` instances (64 channels throughout)
    - Receptive field covers the complete $8 \times 8$ chess board geometry.

### Subtask 1.7.2: Classification Head (`src/models/residual_cnn.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - `AdaptiveAvgPool2d((1, 1))` pooling $64 \times 8 \times 8 \to 64 \times 1 \times 1$
  - Flatten to vector of size 64
  - `LayerNorm(64)`
  - `Dropout(p=0.20)`
  - Fully connected linear layer: `Linear(64, num_classes=6)`
  - Returns raw logits $[B, 6]$ during training, or softmax probabilities $[B, 6]$ during inference.

### Subtask 1.7.3: Single-Board Current-Position Baseline (`src/models/single_board_cnn.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Scientific Ablation Baseline**:
  - Class `CurrentPositionResNet(nn.Module)`
  - Identical backbone and head structure, but `in_channels=14` (only the current board state $B_t$, zero history).
  - Used to answer the primary research question: *"Does recent move history improve recognition over evaluating only the current board?"*

### Subtask 1.7.4: ONNX & TorchScript Export Pipeline (`src/models/export.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Function `export_to_onnx(model, checkpoint_path, output_path)`:
    - Dummy input: `torch.randn(1, 112, 8, 8)`
    - Exports to `models/checkpoints/mindcheck_resnet.onnx`
    - Verifies export via `onnx.checker.check_model()`
  - Function `export_to_torchscript(model, checkpoint_path, output_path)`:
    - Trace model via `torch.jit.trace`
    - Saves to `models/checkpoints/mindcheck_resnet.pt`

---

## 3. Architecture PyTorch Schema

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
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
    def __init__(self, in_channels: int = 112, num_classes: int = 6, channels: int = 64, num_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Dropout(p=0.2),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.blocks(out)
        out = self.gap(out)
        return self.head(out)
```

---

## 4. Claude Code Action Plan
1. Implement `src/models/residual_cnn.py` and `src/models/single_board_cnn.py`.
2. Implement `src/models/export.py`.
3. Implement `tests/test_models.py` verifying:
   - Output shapes `(batch_size, 6)` for arbitrary batch sizes.
   - Forward pass without NaN or exploding activations.
   - ONNX export and runtime verification with `onnxruntime`.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_models.py` -> 100% pass.
- Model parameter count is between 200,000 and 350,000 parameters.
- ONNX CPU inference time per sample is $< 10 \text{ ms}$.
