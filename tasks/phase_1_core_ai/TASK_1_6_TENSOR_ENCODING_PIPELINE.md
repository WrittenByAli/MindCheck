# Task 1.6: Spatial-Temporal Chess Tensor Encoding Pipeline

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 5  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Transform 8-ply chess position histories into high-dimensional spatial-temporal PyTorch tensors of shape `(112, 8, 8)`. Implement canonical perspective normalization (attacker always moves upward) and early-game zero-padding with validity masks. Deliver an efficient memory-mapped PyTorch `Dataset` and `DataLoader`.

---

## 2. Subtask Breakdown

### Subtask 1.6.1: Single-Ply Canonical Plane Encoder (`src/data/tensor_encoder.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Encode a single `chess.Board` into a NumPy array of shape `(14, 8, 8)` with dtype `np.float32`:
    - Planes 0–5: Friendly pieces (`P`, `N`, `B`, `R`, `Q`, `K`)
    - Planes 6–11: Opponent pieces (`p`, `n`, `b`, `r`, `q`, `k`)
    - Plane 12: Castling rights (replicated scalar plane: bit 0: friendly kingside, bit 1: friendly queenside, bit 2: enemy kingside, bit 3: enemy queenside)
    - Plane 13: En-passant target square (1 on valid target square, else 0)

### Subtask 1.6.2: 8-Ply History Stacking & Attacker Normalization (`src/data/tensor_encoder.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Stack 8 consecutive half-move states $[B_{t-7}, \dots, B_t]$ along the channel dimension:
    $$\text{Channels} = 8 \times 14 = 112$$
    $$\text{Final Shape} = (112, 8, 8)$$
  - **Attacker Perspective Normalization**:
    - If side to move at ply $t$ is **White**: keep standard orientation (Rank 1 at row 7, Rank 8 at row 0).
    - If side to move at ply $t$ is **Black**: flip array along row axis (`np.flip(arr, axis=-2)`), invert rank coordinates, and swap Friendly vs Opponent channels (planes 0–5 $\leftrightarrow$ planes 6–11).
    - **Advantage**: Network always learns attacking vectors moving toward Rank 8 regardless of whether White or Black is playing.

### Subtask 1.6.3: History Padding Mask & Early Plies (`src/data/tensor_encoder.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - For positions where ply $t < 7$ (e.g. ply 3):
    - Historical slots $t-7 \dots t-4$ have no preceding moves.
    - Zero-fill the missing planes.
    - Return a 1D validity mask $M \in \mathbb{R}^8$ where $M_i = 1.0$ if ply exists, $0.0$ if zero-padded.

### Subtask 1.6.4: PyTorch Dataset & DataLoader (`src/data/chess_dataset.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Implement class `ChessPlanDataset(torch.utils.data.Dataset)`.
  - Support both on-the-fly dynamic encoding from PGN/FEN records and pre-compiled memory-mapped disk tensors (`np.memmap` / `.npy` shards).
  - DataLoader configuration: `batch_size=64`, `shuffle=True`, `num_workers=2`, `pin_memory=True`.

---

## 3. Function Signatures & Data Model

```python
import numpy as np
import torch
from typing import List, Tuple
import chess

class ChessTensorEncoder:
    def __init__(self, history_length: int = 8, num_planes_per_ply: int = 14):
        self.history_length = history_length
        self.num_planes = num_planes_per_ply
        self.total_channels = history_length * num_planes_per_ply

    def encode_window(
        self, 
        history: List[chess.Board], 
        attacking_side: chess.Color
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            tensor: np.ndarray of shape (112, 8, 8), dtype float32
            mask: np.ndarray of shape (8,), dtype float32
        """
        pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/data/tensor_encoder.py` with `ChessTensorEncoder`.
2. Implement `src/data/chess_dataset.py` with `ChessPlanDataset`.
3. Implement `tests/test_tensor_encoder.py` verifying:
   - Output shape is strictly `(112, 8, 8)`.
   - Black perspective flip properly places Black pawns moving toward row 0.
   - Zero-padding mask matches history length.
4. Verify memory efficiency: 10,000 encoded samples require $< 1 \text{ GB}$ on disk.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_tensor_encoder.py` -> 100% pass.
- Single tensor forward conversion time $< 1.5 \text{ milliseconds}$.
