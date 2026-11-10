# Subtask 1.7.1: Residual Convolutional Backbone Implementation

**Subtask ID**: `1.7.1.RESIDUAL.CNN.BACKBONE`  
**Parent Task**: `Task 1.7`  
**Phase**: Phase 1  
**Milestone**: Milestone 6  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement the custom PyTorch Residual CNN backbone with 4 residual blocks and 64 channels.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Input tensor `(batch_size, 112, 8, 8)`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/models/residual_cnn.py` defining `ResidualBlock` and `HistoryChessResNet`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. `ResidualBlock(nn.Module)`:
   - Conv2D(64, 64, kernel=3, padding=1, bias=False) -> BatchNorm2D(64) -> GELU()
   - Conv2D(64, 64, kernel=3, padding=1, bias=False) -> BatchNorm2D(64)
   - Skip connection: $y = \text{GELU}(x + \mathcal{F}(x))$
2. Stem:
   - Conv2D(112, 64, kernel=3, padding=1, bias=False) -> BatchNorm2D(64) -> GELU()
3. Backbone: 4 sequential Residual Blocks.

---

## 5. Claude Code Action Plan
1. Implement `ResidualBlock` and backbone in `src/models/residual_cnn.py`.
2. Write unit tests in `tests/test_models.py` verifying forward pass and residual addition.
3. Assert parameter count is ~250k-300k.

---

## 6. Unit Testing & Acceptance Criteria
- Forward pass succeeds without shape errors.
- Gradients backpropagate through all residual blocks.
