# Subtask 1.7.3: Single-Board Current-Position Baseline CNN

**Subtask ID**: `1.7.3.SINGLE.BOARD.BASELINE.CNN`  
**Parent Task**: `Task 1.7`  
**Phase**: Phase 1  
**Milestone**: Milestone 6  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement matched single-board CNN architecture (14 input channels) for scientific history ablation.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Current board tensor `(batch_size, 14, 8, 8)`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/models/single_board_cnn.py` defining `CurrentPositionResNet`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Matched Capacity:
   - Class `CurrentPositionResNet(nn.Module)`.
   - In-channels: 14 (only current position, zero history).
   - Identical stem, 4 residual blocks, 64 channels, and 6-class classification head.
2. Directly answers the primary research question by providing a controlled ablation baseline.

---

## 5. Claude Code Action Plan
1. Implement `src/models/single_board_cnn.py`.
2. Write unit test verifying forward pass on `(batch_size, 14, 8, 8)`.
3. Verify parameter parity with history-aware network.

---

## 6. Unit Testing & Acceptance Criteria
- Single-board baseline runs forward passes cleanly.
- Architecture matches history model in all aspects except input channels.
