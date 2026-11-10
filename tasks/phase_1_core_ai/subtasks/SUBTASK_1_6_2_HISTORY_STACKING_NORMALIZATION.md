# Subtask 1.6.2: 8-Ply History Stacking & Attacker Normalization

**Subtask ID**: `1.6.2.HISTORY.STACKING.NORMALIZATION`  
**Parent Task**: `Task 1.6`  
**Phase**: Phase 1  
**Milestone**: Milestone 5  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Stack 8 consecutive board states into shape `(112, 8, 8)` and normalize board orientation relative to attacking side.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: List of 8 `chess.Board` states and attacking player color.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: NumPy array of shape `(112, 8, 8)`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Normalization Logic:
   - If attacking side is White: keep standard orientation.
   - If attacking side is Black: flip array along row axis (`np.flip(arr, axis=-2)`), swap Friendly (0-5) and Opponent (6-11) piece planes.
   - Result: Attacking player is always advancing toward row 0 / Rank 8.
2. Channel Stacking:
   - $8 \times 14 = 112$ channels ordered chronologically from $t-7$ to $t$.

---

## 5. Claude Code Action Plan
1. Implement stacking and normalization in `src/data/tensor_encoder.py`.
2. Write unit test with identical pawn storm played by White and Black; verify identical normalized tensors.
3. Test channel alignment.

---

## 6. Unit Testing & Acceptance Criteria
- Output shape is strictly `(112, 8, 8)`.
- Rotational invariance confirmed between White and Black maneuvers.
