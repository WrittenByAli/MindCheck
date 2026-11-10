# Subtask 1.6.3: History Padding Mask & Sequence Validity Encoding

**Subtask ID**: `1.6.3.HISTORY.PADDING.VALIDITY.MASK`  
**Parent Task**: `Task 1.6`  
**Phase**: Phase 1  
**Milestone**: Milestone 5  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Handle early game plies (< 8 plies) with zero-padding and produce 1D validity masks.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Position history with length $< 8$.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Padded tensor `(112, 8, 8)` and 1D validity mask `(8,)`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Zero-Padding:
   - For missing plies $t-7 \dots t-k$, zero out the corresponding $14 \times 8 \times 8$ channels.
2. Validity Mask:
   - Mask $M \in \mathbb{R}^8$ where $M_i = 1.0$ if ply exists, $0.0$ if padded.
3. Guarantees model can analyze opening positions without crashing.

---

## 5. Claude Code Action Plan
1. Implement padding and mask logic in `src/data/tensor_encoder.py`.
2. Write unit tests for plies 0, 1, 4, 7, and 20.
3. Verify mask values match valid ply counts.

---

## 6. Unit Testing & Acceptance Criteria
- Padded positions produce valid tensors without NaN.
- Validity mask accurately reflects available history depth.
