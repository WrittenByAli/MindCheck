# Subtask 1.6.1: Single-Ply Canonical 14-Plane Board State Encoder

**Subtask ID**: `1.6.1.BOARD.CANONICAL.ENCODER`  
**Parent Task**: `Task 1.6`  
**Phase**: Phase 1  
**Milestone**: Milestone 5  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Encode a single chess board state into 14 binary $8 \times 8$ spatial planes.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.Board` instance and attacking side color.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: NumPy array of shape `(14, 8, 8)`, dtype `float32`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `BoardPlaneEncoder`:
   - Planes 0-5: Friendly P, N, B, R, Q, K
   - Planes 6-11: Opponent p, n, b, r, q, k
   - Plane 12: Castling rights (4 bits replicated across 8x8)
   - Plane 13: En-passant square (1 on target square, else 0)
2. Bitboard conversion: Use `board.pieces_mask()` and bit shifts for fast array filling.

---

## 5. Claude Code Action Plan
1. Implement `BoardPlaneEncoder` in `src/data/tensor_encoder.py`.
2. Write unit tests in `tests/test_tensor_encoder.py` asserting shape, dtypes, and piece counts.
3. Benchmark encoding speed (< 0.1ms per board).

---

## 6. Unit Testing & Acceptance Criteria
- Output shape is strictly `(14, 8, 8)` with dtype `float32`.
- Piece counts match board representation.
