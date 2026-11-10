# Subtask 1.3.1: 64-Bit Bitboard Fact & Piece Mobility Extractor

**Subtask ID**: `1.3.1.BITBOARD.FACT.EXTRACTOR`  
**Parent Task**: `Task 1.3`  
**Phase**: Phase 1  
**Milestone**: Milestone 3  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement high-speed bitboard extraction for piece locations, attack bitmasks, and mobility counts.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.Board` instance.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/rules/fact_extractor.py` returning bitmask dictionaries and mobility counts.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `ChessFactExtractor`:
   - Piece masks: `board.pieces_mask(piece_type, color)` for P, N, B, R, Q, K.
   - Attack bitboards: Compute `attack_mask(color)` using bitwise OR over `board.attacks(sq)`.
   - Mobility: Count legal moves per piece type using bitboard popcount.
   - Material balance: Compute standard centipawn material evaluation.

---

## 5. Claude Code Action Plan
1. Implement bitboard extraction methods in `src/rules/fact_extractor.py`.
2. Write bitwise helper utilities (popcount, square mask, ray attacks).
3. Write unit tests in `tests/test_fact_extractor.py` verifying bitmask accuracy.
4. Benchmark fact extraction speed across 1,000 boards (< 0.5s total).

---

## 6. Unit Testing & Acceptance Criteria
- Bitboard masks match piece positions exactly.
- Extraction latency per board is < 0.5 ms.
