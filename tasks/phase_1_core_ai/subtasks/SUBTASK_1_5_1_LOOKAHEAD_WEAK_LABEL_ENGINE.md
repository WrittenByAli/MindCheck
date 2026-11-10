# Subtask 1.5.1: Lookahead Retrospective Execution Heuristic Engine

**Subtask ID**: `1.5.1.LOOKAHEAD.WEAK.LABEL.ENGINE`  
**Parent Task**: `Task 1.5`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Build the weak-label generator inspecting future plies to verify plan execution for training data.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Full `chess.pgn.Game` and position ply index $t$.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/data/weak_labeler.py` returning weak label class ID and confidence.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `RetrospectiveWeakLabeler`:
   - Inspect future plies $t+1 \dots t+8$:
     - For `OUTPOST`: Did minor piece occupy outpost and remain for $\ge 2$ plies?
     - For `K_STORM`: Did flank pawns advance further or trade off opening king files?
     - For `ROOK_FILE`: Did heavy pieces maintain control or penetrate rank 7?
     - For `CHAIN_UNDERMINE`: Did pawn structure alter through captures or tension resolution?
   - If future confirms execution: assign plan label.
   - If future reveals tactical diversion or immediate blunder: assign `NO_PLAN` or reject.

---

## 5. Claude Code Action Plan
1. Implement `src/data/weak_labeler.py`.
2. Implement future lookahead checkers for all 5 classes.
3. Write `tests/test_weak_labeler.py` verifying retrospective confirmation logic.
4. Benchmark weak-labeling speed on 100 games.

---

## 6. Unit Testing & Acceptance Criteria
- Confirmed executed plans receive high-confidence weak labels.
- Unfinished or abandoned plans are rejected or labeled `NO_PLAN`.
