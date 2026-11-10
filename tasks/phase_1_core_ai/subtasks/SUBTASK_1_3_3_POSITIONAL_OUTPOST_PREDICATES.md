# Subtask 1.3.3: Positional Predicates: Outposts & Heavy Piece Batteries

**Subtask ID**: `1.3.3.POSITIONAL.OUTPOST.PREDICATES`  
**Parent Task**: `Task 1.3`  
**Phase**: Phase 1  
**Milestone**: Milestone 3  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Extract symbolic predicates for minor piece outposts and heavy piece file doubling.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.Board` instance and recent move history.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Predicate outputs: `outpost_squares`, `occupied_outposts`, `rook_batteries`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Outpost Detection:
   - Candidate square $S$ on ranks 4-6 (White) or ranks 3-5 (Black), files c, d, e, f.
   - $S$ is defended by at least one friendly pawn.
   - $S$ cannot be attacked by opponent pawns (enemy pawns on adjacent files are traded or past $S$).
   - Check if friendly Knight or Bishop occupies $S$ or can legally reach $S$ in 1 move.
2. Rook Battery:
   - Identify files containing $\ge 2$ friendly major pieces (Rook+Rook or Rook+Queen).
   - Check if battery controls 7th/8th rank entry squares.

---

## 5. Claude Code Action Plan
1. Implement outpost and battery algorithms in `src/rules/fact_extractor.py`.
2. Write unit tests in `tests/test_fact_extractor.py` using Kasparov-Karpov 1985 G16 (d5 outpost) and Alekhine's Gun FENs.
3. Verify zero false positive outposts on empty or flank squares.

---

## 6. Unit Testing & Acceptance Criteria
- Outpost squares correctly detected only when uncontested by enemy pawns.
- Heavy batteries correctly detected along open/semi-open files.
