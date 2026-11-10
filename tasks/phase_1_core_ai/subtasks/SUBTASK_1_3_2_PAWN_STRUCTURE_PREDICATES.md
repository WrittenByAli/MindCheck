# Subtask 1.3.2: Pawn Structure & King Safety Predicate Engine

**Subtask ID**: `1.3.2.PAWN.STRUCTURE.PREDICATES`  
**Parent Task**: `Task 1.3`  
**Phase**: Phase 1  
**Milestone**: Milestone 3  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Extract symbolic predicates for pawn chains, chain base/apex, open/semi-open files, and king pawn shield.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.Board` instance.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Predicate outputs in `ExtractedBoardFacts` (`white_pawn_chains`, `open_files`, `king_shield_integrity`).

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Open Files: File mask & (white_pawns | black_pawns) == 0.
2. Semi-Open Files: File has opponent pawns but zero friendly pawns.
3. Pawn Chains: Detect diagonal mutual pawn defense ($s$ defended by $s \pm 7$ or $s \pm 9$). Identify the base pawn (root of chain with no pawn defending it).
4. King Safety / Flank Advance: King quadrant pawn shield status and count of advanced flank pawns (f, g, h / a, b, c).

---

## 5. Claude Code Action Plan
1. Implement pawn structure predicates in `src/rules/fact_extractor.py`.
2. Add specific test cases in `tests/test_fact_extractor.py` for pawn chain detection and file classification.
3. Verify with Carlsbad and French Defense pawn structures.

---

## 6. Unit Testing & Acceptance Criteria
- Accurately detects open and semi-open files for both colors.
- Correctly identifies chain bases and flank advances.
