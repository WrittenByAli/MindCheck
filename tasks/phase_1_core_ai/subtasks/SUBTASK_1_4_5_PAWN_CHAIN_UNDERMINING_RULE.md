# Subtask 1.4.5: Pawn-Chain Undermining Operational Rule Template

**Subtask ID**: `1.4.5.PAWN.CHAIN.UNDERMINING.RULE`  
**Parent Task**: `Task 1.4`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Implement rule template evaluating heuristic score for Pawn-Chain Undermining (`CHAIN_UNDERMINE`).

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PositionWindow` and `ExtractedBoardFacts`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `evaluate_chain_undermining(window, facts) -> float` score in [0.0, 1.0].

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Operational Criteria:
   - Opponent maintains a locked or connected pawn chain ($\ge 2$ pawns).
   - Friendly pawn lever was pushed in recent 8 plies attacking the base or apex of the chain.
   - Pawn tension exists with capture possibility.
2. Scoring:
   - 1.0 if lever attacking base pawn.
   - 0.8 if lever attacking apex pawn.
   - 0.0 if no chain or no undermining moves.

---

## 5. Claude Code Action Plan
1. Implement `evaluate_chain_undermining` in `src/rules/rule_engine.py`.
2. Add unit tests with French Defense (...c5 attacking d4).
3. Test closed structures without pawn levers.

---

## 6. Unit Testing & Acceptance Criteria
- Accurately detects pawn levers targeting chain foundations.
- Distinguishes structural levers from random pawn moves.
