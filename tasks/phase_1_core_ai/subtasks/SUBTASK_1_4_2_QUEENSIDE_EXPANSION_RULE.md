# Subtask 1.4.2: Queenside Expansion Operational Rule Template

**Subtask ID**: `1.4.2.QUEENSIDE.EXPANSION.RULE`  
**Parent Task**: `Task 1.4`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement rule template evaluating heuristic score for Queenside Expansion (`Q_EXP`).

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PositionWindow` and `ExtractedBoardFacts`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `evaluate_queenside_expansion(window, facts) -> float` score in [0.0, 1.0].

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Operational Criteria:
   - Recent 8 plies contain $\ge 2$ pawn advances on a, b, or c files.
   - Queenside pawns control space past 4th rank (White) or 5th rank (Black).
   - Minor piece or rook supports queenside pawn mass.
2. Formula:
   $$S = \min(1.0, 0.4 \times \text{queenside\_pushes} + 0.3 \times \text{space\_advantage} + 0.3 \times \text{piece\_support})$$

---

## 5. Claude Code Action Plan
1. Implement `evaluate_queenside_expansion` in `src/rules/rule_engine.py`.
2. Add unit tests for minority attacks and Sicilian queenside pawn pushes.
3. Verify low scores on central or kingside games.

---

## 6. Unit Testing & Acceptance Criteria
- Scores > 0.70 on positions with clear queenside expansion.
- Disregards single isolated pawn pushes.
