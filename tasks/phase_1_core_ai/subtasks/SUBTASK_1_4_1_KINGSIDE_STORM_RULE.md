# Subtask 1.4.1: Kingside Pawn Storm Operational Rule Template

**Subtask ID**: `1.4.1.KINGSIDE.STORM.RULE`  
**Parent Task**: `Task 1.4`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement rule template evaluating heuristic score for Kingside Pawn Storm (`K_STORM`).

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PositionWindow` and `ExtractedBoardFacts`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `evaluate_kingside_storm(window, facts) -> float` score in [0.0, 1.0].

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Operational Criteria:
   - Opponent king is castled kingside (f, g, h files).
   - Recent 8 plies contain $\ge 2$ flank pawn advances on f, g, or h files.
   - Flank pawns have advanced to rank 4+ (White) or rank 5- (Black).
   - Major piece behind advancing pawns adds bonus.
2. Formula:
   $$S = \min(1.0, 0.4 \times \text{pawn\_pushes} + 0.3 \times \text{rank\_depth} + 0.3 \times \text{support})$$

---

## 5. Claude Code Action Plan
1. Implement `evaluate_kingside_storm` in `src/rules/rule_engine.py`.
2. Write unit tests in `tests/test_rule_engine.py` with known storm game positions.
3. Test edge cases (queenside castling, quiet maneuvering).

---

## 6. Unit Testing & Acceptance Criteria
- Scores > 0.70 on active kingside storm positions.
- Scores < 0.20 when kings are castled on opposite wings without pawn advances.
