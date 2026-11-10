# Subtask 1.4.3: Outpost Occupation Operational Rule Template

**Subtask ID**: `1.4.3.OUTPOST.OCCUPATION.RULE`  
**Parent Task**: `Task 1.4`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Implement rule template evaluating heuristic score for Outpost Occupation (`OUTPOST`).

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PositionWindow` and `ExtractedBoardFacts`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `evaluate_outpost_occupation(window, facts) -> float` score in [0.0, 1.0].

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Operational Criteria:
   - Outpost square exists on central files (c, d, e, f) defended by friendly pawn and uncontested by enemy pawns.
   - Friendly Knight or Bishop entered the outpost square during recent 8 plies, or is positioned within 1 legal move.
2. Scoring:
   - 1.0 if minor piece occupies outpost currently.
   - 0.7 if piece maneuvered toward outpost in recent plies.
   - 0.0 if no outpost exists.

---

## 5. Claude Code Action Plan
1. Implement `evaluate_outpost_occupation` in `src/rules/rule_engine.py`.
2. Add unit test with Kasparov Karpov Game 16 (d5 knight).
3. Verify score = 0 when outpost is easily attacked by enemy pawn.

---

## 6. Unit Testing & Acceptance Criteria
- Returns 1.0 on established outposts occupied by knights/bishops.
- Returns 0.0 when target square is vulnerable to enemy pawns.
