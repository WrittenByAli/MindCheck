# Subtask 1.4.4: Rook-File Buildup Operational Rule Template

**Subtask ID**: `1.4.4.ROOK.FILE.BUILDUP.RULE`  
**Parent Task**: `Task 1.4`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement rule template evaluating heuristic score for Rook-File Buildup (`ROOK_FILE`).

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PositionWindow` and `ExtractedBoardFacts`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `evaluate_rook_file_buildup(window, facts) -> float` score in [0.0, 1.0].

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Operational Criteria:
   - File is open (no pawns) or semi-open (no friendly pawns).
   - $\ge 2$ heavy pieces (R+R or R+Q) occupy the file.
   - Moving rook to open file occurred in recent 8 plies.
2. Scoring:
   - 1.0 if doubled on open file.
   - 0.7 if doubled on semi-open file.
   - 0.4 if single rook occupies open file.

---

## 5. Claude Code Action Plan
1. Implement `evaluate_rook_file_buildup` in `src/rules/rule_engine.py`.
2. Add unit tests for open-file doubling and tripling.
3. Test edge cases with closed files.

---

## 6. Unit Testing & Acceptance Criteria
- Returns high score on doubled rooks on open files.
- Returns 0.0 on closed files with pawns blocking rooks.
