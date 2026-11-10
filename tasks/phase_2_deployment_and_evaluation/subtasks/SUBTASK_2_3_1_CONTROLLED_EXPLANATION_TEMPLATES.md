# Subtask 2.3.1: Controlled Explanation Template Library

**Subtask ID**: `2.3.1.CONTROLLED.EXPLANATION.TEMPLATES`  
**Parent Task**: `Task 2.3`  
**Phase**: Phase 2  
**Milestone**: Milestone 11  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build deterministic template library separating observed board facts from model hypotheses.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Detected plan, confidence, and attacking side.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/tracking/explanations.py` defining template formats for all 6 classes.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Strict Fact vs Hypothesis Separation:
   - Clear distinction: 'Observed Evidence' (verified pieces/moves) vs 'Model Hypothesis' (learned strategic intent).
2. Templates:
   - `K_STORM`: Flank pawn depth, king sector target, rook alignment.
   - `Q_EXP`: Queenside pawn space, flank control.
   - `OUTPOST`: Minor piece on hole, defending pawn anchor.
   - `ROOK_FILE`: Heavy piece alignment, open file entry squares.
   - `CHAIN_UNDERMINE`: Pawn lever move, target chain base/apex.
   - `NO_PLAN`: Balanced maneuvering or transitional state.

---

## 5. Claude Code Action Plan
1. Implement template definitions in `src/tracking/explanations.py`.
2. Write unit tests checking that templates render valid sentences without syntax errors.
3. Verify no unformatted template placeholders `{}` remain in rendered outputs.

---

## 6. Unit Testing & Acceptance Criteria
- All templates produce grammatically correct, pedagogical sentences.
- Explicitly labels predictions as hypotheses, not telepathic intentions.
