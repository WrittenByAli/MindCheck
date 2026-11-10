# Subtask 1.9.1: Written Annotation Rubric & Guideline Documentation

**Subtask ID**: `1.9.1.ANNOTATION.RUBRIC.GUIDELINE`  
**Parent Task**: `Task 1.9`  
**Phase**: Phase 1  
**Milestone**: Milestone 8  
**Primary Owner**: **Shared (Both Members)**  
**Collaborator**: **Shahzad Ahmad & Muhammad Ali**  

---

## 1. Context & Objective
Formalize the human annotation rubric and rating instructions for the 250 held-out test windows.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `data/annotations/ANNOTATION_GUIDELINES.md` signed off by both members.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Operational guidelines for annotators:
   - History-only constraint: Inspect ONLY plies $t-7 \dots t$. Never peek forward.
   - Independence constraint: No communication between annotators during scoring.
   - Record: Primary plan, confidence score (0.4-1.0), secondary plausible plan, and rationale.
   - Disclose reviewer chess ratings in final submission.

---

## 5. Claude Code Action Plan
1. Draft and finalize `data/annotations/ANNOTATION_GUIDELINES.md`.
2. Both members review and confirm operational consensus.
3. Commit guideline document to repository.

---

## 6. Unit Testing & Acceptance Criteria
- Guidelines clearly specify boundary criteria for all 5 plans and `NO_PLAN`.
- Both team members understand and follow the identical rubric.
