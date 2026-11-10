# Subtask 2.3.2: Strategic Plan Rationale & Supporting Evidence Formatter

**Subtask ID**: `2.3.2.STRATEGIC.RATIONALE.FORMATTER`  
**Parent Task**: `Task 2.3`  
**Phase**: Phase 2  
**Milestone**: Milestone 11  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Format concrete chess moves, square names, and tactical facts into pedagogical bullet points.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PlayerPlanState`, `ExtractedBoardFacts`, and recent SAN move list.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `ExplanationCard` with summary sentence and grounded factual bullet points.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `DeterministicExplanationGenerator`:
   - Interpolates verified square coordinates (e.g. 'd5', 'g5', 'c-file').
   - Constructs 3-4 bullet points highlighting supporting evidence.
   - Summarizes tactical context (e.g. 'Opponent king castled on g8, White pawn on g5').

---

## 5. Claude Code Action Plan
1. Implement formatting methods in `src/tracking/explanations.py`.
2. Test formatting across all 5 sample master games.
3. Assert that rendered bullet points contain valid algebraic notation.

---

## 6. Unit Testing & Acceptance Criteria
- Bullet points accurately reflect the verified move and board state.
- Pedagogical tone suitable for chess learners.
