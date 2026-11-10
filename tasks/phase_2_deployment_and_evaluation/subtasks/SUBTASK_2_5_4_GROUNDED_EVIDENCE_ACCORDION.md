# Subtask 2.5.4: Grounded Evidence Fact Inspector & Accordion

**Subtask ID**: `2.5.4.GROUNDED.EVIDENCE.ACCORDION`  
**Parent Task**: `Task 2.5`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build expandable accordion presenting pawn structures, outposts, file control, and explanation card.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `ExtractedBoardFacts` and `ExplanationCard`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/evidence_view.py` rendering formatted evidence blocks.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Accordion Sections:
   - *Strategic Summary*: Active plan, onset ply, confidence, explanation text.
   - *Pawn Structures*: Chains, base pawns, advanced storm pawns.
   - *Piece Coordinates*: Outpost squares and piece maneuvers.
   - *File Control*: Open/semi-open files and rook batteries.

---

## 5. Claude Code Action Plan
1. Implement `src/ui/evidence_view.py`.
2. Style with markdown callout boxes and metric widgets.
3. Verify evidence updates correctly at each move step.

---

## 6. Unit Testing & Acceptance Criteria
- Clear, human-readable breakdown of underlying chess facts.
- Allows learners to inspect why the AI made its prediction.
