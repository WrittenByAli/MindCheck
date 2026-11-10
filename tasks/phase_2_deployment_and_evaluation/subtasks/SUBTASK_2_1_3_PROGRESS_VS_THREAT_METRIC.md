# Subtask 2.1.3: Plan Progress vs Threat Severity Metric Formulation

**Subtask ID**: `2.1.3.PROGRESS.VS.THREAT.METRIC`  
**Parent Task**: `Task 2.1`  
**Phase**: Phase 2  
**Milestone**: Milestone 9  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Formulate independent metrics for plan completion progress versus immediate tactical threat severity.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PlayerPlanState` and `ExtractedBoardFacts`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `progress_score` and `threat_score` floats in [0.0, 1.0].

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Plan Progress ($0.0 \dots 1.0$):
   - Measures completion of strategic preconditions (e.g. flank pawns advanced to target rank, pieces placed on outposts).
2. Threat Severity ($0.0 \dots 1.0$):
   - Measures danger to defender (king proximity, attackers vs defenders, open file entry threats).
3. Displayed as separate meters in UI to avoid confusing progress with danger.

---

## 5. Claude Code Action Plan
1. Implement metric evaluators in `src/tracking/state_tracker.py`.
2. Write unit tests asserting bounds [0.0, 1.0].
3. Verify progress increases as pieces advance toward target squares.

---

## 6. Unit Testing & Acceptance Criteria
- Progress score correlates with geometric advancement.
- Threat score reflects positional pressure on defending king.
