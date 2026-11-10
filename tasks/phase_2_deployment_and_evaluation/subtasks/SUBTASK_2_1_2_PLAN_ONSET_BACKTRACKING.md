# Subtask 2.1.2: Earliest Detected Onset Estimation & Timeline History

**Subtask ID**: `2.1.2.PLAN.ONSET.BACKTRACKING`  
**Parent Task**: `Task 2.1`  
**Phase**: Phase 2  
**Milestone**: Milestone 9  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Detect the earliest supporting ply of an activated plan by backtracking through the history window.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Candidate activation event in `TemporalPlanTracker`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `onset_ply` integer attribute and timeline record in `PlayerPlanState`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Onset Backtracking:
   - When candidate plan $C$ activates at ply $t$:
     - Scan backward through the 8-ply window to identify the earliest ply $t_{\text{onset}}$ where supporting moves or candidate confidence $> 0.35$ appeared.
   - Record `onset_ply = t_onset`.
   - Store `consecutive_plies_active = t - t_onset + 1`.

---

## 5. Claude Code Action Plan
1. Implement onset backtracking in `src/tracking/state_tracker.py`.
2. Add unit test verifying onset detection on known games.
3. Test edge case where onset occurs at ply 0.

---

## 6. Unit Testing & Acceptance Criteria
- Onset ply correctly identifies the first supporting ply of the plan.
- Timeline accurately logs plan duration.
