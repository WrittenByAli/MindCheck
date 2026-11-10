# Subtask 2.1.4: Unit Tests & Move-Sequence Simulation for State Tracker

**Subtask ID**: `2.1.4.STATE.TRACKER.SIMULATION.TESTS`  
**Parent Task**: `Task 2.1`  
**Phase**: Phase 2  
**Milestone**: Milestone 9  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build a comprehensive simulation test suite for temporal plan tracking across full game sequences.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `src/tracking/state_tracker.py`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `tests/test_state_tracker.py` with multi-ply simulation scenarios.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Test Scenarios:
   - Sustained plan: 10 plies of `OUTPOST` -> remains active throughout.
   - Alternating noise: `[K_STORM, NO_PLAN, K_STORM, NO_PLAN]` -> filters oscillation.
   - Strategic shift: 5 plies of `K_STORM` followed by 3 plies of `Q_EXP` -> clean transition.
   - Decay: 5 plies of `NO_PLAN` -> active plan gracefully decays to `NO_PLAN`.

---

## 5. Claude Code Action Plan
1. Implement simulation scenarios in `tests/test_state_tracker.py`.
2. Run `pytest tests/test_state_tracker.py`.
3. Check 100% test coverage of state machine branches.

---

## 6. Unit Testing & Acceptance Criteria
- All simulated game sequences pass expected state assertions.
- Tracker correctly handles game resets.
