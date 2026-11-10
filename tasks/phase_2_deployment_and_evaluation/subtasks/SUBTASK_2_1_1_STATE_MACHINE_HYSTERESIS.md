# Subtask 2.1.1: Per-Player State Machine & Hysteresis Logic

**Subtask ID**: `2.1.1.STATE.MACHINE.HYSTERESIS`  
**Parent Task**: `Task 2.1`  
**Phase**: Phase 2  
**Milestone**: Milestone 9  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Implement per-player plan state tracking with 2-ply hysteresis to eliminate prediction flickering.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Stream of per-ply model predictions and active player turn.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/tracking/state_tracker.py` with `TemporalPlanTracker`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Maintain separate state for White and Black.
2. Hysteresis Transition Rules:
   - Require candidate plan $C$ to be predicted with confidence $\ge 0.45$ for 2 consecutive plies before activating.
   - Maintain active plan even if confidence dips for 1 ply.
   - Require 2 consecutive contradictory updates to switch active plans.
   - If `NO_PLAN` persists for $\ge 3$ plies, decay active state to `NO_PLAN`.

---

## 5. Claude Code Action Plan
1. Implement state machine in `src/tracking/state_tracker.py`.
2. Write unit tests in `tests/test_state_tracker.py` simulating move sequences.
3. Verify that isolated 1-ply noise spikes do not trigger plan flips.

---

## 6. Unit Testing & Acceptance Criteria
- State tracker maintains persistent active plan across consecutive plies.
- Hysteresis successfully filters out transient tactical spikes.
