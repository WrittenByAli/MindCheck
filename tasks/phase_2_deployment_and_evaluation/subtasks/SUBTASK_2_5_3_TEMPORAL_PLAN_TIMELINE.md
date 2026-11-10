# Subtask 2.5.3: Whole-Game Temporal Plan Evolution Chart

**Subtask ID**: `2.5.3.TEMPORAL.PLAN.TIMELINE`  
**Parent Task**: `Task 2.5`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build interactive line/area chart plotting plan probabilities across all game plies with onset markers.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Precomputed game history probabilities.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/timeline.py` rendering multi-line temporal chart.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Chart:
   - X-axis: Game ply ($0 \dots T$).
   - Y-axis: Softmax probability ($0.0 \dots 1.0$).
   - 6 lines/curves for all classes.
   - Vertical cursor line at `current_ply`.
   - Star/circle marker at the detected **Plan Onset Ply**.

---

## 5. Claude Code Action Plan
1. Implement `render_game_timeline` in `src/ui/timeline.py`.
2. Precompute whole-game predictions with progress indicator.
3. Verify cursor line syncs with navigation stepper.

---

## 6. Unit Testing & Acceptance Criteria
- Visualizes the birth, dominance, and decline of strategic plans.
- Cursor follows move stepper smoothly.
