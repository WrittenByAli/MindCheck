# Subtask 2.5.2: Multi-Class Probability Bar Chart & Confidence Breakdown

**Subtask ID**: `2.5.2.PROBABILITY.BAR.CHARTS`  
**Parent Task**: `Task 2.5`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Render horizontal probability bar chart across all 6 classes with abstention threshold line.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Probability dictionary from inference runner.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/charts.py` rendering styled probability bars.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Chart: Horizontal bars for `K_STORM`, `Q_EXP`, `OUTPOST`, `ROOK_FILE`, `CHAIN_UNDERMINE`, `NO_PLAN`.
2. Features:
   - Color coding matching each plan class.
   - Vertical dashed threshold line at abstention cut-off (e.g. 40%).
   - Status badge: Active / Uncertain.

---

## 5. Claude Code Action Plan
1. Implement `render_probability_bars` in `src/ui/charts.py`.
2. Test chart rendering across diverse probability profiles.
3. Verify responsive layout in Streamlit right column.

---

## 6. Unit Testing & Acceptance Criteria
- Probabilities render cleanly with percentage labels.
- Status badge updates dynamically based on confidence.
