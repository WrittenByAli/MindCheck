# Subtask 1.5.2: Label Noise Filtering, Ambiguity Rejection & Class Balancing

**Subtask ID**: `1.5.2.NOISE.FILTER.AND.BALANCING`  
**Parent Task**: `Task 1.5`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Filter noisy weak labels, enforce class balancing, and cap samples per game.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Raw extracted weak-labeled records.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Filtered and balanced record collection (target 20,000-50,000 samples).

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Noise Filter:
   - Discard records where weak confidence $< 0.55$.
   - Discard records with extreme material imbalance ($|\Delta| > 800$ centipawns, e.g. queen down).
2. Per-Game Capping:
   - Cap maximum extracted samples from any single game to $\le 25$ positions.
3. Stratified Balancing:
   - Target roughly 15-20% representation per active plan class and ~20% `NO_PLAN`.

---

## 5. Claude Code Action Plan
1. Implement filtering and balancing routines in `src/data/weak_labeler.py`.
2. Write tests asserting sample cap per game and confidence thresholding.
3. Generate distribution audit plots.

---

## 6. Unit Testing & Acceptance Criteria
- No single game contributes > 25 positions.
- No class occupies > 30% of total dataset (balanced distribution).
