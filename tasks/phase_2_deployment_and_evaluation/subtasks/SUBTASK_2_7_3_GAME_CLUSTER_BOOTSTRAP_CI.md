# Subtask 2.7.3: Game-Cluster Bootstrap Confidence Interval Calculator

**Subtask ID**: `2.7.3.GAME.CLUSTER.BOOTSTRAP.CI`  
**Parent Task**: `Task 2.7`  
**Phase**: Phase 2  
**Milestone**: Milestone 14  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Compute 95% Confidence Intervals for Macro-F1 using 1,000 game-cluster bootstrap iterations.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `experiments/evaluation_results.csv` with `game_id` column.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `experiments/bootstrap_confidence_intervals.json`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/models/bootstrap.py`:
   - Resample unique game IDs with replacement $B = 1,000$ times.
   - Compute Macro-F1 on each bootstrap sample.
   - Extract empirical 2.5th and 97.5th percentiles.
   - Report: $\text{Macro-F1} = \hat{\mu} \; [95\% \text{ CI}: L, U]$.

---

## 5. Claude Code Action Plan
1. Implement `src/models/bootstrap.py`.
2. Run 1,000 bootstrap iterations.
3. Log confidence intervals for inclusion in the final report.

---

## 6. Unit Testing & Acceptance Criteria
- Statistically sound confidence intervals accounting for clustered game data.
- Confirms whether History CNN superiority is statistically significant.
