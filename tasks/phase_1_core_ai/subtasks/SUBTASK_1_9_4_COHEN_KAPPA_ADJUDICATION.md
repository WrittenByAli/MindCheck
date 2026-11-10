# Subtask 1.9.4: Cohen's Kappa Inter-Rater Reliability & Adjudication Tool

**Subtask ID**: `1.9.4.COHEN.KAPPA.ADJUDICATION`  
**Parent Task**: `Task 1.9`  
**Phase**: Phase 1  
**Milestone**: Milestone 8  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Calculate Cohen's Kappa score (\kappa), generate conflict report, and produce frozen consensus ground truth.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Completed annotation files from both members.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `data/annotations/test_ground_truth.csv` and `experiments/inter_rater_reliability.json`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/models/adjudicate.py`:
   - Load both member annotation CSVs.
   - Compute observed agreement $p_o$ and Cohen's Kappa:
     $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   - Target $\kappa \ge 0.65$ (substantial agreement).
   - Export disputed positions to `data/annotations/disagreements.csv`.
2. Joint Adjudication: Resolve disputes through discussion and export final frozen `test_ground_truth.csv`.

---

## 5. Claude Code Action Plan
1. Implement `src/models/adjudicate.py`.
2. Run script to calculate Kappa and extract disagreements.
3. Conduct consensus meeting to adjudicate disputed rows.
4. Freeze `test_ground_truth.csv`.

---

## 6. Unit Testing & Acceptance Criteria
- `test_ground_truth.csv` finalized with 250 adjudicated labels.
- Inter-rater Kappa score documented and reported.
