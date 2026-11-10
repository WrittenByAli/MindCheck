# Subtask 1.5.4: Dataset Manifest & Statistical Distribution Audit

**Subtask ID**: `1.5.4.DATASET.AUDIT.MANIFEST`  
**Parent Task**: `Task 1.5`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Generate automated dataset summary report with class counts, rating statistics, and ply distributions.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Split parquet/csv files in `data/processed/`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `data/processed/dataset_summary.json` and statistical bar charts.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/data/audit.py`:
   - Compute: total games, total windows, per-class counts in train/val/test splits.
   - Compute mean player rating, mean ply index, and result breakdown (1-0, 0-1, 1/2-1/2).
   - Export summary to `data/processed/dataset_summary.json`.
   - Generate distribution plots in `experiments/plots/dataset_class_distribution.png`.

---

## 5. Claude Code Action Plan
1. Implement `src/data/audit.py`.
2. Run audit over processed dataset.
3. Verify JSON summary output.

---

## 6. Unit Testing & Acceptance Criteria
- `data/processed/dataset_summary.json` exists with valid metrics.
- Class distribution charts generated.
