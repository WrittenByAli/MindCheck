# Subtask 2.7.2: Confusion Matrix, Macro-F1 & Per-Class Metrics Generator

**Subtask ID**: `2.7.2.METRICS.CONFUSION.MATRICES`  
**Parent Task**: `Task 2.7`  
**Phase**: Phase 2  
**Milestone**: Milestone 14  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Compute classification metrics and generate normalized confusion matrix heatmaps.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `experiments/evaluation_results.csv`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `experiments/plots/confusion_matrix_*.png` and `experiments/metrics_summary.json`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/models/metrics.py`:
   - Compute: Overall Accuracy, Macro-F1, per-class Precision/Recall/F1 for all 6 classes.
   - Plot normalized confusion matrices using Seaborn heatmap.
   - Format publication-ready LaTeX / Markdown tables.

---

## 5. Claude Code Action Plan
1. Implement `src/models/metrics.py`.
2. Generate confusion matrix plots for each model.
3. Verify metric calculation with `scikit-learn.metrics.classification_report`.

---

## 6. Unit Testing & Acceptance Criteria
- Confusion matrix heatmaps saved to `experiments/plots/`.
- Macro-F1 and per-class metrics exported to JSON.
