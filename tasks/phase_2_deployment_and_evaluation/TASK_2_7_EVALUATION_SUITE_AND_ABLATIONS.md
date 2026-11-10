# Task 2.7: Comprehensive Evaluation Suite & Failure Analysis

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 14  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Build the scientific evaluation suite to evaluate MindCheck against baselines on the frozen 250-position ground truth test set. Compute Macro-F1, per-class Precision/Recall/F1, multi-class confusion matrices, game-cluster bootstrap confidence intervals, history ablation differences, and a systematic qualitative audit of model failures.

---

## 2. Subtask Breakdown

### Subtask 2.7.1: Multi-Model Benchmark Runner (`src/models/evaluate.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Evaluate 4 systems side-by-side on `data/annotations/test_ground_truth.csv`:
    1. **MindCheck History-Aware ResNet (8 plies)**
    2. **Current-Position ResNet Baseline (1 ply)**
    3. **Rule-Only Symbolic Detector**
    4. **Neural-Rule Hybrid Model**
  - Save predictions and ground truth labels to `experiments/evaluation_results.csv`.

### Subtask 2.7.2: Metric Calculations & Confusion Matrix Heatmaps (`src/models/metrics.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Calculate:
    - Overall Accuracy
    - Macro-Averaged Precision, Recall, and F1-Score
    - Per-class Precision, Recall, and F1 for all 6 classes
  - Generate normalized confusion matrix heatmaps using Seaborn/Matplotlib:
    - Save to `experiments/plots/confusion_matrix_history_resnet.png`
    - Save to `experiments/plots/confusion_matrix_single_board.png`
    - Save to `experiments/plots/confusion_matrix_rule_baseline.png`

### Subtask 2.7.3: Game-Cluster Bootstrap Confidence Intervals (`src/models/bootstrap.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Cluster Bootstrap Algorithm:
    - Sample unique game IDs with replacement $B = 1,000$ times.
    - Extract all test windows belonging to the sampled games.
    - Compute Macro-F1 on each bootstrap sample.
    - Determine empirical 2.5th and 97.5th percentiles to report 95% Confidence Intervals:
      $$\text{Macro-F1} = \hat{\mu} \; [95\% \text{ CI}: \text{lower}, \text{upper}]$$

### Subtask 2.7.4: Qualitative Failure Analysis & Audit Report (`docs/failure_analysis.md`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Select and document 10 representative test failures:
    - **5 False Positives**: Situations where the model predicted a plan that was not present.
    - **5 False Negatives**: Situations where a strategic plan was underway but missed.
  - For each failure, document: FEN, move history, predicted vs actual class, model probabilities, and chess explanation of why the failure occurred.

---

## 3. Benchmark CLI Command

```bash
# Run full evaluation across all systems and generate plots
python -m src.models.evaluate \
  --test_data data/annotations/test_ground_truth.csv \
  --checkpoint models/checkpoints/best_resnet.pt \
  --output_dir experiments/
```

---

## 4. Claude Code Action Plan
1. Implement `src/models/evaluate.py` orchestrating multi-system inference.
2. Implement `src/models/metrics.py` for confusion matrices and classification reports.
3. Implement `src/models/bootstrap.py` for game-cluster bootstrapping.
4. Write `docs/failure_analysis.md` summarizing qualitative failure patterns.
5. Implement `tests/test_evaluation_pipeline.py` verifying metric calculations on known synthetic inputs.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_evaluation_pipeline.py` -> 100% pass.
- All evaluation figures and CSV tables successfully generated under `experiments/`.
- Macro-F1 and 95% bootstrap intervals logged and ready for inclusion in the project report.
