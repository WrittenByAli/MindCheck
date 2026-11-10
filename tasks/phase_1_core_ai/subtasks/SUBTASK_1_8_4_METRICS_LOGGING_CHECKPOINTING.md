# Subtask 1.8.4: Training Metrics Logging, TensorBoard & Checkpoint Selection

**Subtask ID**: `1.8.4.METRICS.LOGGING.CHECKPOINTING`  
**Parent Task**: `Task 1.8`  
**Phase**: Phase 1  
**Milestone**: Milestone 7  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Log epoch losses, accuracies, and per-class F1 metrics; manage checkpoint files.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Training and validation statistics per epoch.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `models/checkpoints/best_resnet.pt`, `training_metrics.json`, and learning curve plots.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `MetricsLogger`:
   - Logs: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1`, per-class F1.
   - Exports `experiments/training_history.json`.
   - Plots training vs validation loss and Macro-F1 curves to `experiments/plots/learning_curves.png`.
   - Saves checkpoint dict with `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_f1`.

---

## 5. Claude Code Action Plan
1. Implement `src/models/logger.py`.
2. Integrate logger into `src/models/train.py`.
3. Verify plot generation and checkpoint file persistence.

---

## 6. Unit Testing & Acceptance Criteria
- `best_resnet.pt` saved with metadata and retrievable weights.
- Training curve plots generated cleanly.
