# Subtask 1.8.2: Validation Early Stopping, Learning Rate Scheduler & Seeds

**Subtask ID**: `1.8.2.VALIDATION.EARLY.STOPPING.SCHEDULER`  
**Parent Task**: `Task 1.8`  
**Phase**: Phase 1  
**Milestone**: Milestone 7  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement validation Macro-F1 early stopping, cosine annealing schedule, and deterministic seeding.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Validation DataLoader and checkpoint directory.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Early stopping handler and learning rate updates.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Scheduler: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)`.
2. Early Stopping: Monitor **Validation Macro-F1**; if no improvement for 5 consecutive epochs, terminate and reload best weights.
3. Deterministic Seed: Set seed 42 across PyTorch, NumPy, and Python random.

---

## 5. Claude Code Action Plan
1. Implement early stopping and scheduling in `src/models/train.py`.
2. Write test verifying early stopping triggers when validation metrics plateau.
3. Verify deterministic repeatability across duplicate seeded runs.

---

## 6. Unit Testing & Acceptance Criteria
- Training halts automatically when validation metric stops improving.
- Best checkpoint restored and saved to disk.
