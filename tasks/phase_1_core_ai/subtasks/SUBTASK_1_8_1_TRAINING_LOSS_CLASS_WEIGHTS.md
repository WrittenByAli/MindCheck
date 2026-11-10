# Subtask 1.8.1: PyTorch Training Loop, Loss & Class Weighting

**Subtask ID**: `1.8.1.TRAINING.LOSS.CLASS.WEIGHTS`  
**Parent Task**: `Task 1.8`  
**Phase**: Phase 1  
**Milestone**: Milestone 7  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Build the PyTorch training loop with class-weighted Cross-Entropy loss and label smoothing.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: DataLoader instances and configuration parameters.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/models/train.py` with complete optimization step.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `ModelTrainer`:
   - Loss: `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`.
   - Weights: $w_c = \frac{N}{K \cdot N_c}$ to prevent majority class dominance.
   - Optimizer: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`.
   - Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

---

## 5. Claude Code Action Plan
1. Implement training loop in `src/models/train.py`.
2. Write unit tests in `tests/test_training_step.py` verifying loss decreases across 5 iterations on synthetic batch.
3. Test class weight calculation.

---

## 6. Unit Testing & Acceptance Criteria
- Single training step executes without NaN or exploding gradients.
- Class weights correctly penalize rare class errors.
