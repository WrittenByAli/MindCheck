# Task 1.8: Training, Optimization & Ablation Pipeline

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 7  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Build a reproducible, robust PyTorch training and optimization pipeline. Implement weighted Cross-Entropy loss to handle class imbalance, AdamW optimizer with cosine annealing schedules, validation-driven early stopping, and comparative history-length ablation experiments (1-ply vs 4-ply vs 8-ply).

---

## 2. Subtask Breakdown

### Subtask 1.8.1: Training Engine & Loss Function (`src/models/train.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Class `ModelTrainer`.
  - Objective Function: `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`.
    - Class weights inversely proportional to training frequency: $w_c = \frac{N}{K \cdot N_c}$.
  - Optimizer: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`.
  - Mixed Precision Training: Support `torch.cuda.amp.autocast()` if GPU is available.

### Subtask 1.8.2: Schedulers, Early Stopping & Seeds (`src/models/train.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Learning Rate Schedule: `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)`.
  - Early Stopping: Track Validation **Macro-F1** (not just loss). If Validation Macro-F1 does not improve for 5 consecutive epochs, terminate training and restore the best checkpoint.
  - Reproducibility Seed: Deterministic seeding of `torch.manual_seed(42)`, `np.random.seed(42)`, and `torch.backends.cudnn.deterministic = True`.

### Subtask 1.8.3: History-Length Ablation Runner (`src/models/train_ablation.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Script to train and compare three configurations under matched settings (identical seeds, data splits, and optimizer):
    1. **1-ply (Current-Position Baseline)**: Input shape `(14, 8, 8)`
    2. **4-ply History**: Input shape `(56, 8, 8)`
    3. **8-ply History (MindCheck Standard)**: Input shape `(112, 8, 8)`
  - Outputs comparative summary table: Macro-F1, per-class F1, parameter counts.

### Subtask 1.8.4: Metrics Logging & Checkpoint Selection (`src/models/logger.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Log training and validation loss, accuracy, and Macro-F1 at every epoch.
  - Export training curves to `experiments/plots/learning_curves.png`.
  - Save best model checkpoint to `models/checkpoints/best_resnet.pt` along with metadata JSON recording hyperparameters, best epoch, and validation metrics.

---

## 3. CLI Interface & Execution Commands

```bash
# Train primary 8-ply History-Aware ResNet
python -m src.models.train --config config/config.yaml --epochs 30 --batch_size 64

# Train 1-ply Current-Position Ablation Baseline
python -m src.models.train --config config/config.yaml --epochs 30 --ablation_single_board

# Run automated multi-seed ablation comparison
python -m src.models.train_ablation --seeds 42 123 999
```

---

## 4. Claude Code Action Plan
1. Implement `src/models/train.py` with full training/validation loops.
2. Implement `src/models/train_ablation.py`.
3. Implement `src/models/logger.py`.
4. Create `tests/test_training_step.py` verifying that a single forward-backward optimization step successfully reduces loss on a synthetic mini-batch.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_training_step.py` -> 100% pass.
- Primary model achieves $> 0.60$ Validation Macro-F1 on the weak-label validation split.
- Training automatically saves `models/checkpoints/best_resnet.pt` with valid weights.
