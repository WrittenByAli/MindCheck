---
name: mindcheck-ml-pipeline
description: Specialized guidance for MindCheck's chess dataset processing, PyTorch residual CNN training and rigorous ML evaluation. Load when working on PGN ingestion, tensor encoding, weak labeling, dataset splitting, model training, or evaluation/ablation code.
---

# MindCheck ML Pipeline

## Data pipeline
- Validate PGNs while streaming (RAM < 300MB); reject malformed games rather than silently skipping.
- Windows are 8 half-moves of history ($B_{t-7}...B_t$), zero-padded for early plies with a 1D `(8,)` validity mask.
- Tensor encoding: `(112, 8, 8)` float32 per window (see `mindcheck-chess-engineering` for plane layout and orientation rules).
- Weak labels come from a retrospective labeler inspecting plies $t+1...t+8$ — **never** feed future-move information into the neural model's input tensor itself.
- Split at the **game level** (`game_id`), 70/15/15. Verify zero game-id overlap across splits before training.
- Detect and report duplicate windows/games before finalizing a split.

## Model & training
- Residual CNN in PyTorch: 4 residual blocks, 64 channels, 6-way output (5 plans + `NO_PLAN`).
- Loss: cross-entropy with class-imbalance handling (class weights or focal loss) — report which was used.
- Optimizer: Adam. Use early stopping on validation macro-F1; checkpoint the best epoch only.
- Set and log random seeds for every run; store the exact config alongside each checkpoint.
- Maintain the 1-ply (current-position-only) ablation baseline (`single_board_cnn.py`) as a standing comparison, plus a history-length ablation (1 vs 4 vs 8 ply) across multiple seeds.

## Evaluation
- Report macro-F1, per-class precision/recall/F1, and confusion matrices — never a single accuracy number alone.
- Select thresholds/hyperparameters on validation only; touch the held-out test set exactly once, at the end.
- Keep weak-labeled training data and independently annotated reference/test data clearly separate in all reports (report Cohen's kappa for annotator agreement where applicable).

## Hard restrictions
- Never use future moves as neural model inputs.
- Never allow windows from the same game to leak across train/val/test.
- Never fabricate or guess metrics — only report numbers from an actual run's output.
- Never claim a model is "trained" unless a checkpoint file actually exists on disk; check before claiming.
- Never substitute rule-based (`src/rules/`) output for the required neural classifier's predictions.
