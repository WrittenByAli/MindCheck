# Phase 1: Core AI & Machine Learning Pipeline

**Phase Duration**: 11 November 2026 – 21 November 2026  
**Primary Engineering Lead**: **Shahzad Ahmad** (`24I-3090`)  
**Collaborating Engineer**: **Muhammad Ali** (`24I-3158`)  

---

## Executive Summary
Phase 1 delivers the complete scientific machine learning core of MindCheck:
1. Data ingestion, validation, and 8-ply sliding window extraction from raw PGN files.
2. Symbolic 64-bit bitboard fact extraction and 5-plan rule engine.
3. Retrospective lookahead execution heuristics for weak-label generation.
4. Leak-free game-level data splitting (70% Train, 15% Val, 15% Test).
5. Canonical spatial-temporal tensor encoding ($112 \times 8 \times 8$).
6. Deep Residual CNN model in PyTorch alongside single-board ablation baselines.
7. Reproducible training, validation early stopping, checkpointing, and ONNX export.
8. Independent double-blind annotation benchmark (250 positions) and Cohen's Kappa evaluation.

---

## Task Inventory

| Task ID | Task Title | Primary Owner | Status |
| :--- | :--- | :---: | :---: |
| [TASK_1_1](./TASK_1_1_ENVIRONMENT_AND_SCAFFOLDING.md) | Environment, Dependencies & Repository Scaffolding | Shahzad Ahmad / Muhammad Ali | Ready for Claude Code |
| [TASK_1_2](./TASK_1_2_DATASET_ACQUISITION_AND_PGN_PARSING.md) | Dataset Acquisition & PGN Ingestion Pipeline | Shahzad Ahmad | Ready for Claude Code |
| [TASK_1_3](./TASK_1_3_SYMBOLIC_FACT_EXTRACTOR.md) | Symbolic Chess Fact & Predicate Extractor | Shahzad Ahmad / Muhammad Ali | Ready for Claude Code |
| [TASK_1_4](./TASK_1_4_FIVE_PLAN_RULE_ENGINE.md) | 5-Plan Symbolic Rule Engine & Baseline | Shahzad Ahmad / Muhammad Ali | Ready for Claude Code |
| [TASK_1_5](./TASK_1_5_WEAK_LABEL_GENERATOR_AND_SPLIT.md) | Weak Labeling Engine & Dataset Splitting | Shahzad Ahmad / Muhammad Ali | Ready for Claude Code |
| [TASK_1_6](./TASK_1_6_TENSOR_ENCODING_PIPELINE.md) | Spatial-Temporal Chess Tensor Encoding Pipeline | Shahzad Ahmad | Ready for Claude Code |
| [TASK_1_7](./TASK_1_7_RESNET_MODEL_ARCHITECTURE.md) | Deep Residual CNN Architecture Implementation | Shahzad Ahmad / Muhammad Ali | Ready for Claude Code |
| [TASK_1_8](./TASK_1_8_TRAINING_AND_OPTIMIZATION_PIPELINE.md) | Training, Optimization & Ablation Pipeline | Shahzad Ahmad / Muhammad Ali | Ready for Claude Code |
| [TASK_1_9](./TASK_1_9_INDEPENDENT_ANNOTATION_BENCHMARK.md) | Independent Ground Truth Annotation Benchmark | Shared (Both Members) | Ready for Claude Code |

---

## Actionable Subtasks Index (for Claude Code)

| Subtask ID | Specification Document | Primary Owner | Target Deliverable |
| :---: | :--- | :---: | :--- |
| **1.1.1** | [SUBTASK_1_1_1_DEPENDENCY_SPECIFICATION.md](subtasks/SUBTASK_1_1_1_DEPENDENCY_SPECIFICATION.md) | Shahzad Ahmad | `requirements.txt` |
| **1.1.2** | [SUBTASK_1_1_2_DIRECTORY_SCAFFOLDING.md](subtasks/SUBTASK_1_1_2_DIRECTORY_SCAFFOLDING.md) | Muhammad Ali | `src/`, `tests/` scaffolding |
| **1.1.3** | [SUBTASK_1_1_3_CONFIGURATION_MANAGEMENT.md](subtasks/SUBTASK_1_1_3_CONFIGURATION_MANAGEMENT.md) | Shahzad Ahmad | `src/utils/config.py` |
| **1.2.1** | [SUBTASK_1_2_1_DATASET_ACQUISITION_MANIFEST.md](subtasks/SUBTASK_1_2_1_DATASET_ACQUISITION_MANIFEST.md) | Shahzad Ahmad | `data/raw/manifest.json` |
| **1.2.2** | [SUBTASK_1_2_2_STREAMING_PGN_PARSER.md](subtasks/SUBTASK_1_2_2_STREAMING_PGN_PARSER.md) | Shahzad Ahmad | `src/data/pgn_parser.py` |
| **1.2.3** | [SUBTASK_1_2_3_SLIDING_WINDOW_EXTRACTOR.md](subtasks/SUBTASK_1_2_3_SLIDING_WINDOW_EXTRACTOR.md) | Shahzad Ahmad | `src/data/window_extractor.py` |
| **1.3.1** | [SUBTASK_1_3_1_BITBOARD_FACT_EXTRACTOR.md](subtasks/SUBTASK_1_3_1_BITBOARD_FACT_EXTRACTOR.md) | Shahzad Ahmad | `src/rules/fact_extractor.py` |
| **1.3.2** | [SUBTASK_1_3_2_PAWN_STRUCTURE_PREDICATES.md](subtasks/SUBTASK_1_3_2_PAWN_STRUCTURE_PREDICATES.md) | Shahzad Ahmad | Pawn structures & king safety |
| **1.3.3** | [SUBTASK_1_3_3_POSITIONAL_OUTPOST_PREDICATES.md](subtasks/SUBTASK_1_3_3_POSITIONAL_OUTPOST_PREDICATES.md) | Muhammad Ali | Outposts & rook batteries |
| **1.3.4** | [SUBTASK_1_3_4_FACT_EXTRACTOR_UNIT_TESTS.md](subtasks/SUBTASK_1_3_4_FACT_EXTRACTOR_UNIT_TESTS.md) | Muhammad Ali | `tests/test_fact_extractor.py` |
| **1.4.1** | [SUBTASK_1_4_1_KINGSIDE_STORM_RULE.md](subtasks/SUBTASK_1_4_1_KINGSIDE_STORM_RULE.md) | Shahzad Ahmad | Kingside storm template |
| **1.4.2** | [SUBTASK_1_4_2_QUEENSIDE_EXPANSION_RULE.md](subtasks/SUBTASK_1_4_2_QUEENSIDE_EXPANSION_RULE.md) | Shahzad Ahmad | Queenside expansion template |
| **1.4.3** | [SUBTASK_1_4_3_OUTPOST_OCCUPATION_RULE.md](subtasks/SUBTASK_1_4_3_OUTPOST_OCCUPATION_RULE.md) | Muhammad Ali | Outpost occupation template |
| **1.4.4** | [SUBTASK_1_4_4_ROOK_FILE_BUILDUP_RULE.md](subtasks/SUBTASK_1_4_4_ROOK_FILE_BUILDUP_RULE.md) | Shahzad Ahmad | Rook file buildup template |
| **1.4.5** | [SUBTASK_1_4_5_PAWN_CHAIN_UNDERMINING_RULE.md](subtasks/SUBTASK_1_4_5_PAWN_CHAIN_UNDERMINING_RULE.md) | Muhammad Ali | Chain undermining template |
| **1.4.6** | [SUBTASK_1_4_6_RULE_ONLY_BASELINE_CLASSIFIER.md](subtasks/SUBTASK_1_4_6_RULE_ONLY_BASELINE_CLASSIFIER.md) | Shahzad Ahmad | `src/rules/rule_engine.py` |
| **1.5.1** | [SUBTASK_1_5_1_LOOKAHEAD_WEAK_LABEL_ENGINE.md](subtasks/SUBTASK_1_5_1_LOOKAHEAD_WEAK_LABEL_ENGINE.md) | Shahzad Ahmad | `src/data/weak_labeler.py` |
| **1.5.2** | [SUBTASK_1_5_2_NOISE_FILTER_AND_BALANCING.md](subtasks/SUBTASK_1_5_2_NOISE_FILTER_AND_BALANCING.md) | Shahzad Ahmad | Dataset balancing & filtering |
| **1.5.3** | [SUBTASK_1_5_3_GAME_LEVEL_SPLITTER.md](subtasks/SUBTASK_1_5_3_GAME_LEVEL_SPLITTER.md) | Shahzad Ahmad | `src/data/dataset_splitter.py` |
| **1.5.4** | [SUBTASK_1_5_4_DATASET_AUDIT_MANIFEST.md](subtasks/SUBTASK_1_5_4_DATASET_AUDIT_MANIFEST.md) | Muhammad Ali | `data/processed/dataset_summary.json` |
| **1.6.1** | [SUBTASK_1_6_1_BOARD_CANONICAL_ENCODER.md](subtasks/SUBTASK_1_6_1_BOARD_CANONICAL_ENCODER.md) | Shahzad Ahmad | 14-plane board representation |
| **1.6.2** | [SUBTASK_1_6_2_HISTORY_STACKING_NORMALIZATION.md](subtasks/SUBTASK_1_6_2_HISTORY_STACKING_NORMALIZATION.md) | Shahzad Ahmad | 112-channel normalized tensor |
| **1.6.3** | [SUBTASK_1_6_3_HISTORY_PADDING_VALIDITY_MASK.md](subtasks/SUBTASK_1_6_3_HISTORY_PADDING_VALIDITY_MASK.md) | Shahzad Ahmad | Zero-padding & validity masks |
| **1.6.4** | [SUBTASK_1_6_4_PYTORCH_DATASET_DATALOADER.md](subtasks/SUBTASK_1_6_4_PYTORCH_DATASET_DATALOADER.md) | Shahzad Ahmad | `src/data/chess_dataset.py` |
| **1.7.1** | [SUBTASK_1_7_1_RESIDUAL_CNN_BACKBONE.md](subtasks/SUBTASK_1_7_1_RESIDUAL_CNN_BACKBONE.md) | Shahzad Ahmad | `src/models/residual_cnn.py` |
| **1.7.2** | [SUBTASK_1_7_2_CLASSIFICATION_HEAD.md](subtasks/SUBTASK_1_7_2_CLASSIFICATION_HEAD.md) | Shahzad Ahmad | GAP & 6-class output head |
| **1.7.3** | [SUBTASK_1_7_3_SINGLE_BOARD_BASELINE_CNN.md](subtasks/SUBTASK_1_7_3_SINGLE_BOARD_BASELINE_CNN.md) | Shahzad Ahmad | `src/models/single_board_cnn.py` |
| **1.7.4** | [SUBTASK_1_7_4_ONNX_TORCHSCRIPT_EXPORT.md](subtasks/SUBTASK_1_7_4_ONNX_TORCHSCRIPT_EXPORT.md) | Muhammad Ali | `src/models/export.py` |
| **1.8.1** | [SUBTASK_1_8_1_TRAINING_LOSS_CLASS_WEIGHTS.md](subtasks/SUBTASK_1_8_1_TRAINING_LOSS_CLASS_WEIGHTS.md) | Shahzad Ahmad | `src/models/train.py` |
| **1.8.2** | [SUBTASK_1_8_2_VALIDATION_EARLY_STOPPING_SCHEDULER.md](subtasks/SUBTASK_1_8_2_VALIDATION_EARLY_STOPPING_SCHEDULER.md) | Shahzad Ahmad | Early stopping & Cosine Annealing |
| **1.8.3** | [SUBTASK_1_8_3_HISTORY_LENGTH_ABLATION.md](subtasks/SUBTASK_1_8_3_HISTORY_LENGTH_ABLATION.md) | Shahzad Ahmad | `src/models/train_ablation.py` |
| **1.8.4** | [SUBTASK_1_8_4_METRICS_LOGGING_CHECKPOINTING.md](subtasks/SUBTASK_1_8_4_METRICS_LOGGING_CHECKPOINTING.md) | Muhammad Ali | `src/models/logger.py` |
| **1.9.1** | [SUBTASK_1_9_1_ANNOTATION_RUBRIC_GUIDELINE.md](subtasks/SUBTASK_1_9_1_ANNOTATION_RUBRIC_GUIDELINE.md) | Shared (Both) | Annotation guidelines document |
| **1.9.2** | [SUBTASK_1_9_2_TEST_WINDOW_SAMPLER.md](subtasks/SUBTASK_1_9_2_TEST_WINDOW_SAMPLER.md) | Muhammad Ali | `src/data/sample_test_set.py` |
| **1.9.3** | [SUBTASK_1_9_3_INDEPENDENT_DOUBLE_BLIND_ANNOTATION.md](subtasks/SUBTASK_1_9_3_INDEPENDENT_DOUBLE_BLIND_ANNOTATION.md) | Shared (Both) | Raw member annotations CSVs |
| **1.9.4** | [SUBTASK_1_9_4_COHEN_KAPPA_ADJUDICATION.md](subtasks/SUBTASK_1_9_4_COHEN_KAPPA_ADJUDICATION.md) | Shahzad Ahmad | `src/models/adjudicate.py` |

---

## Phase 1 Deliverables & Acceptance Checklist
- [ ] `data/raw/` contains verified PGN archive.
- [ ] `data/processed/` contains memory-mapped `.npy` / `.pt` datasets with game-level splits.
- [ ] `models/checkpoints/best_model.pt` and `models/checkpoints/model.onnx` generated.
- [ ] `data/annotations/test_ground_truth.csv` populated with 250 adjudicated labels ($\kappa \ge 0.65$).
- [ ] All Phase 1 unit tests passing (`pytest tests/test_phase1.py`).
