# MindCheck: Phased Roadmap & Step-by-Step Milestones

**Target Delivery Date**: Fully Functional by Start of December (Final Submission: **06 December 2026**)  
**Current Date**: 11 November 2026  
**Presentation / Demonstration Window**: 07–09 December 2026  

---

## High-Level Phase Division

```
========================================================================================
PHASE 1: Core AI & Machine Learning Pipeline (11 Nov 2026 – 21 Nov 2026)
----------------------------------------------------------------------------------------
Focus: Data engineering, symbolic fact extraction, weak-labeling, tensor pipelines,
       PyTorch ResNet training, baseline comparisons, and independent annotations.
========================================================================================
                                      │
                                      ▼
========================================================================================
PHASE 2: System Integration, Streamlit Deployment & Evaluation (22 Nov 2026 – 06 Dec 2026)
----------------------------------------------------------------------------------------
Focus: Temporal plan tracking, evidence fusion, interactive Streamlit deployment,
       Stockfish counterplay integration, comprehensive ablation benchmarks,
       formal academic report, presentation slides, and single ZIP submission.
========================================================================================
```

---

## Detailed Step-by-Step Milestones

### Phase 1: Core AI & Machine Learning Pipeline

#### Milestone 1: Environment Setup & Architecture Definition (Target: 11 Nov – 12 Nov)
- [x] Configure Python 3.11+ virtual environment and verify CUDA/CPU availability.
- [x] Establish modular directory layout (`src/`, `tasks/`, `docs/`, `tests/`, `config/`).
- [x] Define typed runtime configuration schema (`config/config.yaml` + Pydantic validation).
- [x] Establish unit testing framework with PyTest.

#### Milestone 2: Dataset Acquisition & PGN Ingestion Engine (Target: 13 Nov – 14 Nov)
- [x] Select and acquire public PGN dataset (Lichess Elite Database subset, high-rated games 2200+ Elo).
- [x] Implement memory-safe streaming PGN reader using `chess.pgn.read_game()`.
- [x] Implement game validator: filter out ultra-short games (< 20 plies), malformed sequences, and non-standard variants.
- [x] Construct 8-half-move sliding history window generator across game plies.

#### Milestone 3: Symbolic Board Fact & Predicate Extractor (Target: 15 Nov – 16 Nov)
- [x] Implement 64-bit bitboard fact extractor for piece placement, legal mobility, attacks, and defense.
- [x] Implement pawn structure analyzer: pawn chains, chain bases, backward pawns, open/semi-open files, king shield pawns.
- [x] Implement strategic predicates: outpost detection (uncontested squares for knights/bishops), file control, rook doubling.
- [x] Create comprehensive PyTest test suite verifying chess predicates on classic benchmark positions.

#### Milestone 4: Five-Plan Symbolic Rule Engine & Weak-Labeling (Target: 17 Nov – 18 Nov)
- [x] Formalize operational rule templates for all 5 strategic plans:
  1. Kingside Pawn Storm
  2. Queenside Expansion
  3. Outpost Occupation
  4. Rook-File Buildup
  5. Pawn-Chain Undermining
- [x] Implement lookahead retrospective execution heuristic (inspecting bounded next 6–10 plies to verify plan execution for weak-labeling).
- [x] Implement ambiguity resolver and null class assignment (`NO_PLAN`).
- [x] Partition dataset into strict **game-level** Train/Val/Test splits (70% train, 15% val, 15% test) to prevent window leakage.

#### Milestone 5: Spatial-Temporal Tensor Encoding Pipeline (Target: 19 Nov)
- [x] Implement canonical board state tensor encoding: $14 \times 8 \times 8$ (12 piece planes + castling rights + en-passant + side to move).
- [x] Stack 8 consecutive half-move states into spatial-temporal tensor: shape `(8, 14, 8, 8)` or `(112, 8, 8)`.
- [x] Implement board orientation normalization (flip board so attacking side always advances upward).
- [x] Implement history padding mask for early-game plies (< 8 plies).
- [x] Implement high-throughput PyTorch `Dataset` and `DataLoader` with memory-mapped disk caching.

#### Milestone 6: Deep Residual CNN Architecture (Target: 20 Nov)
- [x] Implement PyTorch Residual Convolutional Neural Network (3–4 residual blocks, 64 channels, BatchNorm/LayerNorm, GELU/ReLU).
- [x] Implement Global Average Pooling (GAP) and 6-class linear classification head.
- [x] Implement Current-Position-Only CNN baseline (evaluating only ply 0) for direct research ablation.
- [x] Verify forward pass, gradient flow, and parameter count (~150k–300k parameters for fast CPU inference).

#### Milestone 7: Model Training, Checkpointing & Baseline Experiments (Target: 21 Nov)
- [x] Implement training loop with Cross-Entropy Loss, AdamW optimizer, and cosine annealing schedule.
- [x] Implement class weighting or focal loss to mitigate class imbalance.
- [x] Implement validation-driven early stopping and model checkpoint serialization (`models/checkpoints/best_model.pt`).
- [x] Export trained checkpoint to ONNX / TorchScript for zero-latency CPU inference.

#### Milestone 8: Independent Ground Truth Annotation Benchmark (Target: 21 Nov – 22 Nov)
- [x] Extract 200–300 held-out test position windows unseen during training.
- [x] Independent double-blind annotation by **Muhammad Ali** and **Shahzad Ahmad** using the formal written rubric.
- [x] Compute inter-rater agreement and Cohen's Kappa score ($\kappa$).
- [x] Resolve discrepancies and establish frozen gold-standard test benchmark.

---

### Phase 2: System Integration, Streamlit Deployment & Evaluation

#### Milestone 9: Temporal Plan State Tracker (Target: 23 Nov – 24 Nov)
- [x] Implement per-player state machine tracking plan state across consecutive plies.
- [x] Implement hysteresis switching logic: require 2 supporting updates to activate a plan, 2 contradictory updates to switch plans.
- [x] Implement plan onset detection (identifying the earliest supporting ply).
- [x] Differentiate between strategic progress (plan completion) and immediate threat severity.

#### Milestone 10: Symbolic Evidence & Neural Prediction Fusion (Target: 25 Nov)
- [x] Calibrate neural softmax scores with temperature scaling.
- [x] Implement score fusion blending normalized symbolic rule confidence with neural probabilities.
- [x] Implement prediction abstention logic: trigger `NO_PLAN` / uncertain when top probability or top-2 margin falls below threshold.

#### Milestone 11: Controlled Deterministic Explanation Generator (Target: 26 Nov)
- [x] Implement strict templated natural language generator separating observed facts from neural hypotheses.
- [x] Map active plan and grounded bitboard facts to clear, human-understandable chess explanations.
- [x] Implement structured claim validator ensuring zero hallucination.

#### Milestone 12: Streamlit Interactive Web Application (Target: 27 Nov – 29 Nov)
- [x] Build multi-panel Streamlit dashboard: Game Input -> Board Visualization -> Neural Prediction -> Strategic Timeline -> Grounded Evidence.
- [x] Integrate interactive SVG chessboard rendering with dynamic arrows and square highlights.
- [x] Implement PGN file upload + pre-loaded famous master games selector.
- [x] Implement move stepper navigation (Previous, Next, Jump to Move, Auto-play).
- [x] Implement real-time inference execution under 20ms using cached ONNX / TorchScript models.

#### Milestone 13: Optional Stockfish Defensive Counterplay Screening (Target: 30 Nov)
- [x] Integrate `chess.engine` UCI client with configurable search budget (depth 10–12).
- [x] Screen defensive candidate moves that disrupt the opponent's detected plan.
- [x] Present engine analysis with clear disclaimer: engine-assisted suggestions, not guaranteed counter-plans.

#### Milestone 14: Quantitative Evaluation Suite & Ablations (Target: 01 Dec – 02 Dec)
- [x] Execute automated benchmark comparing:
  1. MindCheck History-Aware ResNet (8 plies)
  2. Current-Position ResNet (1 ply)
  3. Rule-Only Detector
  4. Neural-Rule Hybrid Fusion
- [x] Generate Macro-F1, per-class Precision/Recall/F1, and Confusion Matrix plots.
- [x] Compute game-cluster bootstrap confidence intervals.
- [x] Conduct systematic qualitative failure analysis on misclassified instances.

#### Milestone 15: Academic Project Report, Slides & GCR Submission (Target: 03 Dec – 06 Dec)
- [x] Write comprehensive academic project report conforming to all AI2002 guidelines (PDF).
- [x] Generate high-resolution end-to-end system architecture & software pipeline diagram.
- [x] Design professional presentation slide deck for live defense.
- [x] Perform clean-install environment verification (`venv` rebuild and cold launch).
- [x] Package complete repository, models, sample PGNs, report, and slides into single GCR ZIP package before 06 Dec 2026.
