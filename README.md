# MindCheck: History-Aware Chess Plan Recognition Using a Neural Network

[![Course](https://img.shields.io/badge/Course-AI2002%20Artificial%20Intelligence-blue.svg)](https://isb.nu.edu.pk/)
[![Semester](https://img.shields.io/badge/Semester-Fall%202026-orange.svg)](#)
[![Path](https://img.shields.io/badge/Project%20Path-A%20(Software--Based)-green.svg)](#)
[![Group](https://img.shields.io/badge/Group-29-purple.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)

---

## Executive Summary
**MindCheck** is an Artificial Intelligence semester project for **AI2002: Artificial Intelligence (Fall 2026)** at the **National University of Computer and Emerging Sciences (FAST NUCES, Islamabad Campus)**, instructed by **Dr. Muhammad Bilal**.

Traditional chess engines output numeric evaluation scores (e.g. `+1.8`) and concrete tactical move variations, but fail to explain high-level strategic intent to learners. **MindCheck** bridges this pedagogical gap by formulating chess strategic plan recognition as a **spatial-temporal supervised learning problem**. Given a sequence of the recent 8 half-moves ($112 \times 8 \times 8$ tensor), a custom **PyTorch Deep Residual Convolutional Neural Network (ResNet)** classifies the underlying strategic theme into one of 5 distinct strategic plan classes (or `NO_PLAN`), tracks plan stability and onset across game plies using a temporal hysteresis state machine, and presents grounded symbolic evidence in an interactive **Streamlit** application.

---

## Group Information & Work Distribution

**Group Number**: Group 29  
**Selected Path**: **Path A - Software-Based** (Strictly Software, Zero Hardware)  

| Team Member | Roll Number | Primary Role & Component Ownership |
| :--- | :---: | :--- |
| **Shahzad Ahmad** | `24I-3090` | **Lead: Data & Neural Core**<br>• PGN Streaming Parser & Game-Level Dataset Splitting<br>• Symbolic Bitboard Fact Extractor & 5-Plan Rule Engine<br>• Spatial-Temporal Tensor Encoding Pipeline ($112 \times 8 \times 8$)<br>• PyTorch Residual CNN Architecture & Baseline Training Loops<br>• Quantitative Evaluation Suite, Confusion Matrices & Ablations |
| **Muhammad Ali** | `24I-3158` | **Lead: Deployment, UI & Temporal Logic**<br>• Interactive Streamlit Web Application (`src/ui/app.py`)<br>• Dynamic SVG Chessboard Component with Strategic Overlays<br>• Stateful Temporal Plan Tracker (Hysteresis & Onset Backtracking)<br>• Neural-Rule Score Fusion & Prediction Abstention Gating<br>• Deterministic Natural Language Explanation Generator<br>• Stockfish UCI Counterplay Screening & GCR Packaging |

**Shared Responsibilities**:
- Written Strategic Plan Rubric Definition ([docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md](docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md))
- Double-Blind Independent Annotation of 250 Held-Out Test Windows & Cohen's Kappa Calculation ($\kappa$)
- Comprehensive Academic Project Report ([docs/02_PROJECT_STATEMENT_COMPLIANCE_MATRIX.md](docs/02_PROJECT_STATEMENT_COMPLIANCE_MATRIX.md))
- Presentation Slide Deck & Live Oral Defense (07–09 Dec 2026)

---

## 5-Plan Strategy Taxonomy (+ Null Class)

1. **Kingside Pawn Storm (`K_STORM`)**: Coordinated advance of flank pawns (g, h, f) against the opponent's castled king.
2. **Queenside Expansion (`Q_EXP`)**: Coordinated advance of queenside pawns (a, b, c) to secure spatial dominance or open flank files.
3. **Outpost Occupation (`OUTPOST`)**: Maneuvering a minor piece onto an uncontested square hole defended by friendly pawns.
4. **Rook-File Buildup (`ROOK_FILE`)**: Heavy piece battery concentration along an open or semi-open file.
5. **Pawn-Chain Undermining (`CHAIN_UNDERMINE`)**: Thematic pawn levers attacking the base or apex of a locked opponent pawn chain.
6. **No Dominant Plan (`NO_PLAN`)**: Tactical skirmishes, opening development, or balanced/transitional positions.

---

## Repository Structure & Documentation Map

```
d:\AI_Project\
├── docs/                                   # Architecture, Compliance & Rubric Specifications
│   ├── 00_PROJECT_OVERVIEW.md              # Problem statement, research questions & CLO mapping
│   ├── 01_TECH_STACK_AND_ENVIRONMENT.md    # Library versions, optimizations & setup guide
│   ├── 02_PROJECT_STATEMENT_COMPLIANCE_MATRIX.md # Clause-by-clause audit vs AI2002 requirements
│   ├── 03_PHASED_ROADMAP_AND_MILESTONES.md # Step-by-step milestone timeline to Dec 6 deadline
│   ├── 04_WORK_DISTRIBUTION_AND_OWNERSHIP.md # Detailed RACI matrix and inter-module contracts
│   ├── 05_ARCHITECTURE_AND_PIPELINE_DESIGN.md # End-to-end tensor schemas & ResNet architecture
│   ├── 06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md # Operational definitions & human annotation rules
│   ├── 07_INDEPENDENT_EVALUATION_PROTOCOL.md # 250-position test protocol, Cohen's Kappa & bootstrap
│   └── 08_STREAMLIT_UI_UX_SPECIFICATION.md # Frontend wireframes, SVG overlays & timeline views
│
├── tasks/                                  # Actionable Markdown Files for Claude Code Execution
│   ├── phase_1_core_ai/                    # Phase 1: Data, Rules, Tensors & Model Training
│   │   ├── README.md                       # Phase 1 overview and deliverables checklist
│   │   ├── TASK_1_1_ENVIRONMENT_AND_SCAFFOLDING.md
│   │   ├── TASK_1_2_DATASET_ACQUISITION_AND_PGN_PARSING.md
│   │   ├── TASK_1_3_SYMBOLIC_FACT_EXTRACTOR.md
│   │   ├── TASK_1_4_FIVE_PLAN_RULE_ENGINE.md
│   │   ├── TASK_1_5_WEAK_LABEL_GENERATOR_AND_SPLIT.md
│   │   ├── TASK_1_6_TENSOR_ENCODING_PIPELINE.md
│   │   ├── TASK_1_7_RESNET_MODEL_ARCHITECTURE.md
│   │   ├── TASK_1_8_TRAINING_AND_OPTIMIZATION_PIPELINE.md
│   │   └── TASK_1_9_INDEPENDENT_ANNOTATION_BENCHMARK.md
│   └── phase_2_deployment_and_evaluation/  # Phase 2: Deployment, Tracking, UI & Evaluation
│       ├── README.md                       # Phase 2 overview and deliverables checklist
│       ├── TASK_2_1_TEMPORAL_TRACKER_ENGINE.md
│       ├── TASK_2_2_EVIDENCE_FUSION_MODULE.md
│       ├── TASK_2_3_DETERMINISTIC_EXPLANATION_GENERATOR.md
│       ├── TASK_2_4_STREAMLIT_CORE_APPLICATION.md
│       ├── TASK_2_5_STREAMLIT_BOARD_AND_EVIDENCE_VIZ.md
│       ├── TASK_2_6_STOCKFISH_COUNTERPLAY_INTEGRATION.md
│       ├── TASK_2_7_EVALUATION_SUITE_AND_ABLATIONS.md
│       ├── TASK_2_8_PROJECT_REPORT_AND_SLIDES.md
│       └── TASK_2_9_FINAL_PACKAGING_AND_DEMO_CHECK.md
│
├── src/                                    # Modular Source Code
│   ├── data/                               # PGN parser, dataset splitter, tensor encoder
│   ├── rules/                              # 64-bit bitboard fact extractor & rule templates
│   ├── models/                             # ResNet backbone, single-board baseline, train, export
│   ├── tracking/                           # Temporal state tracker, hybrid fusion, explanations
│   ├── ui/                                 # Streamlit app, SVG chessboard view, analytical charts
│   └── utils/                              # Pydantic configuration, logging, Stockfish client
│
├── tests/                                  # PyTest Unit & Integration Tests
├── config/                                 # Central YAML configuration
│   └── config.yaml
├── data/                                   # Data storage (raw, processed, annotations)
├── models/checkpoints/                     # Serialized PyTorch weights and ONNX models
├── requirements.txt                        # Pinned dependencies
└── README.md                               # This file
```

---

## Phased Project Roadmap

### Phase 1: Core AI & Machine Learning Pipeline (11 Nov – 21 Nov 2026)
- **Milestone 1**: Environment setup, dependency verification, and typed configuration schema.
- **Milestone 2**: Public master PGN streaming parser and 8-ply sliding window generator.
- **Milestone 3**: 64-bit bitboard fact extractor for pawn chains, outposts, and king safety.
- **Milestone 4**: 5-Plan rule templates, lookahead execution heuristics, and leak-free game-level splits.
- **Milestone 5**: Canonical spatial-temporal tensor encoding ($112 \times 8 \times 8$) and PyTorch DataLoader.
- **Milestone 6**: Deep Residual CNN model implementation and single-board ablation baseline.
- **Milestone 7**: Cross-Entropy training loop, cosine annealing, early stopping, and ONNX export.
- **Milestone 8**: Independent double-blind annotation of 250 held-out windows and Cohen's Kappa score.

### Phase 2: Deployment, Tracking, Evaluation & Submission (22 Nov – 06 Dec 2026)
- **Milestone 9**: Stateful temporal plan tracking with hysteresis and onset estimation.
- **Milestone 10**: Neural-rule score fusion and prediction abstention gating.
- **Milestone 11**: Controlled deterministic natural language explanation generator.
- **Milestone 12**: Mandatory Streamlit web application with SVG board and ply navigation.
- **Milestone 13**: Stockfish UCI integration for defensive counterplay screening.
- **Milestone 14**: Quantitative benchmarking (History CNN vs 1-Ply CNN vs Rules), confusion matrices, and bootstrap CIs.
- **Milestone 15**: Formal academic project report, presentation slides, clean-install verification, and single GCR ZIP submission.

---

## Quick Start & Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated test suite
pytest tests/

# 4. Launch interactive Streamlit web application
streamlit run src/ui/app.py
```

---

## Academic Integrity & Compliance Statement
This project is implemented entirely by **Group 29 (Muhammad Ali and Shahzad Ahmad)** in compliance with the **AI2002 Project Statement (Fall 2026)**. No external project, repository, or tutorial code has been plagiarized. External libraries (`torch`, `python-chess`, `streamlit`) are utilized strictly as documented support infrastructure.
