# MindCheck: Work Distribution & Component Ownership

**Course**: AI2002: Artificial Intelligence | Fall 2026  
**Group Number**: Group 29  
**Team Members**:
- **Muhammad Ali** (Roll No: `24I-3158`)
- **Shahzad Ahmad** (Roll No: `24I-3090`)

---

## 1. Ownership & Responsibility Overview

Following the official **MindCheck Project Proposal (Section 13)** and the **AI Project Statement Requirements**, the engineering responsibilities are divided cleanly between both group members based on system boundaries. Both members are equally responsible for understanding all aspects of the pipeline for the live defense.

```
+---------------------------------------------------------------------------------------+
|                                    GROUP 29 WORK DISTRIBUTION                         |
+---------------------------------------------------------------------------------------+
|  SHAHZAD AHMAD (24I-3090)                 |  MUHAMMAD ALI (24I-3158)                  |
|  Lead: Data & Neural Core                 |  Lead: Deployment, UI & Temporal Logic    |
+-------------------------------------------+-------------------------------------------+
| - PGN Ingestion & Validation Pipeline     | - Streamlit Core Web Application          |
| - Symbolic Bitboard Fact Extractor        | - Interactive SVG Chessboard Component    |
| - 5-Plan Rule Engine & Weak-Labeler       | - Temporal Plan State Tracker             |
| - Spatial-Temporal Tensor Encoder         | - Evidence Fusion & Abstention Engine     |
| - PyTorch Residual CNN Implementation     | - Deterministic Explanation Generator     |
| - Training, Loss & Optimization Loop      | - Stockfish Counterplay Integration       |
| - Model Checkpointing & ONNX Export       | - Live Demo Rehearsal & Clean Packaging   |
+---------------------------------------------------------------------------------------+
|  SHARED RESPONSIBILITIES (Both Members)                                               |
| - Strategic Plan Rubric Definition & Ground-Truth Annotation (200-300 test positions) |
| - Cohen's Kappa Inter-Rater Reliability & Disagreement Resolution                    |
| - Academic Project Report Authoring & Visual Architecture Diagram                     |
| - Presentation Slide Deck & Live Oral Defense                                         |
+---------------------------------------------------------------------------------------+
```

---

## 2. Detailed Task Breakdown & RACI Matrix

**RACI Key**:  
- **R (Responsible)**: The primary owner who writes the code/docs.  
- **A (Accountable)**: The owner with final sign-off on component correctness.  
- **C (Consulted)**: The partner who reviews, tests, or provides inputs.  
- **I (Informed)**: The partner kept updated on progress.

| Task ID | Task Description | Shahzad Ahmad (24I-3090) | Muhammad Ali (24I-3158) | Primary Deliverable |
| :--- | :--- | :---: | :---: | :--- |
| **1.1** | Environment Scaffolding & Config | **R / A** | C | `requirements.txt`, `config.yaml` |
| **1.2** | PGN Ingestion & Sliding Window | **R / A** | C | `src/data/pgn_parser.py` |
| **1.3** | Bitboard Fact Extractor | **R / A** | C | `src/rules/fact_extractor.py` |
| **1.4** | 5-Plan Rule Engine & Weak-Labeler | **R / A** | C | `src/rules/rule_engine.py` |
| **1.5** | Game-Level Dataset Splitting | **R / A** | C | `src/data/dataset_splitter.py` |
| **1.6** | Spatial-Temporal Tensor Pipeline | **R / A** | C | `src/data/tensor_encoder.py` |
| **1.7** | Residual CNN Model Architecture | **R / A** | C | `src/models/residual_cnn.py` |
| **1.8** | Training Loop & Hyperparameters | **R / A** | C | `src/models/train.py` |
| **1.9** | Independent Annotation & Kappa | **R** | **R** | `data/annotations/test_labels.csv` |
| **2.1** | Temporal Plan State Tracker | C | **R / A** | `src/tracking/state_tracker.py` |
| **2.2** | Score Calibration & Fusion | C | **R / A** | `src/tracking/fusion.py` |
| **2.3** | Deterministic Explanation Engine | C | **R / A** | `src/tracking/explanations.py` |
| **2.4** | Streamlit Core App & Navigation | C | **R / A** | `src/ui/app.py` |
| **2.5** | SVG Board & Analytical Visuals | C | **R / A** | `src/ui/board_view.py` |
| **2.6** | Stockfish Counterplay Integration | C | **R / A** | `src/utils/stockfish_client.py` |
| **2.7** | Evaluation Benchmark & Metrics | **R / A** | C | `src/models/evaluate.py` |
| **2.8** | Project Report & Slides | **R** | **R** | `docs/report.pdf`, `docs/slides.pdf` |
| **2.9** | Clean Packaging & GCR ZIP | C | **R / A** | `MindCheck_Group29.zip` |

---

## 3. Inter-Module Interface Contracts

To enable seamless parallel development between Shahzad Ahmad and Muhammad Ali, the following data contracts are strictly established:

### Contract 1: Model Prediction Output (`Shahzad` -> `Muhammad Ali`)
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class PlanPrediction:
    predicted_class: str                    # e.g., "K_STORM"
    class_id: int                           # 0 to 5
    probabilities: Dict[str, float]         # {"K_STORM": 0.78, "Q_EXP": 0.05, ...}
    is_uncertain: bool                      # True if top probability < threshold
    logits: np.ndarray                      # Raw output logits
    model_version: str                      # Checkpoint identifier
```

### Contract 2: Symbolic Facts Output (`Shahzad` -> `Muhammad Ali`)
```python
@dataclass
class BoardFacts:
    ply_index: int
    turn: str                               # "white" or "black"
    king_squares: Dict[str, str]            # {"white": "g1", "black": "g8"}
    pawn_storms: Dict[str, List[str]]       # Advanced flank pawns
    open_files: List[str]                   # e.g., ["c", "e"]
    rook_batteries: List[Dict[str, any]]    # Doubled rooks on files
    outposts: List[Dict[str, any]]          # Supported uncontested squares
    pawn_chains: List[Dict[str, any]]       # Chains and base targets
    rule_scores: Dict[str, float]           # Normalized scores from symbolic detector
```

### Contract 3: Temporal Tracker Output (`Muhammad Ali` -> `Streamlit UI`)
```python
@dataclass
class TemporalPlanState:
    attacking_side: str                     # "white" or "black"
    current_active_plan: str                # e.g., "OUTPOST"
    onset_ply: int                          # Ply where plan first initiated
    consecutive_plies_active: int           # Duration of active plan
    plan_progress: float                    # 0.0 (nascent) to 1.0 (executed)
    plan_threat_severity: float             # Positional danger rating
    recent_plan_history: List[str]          # History across plies
    explanation_text: str                   # Grounded human-readable rationale
```

---

## 4. Academic Integrity & Defense Readiness
Both members must be able to explain, debug, and justify any portion of the codebase during the live oral demonstration. Regular peer code reviews and weekly cross-component walkthroughs will be conducted to ensure 100% dual competency.
