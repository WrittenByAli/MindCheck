# CLAUDE.md: Memory & Developer Guide for MindCheck

> **Persistent AI Context & System Manual for Claude Code**  
> *Project*: MindCheck: History-Aware Chess Plan Recognition Using a Neural Network  
> *Course*: AI2002: Artificial Intelligence | Fall 2026 | FAST NUCES Islamabad  
> *Instructor*: Dr. Muhammad Bilal | *Group*: Group 29  
> *Selected Path*: **Path A — Software-Based** (Strictly Software, Zero Hardware)  
> *Team Members*:
> - **Shahzad Ahmad** (`24I-3090`) — Lead: Data & Neural Core
> - **Muhammad Ali** (`24I-3158`) — Lead: Deployment, UI & Temporal Tracking

---

## 1. Project Mission & Architectural Overview

MindCheck formulates chess strategic plan recognition as a **spatial-temporal supervised learning problem**. Given a sequence of the recent 8 half-moves ($112 \times 8 \times 8$ tensor), a custom **PyTorch Deep Residual Convolutional Neural Network (ResNet)** classifies the underlying strategic theme into one of 5 distinct strategic plan classes (or `NO_PLAN`), tracks plan stability and onset across game plies using a temporal hysteresis state machine, and presents grounded symbolic evidence in an interactive **Streamlit** application.

### End-to-End Pipeline
```
[ PGN Archive (Lichess Elite 2200+) ]
                  │
                  ▼
[ Streaming PGN Parser & Sanity Filter ] (RAM < 300MB)
                  │
                  ▼
[ 8-Ply Sliding Window Extractor ] ([Bt-7 ... Bt])
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
[ 64-Bit Bitboard Facts ]  [ Retrospective Weak-Labeler ] (Inspects plies t+1...t+8)
     │                         │
     │                         ▼
     │            [ Game-Level Split: 70% Train / 15% Val / 15% Test ]
     │                         │
     │                         ▼
     │            [ Tensor Encoder: 112 x 8 x 8 Normalized Attacker Tensor ]
     │                         │
     │                         ▼
     │            [ PyTorch Residual CNN ] (4 blocks, 64 channels)
     │                         │
     └────────────┬────────────┘
                  ▼
     [ Hybrid Score Fusion & Abstention Gate ] (alpha * P_neural + (1-alpha) * S_rule)
                  │
                  ▼
     [ Stateful Temporal Plan Tracker ] (2-ply Hysteresis & Onset Backtracking)
                  │
                  ▼
     [ Deterministic Natural Language Explanation Generator ]
                  │
                  ▼
     [ Interactive Streamlit Web App ] (SVG Board, Move Stepper, Timeline, Charts)
```

---

## 2. Strategic Plan Taxonomy & Class Codes

| Class ID | Code | Plan Name | Core Preconditions & Operational Criteria |
| :---: | :--- | :--- | :--- |
| **0** | `K_STORM` | Kingside Pawn Storm | Flank pawns (g, h, f) advancing against castled king ($\ge$ rank 4/5); heavy piece support. |
| **1** | `Q_EXP` | Queenside Expansion | Coordinated a-, b-, c-pawn pushes past rank 4/5 for space advantage or minority attack. |
| **2** | `OUTPOST` | Outpost Occupation | Minor piece (N/B) occupying a hole defended by friendly pawn, uncontested by enemy pawns. |
| **3** | `ROOK_FILE` | Rook-File Buildup | Doubled/tripled heavy pieces (R+R, R+Q) on an open or semi-open file controlling rank 7/8. |
| **4** | `CHAIN_UNDERMINE` | Pawn-Chain Undermining | Thematic pawn lever (e.g., `...c5` vs `d4`) attacking the base or apex of an enemy pawn chain. |
| **5** | `NO_PLAN` | No Dominant Plan | Balanced maneuvering, uncommitted opening moves, or conflicting tactical tension. |

---

## 3. Team Division & Code Ownership

When developing or modifying modules, strictly adhere to component ownership:

```
+---------------------------------------------------------------------------------------------+
| SHAHZAD AHMAD (24I-3090)                        | MUHAMMAD ALI (24I-3158)                   |
| Lead: Data & Neural Core                        | Lead: Deployment, UI & Temporal Tracking  |
+-------------------------------------------------+-------------------------------------------+
| src/data/pgn_parser.py                          | src/ui/app.py                             |
| src/data/window_extractor.py                    | src/ui/board_view.py                      |
| src/data/weak_labeler.py                        | src/ui/game_loader.py                     |
| src/data/dataset_splitter.py                    | src/ui/navigator.py                       |
| src/data/tensor_encoder.py                      | src/ui/charts.py                          |
| src/data/chess_dataset.py                       | src/ui/timeline.py                        |
| src/rules/fact_extractor.py                     | src/ui/evidence_view.py                   |
| src/rules/rule_engine.py                        | src/tracking/state_tracker.py             |
| src/models/residual_cnn.py                      | src/tracking/fusion.py                    |
| src/models/single_board_cnn.py                  | src/tracking/explanations.py              |
| src/models/train.py                             | src/models/export.py                      |
| src/models/evaluate.py                          | src/utils/stockfish_client.py             |
| src/models/metrics.py                           | scripts/verify_install.py                 |
| src/models/bootstrap.py                         | scripts/package_submission.py             |
+-------------------------------------------------+-------------------------------------------+
| SHARED: data/annotations/, docs/report.pdf, docs/slides.pdf, docs/06_CHESS_...md            |
+---------------------------------------------------------------------------------------------+
```

---

## 4. Key Engineering Constraints & Conventions

### 4.1 Tensor Shapes & Orientation Normalization
- **Canonical Tensor Shape**: `(batch_size, 112, 8, 8)` with dtype `torch.float32`.
- **Channels**: 8 half-moves $\times$ 14 planes per half-move = 112 channels.
  - Planes 0–5: Friendly P, N, B, R, Q, K
  - Planes 6–11: Opponent p, n, b, r, q, k
  - Plane 12: Castling rights (4 bits replicated across $8 \times 8$)
  - Plane 13: En-passant square (1 on target square, else 0)
- **Attacker Perspective Normalization**:
  - If Black is side to move at ply $t$: flip array along the row axis (`np.flip(arr, axis=-2)`) and swap Friendly (0–5) $\leftrightarrow$ Opponent (6–11) planes.
  - *Crucial Rule*: The network MUST ALWAYS perceive the attacking side advancing toward Row 0 (Rank 8).
- **Padding Mask**: For opening moves ($t < 7$), missing history slots are zero-padded, accompanied by a 1D validity mask `(8,)`.

### 4.2 Bitboard Optimizations
- Always leverage `python-chess` 64-bit integer bitboards (`board.pieces_mask(piece, color)`).
- Compute attacks using bitwise OR over `board.attacks(sq)`.
- Use popcount (`bin(mask).count('1')`) for mobility counts.
- Never loop square-by-square over the 64 squares when a bitwise operation (`&`, `|`, `^`) can achieve it in a single CPU cycle.

### 4.3 Temporal State Machine Hysteresis
- Plan activations require **2 consecutive plies** of confidence $\ge 0.45$.
- Switching to a rival plan requires **2 consecutive plies** of contradictory evidence.
- Decay to `NO_PLAN` triggers after $\ge 3$ consecutive ambiguous or low-confidence plies.
- **Onset Backtracking**: Backtrack through the 8-ply window to record the earliest ply where candidate support emerged.

### 4.4 Explanations & Zero Hallucination
- Explanations MUST use **strictly controlled deterministic templates** filled with verified bitboard predicates.
- Clearly separate **Observed Board Facts** from **Model Hypotheses**.
- Never output speculative psychological intents or fabricate non-existent moves/squares.

### 4.5 Data Leakage Prevention
- Splitting MUST occur at the **game level** (`game_id`), never at the position window level.
- Windows from game $G$ belong exclusively to Train, Val, or Test.

---

## 5. Standard Developer & Execution Commands

### 5.1 Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1       # Windows PowerShell
source venv/bin/activate          # Linux/macOS

# Install dependencies (CPU or CUDA)
pip install -r requirements.txt
```

### 5.2 Verification & Testing
```bash
# Run complete test suite
pytest tests/ -v

# Run fast fact extractor tests
pytest tests/test_fact_extractor.py -v

# Run state tracker tests
pytest tests/test_state_tracker.py -v

# Run environment verification script
python scripts/verify_install.py
```

### 5.3 Training & Optimization
```bash
# Train primary 8-ply History-Aware ResNet
python -m src.models.train --config config/config.yaml --epochs 30 --batch_size 64

# Train 1-ply Current-Position Ablation Baseline
python -m src.models.train --config config/config.yaml --epochs 30 --ablation_single_board

# Run multi-seed history length ablation (1-ply vs 4-ply vs 8-ply)
python -m src.models.train_ablation --seeds 42 123 999
```

### 5.4 Evaluation & Benchmarking
```bash
# Run multi-model evaluation on 250-position ground truth
python -m src.models.evaluate --test_data data/annotations/test_ground_truth.csv

# Calculate Cohen's Kappa on independent annotations
python -m src.models.adjudicate

# Compute 1000-iteration game-cluster bootstrap confidence intervals
python -m src.models.bootstrap
```

### 5.5 Streamlit Web Application
```bash
# Launch interactive Streamlit application
streamlit run src/ui/app.py

# Launch on custom port
streamlit run src/ui/app.py --server.port 8501
```

### 5.6 GCR Packaging
```bash
# Verify and build single consolidated submission ZIP (< 80MB)
python scripts/package_submission.py
```

---

## 6. How Claude Code Must Execute Tasks

1. **Locate Target Subtask**: Read the exact specification file in:
   - `tasks/phase_1_core_ai/subtasks/SUBTASK_X_Y_Z_...md`
   - `tasks/phase_2_deployment_and_evaluation/subtasks/SUBTASK_X_Y_Z_...md`
2. **Review Parent Context**: Check corresponding task file in `tasks/phase_1_core_ai/` or `tasks/phase_2_deployment_and_evaluation/`.
3. **Inspect Inputs & Data Contracts**: Check `docs/04_WORK_DISTRIBUTION_AND_OWNERSHIP.md` and `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md`.
4. **Implement Code**: Write clean, modular, typed Python code with comprehensive docstrings.
5. **Write Unit Tests**: Add test cases to `tests/`.
6. **Execute & Verify**: Run `pytest tests/test_...` to guarantee 100% pass rate before marking subtask complete.
