# Task 1.5: Weak Labeling Engine & Dataset Splitting

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 4  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Generate weak training labels for 20,000–50,000 position windows using a retrospective lookahead execution heuristic (inspecting future plies $t+1 \dots t+8$). Filter noisy or ambiguous labels, balance class representations, and execute a strict **game-level split** (70% Train, 15% Validation, 15% Test) to completely avoid data leakage between adjacent position windows.

---

## 2. Subtask Breakdown

### Subtask 1.5.1: Retrospective Execution Weak-Labeler (`src/data/weak_labeler.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Crucial ML Design Principle**: The neural network only ever sees past history $[B_{t-7} \dots B_t]$. However, the *label generator* at training time inspects future plies $[B_{t+1} \dots B_{t+8}]$ to determine whether a strategic pattern was successfully carried out.
- **Verification Rule**:
  - If a player was preparing an outpost, did their Knight actually occupy and remain on that outpost in plies $t+1 \dots t+6$?
  - If advancing flank pawns, did pawns reach the enemy king's sector in $t+1 \dots t+8$?
  - If future moves diverge into unrelated tactical trades, reject the weak label.

### Subtask 1.5.2: Noise Filtering & Class Balancing (`src/data/weak_labeler.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Positions where multiple plans conflict with equal confidence are assigned `NO_PLAN` or filtered out.
  - Implement stratified subsampling to prevent `NO_PLAN` from overwhelming the dataset:
    - Target: ~15–20% per active plan class, ~20% `NO_PLAN`.
  - Cap maximum positions sampled per individual game to $\le 25$ to maintain diversity.

### Subtask 1.5.3: Game-Level Dataset Partitioning (`src/data/dataset_splitter.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Mandatory Leak-Free Requirement**:
  - In chess, consecutive plies of the same game are highly correlated. Splitting by position windows causes massive data leakage.
  - Therefore, the dataset MUST be partitioned **at the game level**:
    - 70% of games $\to$ Training set
    - 15% of games $\to$ Validation set
    - 15% of games $\to$ Test set
  - All windows from game $G$ belong exclusively to one split.

### Subtask 1.5.4: Dataset Manifest & Statistical Audit (`src/data/audit.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Generate `data/processed/dataset_summary.json` containing:
    - Total games and total position windows in train/val/test splits.
    - Class counts and proportions for all 6 classes across each split.
    - Average player rating (Elo) and ply index distributions.
    - Verification check: 0% game overlap between splits.

---

## 3. Data Structures & Signatures

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import chess

@dataclass
class LabeledPositionRecord:
    window_id: str
    game_id: str
    ply_index: int
    turn: str
    label_class: str          # "K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"
    label_id: int             # 0 to 5
    weak_confidence: float    # Heuristic confidence
    split: str                # "train", "val", "test"
    fen: str                  # Current position FEN
```

---

## 4. Claude Code Action Plan
1. Implement `src/data/weak_labeler.py` with future lookahead verification.
2. Implement `src/data/dataset_splitter.py` with deterministic game-level hashing.
3. Write script `scripts/generate_dataset.py` executing the end-to-end labeling and splitting pipeline.
4. Implement `tests/test_weak_labeler.py` and `tests/test_dataset_splitter.py` verifying split disjointness.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_dataset_splitter.py` -> Verify $GameID_{\text{train}} \cap GameID_{\text{val}} \cap GameID_{\text{test}} = \emptyset$.
- Confirm output files generated:
  - `data/processed/train_records.parquet` (or `.csv`)
  - `data/processed/val_records.parquet`
  - `data/processed/test_records.parquet`
  - `data/processed/dataset_summary.json`
