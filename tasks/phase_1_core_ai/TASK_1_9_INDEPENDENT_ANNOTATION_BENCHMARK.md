# Task 1.9: Independent Ground Truth Annotation Benchmark

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 8  
**Primary Owners**: Shared Responsibility (Muhammad Ali & Shahzad Ahmad)  

---

## 1. Objective & Scope
Construct an independent, rigorously reviewed gold-standard test set of 250 held-out chess position windows ($N = 250$). Both group members independently label the windows using a written rubric and history-only visibility, calculate Cohen's Kappa ($\kappa$) inter-rater reliability, adjudicate disagreements, and freeze the final benchmark to evaluate all models.

---

## 2. Subtask Breakdown

### Subtask 1.9.1: Formal Written Rubric Reference
- **Owners**: Muhammad Ali (`24I-3158`) & Shahzad Ahmad (`24I-3090`)
- **Specification**: Ensure both annotators follow the strict definitions codified in [docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md](../../docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md).
- Both reviewers sign off on rubric criteria before beginning independent annotations.

### Subtask 1.9.2: Test Sample Extractor Tool (`src/data/sample_test_set.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Sample 250 distinct position windows exclusively from the 15% held-out test game split.
  - Output files:
    - `data/annotations/raw_annotations_member1.csv` (for Shahzad Ahmad)
    - `data/annotations/raw_annotations_member2.csv` (for Muhammad Ali)
  - Columns: `window_id`, `game_id`, `ply_index`, `turn`, `fen_current`, `history_moves_san`, `primary_label`, `confidence`, `secondary_label`, `notes`.
  - The script masks out all automatic weak labels and model predictions.

### Subtask 1.9.3: Double-Blind Independent Annotation Execution
- **Owners**: Shahzad Ahmad (`24I-3090`) & Muhammad Ali (`24I-3158`)
- **Strict Protocol**:
  - Neither member inspects the other member's file during annotation.
  - Annotators inspect only the past 8 half-moves leading to ply $t$.
  - Annotators must disclose their chess rating/experience level in the final report.

### Subtask 1.9.4: Cohen's Kappa & Adjudication Pipeline (`src/models/adjudicate.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Load `raw_annotations_member1.csv` and `raw_annotations_member2.csv`.
  - Compute raw observed agreement $p_o$ and Cohen's Kappa score $\kappa$.
  - Generate discrepancy review report `data/annotations/disagreements.csv`.
  - Joint adjudication session to resolve conflicts and output the frozen ground truth: `data/annotations/test_ground_truth.csv`.

---

## 3. Annotation Schema & Data Structures

```csv
window_id,game_id,ply_index,turn,primary_label,confidence,secondary_label,rationale
W0001,game_4812,34,white,OUTPOST,1.0,NO_PLAN,Knight established on d5 outpost defended by e4 pawn
W0002,game_9124,28,black,K_STORM,0.8,NO_PLAN,Black pushing g5 and h5 against castled white king
```

---

## 4. Claude Code Action Plan
1. Implement `src/data/sample_test_set.py` to extract 250 unlabelled test rows.
2. Implement `src/models/adjudicate.py` to calculate Cohen's Kappa using `scikit-learn.metrics.cohen_kappa_score` and export conflict lists.
3. Provide a simple CLI review script `scripts/adjudicate_cli.py` to step through disputed positions during the consensus meeting.
4. Implement `tests/test_adjudication.py` verifying that identical annotation files yield $\kappa = 1.00$.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_adjudication.py` -> 100% pass.
- Verified generation of frozen `data/annotations/test_ground_truth.csv` with exactly 250 labeled records.
- Report observed agreement and $\kappa$ in the final project documentation.
