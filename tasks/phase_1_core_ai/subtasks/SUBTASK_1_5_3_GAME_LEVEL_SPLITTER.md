# Subtask 1.5.3: Game-Level Train/Validation/Test Splitter (70/15/15)

**Subtask ID**: `1.5.3.GAME.LEVEL.SPLITTER`  
**Parent Task**: `Task 1.5`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Partition dataset strictly at the game level (70% train, 15% val, 15% test) to prevent window leakage.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: List of labeled position records.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `train_records.parquet`, `val_records.parquet`, `test_records.parquet`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `GameLevelSplitter`:
   - Extract unique `game_id` set.
   - Deterministic hash partitioning: `hash(game_id + seed) % 100`:
     - 0..69: Train (70%)
     - 70..84: Validation (15%)
     - 85..99: Test (15%)
   - Group all position windows by their parent `game_id`.
2. Assertion: $\text{Games}_{\text{train}} \cap \text{Games}_{\text{val}} \cap \text{Games}_{\text{test}} = \emptyset$.

---

## 5. Claude Code Action Plan
1. Implement `src/data/dataset_splitter.py`.
2. Write `tests/test_dataset_splitter.py` verifying zero game overlap across splits.
3. Save split record tables to `data/processed/`.

---

## 6. Unit Testing & Acceptance Criteria
- Game overlap across splits is strictly 0%.
- Class proportions are preserved across train, val, and test splits.
