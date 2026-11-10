# Subtask 1.9.2: 250 Held-Out Position Window Extraction Tool

**Subtask ID**: `1.9.2.TEST.WINDOW.SAMPLER`  
**Parent Task**: `Task 1.9`  
**Phase**: Phase 1  
**Milestone**: Milestone 8  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Extract exactly 250 stratified held-out position windows from the test split games for human annotation.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `test_records.parquet` from 15% test split.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `data/annotations/raw_annotations_member1.csv` and `raw_annotations_member2.csv`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/data/sample_test_set.py`:
   - Sample 250 distinct position windows from test games.
   - Stratify across game phases: Opening (plies 15-25), Middlegame (26-55), Late (56+).
   - Export blank annotation CSV templates for Shahzad Ahmad and Muhammad Ali.
   - Mask out all weak labels and model predictions to prevent bias.

---

## 5. Claude Code Action Plan
1. Implement `src/data/sample_test_set.py`.
2. Run sampling script.
3. Verify that sampled games have 0% overlap with training/validation splits.

---

## 6. Unit Testing & Acceptance Criteria
- Exactly 250 unique test position rows generated.
- No labels or model hints present in annotation sheets.
