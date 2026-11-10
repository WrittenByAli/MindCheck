# Subtask 2.7.1: Multi-Model Benchmark Runner on Test Ground Truth

**Subtask ID**: `2.7.1.MULTI.MODEL.BENCHMARK.RUNNER`  
**Parent Task**: `Task 2.7`  
**Phase**: Phase 2  
**Milestone**: Milestone 14  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Execute automated benchmarking of 4 systems on the frozen 250-position test set.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `data/annotations/test_ground_truth.csv` and trained checkpoints.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `experiments/evaluation_results.csv` containing side-by-side predictions.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/models/evaluate.py`:
   - Evaluates:
     1. MindCheck History-Aware ResNet (8 plies)
     2. Current-Position ResNet Baseline (1 ply)
     3. Rule-Only Symbolic Detector
     4. Neural-Rule Hybrid Model
   - Logs prediction class, probabilities, and latency per sample.

---

## 5. Claude Code Action Plan
1. Implement `src/models/evaluate.py`.
2. Execute full benchmark on 250 test positions.
3. Save raw prediction outputs to CSV.

---

## 6. Unit Testing & Acceptance Criteria
- All 4 systems evaluated on identical test examples.
- Output table saved with complete prediction records.
