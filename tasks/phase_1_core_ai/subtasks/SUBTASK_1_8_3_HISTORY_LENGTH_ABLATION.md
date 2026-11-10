# Subtask 1.8.3: History Length Ablation Engine (1 vs 4 vs 8 Plies)

**Subtask ID**: `1.8.3.HISTORY.LENGTH.ABLATION`  
**Parent Task**: `Task 1.8`  
**Phase**: Phase 1  
**Milestone**: Milestone 7  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Execute controlled ablation comparing 1-ply, 4-ply, and 8-ply history models under matched hyperparameter settings.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Identical train/val splits and random seeds.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/models/train_ablation.py` and `experiments/ablation_results.json`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/models/train_ablation.py`:
   - Trains 3 variants:
     1. 1-ply: Current board only ($14 \times 8 \times 8$)
     2. 4-ply: Recent 4 half-moves ($56 \times 8 \times 8$)
     3. 8-ply: Full MindCheck model ($112 \times 8 \times 8$)
   - Uses matched batch sizes, optimizer, and seeds.
   - Evaluates on validation set and logs Macro-F1 deltas.

---

## 5. Claude Code Action Plan
1. Implement `src/models/train_ablation.py`.
2. Run ablation benchmark.
3. Output comparative summary table and bar chart in `experiments/plots/ablation_history.png`.

---

## 6. Unit Testing & Acceptance Criteria
- Successfully trains all 3 ablation models.
- Produces empirical F1 comparison answering the primary research question.
