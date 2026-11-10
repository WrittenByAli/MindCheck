# Subtask 2.2.1: Score Calibration & Symbolic Rule Normalization

**Subtask ID**: `2.2.1.SCORE.CALIBRATION.NORMALIZATION`  
**Parent Task**: `Task 2.2`  
**Phase**: Phase 2  
**Milestone**: Milestone 10  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Calibrate neural logits with temperature scaling and normalize symbolic rule scores.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Raw model logits and symbolic rule scores.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Calibrated probabilities and normalized rule vectors.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Temperature Scaling:
   - $P_{\text{neural}}(c) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)}$ with $T \approx 1.2$ tuned on validation data.
2. Rule Normalization:
   - Normalize rule scores $S_{\text{rule}}(c) \in [0.0, 1.0]$ across classes: $\hat{S}_{\text{rule}}(c) = \frac{S_{\text{rule}}(c)}{\sum_k S_{\text{rule}}(k) + \epsilon}$.

---

## 5. Claude Code Action Plan
1. Implement calibration functions in `src/tracking/fusion.py`.
2. Write unit tests in `tests/test_fusion.py` asserting probability sums and positive bounds.
3. Test temperature scaling effect on entropy.

---

## 6. Unit Testing & Acceptance Criteria
- Output vectors strictly sum to 1.0.
- Overconfident extreme probabilities are softened appropriately.
