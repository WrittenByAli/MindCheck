# Subtask 2.2.3: Prediction Abstention & Ambiguity Threshold Filter

**Subtask ID**: `2.2.3.UNCERTAINTY.ABSTENTION.GATING`  
**Parent Task**: `Task 2.2`  
**Phase**: Phase 2  
**Milestone**: Milestone 10  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Implement confidence and margin abstention gates to flag ambiguous or low-confidence positions.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Hybrid score vector $S_{\text{hybrid}}$.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `is_abstained: bool`, `abstention_reason: str`, and final class assignment.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Confidence Gate: $\max_c S_{\text{hybrid}}(c) < \theta_{\text{conf}}$ (default: 0.40).
2. Margin Gate: $S_{\text{hybrid}}(c_{(1)}) - S_{\text{hybrid}}(c_{(2)}) < \theta_{\text{margin}}$ (default: 0.10).
3. If either gate triggers:
   - `is_abstained = True`
   - Assign class `NO_PLAN` (Class 5) with status 'Strategically Ambiguous'.

---

## 5. Claude Code Action Plan
1. Implement abstention gates in `src/tracking/fusion.py`.
2. Write unit tests in `tests/test_fusion.py` testing uniform and close-margin distributions.
3. Compute selective accuracy on accepted predictions.

---

## 6. Unit Testing & Acceptance Criteria
- Successfully flags ambiguous and uniform probability profiles.
- Accuracy on non-abstained predictions is significantly higher than full test set.
