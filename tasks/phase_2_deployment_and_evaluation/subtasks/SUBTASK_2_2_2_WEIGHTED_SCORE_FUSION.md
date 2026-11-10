# Subtask 2.2.2: Validation-Weighted Neural-Rule Score Fusion Engine

**Subtask ID**: `2.2.2.WEIGHTED.SCORE.FUSION`  
**Parent Task**: `Task 2.2`  
**Phase**: Phase 2  
**Milestone**: Milestone 10  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Blend neural probabilities with symbolic rule scores using validation-tuned weight \alpha.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Calibrated neural probabilities and normalized rule scores.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `FusedPrediction` object with hybrid score vector.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Fusion Formula:
   $$S_{\text{hybrid}}(c) = \alpha \cdot P_{\text{neural}}(c) + (1 - \alpha) \cdot S_{\text{rule}}(c)$$
2. Weight Parameter: $\alpha = 0.70$ (default, configurable via YAML).
3. Script `scripts/tune_fusion_alpha.py` running grid search $\alpha \in [0.0, 1.0]$ on validation set.

---

## 5. Claude Code Action Plan
1. Implement `HybridFusionEngine.fuse` in `src/tracking/fusion.py`.
2. Implement `scripts/tune_fusion_alpha.py`.
3. Add unit tests verifying fusion arithmetic and bounds.

---

## 6. Unit Testing & Acceptance Criteria
- Hybrid score combines both modalities smoothly.
- Yields higher Macro-F1 than rule baseline or neural model alone.
