# Task 2.2: Neural-Rule Evidence Fusion & Uncertainty Handling

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 10  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Build the explicit neural-rule fusion and uncertainty abstention engine. Blend raw neural network softmax outputs with symbolic fact-based rule scores using a validation-tuned weight $\alpha$. Implement principled prediction abstention to flag ambiguous or uncommitted chess positions rather than forcing false positive predictions.

---

## 2. Subtask Breakdown

### Subtask 2.2.1: Calibration & Normalization (`src/tracking/fusion.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Neural logits $z$ calibrated via temperature scaling: $P_{\text{neural}}(c) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)}$ with $T \approx 1.2$.
  - Symbolic rule scores $S_{\text{rule}}(c) \in [0.0, 1.0]$ normalized across classes.

### Subtask 2.2.2: Weighted Score Fusion (`src/tracking/fusion.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Compute blended score vector:
    $$S_{\text{hybrid}}(c) = \alpha \cdot P_{\text{neural}}(c) + (1 - \alpha) \cdot S_{\text{rule}}(c)$$
  - Optimal parameter $\alpha \in [0, 1]$ selected via grid search on the validation set (typically $\alpha \in [0.65, 0.75]$).
  - Neural predictions and hybrid fused predictions are tracked as distinct channels for comparative inspection.

### Subtask 2.2.3: Uncertainty Abstention & Ambiguity Filter (`src/tracking/fusion.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Let $c_{(1)}$ and $c_{(2)}$ be the classes with the highest and second-highest hybrid scores.
  - **Abstention Criteria**:
    1. Low Confidence: $S_{\text{hybrid}}(c_{(1)}) < \theta_{\text{confidence}}$ (default: $0.40$).
    2. High Ambiguity (Narrow Margin): $S_{\text{hybrid}}(c_{(1)}) - S_{\text{hybrid}}(c_{(2)}) < \theta_{\text{margin}}$ (default: $0.10$).
  - If either condition is met:
    - Mark output as `is_abstained = True`.
    - Set final predicted class to `NO_PLAN` (Class 5) with status *"Strategically Ambiguous / In Transition"*.

---

## 3. Function Signatures & Data Model

```python
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class FusedPrediction:
    primary_class: str
    class_id: int
    hybrid_score: float
    is_abstained: bool
    abstention_reason: str
    class_probabilities: Dict[str, float]
    neural_probabilities: Dict[str, float]
    rule_scores: Dict[str, float]

class HybridFusionEngine:
    def __init__(self, alpha: float = 0.70, confidence_threshold: float = 0.40, margin_threshold: float = 0.10):
        self.alpha = alpha
        self.conf_thresh = confidence_threshold
        self.margin_thresh = margin_threshold

    def fuse(
        self, 
        neural_probs: Dict[str, float], 
        rule_scores: Dict[str, float]
    ) -> FusedPrediction:
        """Blends probabilities, evaluates margin, and determines abstention."""
        pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/tracking/fusion.py` with `HybridFusionEngine`.
2. Implement validation search script `scripts/tune_fusion_alpha.py`.
3. Implement `tests/test_fusion.py` verifying correct fusion arithmetic and abstention triggering on ambiguous inputs.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_fusion.py` -> 100% pass.
- Abstention successfully catches synthetic uniform distribution $[1/6, \dots, 1/6]$ and assigns `NO_PLAN`.
