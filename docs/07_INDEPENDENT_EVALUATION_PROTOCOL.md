# MindCheck: Independent Evaluation Protocol & Benchmark Suite

This document defines the scientific evaluation methodology, statistical rigor, and ablation benchmarks for **MindCheck** to guarantee total compliance with Course Learning Outcome 3 (CLO 3).

---

## 1. Held-Out Ground Truth Test Set Creation

### 1.1 Sampling Strategy
- **Source**: Drawn strictly from the 15% held-out test split of games (games never seen during training or validation).
- **Target Size**: Exactly 250 position windows ($N = 250$).
- **Stratification**: Balanced across game phases (Opening plies 15–25, Middlegame plies 26–55, Late middlegame 56+) and across player ratings (2200 to 2600+ Elo).

### 1.2 Independent Double-Blind Annotation Process
- **Reviewer 1**: Shahzad Ahmad (`24I-3090`)
- **Reviewer 2**: Muhammad Ali (`24I-3158`)
- **Isolation Constraint**: Neither reviewer communicates during labeling. Annotators view only the 8-ply history leading to position $t$. No future moves and no automated predictions are shown.
- **Data Capture File**: `data/annotations/raw_annotations_member1.csv` and `raw_annotations_member2.csv`.

---

## 2. Inter-Rater Reliability: Cohen's Kappa ($\kappa$)

Inter-rater agreement between the two human annotators is quantified using **Cohen's Kappa**:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where:
- $p_o = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_{1, i} = y_{2, i})$ is the observed relative agreement.
- $p_e = \sum_{k=1}^K p_{1, k} \cdot p_{2, k}$ is the hypothetical probability of chance agreement, where $p_{1, k}$ and $p_{2, k}$ are the marginal proportions of class $k$ assigned by Reviewer 1 and Reviewer 2.

### Interpretation Standard:
- $\kappa > 0.75$: Excellent agreement (robust strategic consensus).
- $0.60 \le \kappa \le 0.75$: Substantial agreement (acceptable for abstract strategic tasks).
- $\kappa < 0.60$: Requires refinement of operational rubric before adjudication.

### Adjudication Step:
For any position where $y_{1, i} \ne y_{2, i}$:
1. Both reviewers meet to review the 8-ply move history and bitboard predicates.
2. The final adjudicated consensus label is recorded into `data/annotations/test_ground_truth.csv`.
3. If genuine ambiguity persists, the position is formally adjudicated as `NO_PLAN` (Class 5).

---

## 3. Comparative Benchmark Configurations

The evaluation suite evaluates four distinct systems on the exact same adjudicated test set:

| System ID | System Name | Input Features | Inference Mechanism |
| :--- | :--- | :--- | :--- |
| **M1 (Proposed)** | **MindCheck History CNN** | 8 consecutive half-moves ($112 \times 8 \times 8$) | PyTorch Deep Residual CNN (4 ResNet blocks) |
| **M2 (Ablation)** | **Current-Position CNN** | 1 current board state ($14 \times 8 \times 8$) | Matched PyTorch ResNet (same capacity) |
| **M3 (Baseline)** | **Symbolic Rule Detector** | Current board + 8-ply moves | Deterministic heuristic bitboard rule engine |
| **M4 (Hybrid)** | **Neural-Rule Fusion** | 8 plies + Bitboard Facts | Score fusion ($\alpha \cdot P_{\text{neural}} + (1-\alpha) \cdot S_{\text{rule}}$) |

---

## 4. Quantitative Evaluation Metrics

For each system $M \in \{M1, M2, M3, M4\}$, the following metrics are computed:

1. **Macro-Averaged F1-Score**:
   $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
   *(Ensures equal weighting across rare and frequent strategic plans).*
2. **Per-Class Precision, Recall, and F1**:
   Reported for all 6 classes (`K_STORM`, `Q_EXP`, `OUTPOST`, `ROOK_FILE`, `CHAIN_UNDERMINE`, `NO_PLAN`).
3. **Multi-Class Confusion Matrix**:
   Visualized as a normalized heatmap displaying misclassification patterns.
4. **Game-Cluster Bootstrap Confidence Intervals**:
   - Because multiple test windows originate from the same chess game, observations are not strictly IID.
   - We perform **1,000 game-level cluster bootstrap iterations** (resampling entire games with replacement) to compute 95% Confidence Intervals for Macro-F1.

---

## 5. Abstention Curve & Confidence Calibration

MindCheck provides an abstention mechanism where predictions with low top-1 probability or low top-2 margin can abstain from committing to a plan:
- **Coverage**: Proportion of test positions where the model produces a committed plan:
  $$\text{Coverage}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\max_k P(y=k|x_i) \ge \theta)$$
- **Selective Accuracy / F1**: Evaluated exclusively on the accepted (non-abstained) predictions.
- **Expected Outcome**: As threshold $\theta$ increases, accuracy on accepted predictions must monotonically increase, proving proper confidence calibration.

---

## 6. Qualitative Error & Failure Analysis Protocol

The final evaluation report must document at least **10 specific game positions** with deep qualitative analysis:
- **5 False Positives**: Situations where the model predicted a plan that was nonexistent (e.g. predicting a Kingside Storm when pawn pushes were purely defensive prophylactic moves).
- **5 False Negatives**: Situations where an active plan was underway but the model predicted `NO_PLAN` or a rival category.
- **Pedagogical Takeaway**: Why the mistake occurred, what board geometric feature confused the convolutional kernels, and how the model can be refined.
