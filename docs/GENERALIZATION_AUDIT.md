# MindCheck — Model Generalization, Overfitting & Underfitting Audit

**Audit date:** 2026-11-19
**Auditor:** Claude Opus 4.6 (automated, read-only)
**Checkpoint:** `models/checkpoints/best_resnet.pt` (epoch 5, best_f1 = 0.5319828)
**Branch:** `results/phase1-training-run`
**Scope:** Read-only. No source code, config, dataset, label, or checkpoint was modified.

---

## Tag legend

Every conclusion is tagged with its epistemic status:

| Tag | Meaning |
|-----|---------|
| **VERIFIED** | Independently reproduced or confirmed by running code and comparing artifacts |
| **REPORTED** | Taken from existing logs/outputs that were not independently re-executed |
| **UNVERIFIABLE** | Cannot be confirmed without data or infrastructure not available in this audit |
| **INCONSISTENT** | Contradictory evidence found across sources |

---

## Phase 1 — Repository & Artifact Inspection

### 1.1 Checkpoint integrity

| Property | Value | Source | Tag |
|----------|-------|--------|-----|
| File exists | Yes | `models/checkpoints/best_resnet.pt` | **VERIFIED** |
| Checkpoint epoch | 5 | `ckpt["epoch"]` | **VERIFIED** |
| Checkpoint best_f1 | 0.5319828 | `ckpt["best_f1"]` | **VERIFIED** |
| Trainable parameters | 287,110 | `count_parameters(model)` | **VERIFIED** |
| Optimizer state saved | Yes | `"optimizer_state_dict" in ckpt` | **VERIFIED** |

The checkpoint was loaded successfully and produces deterministic predictions on the test set.

### 1.2 Architecture verification

| Property | Config (`config.yaml`) | Code (`residual_cnn.py`) | Checkpoint | Docs | Tag |
|----------|----------------------|--------------------------|------------|------|-----|
| Residual blocks | 3 | 3 (`num_blocks` loop) | 3 (287,110 params) | **4** (multiple docs) | **INCONSISTENT** |
| Channels | 64 | 64 | matches | 64 | **VERIFIED** |
| Input channels | 112 | 112 | matches | 112 | **VERIFIED** |
| Dropout | 0.20 | 0.20 | matches | 0.20 | **VERIFIED** |
| Num classes | 6 | 6 | matches | 6 | **VERIFIED** |

**INCONSISTENT finding:** `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md`, `docs/PROJECT_REFERENCE_FULL.md`, and the `mindcheck-ml-pipeline` skill all state **4 residual blocks**, but the actual code, config, and checkpoint all use **3 blocks** (287,110 parameters). With 4 blocks the parameter count would be ~344,838. The shipped model has 3 blocks.

### 1.3 Training configuration

| Parameter | Value | Source | Tag |
|-----------|-------|--------|-----|
| Optimizer | AdamW | `train.py` | **VERIFIED** |
| Learning rate | 0.001 | `config.yaml` + training history (epoch 1 lr=0.001) | **VERIFIED** |
| Weight decay | 1e-4 | `config.yaml` | **VERIFIED** |
| Scheduler | CosineAnnealingLR | `train.py` | **VERIFIED** |
| Label smoothing | 0.05 | `config.yaml` | **VERIFIED** |
| Gradient clipping | 1.0 | `config.yaml` | **VERIFIED** |
| Early stopping patience | 5 | `config.yaml` | **VERIFIED** |
| Batch size | 64 | `config.yaml` | **VERIFIED** |
| Max epochs | 30 | `config.yaml` | **VERIFIED** |
| Actual epochs completed | 10 | `training_history.json` (10 entries) | **REPORTED** |
| Random seed | 42 | `config.yaml` | **VERIFIED** |
| Data augmentation | None | No augmentation code in `train.py` or `chess_dataset.py` | **VERIFIED** |
| Input normalization | None (raw 0/1 planes) | `tensor_encoder.py` | **VERIFIED** |

**Note on early stopping:** Best val macro-F1 occurred at epoch 5 (0.5320). With patience 5, training should have stopped at epoch 10 — which matches the 10-entry training history. **VERIFIED** (consistent).

### 1.4 Data pipeline artifacts

| Artifact | Exists | Tag |
|----------|--------|-----|
| `data/raw/lichess_elite_subset.pgn` | Yes | **VERIFIED** |
| `models/checkpoints/best_resnet.pt` | Yes | **VERIFIED** |
| `experiments/training_history.json` | Yes (10 epochs) | **VERIFIED** |
| `experiments/ablation_results.json` | Yes (9 runs: 3 variants × 3 seeds) | **VERIFIED** |
| Phase B comparison output file | **No** (stdout-only, not persisted) | **VERIFIED** |

---

## Phase 2 — Overfitting Diagnosis

### 2.1 Training curve analysis

Data source: `experiments/training_history.json` (10 epochs). **VERIFIED** by direct file read.

| Epoch | Train Loss | Val Loss | Loss Gap | Train Acc | Val Acc | Acc Gap | Val Macro-F1 |
|------:|----------:|--------:|---------:|----------:|--------:|--------:|-------------:|
| 1 | 1.5146 | 1.5959 | +0.081 | 57.9% | 54.2% | +3.7pp | 0.4680 |
| 2 | 1.2526 | 1.6914 | +0.439 | 70.9% | 50.9% | +20.0pp | 0.4480 |
| 3 | 1.1212 | 1.5082 | +0.387 | 77.1% | 62.6% | +14.5pp | 0.5288 |
| 4 | 0.9980 | 1.6554 | +0.657 | 82.8% | 59.6% | +23.2pp | 0.4982 |
| **5** | **0.9087** | **1.5432** | **+0.634** | **86.7%** | **64.3%** | **+22.4pp** | **0.5320** |
| 6 | 0.8267 | 1.6254 | +0.799 | 90.3% | 63.2% | +27.1pp | 0.5243 |
| 7 | 0.7724 | 1.6596 | +0.887 | 92.7% | 62.5% | +30.2pp | 0.5156 |
| 8 | 0.7248 | 1.7316 | +1.007 | 94.9% | 60.9% | +34.0pp | 0.5057 |
| 9 | 0.6935 | 1.7338 | +1.040 | 96.0% | 62.8% | +33.2pp | 0.5155 |
| 10 | 0.6652 | 1.8522 | +1.187 | 97.1% | 61.9% | +35.2pp | 0.4937 |

### 2.2 Overfitting verdict

**VERDICT: Severe overfitting. VERIFIED.**

Evidence:
1. **Train accuracy climbs monotonically** from 57.9% to 97.1% — the model memorizes nearly all training examples.
2. **Val loss diverges after epoch 1** — it never drops below 1.5082 (epoch 3) and climbs to 1.8522 by epoch 10. The train-val loss gap grows from +0.08 to +1.19.
3. **Val macro-F1 peaks at epoch 5** (0.5320), then declines by 7.2% relative to 0.4937 at epoch 10.
4. **Train-val accuracy gap grows from +3.7pp to +35.2pp** across the 10-epoch run.
5. **Val accuracy is non-monotonic** — it oscillates between 50.9% and 64.3%, while train accuracy rises smoothly. This high variance suggests the model is in an overfitting regime where gradient steps improve training fit but randomly help or hurt generalization.

**Underfitting check:** The model is NOT underfitting — train accuracy of 97.1% demonstrates ample capacity to fit the training distribution. The problem is generalization, not representation power.

### 2.3 Regularization assessment

| Technique | Present | Strength | Assessment |
|-----------|---------|----------|------------|
| Dropout | Yes | 0.20 | Applied once before final linear layer. Single dropout layer after global average pool — not applied within residual blocks. **Insufficient.** |
| Weight decay | Yes | 1e-4 | Standard but mild for a model this prone to overfitting. |
| Label smoothing | Yes | 0.05 | Helps calibration but does not address the memorization pattern. |
| Data augmentation | **No** | — | No augmentation of any kind. Board symmetry (vertical flip for color-swapped positions) is handled at encoding time, not as runtime augmentation. |
| Batch normalization | Yes | Per conv layer | Provides implicit regularization but insufficient alone. |
| Early stopping | Yes | Patience 5 | Correctly triggered at epoch 10. Saves the epoch-5 checkpoint. |

**Key gap:** No data augmentation. Chess boards admit limited augmentations (not arbitrary rotations), but the complete absence of any stochastic augmentation (e.g., dropout within residual blocks, random ply masking, Mixup on encoded tensors) leaves the model dependent on early stopping as its sole defense against memorization.

### 2.4 Overfitting timeline

```
Epoch 1  ■         Train acc: 57.9% | Val acc: 54.2% | Gap: +3.7pp
         → Model is learning; train and val are close
Epoch 3  ■■■       Train acc: 77.1% | Val acc: 62.6% | Gap: +14.5pp
         → Overfitting begins; val loss starts non-monotonic
Epoch 5  ■■■■■     Train acc: 86.7% | Val acc: 64.3% | Gap: +22.4pp
         → BEST CHECKPOINT saved (val macro-F1 = 0.532)
Epoch 7  ■■■■■■■   Train acc: 92.7% | Val acc: 62.5% | Gap: +30.2pp
         → Severe overfitting; val performance declining
Epoch 10 ■■■■■■■■■■ Train acc: 97.1% | Val acc: 61.9% | Gap: +35.2pp
         → Training terminated by early stopping; val still declining
```

---

## Phase 3 — Full Test-Set Evaluation

All metrics in this section were computed by running a read-only evaluation script that loaded the saved checkpoint and re-derived the seed=42 split. **VERIFIED** — results match the independent Phase B comparison output.

### 3.1 Dataset sizes

| Split | Windows | Unique games | Tag |
|-------|--------:|-------------:|-----|
| Train (post-balance) | 34,354 | 1,922 | **VERIFIED** |
| Val | 9,452 | 389 | **VERIFIED** |
| Test | 10,461 | 427 | **VERIFIED** |
| **Total** | **54,267** | **2,738** | |

Split disjointness: `GameLevelSplitter.verify_disjoint()` returns `True`. **VERIFIED.**

### 3.2 Neural model — aggregate test metrics

| Metric | Value | Tag |
|--------|------:|-----|
| Accuracy | 0.6438 | **VERIFIED** |
| Macro-F1 | 0.5301 | **VERIFIED** |
| Weighted-F1 | 0.6637 | **VERIFIED** |
| Balanced accuracy | 0.6067 | **VERIFIED** |
| Macro precision | 0.5149 | **VERIFIED** |
| Macro recall | 0.6067 | **VERIFIED** |

**Accuracy vs balanced accuracy gap:** 0.6438 − 0.6067 = +0.037. The model benefits from the class prior (predicting majority classes correctly inflates raw accuracy). Balanced accuracy at 0.6067 means the average per-class recall is only ~61%.

### 3.3 Neural model — per-class test metrics

| Class | Precision | Recall | F1 | Support | % of test | Tag |
|-------|----------:|-------:|---:|--------:|----------:|-----|
| K_STORM | 0.4417 | 0.6842 | 0.5368 | 703 | 6.7% | **VERIFIED** |
| Q_EXP | 0.5552 | 0.7198 | 0.6269 | 1,299 | 12.4% | **VERIFIED** |
| OUTPOST | 0.7410 | 0.5974 | 0.6615 | 2,710 | 25.9% | **VERIFIED** |
| ROOK_FILE | 0.4823 | 0.5884 | 0.5301 | 882 | 8.4% | **VERIFIED** |
| CHAIN_UNDERMINE | 0.0559 | 0.3924 | 0.0978 | 79 | 0.8% | **VERIFIED** |
| NO_PLAN | 0.8135 | 0.6579 | 0.7275 | 4,788 | 45.8% | **VERIFIED** |

### 3.4 Neural model — confusion matrix

```
Predicted →       K_STORM  Q_EXP  OUTPOST  ROOK_F  CHAIN_U  NO_PLAN
K_STORM              481     38      65      42      31       46
Q_EXP                 73    935      65      38      72      116
OUTPOST              185    237    1619     163      85      421
ROOK_FILE             75     63      92     519      12      121
CHAIN_UNDERMINE        8     14       4       4      31       18
NO_PLAN              267    397     340     310     324     3150
```

**Key confusion patterns:**
- **NO_PLAN → CHAIN_UNDERMINE (324 false positives):** The model predicts CHAIN_UNDERMINE 555 times total, but only 31 are true positives. 324 of its CHAIN_UNDERMINE predictions are actually NO_PLAN windows. This yields precision of only 5.6%.
- **OUTPOST → NO_PLAN (421):** 15.5% of OUTPOST windows are predicted as NO_PLAN.
- **NO_PLAN → K_STORM (267), Q_EXP (397), ROOK_FILE (310):** NO_PLAN windows are frequently mis-predicted as active plans, suggesting the model has not learned a reliable "no plan" boundary.

### 3.5 Rule-only baseline — test metrics

| Metric | Rule-Only | Neural | Δ (Rule − Neural) | Tag |
|--------|----------:|-------:|-------------------:|-----|
| Accuracy | 0.7813 | 0.6438 | **+0.1375** | **VERIFIED** |
| Macro-F1 | 0.6570 | 0.5301 | **+0.1269** | **VERIFIED** |
| Weighted-F1 | 0.7852 | 0.6637 | **+0.1215** | **VERIFIED** |
| Balanced accuracy | 0.6660 | 0.6067 | **+0.0593** | **VERIFIED** |

### 3.6 Rule-only baseline — per-class comparison

| Class | Rule F1 | Neural F1 | Δ (Rule − Neural) | Winner |
|-------|--------:|----------:|-------------------:|--------|
| K_STORM | 0.6322 | 0.5368 | +0.0954 | Rule |
| Q_EXP | 0.7151 | 0.6269 | +0.0882 | Rule |
| OUTPOST | 0.8441 | 0.6615 | +0.1826 | Rule |
| ROOK_FILE | 0.7521 | 0.5301 | +0.2220 | Rule |
| CHAIN_UNDERMINE | 0.1892 | 0.0978 | +0.0914 | Rule |
| NO_PLAN | 0.8093 | 0.7275 | +0.0818 | Rule |

**The rule-only baseline wins on every single class.** The neural model provides strictly negative value-add over the rule engine it was trained to approximate. **VERIFIED.**

### 3.7 Rule-only baseline — confusion matrix

```
Predicted →       K_STORM  Q_EXP  OUTPOST  ROOK_F  CHAIN_U  NO_PLAN
K_STORM              404     19       9       7      14      250
Q_EXP                  8    842      25      18      29      377
OUTPOST               30     40    2187      28      27      398
ROOK_FILE             26     25      20     584       9      218
CHAIN_UNDERMINE        4      3       3       2      35       32
NO_PLAN              103    127     228      32     177     4121
```

The rule engine's dominant failure mode is predicting NO_PLAN for active-plan windows (250+377+398+218+32 = 1,275 misses), reflecting conservative thresholds. By contrast, the neural model's failures are more diffuse — it confuses plans with each other and with NO_PLAN in all directions.

### 3.8 Prediction confidence analysis

| Statistic | Value | Tag |
|-----------|------:|-----|
| Mean max-probability | 0.6496 | **VERIFIED** |
| Median max-probability | 0.6827 | **VERIFIED** |
| Std max-probability | 0.1624 | **VERIFIED** |

The model's confidence is moderate — mean max-probability of 0.65 means the winning class typically holds only 65% of the probability mass. The relatively narrow spread (std 0.16) suggests the model rarely expresses strong confidence in any direction.

#### Per-class confidence breakdown

| Class | Mean conf | Correct conf | Wrong conf | Conf gap | Tag |
|-------|----------:|-------------:|-----------:|---------:|-----|
| K_STORM | 0.7279 | 0.8040 | 0.5631 | +0.241 | **VERIFIED** |
| Q_EXP | 0.6911 | 0.7371 | 0.5729 | +0.164 | **VERIFIED** |
| OUTPOST | 0.6402 | 0.6727 | 0.5919 | +0.081 | **VERIFIED** |
| ROOK_FILE | 0.6652 | 0.7338 | 0.5671 | +0.167 | **VERIFIED** |
| CHAIN_UNDERMINE | 0.6109 | 0.6500 | 0.5856 | +0.064 | **VERIFIED** |
| NO_PLAN | 0.6299 | 0.6536 | 0.5842 | +0.069 | **VERIFIED** |

The confidence gap (correct − wrong) is a proxy for calibration quality. K_STORM and ROOK_FILE show the largest gaps (+0.24, +0.17), meaning the model is somewhat more confident when it is right. CHAIN_UNDERMINE and NO_PLAN have the smallest gaps (+0.06, +0.07) — the model is nearly as confident when wrong as when right, indicating poor calibration on these classes.

#### Confidence histogram

| Bin | Count | % |
|-----|------:|--:|
| [0.0, 0.3) | 123 | 1.2% |
| [0.3, 0.4) | 819 | 7.8% |
| [0.4, 0.5) | 1,327 | 12.7% |
| [0.5, 0.6) | 1,468 | 14.0% |
| [0.6, 0.7) | 1,906 | 18.2% |
| [0.7, 0.8) | 2,928 | 28.0% |
| [0.8, 0.9) | 1,531 | 14.6% |
| [0.9, 1.0] | 359 | 3.4% |

The distribution is unimodal, centered around 0.7. Only 3.4% of predictions reach ≥0.90 confidence. 21.7% of predictions have max-probability below 0.50, meaning the model is uncertain about its own top choice for one in five windows.

### 3.9 Prediction distribution analysis

| Class | True count | Predicted count | Ratio (pred/true) | Tag |
|-------|----------:|--------------:|-------------------:|-----|
| K_STORM | 703 | 1,089 | 1.55× | **VERIFIED** |
| Q_EXP | 1,299 | 1,684 | 1.30× | **VERIFIED** |
| OUTPOST | 2,710 | 2,185 | 0.81× | **VERIFIED** |
| ROOK_FILE | 882 | 1,076 | 1.22× | **VERIFIED** |
| CHAIN_UNDERMINE | 79 | 555 | **7.03×** | **VERIFIED** |
| NO_PLAN | 4,788 | 3,872 | 0.81× | **VERIFIED** |

The model **massively over-predicts CHAIN_UNDERMINE** (7× the true count) and **under-predicts OUTPOST and NO_PLAN** (~0.81× each). This aligns with the confusion matrix: CHAIN_UNDERMINE has only 5.6% precision because 524 of its 555 predictions are false positives. The model has not learned the class prior — it predicts minority classes too often and majority classes not often enough.

### 3.10 Validation set metrics and val-test gap

| Metric | Validation | Test | Δ (val − test) | Tag |
|--------|----------:|-----:|----------------:|-----|
| Accuracy | 0.6435 | 0.6438 | −0.0003 | **VERIFIED** |
| Macro-F1 | 0.5320 | 0.5301 | +0.0019 | **VERIFIED** |
| Weighted-F1 | 0.6629 | 0.6637 | −0.0008 | **VERIFIED** |
| Balanced accuracy | 0.5985 | 0.6067 | −0.0082 | **VERIFIED** |

#### Per-class val-test F1 gap

| Class | Val F1 | Test F1 | Δ (val − test) |
|-------|-------:|--------:|----------------:|
| K_STORM | 0.5330 | 0.5368 | −0.0038 |
| Q_EXP | 0.6092 | 0.6269 | −0.0176 |
| OUTPOST | 0.7036 | 0.6615 | +0.0421 |
| ROOK_FILE | 0.5307 | 0.5301 | +0.0005 |
| CHAIN_UNDERMINE | 0.1047 | 0.0978 | +0.0070 |
| NO_PLAN | 0.7106 | 0.7275 | −0.0169 |

**Val-test gap is minimal** (macro-F1 differs by only +0.0019). This is a positive signal: the model's performance is consistent across the two held-out splits, meaning the validation set was not over-tuned by early stopping. The per-class gaps are within ±0.04 F1 points, with no class showing a suspicious discrepancy. This consistency confirms the game-level split is functioning correctly.

### 3.11 Validation vs training-history cross-check

The training history records val_macro_f1 = 0.531983 at epoch 5. The checkpoint's metadata stores best_f1 = 0.5319828. These match to 7 significant figures, confirming the checkpoint corresponds to the epoch-5 state. **VERIFIED.**

---

## Phase 4 — Data Leakage Verification

### 4.1 Game-level split integrity

| Check | Method | Result | Tag |
|-------|--------|--------|-----|
| Split algorithm | SHA-256 hash of `{game_id}:{seed}`, bucket % 100 against 70/15/15 cutpoints | Deterministic, game-level | **VERIFIED** (code read) |
| Split disjointness | `GameLevelSplitter.verify_disjoint(records)` | `True` | **VERIFIED** (executed) |
| Train game count | 1,922 | | **VERIFIED** |
| Val game count | 389 | | **VERIFIED** |
| Test game count | 427 | | **VERIFIED** |
| Total games | 2,738 | | **VERIFIED** |

The split is structurally leak-free: each game is assigned to exactly one split based on a deterministic hash, and windows within a game inherit that assignment. No game-id appears in more than one split.

### 4.2 Runtime disjointness enforcement

**Bug documented:** `prepare_datasets()` in `train.py` calls `GameLevelSplitter.split()` and `GameLevelSplitter.verify_disjoint()` is available but **never called at training time**. The split correctness relies entirely on the algorithm's structural guarantee (hash-based assignment). While the algorithm is correct, the absence of a runtime assertion means a future code change that breaks the split would not be caught. **VERIFIED** (code inspection).

### 4.3 Label leakage analysis

| Potential leak vector | Status | Evidence | Tag |
|----------------------|--------|----------|-----|
| Future moves in input tensor | **No leak** | `tensor_encoder.py` encodes only boards $B_{t-7}...B_t$; labels come from $B_{t+1}...B_{t+8}$ | **VERIFIED** |
| Label confidence in input | **No leak** | `filter_and_balance` confidence threshold (≥0.55) is a filtering criterion, not an input feature | **VERIFIED** |
| Weak label scores in input | **No leak** | Label scores are computed by `RetrospectiveWeakLabeler` and stored in `WindowRecord.scores`, never passed to `TensorEncoder` | **VERIFIED** |
| Material/move metadata in input | **No leak** | Material filter (|material| ≤ 800) and per-game cap (≤25) filter records but don't add features | **VERIFIED** |

### 4.4 Temporal leakage in weak labels

The retrospective weak labeler scores ply $t$ using game state from plies $t+1$ through $t+8$. This is **by design** — the labels represent "what strategic plan was being executed as evidenced by future moves." The model's input tensor uses only plies $t-7$ through $t$. The temporal boundary is clean.

However, this design means the model is trained to predict **retrospective rule-engine judgments** (Task A), not **human strategic intent** (Task B). There is zero evidence in the repository that the model's labels correlate with human grandmaster strategic assessments. **VERIFIED** (no human annotations exist beyond the small Cohen's kappa benchmark).

### 4.5 Balancing leak check

`balance_classes()` is applied **only to the training split** after the game-level split is complete. Val and test retain their natural class distributions. This is correct — no information from val/test influences the balancing decisions. **VERIFIED** (code read of `prepare_datasets()` in `train.py`).

---

## Phase 5 — Class-Specific Diagnosis

### 5.1 Class distribution across pipeline stages

#### 5.1.1 Raw records (before filtering)

| Class | Count | % | Tag |
|-------|------:|--:|-----|
| K_STORM | 16,190 | 6.8% | **VERIFIED** |
| Q_EXP | 27,431 | 11.4% | **VERIFIED** |
| OUTPOST | 39,397 | 16.4% | **VERIFIED** |
| ROOK_FILE | 13,527 | 5.6% | **VERIFIED** |
| CHAIN_UNDERMINE | 1,925 | 0.8% | **VERIFIED** |
| NO_PLAN | 141,150 | 58.9% | **VERIFIED** |
| **Total** | **239,620** | | |

#### 5.1.2 After filtering (confidence ≥ 0.55, |material| ≤ 800, ≤25/game), before split

| Class | Count | % | Retained from raw | Tag |
|-------|------:|--:|------------------:|-----|
| K_STORM | 4,068 | 6.1% | 25.1% | **VERIFIED** |
| Q_EXP | 9,017 | 13.4% | 32.9% | **VERIFIED** |
| OUTPOST | 19,251 | 28.7% | 48.9% | **VERIFIED** |
| ROOK_FILE | 5,737 | 8.6% | 42.4% | **VERIFIED** |
| CHAIN_UNDERMINE | 587 | 0.9% | 30.5% | **VERIFIED** |
| NO_PLAN | 28,383 | 42.3% | 20.1% | **VERIFIED** |
| **Total** | **67,043** | | 28.0% | |

*Note: Class totals verified by summing train+val+test pre-balance counts (e.g., K_STORM: 2828+537+703=4068).*

#### 5.1.3 Train split: pre-balance vs post-balance

| Class | Pre-balance | Post-balance | Change | Tag |
|-------|------------:|-------------:|-------:|-----|
| K_STORM | 2,828 | 2,828 | 0 | **VERIFIED** |
| Q_EXP | 6,536 | 6,536 | 0 | **VERIFIED** |
| OUTPOST | 13,641 | 10,306 | −3,335 | **VERIFIED** |
| ROOK_FILE | 3,960 | 3,960 | 0 | **VERIFIED** |
| CHAIN_UNDERMINE | 418 | 418 | 0 | **VERIFIED** |
| NO_PLAN | 19,747 | 10,306 | −9,441 | **VERIFIED** |
| **Total** | **47,130** | **34,354** | −12,776 | |

The 30% cap applied to the total post-balance count: 34,354 × 0.30 = 10,306.2 → cap at 10,306. This removed 3,335 OUTPOST and 9,441 NO_PLAN training windows. Balancing does not upsample minority classes — CHAIN_UNDERMINE remains at just 418 training windows (1.2% of training data).

### 5.2 CHAIN_UNDERMINE deep dive

#### 5.2.1 The numbers

| Metric | Value |
|--------|------:|
| Training windows | 418 (1.2% of 34,354) |
| Val windows | 90 (1.0% of 9,452) |
| Test windows | 79 (0.8% of 10,461) |
| Neural test precision | 0.0559 |
| Neural test recall | 0.3924 |
| Neural test F1 | 0.0978 |
| Rule-only test precision | 0.1203 |
| Rule-only test recall | 0.4430 |
| Rule-only test F1 | 0.1892 |

#### 5.2.2 Root cause: conjunctive rule scoring

The rule engine's `_chain_undermining()` method (in `rule_engine.py`) returns either **0.0 or a moderate score** — there is no graded partial credit. The conjunctive condition requires:
1. At least one pawn chain exists (≥2 pawns in a connected support graph)
2. A pawn push within the 8-ply window targets the base or apex of an opponent's chain

All other rules (K_STORM, Q_EXP, OUTPOST, ROOK_FILE) use **graded scoring** with partial credit for partially-satisfied conditions. CHAIN_UNDERMINE is binary: either the full conjunctive condition is met or the score is zero.

This creates a **label sparsity problem**: out of 239,620 raw records, only 1,925 (0.8%) receive a CHAIN_UNDERMINE label, and after filtering (confidence ≥ 0.55), only 587 survive. The binary scoring means the confidence threshold eliminates fewer CHAIN_UNDERMINE records proportionally (30.5% retained, comparable to other classes), but the base rate is so low that the absolute count is crippling.

#### 5.2.3 Neural model failure on CHAIN_UNDERMINE

From the confusion matrix, the neural model predicts CHAIN_UNDERMINE 555 times on the test set:
- True CHAIN_UNDERMINE → predicted CHAIN_UNDERMINE: **31** (true positives)
- Other classes → predicted CHAIN_UNDERMINE: **524** (false positives)

The model has learned to predict CHAIN_UNDERMINE liberally (555 predictions for 79 true instances, a ratio of 7.0×), but with catastrophic precision (5.6%). The 324 false positives from NO_PLAN alone suggest the model is confusing the absence of a plan with chain undermining.

#### 5.2.4 Ablation consistency

CHAIN_UNDERMINE F1 across all 9 ablation runs:

| Seed | 1-ply | 4-ply | 8-ply |
|------|------:|------:|------:|
| 42 | 0.1080 | 0.1201 | 0.0871 |
| 123 | 0.1213 | 0.1512 | 0.0997 |
| 999 | 0.0690 | 0.1206 | 0.0785 |
| **Mean** | **0.0994** | **0.1306** | **0.0884** |

CHAIN_UNDERMINE F1 is consistently terrible across all variants and seeds (range: 0.069–0.151). No context length or random seed rescues this class. The root cause is structural (label generation), not architectural. **VERIFIED.**

### 5.3 Per-class diagnosis summary

| Class | Neural F1 | Diagnosis | Root cause |
|-------|----------:|-----------|------------|
| K_STORM | 0.5368 | Moderate underfitting | Low precision (0.44) — confuses NO_PLAN windows as K_STORM |
| Q_EXP | 0.6269 | Moderate underfitting | Decent recall (0.72) but frequent false positives from NO_PLAN (397 FP) |
| OUTPOST | 0.6615 | Best active-plan class | Largest class; recall (0.60) limited by 421 windows lost to NO_PLAN |
| ROOK_FILE | 0.5301 | Moderate underfitting | Low precision (0.48); 310 false positives from NO_PLAN |
| CHAIN_UNDERMINE | 0.0978 | **Catastrophic failure** | Conjunctive rule scoring → extreme label sparsity (79 test, 418 train) |
| NO_PLAN | 0.7275 | Best overall class | Largest class (45.8% of test); recall 0.66 limited by over-prediction of active plans |

---

## Phase 6 — Proposed Generalization Experiments

**All experiments below are designs only — none were executed during this audit.**

### 6.1 Experiment G1: Stronger regularization

**Hypothesis:** The 35pp train-val accuracy gap can be reduced by applying regularization within the residual blocks.

**Design:**
- Add dropout (p=0.10) after each residual block's first Conv2d→BN→GELU
- Add Mixup (α=0.3) on the encoded tensor input
- Increase weight decay to 1e-3
- Compare: same seed=42, same architecture, 30 epochs, early stopping patience 5
- **Metric:** Val macro-F1 at best epoch; train-val accuracy gap at that epoch

**Expected outcome:** Reduced overfitting gap, potentially modestly higher val macro-F1. Will not fix CHAIN_UNDERMINE (that's a label problem, not a regularization problem).

### 6.2 Experiment G2: CHAIN_UNDERMINE rule revision

**Hypothesis:** Replacing the binary conjunctive scoring with graded partial credit will produce more CHAIN_UNDERMINE labels and improve class F1.

**Design:**
- Modify `_chain_undermining()` to award partial credit for: (a) pawn chain exists (0.3), (b) pawn push exists in window (0.2), (c) push targets chain base/apex (0.5 bonus)
- Re-label entire dataset with revised rule engine
- Re-train with same config, seed=42
- **Metric:** CHAIN_UNDERMINE test F1, total CHAIN_UNDERMINE labels before/after, macro-F1

**Expected outcome:** 3–10× more CHAIN_UNDERMINE labels; F1 improvement from ~0.10 to potentially 0.30–0.50, depending on how many positions exhibit partial evidence.

### 6.3 Experiment G3: Capacity-controlled context-length ablation

**Hypothesis:** The current ablation (1/4/8-ply) is confounded by parameter count (230K/255K/287K). A fair comparison requires fixed capacity.

**Design:** (As detailed in `docs/CONTEXT_LENGTH_ABLATION.md`)
- Fix channel count so all variants have ~287K params (1-ply: ~100 channels, 4-ply: ~78 channels, 8-ply: 64 channels)
- 3 variants × 3 seeds × 2 phases (val tuning + single test eval)
- **Metric:** Test macro-F1, per-class F1, with confidence intervals

### 6.4 Experiment G4: Neural-rule fusion investigation

**Hypothesis:** Since the rule engine beats the neural model on every class, understanding *where* the neural model adds value (if anywhere) requires analyzing the intersection of their predictions.

**Design:**
- On the test set, partition into 4 quadrants: both correct, both wrong, neural-only correct, rule-only correct
- Compute: how many windows does the neural model correctly predict that the rule engine misses?
- If neural-only-correct > 0: analyze what distinguishes those windows
- **Metric:** Quadrant sizes, neural value-add count

**Expected outcome:** The neural model may capture soft pattern similarities that rules miss for a small subset of windows, but the signal is likely weak given the macro-F1 gap.

### 6.5 Experiment G5: Human annotation benchmark

**Hypothesis:** We do not know whether the labels themselves are correct. Without human strategic assessment, we cannot distinguish "model fails to generalize" from "labels are noisy/wrong."

**Design:**
- Sample 200 test windows stratified by class (40 per active class, 40 NO_PLAN)
- Have 2 independent annotators (chess players rated 1800+) label each window
- Compute inter-annotator Cohen's kappa and annotator-vs-rule-engine agreement
- **Metric:** Human-human kappa, human-rule kappa, human-neural kappa

**Expected outcome:** This is the single most important experiment for understanding generalization. If human-rule agreement is low, the entire evaluation framework is measuring the wrong thing.

---

## Phase 7 — Final Scientific Assessment

### 7.1 Summary of findings

| Finding | Status | Severity |
|---------|--------|----------|
| Severe overfitting (35pp train-val gap) | **VERIFIED** | High |
| Rule-only baseline beats neural on all classes | **VERIFIED** | Critical |
| Neural model provides negative value-add | **VERIFIED** | Critical |
| CHAIN_UNDERMINE is catastrophically broken (F1=0.10) | **VERIFIED** | High |
| No data augmentation | **VERIFIED** | Medium |
| 3 vs 4 residual blocks doc inconsistency | **INCONSISTENT** | Low (cosmetic) |
| No runtime split-disjointness assertion | **VERIFIED** | Low (structural guarantee suffices) |
| Phase B results not persisted to file | **VERIFIED** | Low (reproducible from script) |
| Task A vs Task B distinction undocumented | **VERIFIED** | High (scientific validity) |
| No human annotation benchmark exists | **VERIFIED** | Critical |

### 7.2 The central question: Does MindCheck genuinely generalize to unseen chess games?

**Answer: No. The evidence does not support a claim of meaningful generalization. VERIFIED.**

The assessment rests on four independent lines of evidence:

#### Evidence 1: The model overfits severely

Train accuracy of 97.1% with val accuracy of 61.9% (35pp gap) and val macro-F1 declining after epoch 5 means the model has memorized training patterns rather than learning generalizable representations. Early stopping at epoch 5 saves the least-bad checkpoint, but even at that point the train-val gap is 22pp and growing.

#### Evidence 2: The model loses to its own teacher

The neural model was trained on labels generated by the symbolic rule engine. On the test set, the rule engine achieves macro-F1 of 0.6570 versus the neural model's 0.5301 — the teacher outperforms the student on every class. This means the neural model has not even successfully distilled the rule engine's knowledge, let alone learned anything beyond it. A system that underperforms its label source has negative generalization value.

#### Evidence 3: One class is non-functional

CHAIN_UNDERMINE F1 of 0.0978 (precision 5.6%) means the model's CHAIN_UNDERMINE predictions are wrong 94.4% of the time. This is worse than random for a class with 0.8% prevalence. The root cause is structural (binary conjunctive rule scoring producing only 418 training examples), so this failure is not fixable by hyperparameter tuning alone.

#### Evidence 4: We don't know if the labels are right

The model is trained and evaluated against retrospective weak labels from the rule engine (Task A). There is no evidence that these labels correspond to human strategic intent (Task B). The Cohen's kappa benchmark (`task_1.9`) establishes inter-annotator agreement for the annotation scheme but does not validate that the rule engine's automated labels match human judgment at scale. If the labels themselves are noisy or systematically wrong, even a perfectly generalizing model would appear to fail — and a model that "succeeds" might be learning the wrong thing.

### 7.3 Nuanced assessment

The model is not worthless — it does extract *some* signal:
- Accuracy of 0.6438 on 6 classes (chance = ~16.7% uniform, ~45.8% majority-class baseline) shows the model has learned patterns beyond random guessing
- Balanced accuracy of 0.6067 confirms this holds even when correcting for class imbalance
- The epoch-5 checkpoint represents genuine (if limited) learning before overfitting dominates

But "better than random" is insufficient for a system that claims to recognize strategic plans. The model's predictions are less accurate than simply running the rule engine — which requires no training, no GPU, and no checkpoint management.

### 7.4 Path to genuine generalization

Ordered by expected impact:

1. **Human annotation** (Experiment G5): Establish ground truth before optimizing against potentially wrong labels
2. **CHAIN_UNDERMINE rule revision** (Experiment G2): Fix the structural label sparsity
3. **Stronger regularization** (Experiment G1): Address the 35pp overfitting gap
4. **Capacity-controlled ablation** (Experiment G3): Determine optimal context length without confounds
5. **Fusion analysis** (Experiment G4): Find whether the neural model adds any value over rules alone

Until at least experiments G5 and G1 are completed, the claim "MindCheck recognizes strategic plans in unseen chess games" is **not supported by the available evidence**.

---

## Appendix A — Evaluation script details

| Script | Purpose | Runtime | Status |
|--------|---------|--------:|--------|
| `eval_audit.py` | Full test metrics, confusion matrix, rule baseline | ~960s | Completed (crashed on format string in confidence section after all metrics were printed) |
| `confidence_fix.py` | Per-class confidence, prediction distribution, val metrics | ~1253s | Completed successfully |
| `pre_balance_counts.py` | Pre-balance class distributions, disjointness check | ~910s | Completed successfully |

Both scripts loaded the checkpoint, re-derived the seed=42 split from the raw PGN, and ran inference from scratch. No cached or pre-computed results were used. All metrics reported in this document are cross-checked against the Phase B comparison output in `docs/FINAL_MODEL_AUDIT.md` and match exactly.

## Appendix B — Ablation summary statistics

| Variant | Params | Mean val macro-F1 (3 seeds) | Std | Best seed |
|---------|-------:|----------------------------:|----:|-----------|
| 1-ply | 230,662 | 0.5239 | 0.0076 | 999 (0.5314) |
| 4-ply | 254,854 | 0.5565 | 0.0134 | 999 (0.5717) |
| 8-ply | 287,110 | 0.5315 | 0.0092 | 999 (0.5416) |

**Caveat:** Different input channel counts (14/56/112) with the same 64-channel architecture mean parameter counts vary. This ablation is **capacity-confounded** — the 4-ply variant's advantage may reflect a better parameter-efficiency sweet spot rather than a genuine context-length effect. All results are validation-only; no test evaluation was performed. **REPORTED** (from `experiments/ablation_results.json`).

## Appendix C — Raw class counts verification

| Stage | K_STORM | Q_EXP | OUTPOST | ROOK_FILE | CHAIN_U | NO_PLAN | Total |
|-------|--------:|------:|--------:|----------:|--------:|--------:|------:|
| Raw labeler output | 16,190 | 27,431 | 39,397 | 13,527 | 1,925 | 141,150 | 239,620 |
| After filter | 4,068 | 9,017 | 19,251 | 5,737 | 587 | 28,383 | 67,043 |
| Train (pre-balance) | 2,828 | 6,536 | 13,641 | 3,960 | 418 | 19,747 | 47,130 |
| Train (post-balance) | 2,828 | 6,536 | 10,306 | 3,960 | 418 | 10,306 | 34,354 |
| Val | 537 | 1,182 | 2,900 | 895 | 90 | 3,848 | 9,452 |
| Test | 703 | 1,299 | 2,710 | 882 | 79 | 4,788 | 10,461 |

All counts **VERIFIED** by running the pipeline scripts.
