# Variant C: Post-Training Analysis

Variant C is the final baseline: 4-ply input (last 56 of 112 channels), mirror augmentation on the
four symmetric classes, dropout 0.3, weight decay 1e-3, 254,854 parameters, trained on the
80,000-game dataset (`data/processed/scaled_80k`) for 8 epochs with seeds 42, 123 and 999.

Scope and rules followed: this analysis used **train and validation data only**. The only test
numbers quoted are the ones already recorded in `experiments/final_80k_test.json`. Nothing was
tuned on test. All labels here are rule-engine **weak labels**, not human labels. Every number
below comes from an executed run; the producing script or file is named next to it.

| Artifact | Producer |
|---|---|
| `experiments/analysis_variant_c.json` | `scripts/analyze_variant_c.py` |
| `experiments/analysis_chain_labels.json` | `scripts/analyze_chain_labels.py` |
| `experiments/analysis_selection_effect_80k_s42.json` | `scripts/analyze_selection_effect.py` |
| `docs/figures/variant_c_confusion_seed42.png`, `variant_c_learning_curves.png` | `scripts/analyze_variant_c.py` |

## 1. Baseline reproduction

For all three seeds, reloading the checkpoint and recomputing full-validation macro-F1 and accuracy
reproduces the stored values exactly (max absolute difference 0.0). Every checkpoint is from
epoch 8, the training cap. Checkpoint configs match the intended Variant C settings.

| Seed | Val macro-F1 (raw) | Val accuracy | Test macro-F1 raw (recorded) | Test macro-F1 with biases (recorded) |
|---|---|---|---|---|
| 42 | 0.6496 | 0.763 | 0.6447 | 0.7119 |
| 123 | 0.6535 | 0.769 | 0.6507 | 0.7144 |
| 999 | 0.6520 | 0.766 | 0.6494 | 0.7106 |
| mean | 0.6517 | 0.766 | 0.6483 | 0.7123 |

## 2. Where the errors are: `CHAIN_UNDERMINE`

**Class distributions** (counts from the dataset metadata):

| Split | `CHAIN_UNDERMINE` windows | Share |
|---|---|---|
| Train, natural | 8,891 | 0.65% |
| Train, balanced (used for training) | 8,891 | 0.92% |
| Validation | 1,962 | 0.67% |

Balancing caps classes at 30% of the data. It does not up-sample, so `CHAIN_UNDERMINE` stays under
1%. Training also used class-weighted loss, which gives it a weight of about 18.

**Seed 42 on validation, raw logits** (seeds 123 and 999 show the same pattern):

- Precision 0.079, recall 0.549, F1 0.138.
- **False positives: 12,610**, which is 18.3% of all 69,080 validation errors. Of those, 9,449
  (75%) are true `NO_PLAN` windows, then `OUTPOST` 1,353, `Q_EXP` 1,095, `K_STORM` 468 and
  `ROOK_FILE` 245. (`analysis_variant_c.json` states 69% for `NO_PLAN`; that is the share of all
  windows *predicted* `CHAIN_UNDERMINE`, including correct ones, not of false positives.)
- **False negatives: 885** (1.3% of errors): predicted `K_STORM` 308, `Q_EXP` 259, `NO_PLAN` 210,
  `ROOK_FILE` 67, `OUTPOST` 41.
- With the validation-fitted bias: F1 0.245, precision 0.222, recall 0.272, and the class is
  involved in 7.2% of errors instead of 19.5%.

So the class is not hard because the model cannot find it (raw recall is 55%). It is hard because
it is rare and the model over-predicts it: roughly 12 false alarms for every correct hit.

**Is the cause too few examples, overlapping labels, weak features or inconsistent labels?**
Evidence from `analysis_chain_labels.json` (1,200 validation games, 1,925 `CHAIN_UNDERMINE`
windows, seed 42):

- **Labels are internally consistent.** Re-deriving the weak labels reproduced the cached label for
  29,008 of 29,008 windows (100%), so inconsistent rule-engine annotation is not the problem.
- **Not simply too few examples.** `CHAIN_UNDERMINE` went from about 2,700 training windows (22k
  games) to 8,891 (80k games) and its F1 did not move (0.146 to 0.138). More of the same data did
  not help.
- **The label partly depends on the future.** 31.8% of the labelled windows have no lever-push
  evidence in their own input window (it appears only in the lookahead). My prediction was 57%, so
  that hypothesis was only partly right. Model recall is **65.8% when the evidence is visible and
  31.5% when it is future-only** (raw; with biases, 37.5% against 4.9%).
- **Label overlap with pawn-push classes.** The runner-up plan for these windows is `Q_EXP` (46%)
  or `K_STORM` (33%), and the model's confusions go mostly to `K_STORM` and `Q_EXP`. A pawn lever
  looks a lot like a storm or expansion.
- **Not an unusual dependence on the future by one measure.** Other classes also have 19% to 30%
  of labelled windows scoring below 0.5 in their own window (`OUTPOST` only 3% to 5%), so the
  future-dependence of `CHAIN_UNDERMINE` (31.8%) is comparable to `K_STORM`, `Q_EXP` and
  `ROOK_FILE`.

What this does **not** establish: that the labels are *correct* as a description of chess plans.
Only human annotation can show that. Even in the visible group the model catches just two thirds.

## 3. Generalization and learning curves

Validation macro-F1 by epoch (raw), from the stored training histories:

| Seed | Epochs 1 to 8 |
|---|---|
| 42 | 0.608, 0.614, 0.634, 0.620, 0.634, 0.649, 0.631, 0.650 |
| 123 | 0.621, 0.628, 0.621, 0.625, 0.637, 0.639, 0.644, 0.654 |
| 999 | 0.612, 0.623, 0.639, 0.631, 0.628, 0.647, 0.635, 0.652 |

- **The 8-epoch cap did constrain training.** All three runs set their best at epoch 8. The gain
  over the last 3 epochs was +0.016, +0.016 and +0.024, compared with epoch-to-epoch noise of
  roughly 0.007 to 0.014. The trend is positive but small and noisy, so I cannot say how much more
  training would add. The 22k runs, by contrast, had flattened by epoch 8 to 12.
- **`CHAIN_UNDERMINE` did not improve with epochs:** its F1 was 0.113 to 0.138 over the last 3
  epochs for every seed. More training is unlikely to fix it.
- **Train/validation gap.** The logged (train-mode) accuracy gap at epoch 8 is 0.131 to 0.136. An
  eval-mode probe gives 0.126 to 0.130 for 80k against 0.161 for the 22k run (seed 42). The macro-F1
  gap is about 0.153.

### The gap is mostly not overfitting

Per-class train-minus-validation **recall** (recall does not depend on class mix; seed 42, seed 123
in brackets):

| Class | Train recall | Val recall | Gap |
|---|---|---|---|
| `CHAIN_UNDERMINE` | 0.793 | 0.549 | +0.244 (+0.249) |
| `NO_PLAN` | 0.870 | 0.665 | +0.205 (+0.204) |
| `OUTPOST` | 0.919 | 0.831 | +0.087 (+0.081) |
| `K_STORM` | 0.916 | 0.861 | +0.056 (+0.059) |
| `Q_EXP` | 0.885 | 0.853 | +0.032 (+0.037) |
| `ROOK_FILE` | 0.876 | 0.854 | +0.022 (+0.030) |

Balancing keeps the *most confident* windows of the classes it caps (`NO_PLAN`, `OUTPOST`).
`analyze_selection_effect.py` tests whether that makes the training sample easy
(6,000 windows per group, seed 42, train split only):

| Class | Recall on kept windows | Recall on discarded windows | Mean weak confidence (kept / discarded) |
|---|---|---|---|
| `NO_PLAN` | 0.880 | **0.496** | 0.878 / 0.716 |
| `OUTPOST` | 0.924 | **0.563** | 0.972 / 0.739 |

Weighting the two groups by their sizes (290,670 kept and 312,447 discarded `NO_PLAN`; 290,670
kept and 86,799 discarded `OUTPOST`) gives a natural-train recall of about 0.68 for `NO_PLAN` and
0.84 for `OUTPOST`, against validation recall of 0.665 and 0.831. So for those two classes the
train/validation gap is **almost entirely a selection effect** of confidence-based balancing, not
memorization. (The weighting uses sampled recalls, so it is approximate.)

That leaves `CHAIN_UNDERMINE` (+0.24), which has no selection effect (all its windows are kept),
as the one class with a large unexplained gap. The model fits its 8,891 training examples far better
than unseen ones. `K_STORM` (+0.06) shows a small residual gap, and `Q_EXP` and `ROOK_FILE` almost none.

This corrects my earlier statements that treated the 13-point accuracy gap as plain moderate
overfitting. Train accuracy is also measured on a class-balanced, confidence-filtered sample while
validation uses natural frequencies, so the raw accuracy gap overstates it.

## 4. Class-bias methodology and verification

- **Method:** coordinate ascent over per-class additive logit biases (`tune_logit_bias.fit_bias`).
- **Objective:** macro-F1 on **validation** logits and labels.
- **Search:** grid from -3 to 3 in steps of 0.25, 3 passes, result centered (mean subtracted).
- **Overfit check inside the tuner:** validation games are split into halves; biases fit on one
  half and scored on the other. Held-out gains match the full fit (e.g. seed 42: 0.651 to 0.717
  and 0.649 to 0.714).
- **Test isolation:** the tuner's only mention of the test split is a docstring line, and the
  training path does not reference it. The three bias files (written 06:14, 08:13, 08:13) were
  created before the test results file (created 08:14).
- **Per-seed, not shared:** each test run used **its own seed's** biases, not one frozen vector.
  The vectors differ by at most 0.25:

| Seed | `K_STORM` | `Q_EXP` | `OUTPOST` | `ROOK_FILE` | `CHAIN_UNDERMINE` | `NO_PLAN` |
|---|---|---|---|---|---|---|
| 42 | -1.50 | -0.50 | 1.50 | -0.25 | -1.50 | 2.25 |
| 123 | -1.46 | -0.46 | 1.29 | -0.21 | -1.46 | 2.29 |
| 999 | -1.50 | -0.50 | 1.25 | -0.25 | -1.25 | 2.25 |

- **Shared-bias equivalence (validation only):** the mean of the three vectors gives validation
  macro-F1 of 0.7165, 0.7161 and about 0.716 for the three seeds, against 0.7164, 0.7165 and 0.7161
  with their own biases. The choice of per-seed versus shared bias does not matter.
- **What the bias is:** the biases re-inject the natural class frequencies (`NO_PLAN` +2.25,
  minority classes negative) that training removed by balancing and loss weighting together. They
  raise the score by calibrating to the weak-label class prior. They do not add any understanding.

**A hygiene finding, now fixed.** `load_scaled_dataset` opened `test_labels.npy` even in training
and bias tuning. The test labels were never used to tune anything, but strict separation means not
reading them at all. I added a `splits` argument and the training, tuning and test phases now pass
only the splits they need, with a regression test that fails if test files are opened
(`tests/test_scaled_dataset.py::test_restricted_splits_never_open_test_files`). The earlier runs
used the old behaviour; their results are unaffected.

## 5. Evidence-backed explanation of the remaining weaknesses

1. **`CHAIN_UNDERMINE` is a precision problem** (about 12 false alarms per hit), mostly against
   `NO_PLAN`, driven by rarity (under 1% of data) and an 18x loss weight.
2. **It is also a label problem.** About a third of its labels depend on future moves the model
   cannot see, and the runner-up plan is almost always another pawn-push class.
3. **It is the one class that genuinely overfits** (+0.24 recall gap), because there are only 8,891
   examples.
4. **The 13-point accuracy gap is mostly a measurement and selection effect**, not general
   overfitting.
5. **The raw output is miscalibrated** by double class compensation, which a validation-fit bias
   corrects (0.652 to 0.716 validation macro-F1).

## 6. Prioritized recommendations (none of the expensive ones have been run)

| # | Experiment | Evidence | Expected value | Cost |
|---|---|---|---|---|
| 1 | **Human annotation of the 250 test windows** (already planned), reporting `CHAIN_UNDERMINE` agreement | Only way to tell whether the label or the model is wrong | Highest. It decides whether `CHAIN_UNDERMINE` is worth modelling at all. | Human time |
| 2 | **Train longer, with validation early stopping** (e.g. cap 16, patience 3), one seed first | Val F1 still rising at epoch 8 (+0.016 to +0.024 over 3 epochs) | Small: perhaps +0.005 to +0.015 raw macro-F1. No expected help for `CHAIN_UNDERMINE`. | About 2.6 hours for one seed |
| 3 | **Remove double class compensation** (drop loss weights, keep the balanced sampler), all else fixed | Minority classes have high recall and low precision; biases add +0.067 | Cleaner raw calibration. After bias correction I expect little change. | About 80 minutes per seed |
| 4 | **Better `CHAIN_UNDERMINE` representation** (explicit pawn-push planes) | Recall 66% when evidence visible against 31% when future-only; variant C sees 4 boards while the rule sees 8 | Limited by the 31.8% of labels that are unobservable. Changes the 112-channel input spec in `CLAUDE.md`, so it needs a deliberate decision. | Code change plus retraining |

I would not change labels or the architecture before experiment 1. Experiment 3 is the cheapest
controlled change; experiment 2 is the most likely to help the headline score.

## 7. Limitations and threats to validity

- **Weak labels.** Every score is against rule-engine labels. High agreement with them does not mean
  agreement with human judgement.
- **Label analysis used one seed (42)** and only validation windows after filtering (windows kept by
  the confidence and per-game filters), from games chosen because they contain `CHAIN_UNDERMINE`.
- **Visibility mismatch.** "Visible evidence" in the label analysis uses the rule engine's 8-board
  window; variant C sees 4 boards. A push 4 to 6 plies back is counted visible but the model cannot
  see it, so the true effect may be larger.
- **The selection-effect estimate** uses 6,000-window samples and one seed, so the weighted recall
  is approximate.
- **Single shared bias per test run:** biases were tuned on validation, which has the same natural
  class mix as test. They would not transfer to a different class mix.
- **Test split was evaluated once per seed** for the recorded numbers; this analysis did not look at
  it again.
- **Seeds share data and splits.** The three seeds vary initialization and sampling only, so the
  low seed variance (about 0.003) does not cover data-split variance.
