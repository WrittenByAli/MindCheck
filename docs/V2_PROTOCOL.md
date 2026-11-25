# Model v2: pre-registered protocol

Written **before** any v2 result existed. Thresholds are not changed after seeing results.

## Problems being fixed (observed on v1, `mindcheck_final`)

1. All-position fresh holdout macro-F1 is 0.553 (bias-adjusted) against 0.723 on the filtered
   holdout. v1 trained and validated only on filtered windows.
2. `CHAIN_UNDERMINE` overfits: trained-on recall minus validation recall is 15 to 22 points.

## Data

- Same games and the same game-level split as the 80,000-game build. The old 80k test games are
  never read.
- Validation: 25 random positions per validation game from ply 12, no confidence or material
  filter. This is the selection population.
- Training pool: 30 random positions per training game ("uniform"), plus every other
  `CHAIN_UNDERMINE` position of those games ("enriched").
- Anything matching the 250 annotation windows is removed: training games with identical moves,
  and windows with an identical 4-ply model input.

## Baseline B0

v1's weights (`D_random_balance`, seed 999) with biases re-fitted on the new unfiltered
validation set. This measures what recalibration alone gives, with no retraining.

## Rounds

- **Round 1** (seed 42, up to 16 epochs, patience 3). Both keep v1's hyperparameters; only the
  training data differs:
  - `F_allpos`: uniform + enriched rows.
  - `G_allpos_uniform`: uniform rows only.
- Winner: higher mean held-out-fold bias-adjusted macro-F1 on unfiltered validation. Within
  0.005, the smaller `CHAIN_UNDERMINE` recall gap wins.
- **Round 2:** the winner with seeds 42, 123 and 999.
- No further rounds without reporting back first.

## Gates

| Gate | Condition |
|---|---|
| A: real improvement | Mean over seeds of bias-adjusted held-out-fold macro-F1 is at least B0 + 0.010 |
| B: overall overfit | Every seed: same-class-mix gap at most 3 macro-F1 points and 2 accuracy points (uniform training rows vs unfiltered validation) |
| C: stability | Standard deviation of raw validation macro-F1 across seeds at most 0.005 |
| D: `CHAIN_UNDERMINE` overfit | Every seed: trained-on minus validation recall at most 10 points. A target: reported pass or fail, does not block promotion |
| E: unseen games (once) | A new fresh holdout, all positions: for every seed, raw macro-F1 within 0.020 of raw validation |

The holdout also scores B0 on the same windows, so v1 and v2 are compared directly.

## Promotion

v2 replaces `mindcheck_final` only if A, B, C and E pass. v1 is kept as `mindcheck_v1` first.
Otherwise v1 stays as the final model and the results are reported. Model and seed are chosen on
validation, never on the holdout.

## Amendment 1 (made before any v2 result existed)

An independent review found gaps in the rules above. These replace or define them.

- **Gap measure (gates B and D).** The earlier measure sampled all cached training rows,
  including rows that class balancing discarded and the model never trained on. It therefore
  **understated** the gap, for v1 as well. The training side is now only rows the model trained
  on, resampled to validation's class mix. v1's earlier gate G1 and round-1 eligibility used the
  old measure; corrected v1 values are reported alongside the v2 results.
- **Definitions.** B and D use raw predictions (no bias). C is the sample standard deviation
  (n-1) of raw validation macro-F1. For A, each seed's score is the mean of its two held-out-fold
  bias-adjusted values, then averaged over seeds.
- **E, same weighting on both sides.** Validation holds 25 positions per game, so E compares it
  with a seeded sample of 25 positions per game from the fresh holdout, two-sided, per seed.
  Every-position holdout results are also reported.
- **New gate F: must beat v1 on unseen games.** Gate A uses the validation set that also chose
  the v2 variant and epochs, so it favours v2. On the 25-per-game holdout sample, with biases
  fitted on validation: the mean over v2 seeds of bias-adjusted macro-F1 must be at least B0's
  plus 0.010, and the validation-chosen seed must beat B0.
- **Promotion** now requires A, B, C, E and F.

## Amendment 2 (made before any v2 result existed)

Round 1 gets a third candidate, `H_allpos_strongreg`: the same data as `F_allpos`, with stronger
regularization against overfitting (dropout 0.3 → 0.4, weight decay 1e-3 → 3e-3). The winner
rule, gates and promotion are unchanged and apply to all three candidates.

## Limits that do not change

All labels are rule-engine weak labels. A higher score on them does not show agreement with
human judgement.
