# Model v2: Results

Protocol: [V2_PROTOCOL.md](V2_PROTOCOL.md), with all thresholds fixed before results. All scores are
macro-F1 against rule-engine **weak labels**, not human judgement.

## Decision

**v2 was not promoted.** Under the protocol, `mindcheck_final` stays **v1**. v2 failed one gate (B,
the train/validation accuracy gap) and passed every other blocking gate. It is saved separately as
`models/checkpoints/mindcheck_v2_candidate.*` so the team can choose.

## What v2 is

`G_allpos_uniform`, seed 42. It uses v1's architecture and hyperparameters (4-ply input, dropout
0.3, weight decay 1e-3, mirror augmentation, random class balancing), and trains on **unfiltered**
positions: up to 30 random positions per training game from ply 12, with no confidence or material
filter. Training data: 55,866 games, 1.65M positions, of which 968,901 remained after balancing.
Every annotation game and identical model input was excluded.

## Unseen games: second fresh holdout

6,000 games no model or earlier evaluation had used (463,408 positions). Biases were fitted on
validation only. Each model was evaluated once.

| Model | All positions | All positions, accuracy | 25 per game | Filtered positions |
|---|---|---|---|---|
| v1 as shipped | 0.554 | 0.688 | 0.559 | **0.722** |
| v1, biases refitted on unfiltered validation | 0.556 | 0.700 | 0.560 | 0.720 |
| **v2, seed 42** | **0.583** | **0.721** | **0.581** | 0.705 |
| v2, seed 123 | 0.580 | 0.717 | 0.577 | 0.703 |
| v2, seed 999 | 0.580 | 0.722 | 0.577 | 0.702 |

All positions, per-class F1:

| Class | v1, biases refitted | v2 |
|---|---|---|
| `NO_PLAN` | 0.762 | 0.778 |
| `OUTPOST` | 0.725 | 0.741 |
| `Q_EXP` | 0.592 | 0.613 |
| `ROOK_FILE` | 0.558 | 0.577 |
| `K_STORM` | 0.522 | 0.550 |
| `CHAIN_UNDERMINE` | 0.179 | 0.241 |

**v2 is better on unselected positions** (+2.7 points against v1 as shipped), and better in every
class. **v1 is better on filtered positions** (−1.5 points for v2), the population v1 was trained
on.

## Gates

| Gate | Result | Status |
|---|---|---|
| A: beats v1 on validation by at least 0.010 | 0.580 against 0.560 | Pass |
| B: train/validation gap, every seed (3 macro-F1 / 2 accuracy points) | macro-F1 2.8 to 3.2, accuracy 2.4 to 2.6 | **Fail** |
| C: seed standard deviation at most 0.005 | 0.0017 | Pass |
| D: `CHAIN_UNDERMINE` recall gap at most 10 points (target only) | 15 to 20 | Missed (non-blocking) |
| E: holdout within 0.020 of validation, every seed | largest difference 0.0023 | Pass |
| F: beats v1 on holdout by at least 0.010 | 0.578 against 0.560 | Pass |

The gap is measured on rows the model trained on, at validation's class mix. v1 measured the same
way: macro-F1 1.9 to 3.1, accuracy 1.3 to 2.1. v1 also fails gate B, by about 0.1 points on one
seed.

## What was learned about overfitting

Round 1, seed 42, unfiltered validation:

| Variant | Change | Score | Macro-F1 gap | Accuracy gap | `CHAIN_UNDERMINE` gap |
|---|---|---|---|---|---|
| F | unfiltered + every extra `CHAIN_UNDERMINE` position | 0.581 | 4.9 | 3.6 | 30.0 |
| **G** | **unfiltered only** | **0.581** | **3.2** | **2.6** | **20.2** |
| H | F + dropout 0.4, weight decay 3e-3 | 0.580 | 5.0 | 4.0 | 28.1 |

- Oversampling the rare class caused memorization without improving its score.
- Stronger dropout and weight decay did not reduce that gap.
- Removing the oversampling reduced it most.
- The remaining gap is concentrated in `CHAIN_UNDERMINE`, where earlier analysis found about a
  third of labels depend on moves the model cannot see.

## Limitations

- Weak labels only. The human benchmark (227 scorable windows) is still needed.
- Gate B's limits were chosen by us. v2 misses the accuracy limit by about 0.5 points while
  generalizing to unseen games as well as it does to validation (gate E).
- Validation selected the variant, epochs and seed, so validation numbers are slightly optimistic.
  The holdout numbers are not.
- Seeds share one data split.
- The second holdout has now been used; any further model change needs a new one.
