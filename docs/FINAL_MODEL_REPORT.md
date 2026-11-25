# MindCheck: Final Model Report (before human annotation)

**All scores are against rule-engine weak labels.** Nothing here measures agreement with human
judgement. That needs the annotation benchmark.

## The model

| Item | Value |
|---|---|
| Files | `models/checkpoints/mindcheck_final.{pt,onnx,ts.pt,json}` (the weights are gitignored) |
| Input | `(B, 112, 8, 8)` float32, per the project spec. The newest 56 channels (4 plies) are sliced internally. |
| Architecture | `HistoryChessResNet`, 64 channels, 3 residual blocks, 254,854 trainable parameters |
| Training variant | `D_random_balance`: dropout 0.3, weight decay 1e-3, mirror augmentation (classes 2 to 5, only positions with no castling rights), classes capped by **random** sampling |
| Chosen seed | 999, selected on validation (the three seeds are practically equal) |
| Output | Logits with a validation-fitted per-class bias already added |
| Training data | 80,000-game build: 55,867 training games, 968,901 windows after balancing |
| Training hardware | RTX 4060 Laptop GPU |

Load it with `src.models.inference.load_deployable`.

## Headline results

Final test: a **fresh holdout** of 6,000 games taken from after the part of the archive the
training build read. Zero move-sequence overlap with the 80,000-game set. Chosen model, seed 999:

| Evaluation | Macro-F1 raw | Macro-F1 with bias | Accuracy with bias |
|---|---|---|---|
| Filtered holdout (same selection rules as validation) | 0.670 | **0.723** | 0.856 |
| **All positions from ply 12** (closest to real use) | 0.505 | **0.553** | 0.688 |
| Baseline: always predict `NO_PLAN`, all positions | 0.124 | n/a | 0.593 |

**Read both rows.** The first shows the model generalizes to unseen games. The second shows it is
much weaker on unselected positions. Quoting only the first would overstate real performance.

All positions, three seeds with bias: macro-F1 0.549, 0.550, 0.553; accuracy 0.688, 0.684, 0.688.

### Per-class F1, seed 999, with bias

| Class | Filtered holdout | All positions |
|---|---|---|
| `OUTPOST` | 0.896 | 0.725 |
| `NO_PLAN` | 0.886 | 0.747 |
| `Q_EXP` | 0.796 | 0.589 |
| `ROOK_FILE` | 0.797 | 0.563 |
| `K_STORM` | 0.741 | 0.512 |
| `CHAIN_UNDERMINE` | 0.223 | 0.185 |

## Why all-position scores are much lower

Training, validation and the filtered holdout keep only windows with weak-label confidence of at
least 0.55, limited material imbalance, and at most the 25 most confident windows per game. The
confidence is computed with look-ahead moves. So the model never trained on ambiguous positions,
and validation never tested it on them. An independent audit found this before the final run,
which is why the all-position holdout was added.

This is the most likely explanation. It has **not** been tested by training on unfiltered data.
The class mix also differs (`NO_PLAN` is 59% of all positions and 44% of the filtered ones), and
the biases were fitted on filtered validation data.

## How the model was chosen (rules fixed before training)

All selection used validation and training data only.

**Round 1** (seed 42, up to 8 epochs):

| Variant | Score | Macro-F1 gap | Accuracy gap |
|---|---|---|---|
| C: confidence-based balancing | 0.712 | 3.30 | 1.71 |
| D: random balancing | 0.731 | 3.39 | 1.81 |
| E: random balancing, square-root loss weights | 0.735 | 5.76 | 2.51 |

Gaps are in points, train minus validation, measured on the same natural class mix. Limits: 3
macro-F1 and 2 accuracy points. **No variant passed both limits.** The pre-set fallback took the
best score; D and E were within the 0.005 tie margin, so the smaller gap (D) won.

> **Correction.** The gap measure used for the tables in this section sampled all cached
> training rows, including rows that class balancing had discarded and the model never trained
> on, so it **understated** the gap. Re-measured on trained rows only, at validation's class mix:
>
> | Seed | Macro-F1 gap (reported) | Accuracy gap (reported) | Within 3 and 2? |
> |---|---|---|---|
> | 999 (the saved model) | 2.4 (1.9) | 1.8 (1.1) | Yes |
> | 42 | 3.1 (2.9) | 2.1 (1.7) | **No, both over by about 0.1** |
> | 123 | 1.9 (1.7) | 1.3 (1.0) | Yes |
>
> **Gate G1 therefore does not pass**, because it required every seed to be within the limits.
> The round 1 gap values below are understated in the same way. G2 and G3 are unaffected.
> Files: `experiments/gap_corrected_loopr2_D_s{seed}.json`.

**Round 2** (D, seeds 42, 123, 999, up to 16 epochs with early stopping; all stopped early):

| Gate | Result | Limit | Status |
|---|---|---|---|
| G1: train/validation gap, every seed | corrected worst 3.1 macro-F1, 2.1 accuracy | 3 and 2 | **Fail** (see correction above) |
| G2: spread of raw validation macro-F1 | standard deviation 0.0030 | at most 0.005 | Pass |
| G3: filtered holdout vs validation, every seed | largest difference 0.0010 | at most 0.02 | Pass |

A round-3 variant was defined and was not run, because the original (understated) measure
reported a G1 pass.

## Is it overfitting?

- **Overall:** only slightly. For the chosen seed, the corrected same-class-mix gap is 2.4
  macro-F1 and 1.8 accuracy points, and filtered-holdout scores match validation.
- **`CHAIN_UNDERMINE`:** yes. Recall on windows it trained on is 16 to 24 points higher than on
  validation across the three seeds (about 29 points in round 1). It stayed weak under every
  variant tried. Earlier analysis ([VARIANT_C_ANALYSIS.md](VARIANT_C_ANALYSIS.md)) found about a
  third of its labels depend on moves the model cannot see.

## Verification performed

- The exported model, loaded from disk and fed 112-channel tensors, reproduces the recorded
  scores on both holdouts to four decimals.
- On 512 real windows, ONNX differs from PyTorch by at most 3.1e-6 and TorchScript by 0; the
  predicted class is the same everywhere.
- 80,000-game build: no game leaks across splits and no duplicate games.
- Fresh holdout: no move-sequence overlap with the 80,000-game set, no internal duplicates.
- Independent audit: no future moves in input tensors; the rule-engine speed-up matched the old
  behaviour on 15,754 move pairs. Its two findings (filtered evaluation, mirrored positions with
  castling rights) were addressed before training.

## Human annotation benchmark: read before scoring

- The 250 windows come from games assigned to the test split, never used for training or
  validation by game ID.
- **Exclude 23 windows** listed in `data/annotations/benchmark_exclusions.csv`, leaving **227**.
  4 come from games whose exact moves also appear in training under another game ID, and 22 have
  the exact same 4-ply model input in training (3 are in both groups). The sheets themselves are
  unchanged, and the labels were blank when this was checked.
- **Expect lower scores than the filtered numbers.** The annotation windows were sampled without
  the confidence filter, from ply 15 on and balanced across game phases, so the all-position
  results are the closer guide. They are not a prediction: human labels may differ from the rule
  engine.

## Limitations

- Weak labels only.
- All-position performance is far below filtered performance.
- `CHAIN_UNDERMINE` is unreliable.
- The three seeds share one data split, so seed spread does not cover data-sampling variance.
- Validation was used both to select the model and to fit the biases, so validation numbers are
  slightly optimistic. The holdout numbers do not have this problem.
- The fresh holdout has now been used once. A future model change needs another one (the archive
  has many unused games).
- The old 80,000-game test split was not evaluated in this run.

## Reproduce

```powershell
python scripts/build_scaled_dataset.py --max-games 80000 --workers 14 --out-dir data/processed/scaled_80k
.venv\Scripts\python.exe scripts/train_loop.py --wait-for-data
```

`.venv` is a local environment with CUDA PyTorch (2.14.1+cu126) layered on the system packages.
Loop state and results are in `experiments/loop/` (gitignored).
