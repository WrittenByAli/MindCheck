# MindCheck — Final Model Audit

> **Update.** This audit covers the original 3,000-game model. The current baseline (Variant C,
> trained on 80,000 games) is analysed in [VARIANT_C_ANALYSIS.md](VARIANT_C_ANALYSIS.md).

> **Methodology note.** Every claim below is tagged **VERIFIED** (directly
> reproduced from an existing artifact or source file during this audit),
> **REPORTED** (present in prior documentation/commit messages but not
> independently reproduced here), **UNVERIFIABLE** (no artifact exists to
> check it), or **INCONSISTENT** (artifacts contradict each other or prior
> documentation). This audit is read-only: no source file, config, dataset,
> label, checkpoint, or experiment result was modified to produce it.

## 1. Executive summary

MindCheck's data pipeline, rule engine, weak labeler, game-level splitter,
tensor encoder, `HistoryChessResNet` model, and training loop are all real,
implemented, unit-tested code — not scaffolding. One full training run exists
on the real 3,000-game Lichess Elite dataset, with a real saved checkpoint
(`models/checkpoints/best_resnet.pt`) and a real per-epoch history
(`experiments/training_history.json`). The project's headline numbers (best
epoch 5, validation macro-F1 0.5320, per-class F1 table) are **VERIFIED**
directly from the checkpoint's own stored metadata, independent of any
document that merely repeats them.

The project's weakest point is not the code — it is the **evidence gap
between "the model agrees with its own training-label source" (Task A) and
"the model recognizes human strategic intent" (Task B)**. No human annotation
has been collected (0/250 rows filled in either annotator's sheet, despite
complete tooling), so there is currently no evidence for Task B at all. The
`CHAIN_UNDERMINE` class (F1 ≈ 0.10) is a confirmed, structurally explainable
weak point: its rule definition is conjunctive and all-or-nothing (chain
existence AND a *recent* lever push AND no partial credit), unlike every other
rule, which award partial credit for partial evidence — this plausibly both
starves the class of positive training examples and makes it a fine-grained
temporal-spatial pattern for a global-average-pooled CNN to learn from few
examples.

A real, but methodologically limited, 1/4/8-ply history-length ablation
already exists in the repository (`experiments/ablation_results.json`) and
shows a directional (not rigorously controlled) signal that 4-ply history may
outperform 8-ply — i.e., **8-ply has not been proven optimal**. A rigorous
follow-up design, explicitly not executed, is specified in
`docs/CONTEXT_LENGTH_ABLATION.md`.

One confirmed **documentation bug** (not a code bug) was found: `CLAUDE.md`,
`docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md`, `docs/PROJECT_REFERENCE_FULL.md`,
and the `mindcheck-ml-pipeline` skill all state the residual CNN has **4**
residual blocks; the actual, tested, checkpointed model has **3** (verified by
loading the real checkpoint with `num_blocks=3` and getting an exact parameter
match, and by the test suite's own 250k–300k parameter budget, which only a
3-block model satisfies). See §6 and §13.

## 2. Verified project architecture

Pipeline, as implemented (file:function), cross-checked against
`docs/PROJECT_REFERENCE_FULL.md` §1 and `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md`:

```
lichess_elite_subset.pgn (3,000 games, Elo ≥ 2200)
  -> StreamingPGNParser (src/data/pgn_parser.py)
  -> extract_windows (src/data/window_extractor.py) — windows from ply 12 onward, history up to 8 boards
  -> RetrospectiveWeakLabeler.label_game (src/data/weak_labeler.py) — labels using ply t..t+8 via SymbolicRuleEngine
  -> filter_and_balance / balance_classes (src/data/weak_labeler.py) — confidence/material filter, per-game cap, train-only balancing
  -> GameLevelSplitter.split (src/data/dataset_splitter.py) — deterministic sha256(game_id:seed) hash bucketing
  -> ChessTensorEncoder.encode_window (src/data/tensor_encoder.py) — (112,8,8) + (8,) mask, attacker-perspective via board.mirror()
  -> ChessPlanDataset / create_dataloaders (src/data/chess_dataset.py)
  -> HistoryChessResNet (src/models/residual_cnn.py) — 3 residual blocks, 64 channels, 287,110 params
  -> ModelTrainer.fit (src/models/train.py) — AdamW, cosine annealing, early stopping on val macro-F1
  -> save_checkpoint / load_checkpoint (src/models/logger.py)
  -> export_to_onnx / export_to_torchscript (src/models/export.py)
```

This matches the documented pipeline diagram in structure and order. It does
**not** match the documented model depth (3 blocks built/shipped vs. 4 blocks
documented) — see §6, §13.

## 3. Dataset preparation pipeline

| Component | File:function | Verified behavior |
|---|---|---|
| `StreamingPGNParser` | `src/data/pgn_parser.py` | Streams games; `parse_elo` helper used by both window extractor and weak labeler for consistent Elo parsing. |
| `extract_windows` | `src/data/window_extractor.py:extract_windows` | Yields one `PositionWindow` per ply from `start_ply=12`, history up to 8 boards, **never includes boards after `ply_index`** — confirmed by reading the function: the rolling `deque` only ever receives `board.copy(stack=False)` *after* each move is pushed, and the window is yielded immediately with the history built so far; there is no look-ahead in this function. **VERIFIED by code inspection.** |
| `RetrospectiveWeakLabeler` | `src/data/weak_labeler.py:RetrospectiveWeakLabeler.label_game` | Computes `per_ply_scores` for every ply via `SymbolicRuleEngine.evaluate_window` (history-only per call), then **averages scores over `[t, t+1, ..., t+8]`** to decide the label at `t`. This is the one place the pipeline legitimately looks forward — and only for the **label**, never the window/tensor. **VERIFIED by code inspection.** |
| `filter_and_balance` / `balance_classes` | `src/data/weak_labeler.py` | Confidence ≥ 0.55, `|material| ≤ 800`, ≤25 windows/game, then (train-split-only, per `src/models/train.py:prepare_datasets`) downsample any class above 30% of the train split. **VERIFIED**: `prepare_datasets` explicitly calls `filter_and_balance(..., balance=False)` before the split, then `balance_classes(...)` only on `[r for r in kept if r.split == "train"]` — val/test never pass through `balance_classes`. |
| `ChessTensorEncoder` | `src/data/tensor_encoder.py` | `(112,8,8)` + `(8,)` mask; attacker-perspective via `board.mirror()` (flips ranks and swaps colors/castling/en-passant together, in one call) rather than a manual flip+swap — functionally equivalent to the spec's "flip rows + swap planes," **and arguably more robust** since `python-chess`'s own mirror implementation is the one place castling/en-passant edge cases can't be forgotten. **VERIFIED by code inspection.** |

### 3.1 Reported dataset sizes (34,354 / 9,452 / 10,461 train/val/test windows)

**STATUS: VERIFIED by independent reproduction.** No artifact in the
repository persists these counts; `src/models/train.py:main` only prints
them to stdout during an interactive run and they were never redirected to a
file, and `src/data/audit.py:write_summary` (which *would* persist exactly
this, plus a `game_overlap.disjoint` leakage check) was never invoked for the
real run — `data/processed/` contains only a `.gitkeep`. This audit
independently re-ran `scripts/phase_b_baseline_comparison.py` (unmodified),
which calls the same `prepare_datasets(seed=42)` and prints the three split
sizes. The reproduction completed successfully:

| Split | Window count |
|---|---|
| Train | 34,354 |
| Val | 9,452 |
| Test | 10,461 |

These match the spec's reported figures exactly. **VERIFIED.**

## 4. Data splitting and leakage analysis

- **Game-level split, not window-level**: `GameLevelSplitter.assign_split`
  hashes `f"{game_id}:{seed}"` with SHA-256 and buckets `digest % 100` against
  the 70/15/15 cut points. Because the assignment is a pure function of
  `game_id`, **every window of a game maps to the identical split by
  construction** — this is not merely tested, it is structurally guaranteed.
  **VERIFIED by code inspection.**
- **Disjointness is tested** (`tests/test_dataset_splitter.py::test_splits_are_game_disjoint`,
  `test_all_windows_of_a_game_share_one_split`) but **the production
  `prepare_datasets` path in `src/models/train.py` never calls
  `GameLevelSplitter.verify_disjoint` at runtime** — it relies entirely on the
  hash function's correctness rather than asserting it on the live dataset
  each run. Low risk (the hash-bucketing logic is simple and unit-tested) but
  worth flagging: there is no runtime guard that would catch a future
  regression (e.g., someone changing `assign_split` to depend on something
  ply-specific) before it silently leaked. **Recommendation (not applied):**
  add a `splitter.verify_disjoint(kept)` assertion inside `prepare_datasets`.
- **Duplicate/near-duplicate positions across different games**: not checked
  anywhere in the pipeline. Two different games reaching a transpositionally
  identical or near-identical position (common in heavily-booked openings)
  would receive different `game_id`s and could legitimately land in different
  splits while being highly similar inputs. This is a **plausible, unverified
  contributing factor to the train/validation generalization gap** (§10) —
  it is a HYPOTHESIS, not confirmed, since no duplicate-detection artifact
  exists to check it against.

## 5. Tensor encoding

- Plane layout (0–5 friendly, 6–11 opponent, 12 castling, 13 en-passant) and
  shape `(112,8,8)` match `config/config.yaml`'s `model.in_channels: 112`,
  `model.planes_per_ply: 14`. **VERIFIED** by reading `tensor_encoder.py`
  against `config.yaml`.
- Padding: front-padded (oldest slot zeroed first, newest board always in the
  last 14-channel slot), matching the documented "zero-pad early plies"
  behavior, with an explicit `(8,)` validity mask returned alongside.
  **VERIFIED by code inspection**; also exercised by
  `tests/test_tensor_encoder.py` (not fully read line-by-line in this audit,
  but present and part of the passing suite per `TRAINING_IMPLEMENTATION_PLAN.md`'s
  "156/156 tests passing" claim — that specific count is **REPORTED**, not
  re-run in this audit pass).
- **No future-ply information enters the tensor.** `ChessTensorEncoder.encode_window`
  and `encode_position_window` only ever receive `window.board_history`, which
  is built exclusively from `extract_windows` (§3) — there is no code path by
  which `RetrospectiveWeakLabeler`'s look-ahead scores, or any board state
  after `ply_index`, could reach the tensor encoder. **VERIFIED by tracing the
  call graph**: `prepare_datasets` builds `windows_by_id` from `extract_windows`
  output only, and `ChessPlanDataset.__getitem__` calls
  `self.encoder.encode_position_window(self.windows[index])` — the label
  (`self.labels[index]`) and the window are stored as parallel lists populated
  from `LabeledPositionRecord`/`PositionWindow` objects that never reference
  each other's internals.

## 6. Model architecture

`HistoryChessResNet` (`src/models/residual_cnn.py`): stem `Conv2d(112→64, 3×3)`
+ BatchNorm + GELU, then **3** `ResidualBlock`s (each: `Conv2d(64→64,3×3)` +
BN + GELU, `Conv2d(64→64,3×3)` + BN, skip-add, GELU), then
`AdaptiveAvgPool2d(1,1)` → `LayerNorm(64)` → `Dropout(0.20)` → `Linear(64→6)`.

- **Parameter count: 287,110 — VERIFIED** by loading
  `models/checkpoints/best_resnet.pt` into
  `HistoryChessResNet(in_channels=112, num_classes=6, channels=64, num_blocks=3, dropout=0.20)`
  and calling `count_parameters` during this audit. This matches the spec's
  reported ~287,110 exactly, **and only with `num_blocks=3`**
  (`num_blocks=4` on the same config yields 361,094 params, which does not
  match the checkpoint and falls outside `tests/test_models.py::test_parameter_count_in_budget`'s
  asserted 250,000–300,000 target band).
- **INCONSISTENCY (documentation, not code)**: `CLAUDE.md` §1 ("4 blocks, 64
  channels"), `docs/PROJECT_REFERENCE_FULL.md` line 39 ("4 blocks, 64
  channels"), `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md` §3 (draws 4
  residual blocks explicitly), and the `mindcheck-ml-pipeline` skill ("4
  residual blocks") all state 4 blocks. `config/config.yaml`
  (`model.num_blocks: 3`), `src/models/residual_cnn.py` (default
  `num_blocks: int = 3`), and the actual checkpoint (287,110 params, only
  consistent with 3 blocks) all agree on 3. `docs/03_PHASED_ROADMAP_AND_MILESTONES.md`
  hedges with "3–4 residual blocks," which is the only document consistent
  with what was actually built and shipped.
  - **Impact**: Low for correctness (the shipped model is internally
    consistent and tested at 3 blocks; this is a stale/inaccurate top-level
    description, not a functional defect) but **non-trivial for anyone
    onboarding from `CLAUDE.md`/the architecture doc** — they would expect a
    4-block model and a ~361k-parameter checkpoint that doesn't exist.
  - **Proposed fix (not applied)**: update `CLAUDE.md` §1,
    `docs/PROJECT_REFERENCE_FULL.md` line 39, `docs/05`'s diagram (drop
    Residual Block 4), and the `mindcheck-ml-pipeline` skill to say 3 blocks,
    or alternatively retrain a genuine 4-block model if 4 blocks was the
    actually-intended design — that choice is a product decision, not this
    audit's to make.

## 7. Training configuration

Reported vs. `config/config.yaml` vs. `src/models/train.py`/`ModelTrainer.__init__`
defaults — **all VERIFIED as consistent**:

| Setting | Reported | config.yaml | Code default | Verdict |
|---|---|---|---|---|
| Seed | 42 | `training.seed: 42` | `set_seed(seed=42)` | VERIFIED |
| Optimizer | AdamW | — | `torch.optim.AdamW` | VERIFIED |
| Learning rate | 0.001 | `learning_rate: 0.001` | `lr: float = 1e-3` | VERIFIED |
| Weight decay | 1e-4 | `weight_decay: 0.0001` | `weight_decay: float = 1e-4` | VERIFIED |
| Scheduler | CosineAnnealingLR | — | `CosineAnnealingLR(T_max=max(epochs,1), eta_min=1e-5)` | VERIFIED |
| Min LR | 1e-5 | — | `eta_min=1e-5` | VERIFIED |
| Loss | Weighted CE | — | `nn.CrossEntropyLoss(weight=..., label_smoothing=...)` | VERIFIED |
| Label smoothing | 0.05 | `label_smoothing: 0.05` | `label_smoothing: float = 0.05` | VERIFIED |
| Grad clip | 1.0 | — | `max_grad_norm: float = 1.0` | VERIFIED |
| Early stop patience | 5 | `early_stopping_patience: 5` | `EarlyStopper(patience=5)` default | VERIFIED |
| Selection metric | Val macro-F1 | — | `ModelTrainer.fit` tracks `val_f1`/`best_f1` | VERIFIED |
| Batch size | 64 | `batch_size: 64` | CLI/config default | VERIFIED |
| Max epochs | 30 | `max_epochs: 30` | CLI/config default | VERIFIED |
| Restore best checkpoint | yes | — | `self.model.load_state_dict(best_state)` at end of `fit` | VERIFIED |

## 8. Actual training results

### 8.1 From `experiments/training_history.json` and the checkpoint (fully reproducible artifacts)

**VERIFIED directly from the checkpoint's stored metadata** (loaded read-only
during this audit):

- Checkpoint epoch: **5**. Checkpoint `best_f1`: **0.5319828...** (rounds to
  0.532). Checkpoint `metadata.per_class_f1`: `[0.5330, 0.6092, 0.7036,
  0.5307, 0.1047, 0.7106]` for `[K_STORM, Q_EXP, OUTPOST, ROOK_FILE,
  CHAIN_UNDERMINE, NO_PLAN]` respectively — **matches the reported per-class
  F1 table to 3 decimal places exactly.**
- `experiments/training_history.json` (10 logged epochs; training stopped
  after epoch 10 because epoch 5 was the last improvement and
  `early_stopping_patience=5` — 10−5=5 bad epochs triggers the stop,
  consistent with `EarlyStopper.step`'s `>= patience` condition):
  - Epoch 5: `train_acc=0.867206` (86.7%, **matches reported exactly**),
    `val_macro_f1=0.531983` (**matches**).
  - Epoch 10: `train_acc=0.971415` (97.1%, **matches reported exactly**),
    `val_macro_f1=0.493679` (rounds to 0.494, **matches reported**).
  - All ten epochs' `val_macro_f1` values stay in a narrow 0.448–0.532 band
    while `train_acc` climbs monotonically from 57.9% to 97.1% — see §10.

### 8.2 Test-set Macro-F1 (0.530) and rule-engine baseline (0.657)

These two numbers appear in the audit request but **exist in no saved
artifact** — `scripts/phase_b_baseline_comparison.py` only prints its results
to stdout and was never redirected to a file in the commit that added it
(`fd72879`); the commit message itself only states the ablation and training
findings, not the Phase B numbers.

**This audit re-ran `scripts/phase_b_baseline_comparison.py` unmodified** (a
read-only evaluation: it loads the existing checkpoint, re-derives the
deterministic seed=42 split, and evaluates both the neural model and the
rule-only `SymbolicRuleEngine` on the resulting test split — it trains
nothing and writes nothing to disk) to independently check these figures.

The reproduction completed successfully (data prep took 1021.0s; neural
evaluation 12.7s; rule-only evaluation 40.7s). All figures below are
**VERIFIED by independent reproduction** during this audit session.

### Neural model (HistoryChessResNet, epoch-5 checkpoint)

| Metric | Value | Verdict |
|---|---|---|
| Test accuracy | 0.6438 | **VERIFIED** |
| Test macro-F1 | 0.5301 | **VERIFIED** — matches spec's "0.530" to 4 s.f. |
| K_STORM F1 | 0.5368 | **VERIFIED** |
| Q_EXP F1 | 0.6269 | **VERIFIED** |
| OUTPOST F1 | 0.6615 | **VERIFIED** |
| ROOK_FILE F1 | 0.5301 | **VERIFIED** |
| CHAIN_UNDERMINE F1 | 0.0978 | **VERIFIED** — worst class by a wide margin |
| NO_PLAN F1 | 0.7275 | **VERIFIED** |

### Rule-only baseline (SymbolicRuleEngine)

| Metric | Value | Verdict |
|---|---|---|
| Test accuracy | 0.7813 | **VERIFIED** |
| Test macro-F1 | 0.6570 | **VERIFIED** — matches spec's "0.657" |
| K_STORM F1 | 0.6322 | **VERIFIED** |
| Q_EXP F1 | 0.7151 | **VERIFIED** |
| OUTPOST F1 | 0.8441 | **VERIFIED** |
| ROOK_FILE F1 | 0.7521 | **VERIFIED** |
| CHAIN_UNDERMINE F1 | 0.1892 | **VERIFIED** |
| NO_PLAN F1 | 0.8093 | **VERIFIED** |

### Verdict

Neural macro-F1 (0.530) − rule-only macro-F1 (0.657) = **−0.127**.
The neural model does **not** beat the rule-only baseline on any class.
The rule-only engine outperforms the neural classifier on **every single
per-class F1**, including CHAIN_UNDERMINE (0.189 vs. 0.098).

This is a significant finding: the current 8-ply `HistoryChessResNet` does
not add value over the symbolic rules it was trained to replicate. This is
consistent with the overfitting diagnosis in §10 — the model memorizes the
training distribution but fails to generalize the rules' decision boundaries
to held-out games.

## 9. Generalization analysis

- **Validation vs. test consistency**: The gap between the epoch-5
  validation macro-F1 (0.5320, from `training_history.json`, VERIFIED) and
  the reproduced test macro-F1 (0.5301, from §8.2, VERIFIED) is **−0.0019**
  — effectively zero. This is a healthy sign: the model generalizes from
  validation to test about as well as expected, and the test split was not
  used for any model selection (only touched once via Phase B). The problem
  is not val→test degradation; it is that **both** val and test macro-F1 are
  low relative to the rule-only baseline (0.657), confirming the model has
  not learned the rules' decision boundaries well enough to generalize them.
- **Class-wise generalization, per-game/per-player variation, confidence
  calibration, prediction entropy, out-of-distribution behavior, robustness to
  incomplete history**: **UNVERIFIABLE** — no artifact in the repository
  contains any of these. No calibration dump, no per-game breakdown, no
  entropy analysis exists. This is not a case of "the numbers are bad," it is
  a case of "the numbers do not exist." Per this audit's standing
  instruction, this is documented as a missing evaluation, not invented.
- **Inference latency / model size**: `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md`
  §3 claims "~8ms on CPU via ONNX Runtime / TorchScript; < 2ms on CUDA GPU" —
  this is **REPORTED, not reproduced** in this audit (no GPU exists in this
  environment per `TRAINING_IMPLEMENTATION_PLAN.md`, and the CPU figure was
  not independently timed here). Model size: 287,110 params × 4 bytes ≈
  1.1 MB of weights, consistent with the on-disk `best_resnet.pt` (3.48 MB,
  which also includes optimizer state and metadata — the ONNX export,
  1.16 MB, is closer to the pure-weights size) — **VERIFIED by file size
  inspection**, consistent with but not a precise confirmation of exact
  weight-only size.
- **Reliability for intended use**: given the above, the honest answer is
  **not yet determined** — macro-F1 ≈ 0.53 on validation with one class at
  F1 ≈ 0.10 is far from the bar needed for the explanation-generation and
  temporal-tracking layers downstream to be trustworthy, and no calibration
  evidence exists to say whether the model's confidence scores (which the
  fusion/abstention gate in `config.yaml` directly depends on,
  `abstention_threshold: 0.40`) are meaningful.

## 10. Overfitting investigation

**Observed (VERIFIED from `training_history.json`)**: training accuracy rises
monotonically and smoothly (57.9% → 97.1% over 10 epochs) while validation
macro-F1 peaks at epoch 5 (0.532) and then *declines* with increasing noise
(0.524 → 0.516 → 0.506 → 0.515 → 0.494 across epochs 6–10) even as validation
loss also rises monotonically after epoch 3 (1.508 → 1.655 → 1.543 → 1.625 →
1.660 → 1.732 → 1.734 → 1.852). This is the textbook train/val divergence
signature of overfitting, not a measurement artifact — both the loss and the
F1 curves agree on the same story, and the divergence onset is sharp and
unambiguous (epoch 3 for loss, epoch 5 for F1).

**Confirmed causes** (direct evidence from artifacts/code):
1. **The gap is a genuine optimization/generalization divergence, not a
   bug in evaluation**: train and val use the same `ModelTrainer.evaluate`
   code path with identical loss/metric computation; the only difference is
   the data, and val is never balanced (train-only balancing, §3) while val
   keeps its natural class distribution — so train and val are not even
   measuring the same label distribution, which on its own would create a
   different loss *level*, but cannot explain a monotonically *rising* val
   loss over epochs on a fixed val set with a fixed distribution — that part
   is overfitting, not a distribution mismatch artifact.
2. **Early stopping and best-checkpoint restoration worked as designed**: the
   run did not ship the overfit epoch-10 weights; it restored epoch 5's
   weights (VERIFIED: checkpoint epoch=5). The regularization/early-stopping
   *mechanism* is appropriate and functioning correctly — the overfitting
   problem is about the model's intrinsic generalization ceiling on this
   dataset, not a failure to catch overfitting when it happens.

**Likely contributing factors** (plausible, consistent with evidence, not
isolated by a dedicated ablation):
1. **Dataset size relative to problem difficulty**: on the order of 30-40k
   training windows (34,354 — VERIFIED §8.2) across 6 classes with one rare,
   narrowly-defined class (`CHAIN_UNDERMINE`) is a modest amount of data for a
   287k-parameter CNN to learn a genuinely general notion of "strategic plan"
   from, as opposed to memorizing idiosyncrasies of the ~2,100 training games
   (70% of 3,000) it was shown.
2. **Label noise from the weak-labeling process itself**: the retrospective
   label is the *average* rule score over a 9-ply window (t through t+8) — a
   plan that is building but hasn't yet crossed the rule engine's own
   threshold (0.50) and margin (0.15) gates at most of those plies will be
   labeled `NO_PLAN` even on genuinely transitional positions, and nearby plies
   of the *same* game can receive different labels despite near-identical
   board states a few tempi apart. This adds structured noise correlated
   within a game, which validation (different games) cannot be tricked by in
   the same way training data can be overfit to.
3. **Existing regularization is present but modest**: dropout 0.20 (head
   only, not in the residual blocks), weight decay 1e-4, label smoothing
   0.05 — all standard choices, none unusually weak, but none unusually
   strong either for a model that starts overfitting by epoch 3 (loss) / 5
   (F1) out of a 30-epoch budget. This is *consistent with* mild
   under-regularization relative to this dataset size, but calling the
   capacity "too large" outright would be premature — per this audit's
   explicit instruction, and because `tests/test_models.py` already
   constrains the model to a deliberately small 250k–300k-parameter budget;
   the issue is more plausibly data/label volume than raw capacity.

**Hypotheses requiring additional experiments** (not evidenced either way
yet):
1. Whether more/stronger regularization (higher dropout, higher weight decay,
   or mixup/cutmix-style augmentation on the board tensor) would raise the
   epoch-5 peak rather than merely delaying the epoch-5-to-10 decline.
2. Whether the near-duplicate-position concern (§4) measurably inflates
   training accuracy without a corresponding val/test improvement — untested.
3. Whether a larger PGN sample (beyond 3,000 games) would raise the
   validation ceiling — this project's dataset size was a fixed design choice
   (`data/raw/manifest.json`), not experimentally varied.

## 11. CHAIN_UNDERMINE investigation

**F1 ≈ 0.105 (epoch 5, VERIFIED from checkpoint metadata)** — the clear
outlier among the 6 classes (next-lowest is ROOK_FILE at 0.531).

### Dataset distribution
**UNVERIFIABLE at the time of writing** — exact per-class counts (original,
post-balance train, val, test) require a `prepare_datasets` run with
`src/data/audit.compute_summary` invoked on the result, which was never
persisted for the real 3,000-game run (`data/processed/dataset_summary.json`
does not exist — only a `.gitkeep`). This is itself a finding: the audit
tooling to answer this question (§3, `src/data/audit.py`) exists and is
tested, but was simply never run and saved for the actual dataset.

### Classification performance
Per-class F1 = 0.105 is the only directly available number (checkpoint
metadata). Precision/recall breakdown and the confusion matrix are
**UNVERIFIABLE** without a saved per-sample prediction dump, which does not
exist for this run.

### Label-quality analysis (CONFIRMED, by code/spec inspection)

The `CHAIN_UNDERMINE` rule (`src/rules/rule_engine.py:_chain_undermining`) is
structurally different from every other rule in a way that plausibly explains
both its rarity as a label and its difficulty as a learning target:

1. **All other rules give graded partial credit for partial evidence.**
   `_kingside_storm` and `_queenside_expansion` return a weighted sum of
   independent components (pawn pushes, depth, support, king exposure) that
   can accumulate to a moderate score from a single piece of evidence.
   `_outpost_occupation` gives 0.7 just for a *reachable* (not yet occupied)
   outpost. `_rook_file_buildup` gives 0.4 for a lone rook on an open file.
   **`_chain_undermining` returns exactly 0.0 unless a friendly pawn that was
   *specifically pushed within the current 8-ply window* lands on a square
   attacking the base or apex of an opponent pawn chain that the fact
   extractor's connectivity search (`_pawn_chains`) has already identified.**
   There is no partial-credit path. This is a direct, verifiable asymmetry
   in the rule engine's design (confirmed by reading all five `_*` scoring
   methods side by side), and it is **consistent with** (not merely
   speculative about) `CHAIN_UNDERMINE` being both labeled far less often and
   labeled with a much narrower, more specific board pattern than the other
   five classes — which would starve both the rule-derived weak label and the
   model of positive, generalizable examples.
2. **The operational definition itself is also the most conjunctive of the
   six**, per `docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md` §5.2: "Chain
   Existence" AND "Pawn Lever Move ... within the 8-ply history" AND "Context:
   the push is made ... to force pawn trades" (not merely tempo) — three
   conjunctive conditions, versus, e.g., `K_STORM`'s spec which is
   effectively a disjunction of "advancement trigger" OR "supporting
   context." This is the documented *design intent*, not an implementation
   accident — the narrowness is deliberate, but its downstream statistical
   consequence (a rare, hard-to-learn class) was evidently not weighed against
   the project's stated goal of "robust class-wise performance."
3. **Temporal specificity is a plausible model-side difficulty too**: the
   defining signal is not merely "a pawn chain exists and is adjacent to an
   enemy pawn," which is a static, easily-encoded spatial fact — it is "a
   pawn was *just* pushed (within the last 8 plies) to create that
   relationship." The tensor encoding (§5) does preserve per-ply history, so
   this information is *available* to the model in principle, but
   `HistoryChessResNet`'s `AdaptiveAvgPool2d(1,1)` head collapses all spatial
   structure before the classifier, which is friendly to "is there a piece
   configuration like X anywhere on the board" questions but structurally
   harder for "did a specific recent transition happen at a specific square"
   questions — this is a **HYPOTHESIS**, not confirmed by an ablation, but it
   is architecturally plausible given how global average pooling discards
   positional/temporal localization before the linear head.

### Model limitations
The representation (§5) does carry pawn-chain-relevant spatial information
(piece-type planes per ply) and sufficient temporal context (8 full board
snapshots) in principle — so this is not a case of the input literally
lacking the needed facts. The bottleneck is more plausibly (a) the rarity and
narrowness of positive training examples described above, and/or (b) the
architecture's global pooling discarding the fine spatial-temporal
localization the rule's own logic depends on. Both are **HYPOTHESES**
requiring the class-count / confusion-matrix instrumentation described above
(and already proposed as research-roadmap task M2) to confirm.

### Attribution across the requested causal categories
| Candidate cause | Verdict |
|---|---|
| 1. Class rarity | **Likely** (structural — the rule's all-or-nothing scoring almost certainly yields fewer positive labels than the other five classes' graded scoring; exact counts UNVERIFIABLE without a fresh run) |
| 2. Weak-label noise | Likely, to the same degree it affects every class (§10.2) — not shown to be worse specifically for this class |
| 3. Ambiguous strategic definition | **Not confirmed** — the definition is narrow/conjunctive but not vague; it is arguably the *most* precisely specified of the six rules, which cuts against "ambiguity" as the cause and toward "rarity/narrowness" instead |
| 4. Feature limitations | Hypothesis (global average pooling vs. fine temporal localization, above) |
| 5. Class overlap | Not evidenced either way — no confusion matrix exists to check which classes this one is mistaken for |
| 6. Model limitations | Hypothesis (architecture-level, above) |
| 7. Evaluation problems | Not evidenced — the F1 computation itself is standard `sklearn.metrics.f1_score`, no reason to suspect a computation bug |

**This audit explicitly does not recommend oversampling or raising class
weights for `CHAIN_UNDERMINE`** without first running the confusion-matrix
and class-count instrumentation (roadmap M2) — per the request's own
caution, doing so without understanding *why* the class is hard risks
amplifying label noise rather than fixing a genuine learnability gap.

## 12. Weak-labeling limitations: Task A vs. Task B

**Task A** (what this project currently measures): predict
`RetrospectiveWeakLabeler`'s aggregate-rule-score label, which is itself
derived from `SymbolicRuleEngine` evaluated over historical boards *the model
never sees directly* but whose aggregate judgment the model is trained to
approximate. All of this project's current quantitative results — training
accuracy, validation macro-F1, the test macro-F1 (0.530, VERIFIED §8.2), the
neural-vs-rule-only comparison (−0.127 margin, VERIFIED §8.2) — are Task A
measurements.

**Task B** (recognizing actual human strategic intention): **no evidence
currently exists for this task.** The annotation infrastructure
(`src/data/sample_test_set.py`, `ANNOTATION_GUIDELINES.md`,
`src/models/adjudicate.py`, `scripts/adjudicate_cli.py`) is built and
consistent with the taxonomy, but **both annotator CSVs are empty templates
(0/250 rows filled, VERIFIED by direct inspection)** and no
`test_ground_truth.csv` exists anywhere in the repository.

This distinction matters because a high Task A score is explicitly *not*
evidence of Task B competence — if anything, a sufficiently high Task A score
(the neural model closely reproducing the rule engine's own labels) is
somewhat expected *by construction*, since the rule engine supplied the
training labels. The one piece of evidence that would distinguish "the model
learned something beyond the rule engine" from "the model learned to imitate
the rule engine" is the planned neural-vs-rule-only comparison on the test set
(Phase B, §8.2) — if the neural model doesn't clearly beat the rule-only
baseline there, per `TRAINING_IMPLEMENTATION_PLAN.md` §4 Phase B's own stated
rationale, "it has learned nothing beyond memorizing its own label source."

**Retrospective-label systematic bias**: because the label looks at plies
`t..t+8` but the model only sees `t-7..t`, the model is implicitly being
asked to predict, from a position's history, what the position's *near
future* will retrospectively be judged to have been building toward. This is
a reasonable supervised-learning proxy for "plan recognition" but is not the
same claim as "the model perceives the player's intention at time `t`" — it
is closer to "the model perceives which historical patterns correlate with
plans that, in this dataset, went on to be confirmed by the rule engine
shortly afterward." Framing results with this distinction stated explicitly
is a key honesty requirement for any report derived from this project.

## 13. Current model strengths

- A real, leak-free-by-construction (§4), fully-tested data pipeline exists
  end to end — no fabricated or placeholder stage.
- The architecture, training loop, checkpointing, and export paths are all
  genuinely exercised: a real checkpoint exists, a real training history
  exists, and ONNX/TorchScript exports exist on disk
  (`models/checkpoints/best_resnet.onnx`, `.ts.pt`).
- Early stopping and best-epoch checkpoint restoration work correctly and
  were verified to have actually fired as designed (§8.1, §10).
- Four of six classes (`K_STORM`, `Q_EXP`, `OUTPOST`, `NO_PLAN`) reach
  F1 ≥ 0.53 at the selected checkpoint, with `OUTPOST` and `NO_PLAN` above
  0.70 — the model is not failing uniformly; it has learned a real, if
  imperfect, signal for most of the taxonomy.
- Class balancing is correctly restricted to the training split only
  (VERIFIED, §3) — a common and easy-to-get-wrong mistake that this project
  avoided.

## 14. Current model weaknesses

- `CHAIN_UNDERMINE` F1 ≈ 0.10 — the clearest, most severe class-level
  weakness (§11).
- Overfitting past epoch 5, with validation macro-F1 never exceeding ~0.53
  across the whole run (§10) — the model's current generalization ceiling on
  this dataset/architecture is modest in absolute terms, whatever the
  eventual test-set number turns out to be.
- Zero human-validated evidence that the model's predictions correspond to
  actual human-perceived strategic plans (§12) — the project's single
  largest open scientific question.
- No calibration, robustness, cross-player, or cross-game generalization
  evidence exists (§9) — not necessarily because the model performs badly on
  these axes, but because no one has measured them yet.
- One real but methodologically limited history-length ablation exists
  (capacity-confounded, validation-only, mismatched training budget vs. the
  primary run) — 8-ply has not been shown to be the right choice, and a
  preliminary signal points the other way (4-ply) without being conclusive
  (see `CONTEXT_LENGTH_ABLATION.md` §13).
- Documentation (`CLAUDE.md`, `docs/05`, `docs/PROJECT_REFERENCE_FULL.md`,
  the ML-pipeline skill) states a 4-residual-block architecture that does not
  match the actual 3-block, 287,110-parameter shipped model (§6) — not a
  functional bug, but a real risk of future contributors building against a
  description that doesn't match the code.
- The Phase B neural-vs-rule-only comparison — the single most important
  sanity check this project's own training plan identifies as
  "non-negotiable" — was designed and is runnable, but its result was never
  persisted anywhere; this audit is the first time it has been reproduced
  with saved evidence (§8.2).

## 15. Verified versus unverified claims (summary table)

| Claim | Verdict | Evidence |
|---|---|---|
| 8-ply window → `(112,8,8)` tensor, attacker-perspective | VERIFIED | `tensor_encoder.py` read + traced against `config.yaml` |
| Game-level 70/15/15 split, leak-free by construction | VERIFIED | `dataset_splitter.py` read; hash-based assignment is a pure function of `game_id` |
| Train-only class balancing | VERIFIED | `train.py:prepare_datasets` calls `balance_classes` only on the train subset |
| `HistoryChessResNet`, ~287,110 parameters | VERIFIED | Checkpoint loaded read-only; `count_parameters` matches exactly with `num_blocks=3` |
| Architecture has 4 residual blocks | **INCONSISTENT** | Docs/skill say 4; config, code default, and checkpoint-matching param count say 3 |
| Seed 42, AdamW, lr 0.001, wd 1e-4, cosine schedule, label smoothing 0.05, grad clip 1.0, early-stop patience 5, batch 64, max epochs 30 | VERIFIED | `config.yaml` + `train.py`/`ModelTrainer` defaults all agree |
| Best epoch 5, val macro-F1 0.532 | VERIFIED | Checkpoint metadata + `training_history.json` |
| Train acc 86.7% (epoch 5), 97.1% (epoch 10); val macro-F1 0.494 (epoch 10) | VERIFIED | `training_history.json` |
| Per-class F1 table (K_STORM 0.533, Q_EXP 0.609, OUTPOST 0.704, ROOK_FILE 0.531, CHAIN_UNDERMINE 0.105, NO_PLAN 0.711) | VERIFIED | Checkpoint `metadata.per_class_f1`, exact match to 3 decimals |
| Train/val/test sizes 34,354 / 9,452 / 10,461 | **VERIFIED** | Independently reproduced by re-running `scripts/phase_b_baseline_comparison.py` (§8.2); exact match |
| Test macro-F1 0.530 | **VERIFIED** (reproduced: 0.5301) | Same reproduction run; matches spec to 3 s.f. |
| Rule engine 0.657 macro-F1 on test | **VERIFIED** (reproduced: 0.6570) | Same reproduction run; exact match |
| Neural model does NOT beat rule-only baseline | **VERIFIED** (margin: −0.127) | Rule-only outperforms on every per-class F1; see §8.2 verdict |
| 1/4/8-ply ablation exists with real results | VERIFIED (as a limited, capacity-confounded, validation-only run) | `experiments/ablation_results.json` read directly; see `CONTEXT_LENGTH_ABLATION.md` §13 for its scope limits |
| Human annotation / independent benchmark exists | **INCONSISTENT with "exists"** — infrastructure exists, data does not | `raw_annotations_member{1,2}.csv`: 0/250 rows filled each, verified by direct inspection; no `test_ground_truth.csv` |
| Inference ~8ms CPU / <2ms GPU | REPORTED, not reproduced | `docs/05` §3; no timing artifact in repo; no GPU in this environment |
| 156/156 tests passing (training plan's pre-flight checklist) | REPORTED, not re-run in this audit pass | `TRAINING_IMPLEMENTATION_PLAN.md` §3 |

---

*This audit was produced without retraining any model, without launching the
context-length ablation, and without modifying any source file, config,
dataset, label, checkpoint, or existing experiment result. The one
script actually executed (`scripts/phase_b_baseline_comparison.py`) is a
pre-existing, unmodified, read-only evaluation script that loads the existing
checkpoint and writes nothing to disk.*
