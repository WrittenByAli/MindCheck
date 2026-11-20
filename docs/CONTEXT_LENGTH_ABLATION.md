# MindCheck — Context-Length Ablation: Experimental Design

> **Status: DESIGN ONLY. No experiment described in this document has been
> executed under this design.** A preliminary, *differently-scoped* ablation
> already exists in the repository (`experiments/ablation_results.json`,
> produced by `src/models/train_ablation.py`) and is analyzed in
> §13 for what it does and does not tell us. This document specifies the
> rigorous follow-up required to actually answer the research question, and
> explicitly does not authorize running it.

## 1. Research question

Does giving `HistoryChessResNet` more move history (more plies of board state)
genuinely improve chess strategic-plan recognition, or does the 8-ply window
mandated by the project spec provide no additional signal over a shorter
window — or even hurt performance by adding noise/capacity without benefit?

## 2. Hypotheses

- **H1 (history helps monotonically)**: Macro-F1 increases with history length
  (1 < 4 < 8 plies) because more plies let the model see pawn pushes,
  piece maneuvers, and committed advances rather than a single static frame.
- **H2 (diminishing/negative returns)**: Macro-F1 improves from 1→4 plies but
  plateaus or regresses from 4→8 plies, because (a) `start_ply=12` already
  guarantees substantial pre-window development, (b) most of the taxonomy's
  rule predicates (`_kingside_storm`, `_queenside_expansion`, etc.) look at
  pawn pushes within the window regardless of window length beyond ~4 plies,
  and (c) longer input increases first-layer parameter count and nominal
  model capacity on a fixed, modest-sized dataset (~34k train windows),
  raising overfitting risk.
- **H3 (no meaningful effect)**: Differences across history lengths are within
  seed-to-seed noise, i.e., history length is not the dominant lever for this
  architecture/dataset size.

The existing (differently-scoped, 15-epoch, no-test-set) preliminary run in
`experiments/ablation_results.json` is weakly *consistent* with H2 (4-ply beat
8-ply at all three seeds there), but that run has the capacity confound
described in §5 baked in and used only 15 epochs with no early stopping
verification against the primary run's protocol — so it cannot be treated as
evidence for or against H1/H2/H3 at the rigor this project's roadmap needs.

## 3. Experimental design — overview

Two complementary, separately-reported experiments, run across history
lengths **{1, 4, 8}** and seeds **{42, 123, 999}** → 9 runs per experiment,
18 runs total.

| | Experiment A (fixed-capacity) | Experiment B (native-architecture) |
|---|---|---|
| Purpose | Isolate the effect of *information available*, holding model capacity constant | Measure the effect under each length's *natural* architecture, as actually deployed |
| In-channels seen by stem conv | Always 112 (zero-padded/projected) | 14 / 56 / 112 respectively |
| First-conv parameter count | Identical across lengths | Differs across lengths (confound) |
| Confound control | Controls for capacity; cannot claim "this is what we'd ship" | Reflects real deployed variants; cannot isolate information from capacity |

Both experiments must be run and reported together — neither alone answers
the research question cleanly, per the model-capacity confound in §5.

## 4. Fixed-capacity comparison (Experiment A)

**Mechanism**: Keep `HistoryChessResNet`'s stem (`Conv2d(in_channels=112, ...)`)
architecture and parameter count fixed at the 8-ply configuration for all three
variants. For 1-ply and 4-ply runs, zero-pad the unused history slots up to 112
channels *before* the stem, exactly as `ChessTensorEncoder.encode_window`
already zero-pads short early-game windows today (front-padding, mask-tracked).
Concretechanges needed (not applied by this document):

- Reuse `ChessTensorEncoder` with `history_length=8` unconditionally; before
  returning a window to the dataset for the 1-ply/4-ply arms, instead of
  slicing to fewer channels (as `train_ablation.py`'s `ChannelSliceDataset`
  currently does), zero out the planes for the plies the arm is not allowed to
  see, while keeping the tensor at 112 channels and the stem untouched.
- This guarantees identical parameter counts (287,110 for all three arms) and
  identical compute per forward pass, so any F1 difference is attributable
  only to *information content*, not capacity or compute.
- Mask plies must never be filled with real future information — this is a
  *shorter-history* ablation, not a data-leakage vector. The existing 2-D
  validity mask semantics already distinguish "no data available" from "data
  available"; this reuses that contract for deliberately-withheld (not
  unavailable) history.

**Risk to flag in the report**: masking-via-zeros changes the input
*distribution* the model sees (long runs of all-zero planes) relative to the
8-ply model's normal zero-padding (which only ever occurs for the first ≤7
plies of a game). This is a legitimate design tradeoff, not a hidden flaw, but
it must be stated plainly in the final write-up as Experiment A's own
limitation: it isolates capacity but introduces a (different, out-of-distribution-for-training)
input pattern Experiment B's variants never see from their own training
distribution's perspective.

## 5. Native-architecture comparison (Experiment B)

**Mechanism**: Exactly `train_ablation.py`'s current approach — each variant
gets its own `HistoryChessResNet(in_channels={14,56,112})`, so the stem
convolution's parameter count differs:

| Variant | In-channels | Parameters (measured, seed-independent) |
|---|---:|---:|
| 1-ply | 14 | 230,662 |
| 4-ply | 56 | 254,854 |
| 8-ply | 112 | 287,110 |

(Parameter counts are VERIFIED — reproduced directly from
`count_parameters(HistoryChessResNet(in_channels=N))`, see
`FINAL_MODEL_AUDIT.md` §6.)

**Model-capacity confound**: the stem conv alone contributes
`in_channels × 64 × 3 × 3` weights, so going from 1-ply to 8-ply adds
`(112-14) × 64 × 9 = 56,448` parameters before any residual block is counted —
a ~24% increase in total model size. Any F1 gain attributed to "more history"
in Experiment B could instead (or partly) be a capacity effect. This is why
Experiment A exists.

**This is the deployable configuration** — whichever variant is recommended
for production must come from Experiment B's numbers (native architecture),
with Experiment A used only to attribute *why* the effect occurs.

## 6. Controlled variables (held identical across all 18 runs)

- Game-level split: identical `GameLevelSplitter(seed=S)` output for all three
  history-length variants within a given seed `S` — i.e., exactly the same
  train/val/test game sets regardless of history length, so length is the only
  varying factor relative to a fixed seed.
- Weak-label generation: identical `RetrospectiveWeakLabeler(lookahead=8)`
  output per window regardless of the history-length arm — the label always
  looks at the full 8-ply lookahead; only the *model's input* history length
  varies. (Today's `train_ablation.py` already does this correctly — labeling
  happens once per seed, before the per-variant channel slicing.)
- Optimizer, loss, class weighting, gradient clipping, label smoothing,
  cosine schedule: identical to the primary 8-ply run's configuration
  (`AdamW`, `lr=1e-3`, `weight_decay=1e-4`, label smoothing `0.05`, grad clip
  `1.0`, inverse-frequency class weights computed from the variant's own
  *training* split).
- Training budget: identical max epochs and identical early-stopping patience
  across all 18 runs (not the ad-hoc `epochs=15` with no early stopping used
  by the existing preliminary ablation — see §13). Use the primary run's
  `early_stopping_patience=5`, `max_epochs=30`, selecting on validation
  macro-F1, exactly as `ModelTrainer.fit` already implements for the main run.
- Preprocessing: identical `min_confidence=0.55`, `max_material=800`,
  `max_per_game=25`, `max_class_fraction=0.30` filter/balance parameters
  (`filter_and_balance` / `balance_classes` defaults), applied train-only.
- Evaluation: touch the **test** split exactly once per run, at the end, after
  model selection on validation — never for early stopping or hyperparameter
  choice.

## 7. Dataset requirements

- Same `data/raw/lichess_elite_subset.pgn` (3,000 games, Elo ≥ 2200) for every
  run — no resampling or re-filtering between variants or seeds.
- Pre-encode windows once per seed (not per variant) using the existing
  memmap/pre-encoded `ChessPlanDataset` mode (`encoded_tensors`/`encoded_masks`),
  as already flagged as a to-do in
  `tasks/phase_1_core_ai/TRAINING_IMPLEMENTATION_PLAN.md` §4 Phase C ("switch
  `ChessPlanDataset` from dynamic mode to pre-encoded/memmap mode for this
  phase"), to avoid paying the PGN-labeling cost 3× (Experiment B) or 6×
  (A+B) per seed.
- Record, per seed, the exact game-id sets assigned to train/val/test (e.g. a
  JSON manifest) so Experiment A and B share identical splits and the run is
  independently reproducible.

## 8. Random seed strategy

- Seeds: `{42, 123, 999}` (matching the project's existing convention and the
  preliminary ablation already in the repo).
- Each seed controls: `GameLevelSplitter` game assignment, `balance_classes`
  downsampling/shuffling, PyTorch/NumPy/Python RNG state
  (`src.models.train.set_seed`), and `DataLoader` shuffling.
- **Limitation to state explicitly in the final report**: 3 seeds give a
  rough sense of run-to-run variance but are **not sufficient for a
  statistically rigorous significance claim** (a paired t-test or Wilcoxon
  signed-rank test over n=3 pairs has very low power; a single seed's outlier
  can flip an apparent ranking). Treat 3-seed results as *directional*
  evidence only. A credible significance claim would need on the order of
  10+ seeds per arm, which was explicitly out of scope for compute/time
  reasons at this project's scale — state this limitation in the roadmap
  rather than paper over it with an unsupported p-value.

## 9. Evaluation metrics (per run, both experiments)

- Validation and test macro-F1 (primary selection/reporting metric).
- Validation and test balanced accuracy.
- Per-class precision, recall, F1 (all 6 classes, both splits).
- Confusion matrix (test split).
- Training loss, validation loss, test loss (final/best-epoch values and full
  per-epoch curves, as `MetricsLogger` already records for the primary run).
- Training wall-clock time (data-prep separate from training, since data prep
  is currently shared/amortizable across variants within a seed).
- Inference latency: mean/P95 wall-clock per single-window forward pass on
  CPU (the project's only available compute — see
  `TRAINING_IMPLEMENTATION_PLAN.md` §1, CPU-only, no GPU).
- Peak memory: process RSS during training (CPU — there is no GPU in this
  project's environment, so "peak GPU memory" from the original audit
  request's metric list is **not applicable**; report peak CPU RSS instead
  and note the substitution).

## 10. Statistical methodology

- Within each experiment (A, B), compute the mean and standard deviation of
  test macro-F1 across the 3 seeds, per history length.
- Report paired differences (same-seed, 8-ply minus 4-ply minus 1-ply) rather
  than only comparing means across seeds, since splits/labels are seed-matched
  by design (§6).
- Do not report a p-value from a 3-sample paired test as if it were
  conclusive; report effect sizes (mean paired difference, range across the 3
  seeds) and describe the result qualitatively (consistent direction across
  all 3 seeds vs. seed-dependent sign flips) per the limitation in §8.
- If the project later accumulates more seeds, a paired Wilcoxon signed-rank
  test across seeds, per history-length-pair, is the appropriate non-parametric
  test given the small sample size.

## 11. Computational measurements

- Record wall-clock for: dataset preparation (shared per seed), training
  (per variant), and test-set evaluation (per variant).
- Record parameter counts via `count_parameters` for every variant (already
  done for the native-architecture variants, see §5 table; Experiment A's
  fixed-capacity variants should also log their — identical — count as a
  sanity check that the capacity-control actually held).
- Record CPU core count and environment (already pinned in
  `TRAINING_IMPLEMENTATION_PLAN.md` §3: 16 cores, CPU-only PyTorch 2.15).

## 12. Reproducibility requirements

- Pin and log: PyTorch version, python-chess version, scikit-learn version,
  the exact PGN file hash, the exact `config/config.yaml` used, and all nine
  (Experiment B) + nine (Experiment A) per-run seeds/variants in a single
  manifest JSON alongside the results JSON.
- Save every run's best checkpoint and per-epoch history (as the primary run
  already does via `MetricsLogger.save_history` / `save_checkpoint`), not just
  the final summary row — the existing `experiments/ablation_results.json`
  only stores a single summary row per run with no per-epoch curve, which
  prevents diagnosing *when* each variant overfits (a direct analogue of the
  overfitting investigation already done for the 8-ply primary run in
  `FINAL_MODEL_AUDIT.md` §10 — each ablation arm deserves the same).
- Document the exact commit hash the ablation was run against.

## 13. What the existing preliminary ablation can and cannot tell us

`experiments/ablation_results.json` (9 rows: 3 variants × 3 seeds, produced by
the current `src/models/train_ablation.py --epochs 15`) already exists in the
repository as of this audit. It is real, non-fabricated output — but it does
**not** satisfy this design document's bar, for concrete, checkable reasons:

1. **No fixed-capacity arm.** It only ever ran the native-architecture
   configuration (Experiment B); it cannot separate "more history helps" from
   "more parameters help."
2. **Fixed 15-epoch budget, no early stopping, no checkpoint-restore.**
   `_train_variant` in `train_ablation.py` calls `trainer.fit()` with
   `epochs=15` and never stops early or restores a best epoch's weights for
   the *reported* `final_val_macro_f1` metric — it reports `trainer.evaluate`
   on whichever weights are in memory after `fit()` returns (which *does*
   restore best-epoch weights internally per `ModelTrainer.fit`'s
   `self.model.load_state_dict(best_state)`, so `best_val_macro_f1` and
   `final_val_macro_f1` are consistent with each other in practice — but the
   budget itself, 15 fixed epochs with patience=5 default, is not matched to
   the primary run's actual stopping epoch (5) or its 30-epoch ceiling, so
   it is not a controlled comparison against the primary 8-ply result
   reported elsewhere in this project).
3. **Validation-only, no test-set numbers.** Every row reports
   `best_val_macro_f1`/`final_val_macro_f1` only; there is no test-split
   evaluation, so it cannot be compared against the primary run's reported
   test macro-F1.
4. **Only one `max_games` configuration recorded** (apparently the full
   3,000-game set, inferred from `train_ablation.py`'s default
   `max_games=None`, but the results file does not self-document this — no
   manifest of the run's exact invocation accompanies it, violating §12 of
   this design for the run that already happened).

The directional pattern in that file (4-ply mean val macro-F1 ≈ 0.557 >
8-ply ≈ 0.532 > 1-ply ≈ 0.524, per the commit message in `fd72879`) is a
legitimate, real, non-fabricated preliminary signal consistent with **H2**
(diminishing/negative returns beyond 4 plies) — but it is **validation-only,
single-protocol, capacity-confounded, and not reported alongside per-epoch
overfitting curves**, so it should be cited as *suggestive*, not as the
answer to this project's central research question. It is reasonable
evidence to prioritize the rigorous Experiment A design (§4) as the next
research task (see `RESEARCH_ROADMAP.md`), precisely because it already hints
H1 is unlikely without the capacity confound still being a live alternative
explanation.

## 14. Decision criteria

Before recommending any specific history length for production:

1. Experiment B (native) must show the candidate length beating the
   alternatives on **test** macro-F1 in at least 2 of 3 seeds, with the
   direction stated explicitly as directional (not statistically significant,
   per §8/§10).
2. Experiment A (fixed-capacity) must corroborate the same ranking — if
   Experiment A's ranking disagrees with Experiment B's, the honest
   conclusion is "model capacity, not history length, explains the
   native-architecture result," and the roadmap should prioritize capacity
   tuning over history-length changes.
3. Per-class F1 for `CHAIN_UNDERMINE` (the project's known weakest class,
   see `FINAL_MODEL_AUDIT.md` §11) must be reported per arm — a history
   length that improves macro-F1 by helping already-strong classes while
   leaving `CHAIN_UNDERMINE` at its current near-zero F1 does not represent a
   win for the project's actual stated goal of reliable class-wise
   performance.
4. No history-length change should be adopted as "the model" without a
   checkpoint file, a training-history JSON, and a test-set evaluation for
   that exact configuration existing on disk — per this project's standing
   rule against claiming a model is trained without checkpoint evidence.

**These experiments have not been executed under this design. No run has
started. No results exist for this design as written.**
