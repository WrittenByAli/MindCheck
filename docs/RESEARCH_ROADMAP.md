# MindCheck — Research Roadmap

> Companion to `FINAL_MODEL_AUDIT.md` (what exists and what it's worth) and
> `CONTEXT_LENGTH_ABLATION.md` (the one proposed experiment detailed in full).
> This document prioritizes *all* remaining research tasks and states the
> single recommended next experiment. Nothing in this document has been
> executed as described here unless explicitly marked "completed."

## 1. Current project maturity

MindCheck has a complete, working, non-fabricated software pipeline: PGN
ingestion → windowing → symbolic fact extraction → rule engine → retrospective
weak labeling → game-level split → tensor encoding → `HistoryChessResNet`
training → checkpoint export (ONNX + TorchScript). One full training run exists
on real data (3,000 Lichess Elite games, Elo ≥ 2200) with a real saved
checkpoint, a real per-epoch training history, and a real (if differently-
scoped) history-length ablation. Independent-annotation *infrastructure*
exists (sampler, guidelines, adjudication script) but has collected **zero**
actual human labels.

This is a working **Task A** system (predict a rule-engine-derived weak label
from historical board information) with essentially **no evidence yet** for
**Task B** (recognize an actual human strategic intention) — see
`FINAL_MODEL_AUDIT.md` §12 for the full Task A/B distinction. That gap is the
single largest scientific uncertainty in the project.

## 2. Verified completed work

- Data pipeline (PGN parsing → windowing → weak labeling → game-level split →
  tensor encoding) is implemented, unit-tested, and structurally leak-free by
  construction (verified by code inspection, see audit §4–§5).
- `HistoryChessResNet` (3 residual blocks, 64 channels, 287,110 parameters)
  is implemented, tested for shape/gradient/parameter-budget invariants, and
  has one real trained checkpoint (`models/checkpoints/best_resnet.pt`,
  best epoch 5, val macro-F1 0.5320, VERIFIED directly from the checkpoint
  file).
- Training produced a real per-epoch history
  (`experiments/training_history.json`, 10 epochs, early-stopped at epoch 10
  per patience-5 on no improvement since epoch 5) — VERIFIED.
- A preliminary (not rigorous — see `CONTEXT_LENGTH_ABLATION.md` §13)
  1/4/8-ply × 3-seed ablation exists (`experiments/ablation_results.json`,
  9 rows, real non-fabricated run output) — VERIFIED as an artifact, but
  capacity-confounded and validation-only.
- A Phase B neural-vs-rule-baseline comparison script
  (`scripts/phase_b_baseline_comparison.py`) exists and is runnable; its
  test-set numbers are reproduced in this audit cycle (see
  `FINAL_MODEL_AUDIT.md` §8 for the reproduced figures) rather than only
  quoted from an unsaved prior console run.
- Independent human-annotation tooling (sampler, `ANNOTATION_GUIDELINES.md`,
  `adjudicate.py`, `adjudicate_cli.py`) exists and is wired to
  `data/annotations/raw_annotations_member{1,2}.csv` — but those two files
  are empty 250-row templates (VERIFIED: 0/250 rows filled in either file).
  No `test_ground_truth.csv` exists.

## 3. Remaining scientific uncertainties

1. Does the neural model actually outperform the rule-only baseline by a
   margin that isn't noise? (Phase B was designed but its numbers were never
   persisted to a file — this audit reproduces them; see
   `FINAL_MODEL_AUDIT.md` §8.)
2. Is 8-ply history actually better than fewer plies, net of the model-
   capacity confound? (Unresolved — `CONTEXT_LENGTH_ABLATION.md`.)
3. Why does `CHAIN_UNDERMINE` F1 collapse to ~0.10 when every other class
   reaches 0.53–0.71? (Partially diagnosed — see audit §11; several plausible
   contributing factors identified, none yet isolated by a dedicated
   experiment.)
4. Does the model generalize to human-intended plans, or only to the rule
   engine's own output (which supplied its training labels)? Zero human-
   annotated evidence currently exists to answer this.
5. Is the model calibrated (do its confidence scores mean anything), and is
   the fusion/abstention layer's `abstention_threshold=0.40` /
   `margin_threshold=0.10` (from `config/config.yaml`) justified by any
   calibration study? No calibration artifact exists.
6. Does performance hold up across different players/rating bands/game
   phases, or is validation/test macro-F1 dominated by a few over-represented
   games/openings? Not yet measured.

## 4. Prioritized research tasks

Scored qualitatively (High/Medium/Low) on scientific value, expected impact,
implementation complexity, and compute cost; dependencies noted.

| # | Task | Scientific value | Impact | Complexity | Compute cost | Dependencies |
|---|---|---|---|---|---|---|
| 1 | Human annotation benchmark (collect the 250 real labels) | **High** | **High** | Low (tooling exists) | None (human time, not compute) | None — the sampler, CSVs, and adjudication code already exist and are untested only for lack of actual annotation effort |
2 | `CHAIN_UNDERMINE` root-cause isolation experiments | **High** | Medium-High | Medium | Low (CPU minutes–hours) | Needs dataset_summary / confusion-matrix instrumentation (partially done in this audit cycle) |
3 | Controlled context-length ablation (fixed + native capacity) | High | Medium | Medium-High | Medium (18 training runs, ~CPU hours, per `TRAINING_IMPLEMENTATION_PLAN.md`'s ~45 min/run baseline) | Needs pre-encoded/memmap dataset mode wired for ablation reuse |
4 | Weak-label quality / rule-engine audit beyond CHAIN_UNDERMINE | Medium-High | Medium | Low-Medium | Low | None |
5 | Calibration analysis (reliability diagrams, abstention coverage curve) | Medium | Medium (directly informs the shipped fusion/abstention thresholds) | Low | Low | Needs a saved per-sample prediction + probability dump from the test split |
6 | Cross-game / cross-player generalization breakdown | Medium | Medium | Low | Low | Needs per-game / per-player metadata already captured in `LabeledPositionRecord` |
7 | Robustness evaluation (incomplete history, out-of-distribution positions) | Medium | Low-Medium | Medium | Low-Medium | None |
8 | Architecture improvements (beyond fixing the 3-vs-4-block doc bug) | Low-Medium | Uncertain | Medium-High | Medium-High | Should follow, not precede, tasks 1–3 — premature architecture tuning without knowing whether the bottleneck is labels, history length, or capacity risks optimizing the wrong thing |
9 | Inference optimization (latency/footprint) | Low | Low | Low | Low | Not urgent — the project already exports ONNX/TorchScript and is CPU-only by design; no deployment latency problem has been reported |

## 5. Human annotation benchmark design

Infrastructure already exists and should be used as-is rather than redesigned
(it already satisfies nearly every requirement the original audit asked this
section to check for):

- **Representative sampling**: `src/data/sample_test_set.py` (not read in
  full detail in this audit pass, but referenced by
  `docs/07_INDEPENDENT_EVALUATION_PROTOCOL.md` §1.1) draws 250 windows from
  the held-out **test** split only, stratified by game phase (plies 15–25 /
  26–55 / 56+) and by rating band (2200–2600+).
- **Balanced coverage**: stratification is by phase/rating, not by weak-label
  class — this is a deliberate, defensible choice (the whole point is to
  avoid letting the rule engine's own label distribution bias which positions
  get shown to humans), but it means some classes may end up under-sampled in
  the 250-row benchmark. **Action item**: before annotation starts, run a
  class-count check against the weak labels assigned to the sampled 250
  (not used for annotation, only for post-hoc stratification awareness) and
  flag if any of the 6 classes has fewer than ~15 likely-representative rows.
- **Guidelines**: `data/annotations/ANNOTATION_GUIDELINES.md` — history-only,
  independence, single primary label + confidence + secondary label + notes.
  Already matches `docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md`'s
  definitions.
- **Independent annotation**: two files
  (`raw_annotations_member{1,2}.csv`), one per annotator, no shared visibility
  — already structured for this; currently both are empty templates
  (0/250 filled).
- **Agreement measurement**: Cohen's Kappa, implemented in
  `src/models/adjudicate.py` (not line-audited in this pass, but its CLI
  entry point and the κ formula are specified in
  `docs/07_INDEPENDENT_EVALUATION_PROTOCOL.md` §2). Standard caveats apply —
  Cohen's κ corrects for chance agreement under the observed marginal label
  distributions, but (a) it treats all 6 classes as equally confusable a
  priori, which is not true (`NO_PLAN` vs. a committed plan is an easier
  distinction than `K_STORM` vs. `Q_EXP` on an ambiguous flank), and (b) with
  only 2 annotators and 250 items, the κ point estimate itself has non-trivial
  sampling uncertainty that should be reported with a confidence interval,
  not as a bare number.
- **Adjudication**: `scripts/adjudicate_cli.py` + the documented tie-breaking
  rule (disagreements default to `NO_PLAN` if genuinely unresolved) — already
  specified in both `docs/07` and `ANNOTATION_GUIDELINES.md` §3.
- **Held-out separation**: sampling is explicitly restricted to the test
  split's games, which were never used for training or validation-based model
  selection — structurally sound by construction, given the game-level split
  is leak-free (audit §4–§5).
- **Evaluation against human judgment**: once `test_ground_truth.csv` exists,
  compare M1 (neural), M2 (1-ply ablation), M3 (rule engine), and M4 (fusion)
  against it — this is exactly `docs/07`'s existing M1–M4 benchmark design,
  just waiting on real annotations to run against.

**The only missing step is doing the annotation.** This is why it is ranked
priority #1 in §4: the tooling cost is sunk, the remaining cost is two
people's time, and it is the only available evidence for Task B (human-intent
recognition) as opposed to Task A (rule-label reproduction) anywhere in this
project.

## 6. Proposed milestones

1. **M1 — Human benchmark collection** (no new code; 2 people annotate 250
   rows each, independently). Deliverable: filled
   `raw_annotations_member{1,2}.csv`, Cohen's κ report, adjudicated
   `test_ground_truth.csv`.
2. **M2 — CHAIN_UNDERMINE deep-dive** (confusion matrix + label-confidence
   distribution + rule-vs-model agreement on this class specifically, using
   the instrumentation partially built in this audit cycle). Deliverable: a
   short findings memo distinguishing confirmed/likely/hypothesis causes
   (audit §11 is the starting point, not the final word).
3. **M3 — Context-length ablation, rigorous** (per
   `CONTEXT_LENGTH_ABLATION.md`, both Experiment A and B, 18 runs).
   Deliverable: a decision on whether 8-ply, 4-ply, or neither should be the
   shipped configuration, with the capacity confound explicitly resolved.
4. **M4 — Calibration + abstention validation**. Deliverable: a reliability
   diagram and coverage-vs-accuracy curve for the fusion/abstention gate's
   existing `alpha=0.70`, `abstention_threshold=0.40`, `margin_threshold=0.10`
   (`config/config.yaml`), confirming or revising those hand-set values with
   evidence.
5. **M5 — Human-benchmark evaluation**. Deliverable: M1–M4 system comparison
   against the frozen `test_ground_truth.csv` from M1 — the first real
   evidence of Task B performance.

Milestones M2 and M3 can run in parallel with M1 (M1 has no compute
dependency and is gated only by human time). M4 depends on M3's chosen
configuration being final. M5 depends on M1's ground truth existing.

## 7. Dependencies

- M1 depends on nothing (tooling exists); gated by annotator availability.
- M2 depends on being able to re-run `prepare_datasets` once to get labeled
  per-class test-set predictions and confidences (CPU time only).
- M3 depends on the pre-encoded/memmap dataset path being wired for the
  ablation script (currently dynamic-encoding-only is used in
  `train_ablation.py`, which is correct but slower — not a correctness
  blocker, a throughput one).
- M4 depends on M3's chosen final configuration (re-running calibration on a
  configuration that's about to be replaced wastes the human-readable
  reliability diagram).
- M5 depends on M1.

## 8. Computational requirements

All work is CPU-only (per `TRAINING_IMPLEMENTATION_PLAN.md` §1 — no GPU
available, none required at this project's scale). Using that document's own
measured rates (586.7 ms/game data prep, ~27s/epoch training on the full
3,000-game set):

- M2: effectively free — reuses already-labeled data; at most one more
  `prepare_datasets` pass (~30 min) if a fresh run is needed instead of
  reusing this audit cycle's artifacts.
- M3: 18 runs × ~45 min/run worst case (native 8-ply scale) ≈ up to 13.5
  hours of CPU time if run sequentially; trivially parallelizable across the
  16 available cores by running several variant/seed combinations
  concurrently, since each run is independent and CPU-only.
- M4: free (post-hoc analysis of existing test-set predictions).
- M1/M5: no project compute cost.

## 9. Completion criteria

- **M1 complete** when both CSVs are 250/250 filled, κ is computed and
  reported with its interpretation band (per `docs/07`'s thresholds), and
  `test_ground_truth.csv` exists with adjudicated labels for every row.
- **M2 complete** when the `CHAIN_UNDERMINE` findings memo distinguishes, with
  evidence, how much of the ~0.10 F1 is attributable to (a) class rarity,
  (b) weak-label noise, (c) the rule definition's narrow all-or-nothing
  scoring (audit §11's leading hypothesis), (d) feature/architecture
  limitations, and (e) genuine semantic overlap with other classes.
- **M3 complete** when both Experiment A and B have full per-epoch histories,
  test-set metrics, and a stated decision (which history length, if any,
  should ship) that explicitly addresses the capacity confound.
- **M4 complete** when a calibration report exists showing whether the
  current `abstention_threshold`/`margin_threshold` values are empirically
  justified or need revision.
- **M5 complete** when M1–M4 systems are compared against real human labels
  and the gap (if any) between rule-label performance and human-label
  performance is reported honestly, including the possibility that it's
  large.

## 10. Recommended next experiment

**Start the human annotation benchmark (M1) immediately, in parallel with the
`CHAIN_UNDERMINE` deep-dive (M2).**

Justification:

1. M1 has zero compute cost and zero code-risk (the infrastructure is
   already built and tested) — it is purely a matter of two people spending
   their already-budgeted annotation time (`TRAINING_IMPLEMENTATION_PLAN.md`
   Phase E is already on the plan, just "not yet started").
2. It resolves the single largest scientific-credibility gap in the project:
   every number reported so far (training accuracy, validation/test macro-F1,
   the rule-vs-neural comparison) measures agreement with the *rule engine's
   own labels*, not with human judgment. Per this project's own research
   integrity goal, a system that has never been checked against an
   independent human ground truth cannot yet claim to recognize "strategic
   plans" in any sense beyond "reproduce this specific symbolic rule engine."
3. It is a strict prerequisite for M5, which is the only milestone that can
   answer Task B (human-intent recognition) rather than Task A (rule-label
   reproduction) — see audit §12. Every other task on this roadmap
   (context-length ablation, CHAIN_UNDERMINE fixes, calibration) only ever
   improves Task A performance until M5 exists to check whether Task A
   improvements transfer to Task B at all.
4. Unlike M3 (context-length ablation), it requires no new code and carries
   no risk of silently introducing a new confound — it is the lowest-risk,
   highest-information task available.

The context-length ablation (M3) remains the right **second** priority once
M1 is underway, because the existing preliminary ablation already shows a
directional, if confounded, signal (§3 and `CONTEXT_LENGTH_ABLATION.md` §13)
that the project's current 8-ply default may not be optimal — this is cheap
to resolve rigorously and currently rests on compute-confounded evidence.
