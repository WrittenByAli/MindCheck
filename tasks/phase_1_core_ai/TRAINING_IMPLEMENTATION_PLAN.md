# Training Implementation Plan (Locked In)

**Scope**: Execution runbook for running the already-built Task 1.7/1.8/1.9 code
on the real dataset. This is not a new task spec — the code it runs already
exists, tested, and merged on `main` (commit `1168032`). This document fixes
the exact tools, sequence, commands, and success criteria so the run is
unambiguous when started.

**Status at lock-in**: All prerequisites verified. Not yet started.

---

## 1. Tooling — locked in

| Need | Tool | Rationale |
|---|---|---|
| Model + training loop | PyTorch 2.15 (CPU) | Already built (`ModelTrainer`, `HistoryChessResNet`); no GPU available, none needed (see §2) |
| Metrics | scikit-learn | Already wired into `evaluate()` (macro-F1, accuracy, per-class F1) |
| Tensor/array ops | NumPy | Already in `tensor_encoder.py` |
| Chess logic | python-chess | Backbone of the whole pipeline |
| Logging / learning curves | Custom `MetricsLogger` (JSON + matplotlib) | Already built; sufficient for this project's scope |
| Export for inference | ONNX + TorchScript | Already built with output-parity checks (Task 1.7.4) |
| Progress visibility | `tqdm` (new) | Only new dependency — see §1.1 |

**Explicitly rejected, with reasons:**
- **GPU / cloud compute** — measured total runtime is ~45 minutes on CPU; not justified.
- **Weights & Biases / MLflow** — single contributor, local runs, a few experiments. Existing JSON+matplotlib logger covers reporting needs.
- **Optuna / Ray Tune (automated HPO)** — hyperparameters were already deliberately chosen and documented (Task 1.7/1.8 commits). Tune manually only if the baseline run shows a specific, diagnosed problem.

### 1.1 `tqdm` addition

`train.py`'s `prepare_datasets` and `train_epoch` currently print nothing during
their ~30 min and ~27s/epoch loops respectively. Add `tqdm` progress bars to:
- the per-game loop in `prepare_datasets` (`src/models/train.py`)
- the per-batch loop in `train_epoch` (`src/models/train.py`)

Single new dependency (`pip install tqdm`), zero architectural impact, additive only.

---

## 2. Measured performance baseline (not estimates — timed on real data)

| Stage | Measured rate | Full-dataset projection (3000 games) |
|---|---|---|
| Data prep (`prepare_datasets`) | 586.7 ms/game | ~29–30 min |
| Training | 0.9s/epoch per ~1,742 windows | ~27s/epoch → ~13.5 min for 30 epochs |
| **Total** | | **~45 minutes** |

Smoke test (100 games, 2 epochs) produced: 1467 train / 275 val / 350 test
windows, val macro-F1 0.1889 — confirms the pipeline runs end-to-end on real
data; not a result to draw conclusions from (too few games/epochs).

---

## 3. Pre-flight checklist (all verified complete)

- [x] Real PGN present: `data/raw/lichess_elite_subset.pgn` (3000 games, Elo 2301–3099, zero below 2200 floor, no `Variant` tags)
- [x] `config/config.yaml` → `dataset.raw_pgn_path` points at the real file
- [x] All Task 1.6–1.9 code merged on `main` (PR #4), 156/156 tests passing
- [x] `balance_classes` runs train-split-only (PR #4 fix) — val/test keep natural class distribution
- [x] CPU-only confirmed (`torch.cuda.is_available() == False`), 16 cores, 76 GB free disk
- [x] `models/checkpoints/` writable

---

## 4. Execution phases

### Phase A — Baseline training run (Task 1.8.1/1.8.2 deliverable)

```bash
python -m src.models.train --config config/config.yaml --epochs 30 --batch_size 64
```

- Full 3000-game dataset, dynamic tensor encoding (encode-on-access, not pre-cached)
- Class-weighted (inverse frequency, train-split-only) + label-smoothed (0.05) cross-entropy
- AdamW (`lr=0.001`, `weight_decay=1e-4`) + cosine annealing (`T_max=30`, `eta_min=1e-5`)
- Gradient clipping (`max_grad_norm=1.0`), early stopping (patience 5) on val macro-F1
- Seed fixed at 42 (`set_seed`)

**Expected duration**: ~45 minutes.

**Outputs**:
- `models/checkpoints/best_resnet.pt` (model + optimizer state + epoch + per-class F1 metadata)
- `experiments/training_history.json` (per-epoch loss/accuracy/macro-F1/per-class-F1/lr)

**Success criterion**: a real checkpoint exists, val macro-F1 is logged per epoch,
and best-epoch weights (not necessarily the last epoch) are restored at the end.

**Failure handling**: if the run crashes mid-way, no partial checkpoint is lost —
`save_checkpoint` only fires when val macro-F1 improves, so the last-saved
checkpoint before a crash is always a genuine local optimum, not a half-written file.

---

### Phase B — Sanity check against the rule-only baseline

Compare the trained model's test-set macro-F1 against the **rule-only classifier**
(Task 1.4.6, `SymbolicRuleEngine` run standalone with no learned weights) on the
exact same test split.

**Why this is non-negotiable**: the neural model's training labels came from
this same rule engine's output. If the neural model does not beat the rule-only
baseline, it has learned nothing beyond memorizing its own label source — the
single biggest risk flagged earlier in this project's review.

**Success criterion**: neural macro-F1 > rule-only macro-F1 on test, by a
margin large enough to not be noise (report both numbers, don't round away the gap).

---

### Phase C — History-length ablation (Task 1.8.3)

```bash
python -m src.models.train_ablation
```

- Compares 1-ply / 4-ply / 8-ply history on matched splits/seed
- Output: `experiments/ablation_results.json`

**Before running**: switch `ChessPlanDataset` from dynamic mode to its existing
pre-encoded/memmap mode for this phase only. Rationale: this phase trains 3
times on the same underlying windows — paying the tensor-encoding cost 3×
over is pure waste once the windows are already known from Phase A.

**Success criterion**: a monotonic (or explainable non-monotonic) relationship
between history length and macro-F1, reported with per-length numbers — this
is the evidence for why history-awareness matters, the project's core thesis.

---

### Phase D — Export the final checkpoint (Task 1.7.4, already built)

```python
from src.models.export import export_to_onnx, export_to_torchscript
```

Run against the real checkpoint produced in Phase A (once it exists). Both
exporters already verify output parity against the live PyTorch model
(`atol=1e-4` ONNX, `atol=1e-5` TorchScript) — rerun this check on the real
checkpoint, don't assume the synthetic-checkpoint parity test generalizes.

**Success criterion**: `models/checkpoints/*.onnx` / `*.pt` (TorchScript) exist
and pass the parity check against the real trained model.

---

### Phase E — Independent human validation (Task 1.9.3/1.9.4, separate manual track)

Not blocking Phases A–D, but the actual ground-truth check:
1. Shahzad and Ali independently fill in `data/annotations/raw_annotations_member{1,2}.csv` (250 rows each)
2. Run `python -m src.models.adjudicate` → Cohen's Kappa + disagreement list
3. Resolve disputes via `scripts/adjudicate_cli.py` → frozen `test_ground_truth.csv`
4. Compare the trained model's predictions against this frozen ground truth — the real, human-validated accuracy number, independent of the weak labels used in training.

---

## 5. Execution order

```
Phase A (baseline, ~45 min)
   │
   ▼
Phase B (compare to rule-only baseline, minutes)
   │
   ▼
Phase C (ablation — pre-encode first, then run)
   │
   ▼
Phase D (export real checkpoint, seconds)
   │
   ▼
Phase E (manual annotation — independent timeline, whenever ready)
```

## 6. Sign-off

This plan is locked in as of 2026-11-18. No further design decisions are
pending — the next action is to start Phase A.
