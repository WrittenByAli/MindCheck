# CLAUDE.md: MindCheck

> Full original spec (taxonomy preconditions, pipeline diagram, team table, all commands) preserved at `docs/PROJECT_REFERENCE_FULL.md` — read it when you need detail beyond this file.

## Project
MindCheck: history-aware chess strategic plan recognition. 8-ply window -> `(112,8,8)` tensor -> PyTorch residual CNN -> 6-class plan (`K_STORM`, `Q_EXP`, `OUTPOST`, `ROOK_FILE`, `CHAIN_UNDERMINE`, `NO_PLAN`) -> rule/neural fusion -> temporal tracker -> Streamlit UI. AI2002, Fall 2026, Group 29. Path A (software-only).

## Ownership (see `docs/PROJECT_REFERENCE_FULL.md` §3 for full file list)
- **Shahzad Ahmad**: data pipeline, rules, models/training.
- **Muhammad Ali**: UI, temporal tracking, fusion, explanations, deployment.
- Shared: `data/annotations/`, `docs/`.

## Mandatory technical constraints
- Tensor shape `(batch, 112, 8, 8)` float32; attacker-perspective orientation (flip + swap planes if Black to move).
- Game-level train/val/test split only — never leak a `game_id` across splits.
- Weak labels from future-ply lookahead must never enter the model's input tensor.
- Plan hysteresis: 2 plies to activate/switch, 3 ambiguous plies to decay to `NO_PLAN`.
- Explanations are deterministic templates over verified bitboard facts — never speculative intent.
- Never mock/fabricate model predictions, metrics, or checkpoints; never let rule-engine output silently replace the neural classifier.

## Skill discovery
Load the matching skill before non-trivial work in its area — don't load skills outside the task at hand:
- `mindcheck-ml-pipeline` — dataset processing, tensor encoding, training, evaluation.
- `mindcheck-chess-engineering` — python-chess, bitboards, strategic predicates, temporal tracking.
- `mindcheck-streamlit-qa` — UI, model integration, end-to-end testing.

## Key commands
See `docs/PROJECT_REFERENCE_FULL.md` §5 for the full list. Most used:
```bash
pytest tests/ -v
python scripts/check_env.py
python -m src.models.train --config config/config.yaml --epochs 30 --batch_size 64
streamlit run src/ui/app.py
```

## Testing requirement
Any subtask is complete only when its tests pass: `pytest tests/test_<module>.py -v` at 100% pass rate, with real (non-fabricated) output as evidence.

## Git safety
Never commit without being asked. Never push, force-push, or amend published commits without explicit instruction.

## Task execution
1. Read the subtask spec in `tasks/phase_*/subtasks/SUBTASK_X_Y_Z_*.md`.
2. Check parent task + `docs/04_WORK_DISTRIBUTION_AND_OWNERSHIP.md` / `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md`.
3. Implement, typed and documented; add tests under `tests/`; run them before marking done.
