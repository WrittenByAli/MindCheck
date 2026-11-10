# Phase 2: System Integration, Streamlit Deployment & Evaluation

**Phase Duration**: 22 November 2026 – 06 December 2026  
**Primary Engineering Lead**: **Muhammad Ali** (`24I-3158`)  
**Collaborating Engineer**: **Shahzad Ahmad** (`24I-3090`)  

---

## Executive Summary
Phase 2 integrates the trained neural network and symbolic rules into an end-to-end interactive system:
1. Temporal plan state tracking across game plies with hysteresis and onset estimation.
2. Symbolic evidence fusion and prediction abstention gating.
3. Deterministic natural language explanations grounded in verified board facts.
4. Mandatory interactive Streamlit web application with SVG board visualization, ply stepper, and analytical charts.
5. Optional Stockfish engine integration for defensive counterplay analysis.
6. Scientific evaluation suite producing Macro-F1, confusion matrices, bootstrap confidence intervals, and ablation results.
7. Final academic report, presentation slides, clean-install verification, and GCR ZIP packaging.

---

## Task Inventory

| Task ID | Task Title | Primary Owner | Status |
| :--- | :--- | :---: | :---: |
| [TASK_2_1](./TASK_2_1_TEMPORAL_TRACKER_ENGINE.md) | Temporal Plan State Tracker Engine | Muhammad Ali | Ready for Claude Code |
| [TASK_2_2](./TASK_2_2_EVIDENCE_FUSION_MODULE.md) | Neural-Rule Evidence Fusion & Uncertainty Handling | Muhammad Ali / Shahzad Ahmad | Ready for Claude Code |
| [TASK_2_3](./TASK_2_3_DETERMINISTIC_EXPLANATION_GENERATOR.md) | Deterministic Natural Language Explanation Generator | Muhammad Ali | Ready for Claude Code |
| [TASK_2_4](./TASK_2_4_STREAMLIT_CORE_APPLICATION.md) | Streamlit Interactive Web Application - Core Engine | Muhammad Ali | Ready for Claude Code |
| [TASK_2_5](./TASK_2_5_STREAMLIT_BOARD_AND_EVIDENCE_VIZ.md) | Streamlit Board Visualization & Rich Analytical Views | Muhammad Ali | Ready for Claude Code |
| [TASK_2_6](./TASK_2_6_STOCKFISH_COUNTERPLAY_INTEGRATION.md) | Stockfish Defensive Counterplay Integration | Muhammad Ali | Ready for Claude Code |
| [TASK_2_7](./TASK_2_7_EVALUATION_SUITE_AND_ABLATIONS.md) | Comprehensive Evaluation Suite & Failure Analysis | Shahzad Ahmad | Ready for Claude Code |
| [TASK_2_8](./TASK_2_8_PROJECT_REPORT_AND_SLIDES.md) | Academic Project Report & Presentation Slides | Shared (Both Members) | Ready for Claude Code |
| [TASK_2_9](./TASK_2_9_FINAL_PACKAGING_AND_DEMO_CHECK.md) | Final Quality Assurance, Packaging & GCR Submission | Muhammad Ali / Shahzad Ahmad | Ready for Claude Code |

---

## Actionable Subtasks Index (for Claude Code)

| Subtask ID | Specification Document | Primary Owner | Target Deliverable |
| :---: | :--- | :---: | :--- |
| **2.1.1** | [SUBTASK_2_1_1_STATE_MACHINE_HYSTERESIS.md](subtasks/SUBTASK_2_1_1_STATE_MACHINE_HYSTERESIS.md) | Muhammad Ali | `src/tracking/state_tracker.py` |
| **2.1.2** | [SUBTASK_2_1_2_PLAN_ONSET_BACKTRACKING.md](subtasks/SUBTASK_2_1_2_PLAN_ONSET_BACKTRACKING.md) | Muhammad Ali | Onset backtracking logic |
| **2.1.3** | [SUBTASK_2_1_3_PROGRESS_VS_THREAT_METRIC.md](subtasks/SUBTASK_2_1_3_PROGRESS_VS_THREAT_METRIC.md) | Muhammad Ali | Progress & threat severity metrics |
| **2.1.4** | [SUBTASK_2_1_4_STATE_TRACKER_SIMULATION_TESTS.md](subtasks/SUBTASK_2_1_4_STATE_TRACKER_SIMULATION_TESTS.md) | Muhammad Ali | `tests/test_state_tracker.py` |
| **2.2.1** | [SUBTASK_2_2_1_SCORE_CALIBRATION_NORMALIZATION.md](subtasks/SUBTASK_2_2_1_SCORE_CALIBRATION_NORMALIZATION.md) | Shahzad Ahmad | Temperature scaling & normalization |
| **2.2.2** | [SUBTASK_2_2_2_WEIGHTED_SCORE_FUSION.md](subtasks/SUBTASK_2_2_2_WEIGHTED_SCORE_FUSION.md) | Muhammad Ali | `src/tracking/fusion.py` |
| **2.2.3** | [SUBTASK_2_2_3_UNCERTAINTY_ABSTENTION_GATING.md](subtasks/SUBTASK_2_2_3_UNCERTAINTY_ABSTENTION_GATING.md) | Muhammad Ali | Margin & confidence gates |
| **2.3.1** | [SUBTASK_2_3_1_CONTROLLED_EXPLANATION_TEMPLATES.md](subtasks/SUBTASK_2_3_1_CONTROLLED_EXPLANATION_TEMPLATES.md) | Muhammad Ali | `src/tracking/explanations.py` |
| **2.3.2** | [SUBTASK_2_3_2_STRATEGIC_RATIONALE_FORMATTER.md](subtasks/SUBTASK_2_3_2_STRATEGIC_RATIONALE_FORMATTER.md) | Muhammad Ali | Grounded pedagogical bullets |
| **2.3.3** | [SUBTASK_2_3_3_STRUCTURED_CLAIM_VERIFIER.md](subtasks/SUBTASK_2_3_3_STRUCTURED_CLAIM_VERIFIER.md) | Muhammad Ali | Zero-hallucination verifier |
| **2.4.1** | [SUBTASK_2_4_1_APP_STRUCTURE_SESSION_STATE.md](subtasks/SUBTASK_2_4_1_APP_STRUCTURE_SESSION_STATE.md) | Muhammad Ali | `src/ui/app.py` shell & state |
| **2.4.2** | [SUBTASK_2_4_2_PGN_LOADER_SAMPLE_LIBRARY.md](subtasks/SUBTASK_2_4_2_PGN_LOADER_SAMPLE_LIBRARY.md) | Muhammad Ali | `src/ui/game_loader.py` |
| **2.4.3** | [SUBTASK_2_4_3_MOVE_STEPPER_NAVIGATOR.md](subtasks/SUBTASK_2_4_3_MOVE_STEPPER_NAVIGATOR.md) | Muhammad Ali | `src/ui/navigator.py` stepper |
| **2.4.4** | [SUBTASK_2_4_4_INFERENCE_RUNNER_CACHING.md](subtasks/SUBTASK_2_4_4_INFERENCE_RUNNER_CACHING.md) | Muhammad Ali | `src/ui/inference_runner.py` |
| **2.5.1** | [SUBTASK_2_5_1_SVG_CHESSBOARD_OVERLAYS.md](subtasks/SUBTASK_2_5_1_SVG_CHESSBOARD_OVERLAYS.md) | Muhammad Ali | `src/ui/board_view.py` SVG |
| **2.5.2** | [SUBTASK_2_5_2_PROBABILITY_BAR_CHARTS.md](subtasks/SUBTASK_2_5_2_PROBABILITY_BAR_CHARTS.md) | Muhammad Ali | `src/ui/charts.py` probability view |
| **2.5.3** | [SUBTASK_2_5_3_TEMPORAL_PLAN_TIMELINE.md](subtasks/SUBTASK_2_5_3_TEMPORAL_PLAN_TIMELINE.md) | Muhammad Ali | `src/ui/timeline.py` whole game chart |
| **2.5.4** | [SUBTASK_2_5_4_GROUNDED_EVIDENCE_ACCORDION.md](subtasks/SUBTASK_2_5_4_GROUNDED_EVIDENCE_ACCORDION.md) | Muhammad Ali | `src/ui/evidence_view.py` |
| **2.6.1** | [SUBTASK_2_6_1_STOCKFISH_UCI_CLIENT.md](subtasks/SUBTASK_2_6_1_STOCKFISH_UCI_CLIENT.md) | Muhammad Ali | `src/utils/stockfish_client.py` |
| **2.6.2** | [SUBTASK_2_6_2_DEFENSIVE_COUNTERPLAY_SCREENING.md](subtasks/SUBTASK_2_6_2_DEFENSIVE_COUNTERPLAY_SCREENING.md) | Muhammad Ali | Screened counterplay annotations |
| **2.6.3** | [SUBTASK_2_6_3_THREAT_DELTA_SIMULATION.md](subtasks/SUBTASK_2_6_3_THREAT_DELTA_SIMULATION.md) | Muhammad Ali | Threat delta calculation |
| **2.7.1** | [SUBTASK_2_7_1_MULTI_MODEL_BENCHMARK_RUNNER.md](subtasks/SUBTASK_2_7_1_MULTI_MODEL_BENCHMARK_RUNNER.md) | Shahzad Ahmad | `src/models/evaluate.py` |
| **2.7.2** | [SUBTASK_2_7_2_METRICS_CONFUSION_MATRICES.md](subtasks/SUBTASK_2_7_2_METRICS_CONFUSION_MATRICES.md) | Shahzad Ahmad | `src/models/metrics.py` & heatmaps |
| **2.7.3** | [SUBTASK_2_7_3_GAME_CLUSTER_BOOTSTRAP_CI.md](subtasks/SUBTASK_2_7_3_GAME_CLUSTER_BOOTSTRAP_CI.md) | Shahzad Ahmad | `src/models/bootstrap.py` (1000 iter) |
| **2.7.4** | [SUBTASK_2_7_4_QUALITATIVE_FAILURE_ANALYSIS.md](subtasks/SUBTASK_2_7_4_QUALITATIVE_FAILURE_ANALYSIS.md) | Shahzad Ahmad | `docs/failure_analysis.md` (10 cases) |
| **2.8.1** | [SUBTASK_2_8_1_ACADEMIC_REPORT_AUTHORING.md](subtasks/SUBTASK_2_8_1_ACADEMIC_REPORT_AUTHORING.md) | Shared (Both) | `docs/report.pdf` / `REPORT.md` |
| **2.8.2** | [SUBTASK_2_8_2_SYSTEM_PIPELINE_DIAGRAM.md](subtasks/SUBTASK_2_8_2_SYSTEM_PIPELINE_DIAGRAM.md) | Muhammad Ali | `docs/figures/system_pipeline.png` |
| **2.8.3** | [SUBTASK_2_8_3_PRESENTATION_SLIDE_DECK.md](subtasks/SUBTASK_2_8_3_PRESENTATION_SLIDE_DECK.md) | Shared (Both) | `docs/slides.pdf` |
| **2.9.1** | [SUBTASK_2_9_1_CLEAN_ENVIRONMENT_VERIFIER.md](subtasks/SUBTASK_2_9_1_CLEAN_ENVIRONMENT_VERIFIER.md) | Muhammad Ali | `scripts/verify_install.py` & bat |
| **2.9.2** | [SUBTASK_2_9_2_LIVE_DEMO_REHEARSAL_CHECKLIST.md](subtasks/SUBTASK_2_9_2_LIVE_DEMO_REHEARSAL_CHECKLIST.md) | Shared (Both) | Rehearsal script & timed cues |
| **2.9.3** | [SUBTASK_2_9_3_SINGLE_GCR_SUBMISSION_ZIP.md](subtasks/SUBTASK_2_9_3_SINGLE_GCR_SUBMISSION_ZIP.md) | Muhammad Ali | `scripts/package_submission.py` |

---

## Phase 2 Deliverables & Acceptance Checklist
- [ ] Working Streamlit application launching via `streamlit run src/ui/app.py`.
- [ ] End-to-end usable flow demonstrated: PGN upload/selection $\to$ move navigation $\to$ model execution $\to$ board & plan display.
- [ ] Multi-model live comparison (History ResNet vs Single-Board CNN vs Hybrid).
- [ ] Full evaluation results logged with confusion matrix plot and bootstrap confidence intervals.
- [ ] Final Project Report (PDF) and Presentation Slides (PDF) compiled and placed in `docs/`.
- [ ] Clean installation tested in fresh Python virtual environment.
- [ ] Single consolidated ZIP file created for GCR submission.
