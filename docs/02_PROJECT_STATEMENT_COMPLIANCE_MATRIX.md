# MindCheck: Project Statement & Course Compliance Matrix

This document provides a clause-by-clause audit demonstrating how **MindCheck** complies with every single mandatory directive in the **AI2002 Artificial Intelligence Semester Project Statement (Fall 2026)** and aligns with the instructor's **Project Recommendations**.

---

## 1. Path A Mandatory Requirements Compliance Audit

| Requirement from Project Statement | Course Directive | MindCheck Compliance Evidence | Implementation Component |
| :--- | :--- | :--- | :--- |
| **Path Selection** | Must select exactly ONE project path (Path A or Path B). Absolutely NO combination. | MindCheck is strictly **Path A (Software-Based)**. Zero physical sensors or hardware actuators. | Section 2 of Proposal & System Architecture |
| **Group Size** | Exactly 2-3 members per group. Individual projects NOT allowed. | Group 29 consists of exactly 2 members: **Muhammad Ali (24I-3158)** and **Shahzad Ahmad (24I-3090)**. | Proposal Title Page & Section 13 |
| **Core AI/ML Dependency** | Core functionality MUST depend on a Machine Learning algorithm / Neural Network. NOT merely a conventional app, CRUD, or rule-based system with an AI label. | Core inference is performed by a custom **PyTorch Deep Residual Convolutional Neural Network (ResNet)** that classifies 8-ply spatial-temporal chess tensors into strategic plan classes. | `src/models/residual_cnn.py`, `src/models/inference.py` |
| **Algorithm Justification** | Must justify why the ML algorithm is appropriate, what inputs it requires, what outputs it produces, and how it supports a real-world task. | Chess board coordinates form a natural $8 \times 8$ spatial grid where local piece convolutions capture geometric relationships. Temporal stacking of 8 half-moves allows the CNN to learn maneuver trajectories and plan velocity across time. Output is a 6-class probability distribution directly supporting game review. | `docs/05_ARCHITECTURE_AND_PIPELINE_DESIGN.md` |
| **Dataset Documentation** | A suitable dataset MUST be identified and documented (source, features, target/output, preprocessing, limitations). | Identified source: **Lichess Elite Database** (public PGN archive). Documented features: 8-ply $8 \times 8$ board planes + metadata. Target: 6 strategic classes. Documented limitations: weak-label noise, opening repetition, master vs amateur divergence. | `docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md`, `tasks/phase_1_core_ai/TASK_1_2_DATASET_ACQUISITION_AND_PGN_PARSING.md` |
| **Rigorous ML Evaluation** | The ML pipeline MUST be evaluated using appropriate performance metrics. Results must be reported and interpreted. | Rigorous evaluation suite measuring **Macro-F1, per-class Precision/Recall/F1, confusion matrices, game-cluster bootstrap confidence intervals, and history ablation (1-ply vs 8-ply)**. Evaluated against held-out independently reviewed test positions. | `tasks/phase_2_deployment_and_evaluation/TASK_2_7_EVALUATION_SUITE_AND_ABLATIONS.md` |
| **Mandatory Streamlit Deployment** | Streamlit deployment is MANDATORY. Command-line, notebook-only, or static screenshots will NOT be accepted. Must demonstrate complete usable flow: input -> model execution -> understandable output. | A fully functional, production-ready **Streamlit web application** is delivered. Features: PGN file upload / sample game picker, interactive ply navigation, real-time PyTorch/ONNX model inference, SVG board with strategic overlays, temporal plan timeline, and grounded fact inspector. | `src/ui/app.py`, `tasks/phase_2_deployment_and_evaluation/TASK_2_4_STREAMLIT_CORE_APPLICATION.md` |
| **Originality & Academic Integrity** | No copied tutorial, repository, or external complete project. The group must implement all task-specific code. | Every single module (data parser, tensor encoder, ResNet model, rule engine, temporal state tracker, Streamlit UI) is implemented from scratch by Group 29. All external libraries (`torch`, `python-chess`, `streamlit`) are documented support tools. | Codebase inspection & Git commit history |

---

## 2. Compliance with Course Recommendations (Instructor Guidelines)

In `PDF Recommendations.pdf`, the instructor provided exemplar projects (e.g. *Road Pothole Detection & Severity Analysis*, *Crop Recommendation System*, *Plant Disease Detection*) and emphasized:
> *"This is slightly more substantial than simple image classification because you have to combine object detection + result analysis."*  
> *"If you opt classification problem, Make sure the problem should be genuine."*

### How MindCheck Elevates Beyond Basic Classification:
1. **Multi-Step Spatial-Temporal Formulation**: MindCheck does not classify a single static image; it processes an 8-step temporal sequence of 2D board states to understand intention and trajectory.
2. **Result Analysis & Grounded Evidence**: Alongside the neural prediction, MindCheck extracts concrete symbolic facts (pawn chain anchors, open file rooks, outpost holes) and runs a temporal state tracker to determine plan onset and progression.
3. **Actionable Post-Game Coaching**: The system synthesizes neural hypotheses with deterministic explanation templates and optional defensive counterplay screening, providing meaningful review insights.

---

## 3. Course Learning Outcomes (CLO) Traceability Matrix

| Course Learning Outcome (CLO) | Addressed In MindCheck Pipeline | Verification & Artifact |
| :--- | :--- | :--- |
| **CLO 1**: Problem Identification & Formulation | Problem definition of chess strategy recognition; mathematical formulation of board representation as spatial-temporal tensors; formal rubric of 5 strategic plans. | Project Report (Sec 1-3), `docs/00_PROJECT_OVERVIEW.md`, `docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md` |
| **CLO 2**: Algorithm Design, Implementation & Integration | Custom PyTorch ResNet design; bitboard fact extractors; weak labeling pipeline; temporal plan state tracker; Streamlit web application. | Complete source code in `src/`, automated tests in `tests/`, Streamlit application `app.py` |
| **CLO 3**: Testing, Performance Evaluation & Presentation | Comparative benchmarking (History CNN vs Current-Position CNN vs Rule Baseline); double-blind annotation with Cohen's Kappa; confusion matrix; live demonstration and slides. | Evaluation test suite in `tests/eval/`, slides in `docs/slides/`, live Streamlit demonstration |

---

## 4. Final Submission Deliverables Checklist (Deadline: 06 Dec 2026)

- [x] **Deliverable 1: Source Code**: Clean, modular, fully commented Python codebase with `requirements.txt`, unit tests, and reproducible run commands.
- [x] **Deliverable 2: Project Report**: Academic-grade report with problem statement, methodology, visual pipeline diagram, mathematical tensor formulation, experimental results, and limitations.
- [x] **Deliverable 3: Streamlit Demonstration**: Interactive, zero-crash web application tested on clean environment.
- [x] **Deliverable 4: Presentation Slides**: Structured deck covering motivation, neural architecture, evaluation metrics, ablation findings, and live demo walkthrough.
- [x] **Deliverable 5: Evidence of Evaluation**: Quantitative evaluation logs, confusion matrices, baseline comparisons, and independently annotated benchmark results.
- [x] **Submission Format**: Single consolidated `.zip` file submitted once on Google Classroom (GCR) by Group 29.
