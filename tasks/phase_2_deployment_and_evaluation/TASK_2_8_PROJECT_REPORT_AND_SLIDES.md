# Task 2.8: Academic Project Report & Presentation Slides

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 15  
**Primary Owners**: Shared Responsibility (Muhammad Ali & Shahzad Ahmad)  

---

## 1. Objective & Scope
Produce the mandatory academic documentation and presentation deliverables required by the AI2002 Semester Project Statement: a complete, publication-grade **Project Report** and a professional **Presentation Slide Deck** for the live defense (7–9 Dec 2026).

---

## 2. Subtask Breakdown

### Subtask 2.8.1: Formal Academic Project Report (`docs/REPORT.md` / `docs/report.pdf`)
- **Owners**: Muhammad Ali (`24I-3158`) & Shahzad Ahmad (`24I-3090`)
- **Mandatory Course Deliverable Structure**:
  1. **Title Page**: MindCheck, Group 29, Muhammad Ali (24I-3158), Shahzad Ahmad (24I-3090), Instructor: Dr. Muhammad Bilal, Path A - Software-Based.
  2. **Problem Statement & Real-World Justification**: Chess learners' strategic bottleneck, pedagogical gap in engine evaluations, ethical post-game review scope.
  3. **Selected Path & Objectives**: Path A declaration, primary and secondary research questions.
  4. **Methodology & Mathematical Formulation**: Spatial-temporal tensor formulation ($112 \times 8 \times 8$), attacker perspective normalization, validity masks.
  5. **Algorithm Description**: Deep Residual CNN (4 ResNet blocks), classification head, single-board ablation model, loss function, and optimizer.
  6. **System Architecture & Workflow**: Vector pipeline diagram, data flow from PGN to Streamlit.
  7. **Dataset Description**: Lichess Elite archive, ratings, filters, retrospective weak-labeling, game-level split (70/15/15), limitations.
  8. **Implementation Details**: Python, PyTorch, python-chess bitboards, Streamlit frontend, temporal state tracker with hysteresis.
  9. **Testing & Performance Evaluation**:
     - 250 held-out independent test set, Cohen's Kappa score ($\kappa$).
     - Quantitative table: History ResNet vs Single-Board ResNet vs Rule vs Hybrid.
     - Confusion matrices and per-class Precision/Recall/F1.
     - Bootstrap 95% confidence intervals.
     - Abstention coverage curve.
  10. **Ablation Studies**: Impact of history length (1 vs 4 vs 8 plies).
  11. **Qualitative Failure Analysis**: 5 false positives and 5 false negatives analyzed from a chess perspective.
  12. **Limitations & Future Work**: Opening book edge cases, compute constraints, amateur vs master divergence.
  13. **Conclusion & References**: Summary of findings and formal bibliography.

### Subtask 2.8.2: Publication-Grade Pipeline Diagram
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - High-resolution SVG / PNG diagram detailing the complete software architecture.
  - Placed into `docs/figures/system_pipeline.png` and embedded directly in the report and slides.

### Subtask 2.8.3: Presentation Slide Deck (`docs/slides.pdf`)
- **Owners**: Shahzad Ahmad (`24I-3090`) & Muhammad Ali (`24I-3158`)
- **Deck Outline (12–15 Slides)**:
  - Slide 1: Title & Group 29 Introduction
  - Slide 2: Real-World Motivation & The Problem of Plan Recognition
  - Slide 3: Path A Scope & Core Research Questions
  - Slide 4: Strategic Plan Taxonomy (5 Plans + No Plan)
  - Slide 5: Data Engineering & Weak-Label Lookahead Heuristics
  - Slide 6: Spatial-Temporal Tensor Representation ($112 \times 8 \times 8$)
  - Slide 7: Deep Residual CNN Architecture
  - Slide 8: Temporal Plan State Tracking & Hysteresis
  - Slide 9: Mandatory Streamlit Application Architecture & Demo Video/Screens
  - Slide 10: Experimental Setup & Independent Annotation ($\kappa$ calculation)
  - Slide 11: Benchmark Results: History CNN vs Current-Position vs Rules
  - Slide 12: Ablation Study: Does History Window Length Matter?
  - Slide 13: Qualitative Failure Analysis & Limitations
  - Slide 14: Work Distribution & Team Contributions
  - Slide 15: Conclusion & Live Demonstration Q&A

---

## 3. Claude Code Action Plan
1. Draft comprehensive report source in Markdown: `docs/REPORT.md`.
2. Convert `docs/REPORT.md` to publication PDF using `weasyprint`, `pandoc`, or export via browser print.
3. Create slide deck in Markdown/Marp or PowerPoint: `docs/slides/presentation_deck.pptx` / `docs/slides.pdf`.
4. Review against AI Project Statement rubric to ensure 100% compliance.

---

## 4. Verification & Acceptance Criteria
- Both PDF files (`docs/report.pdf` and `docs/slides.pdf`) exist and are visually verified.
- Report contains actual quantitative results, confusion matrices, and ablation figures.
- Both team members review and agree on all content for the live presentation.
