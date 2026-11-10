# Task 2.9: Final Quality Assurance, Packaging & GCR Submission Verification

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 15  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Perform final quality assurance and end-to-end verification. Ensure the application launches flawlessly from a clean virtual environment using a single command, rehearse the live demonstration flow, and generate the final single consolidated `.zip` file for submission on Google Classroom (GCR) before the **06 December 2026** hard deadline.

---

## 2. Subtask Breakdown

### Subtask 2.9.1: Clean Environment Verification & One-Click Launchers
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Implement `scripts/verify_install.py` testing that all required packages, checkpoints, and datasets are present and operable.
  - Create one-click launch scripts:
    - Windows: `run_demo.bat`
    - Linux/macOS: `run_demo.sh`
  - Test clean installation workflow in a freshly created temporary virtual environment (`test_clean_venv`):
    ```bash
    python -m venv test_clean_venv
    .\test_clean_venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python scripts/verify_install.py
    streamlit run src/ui/app.py
    ```
  - Verify that the Streamlit application starts up and renders without crashing or missing dependencies.

### Subtask 2.9.2: Live Demo Rehearsal Checklist (`docs/demo_rehearsal_checklist.md`)
- **Owners**: Muhammad Ali (`24I-3158`) & Shahzad Ahmad (`24I-3090`)
- **Live Demonstration Script (5–7 Minute Defense Window)**:
  - **Minute 1**: Introduction, group members (Muhammad Ali & Shahzad Ahmad), real-world problem statement, and 5-plan taxonomy.
  - **Minute 2**: Dataset engineering, 8-ply sliding window extraction, and spatial-temporal tensor encoding.
  - **Minute 3**: Live Streamlit App Demo - Load *Kasparov vs Karpov 1985* $\to$ step to ply 26 $\to$ show Outpost Occupation detected with 85% probability, show gold highlight on d5, show temporal onset timeline.
  - **Minute 4**: Live Streamlit App Demo - Load *Fischer vs Larsen 1958* $\to$ step to ply 22 $\to$ show Kingside Pawn Storm, show crimson vector arrows and grounded evidence.
  - **Minute 5**: Toggle model engines: show History-Aware ResNet vs Current-Position Baseline live, illustrating how history prevents plan flickering.
  - **Minute 6**: Quantitative evaluation summary (Macro-F1, confusion matrix, Cohen's Kappa $\kappa = 0.72$), failure analysis, and conclusion.
  - **Minute 7**: Q&A defense.

### Subtask 2.9.3: Single GCR Submission ZIP Packager (`scripts/package_submission.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Strict Course Directive**:
  > *"All required files MUST be zipped together and submitted on GCR only (a single zip file per group)."*
- **Specification**:
  - Implement `scripts/package_submission.py` to create `MindCheck_Group29_Submission.zip`.
  - **Inclusions**:
    - `src/` (All source code with comments and docstrings)
    - `docs/` (Report PDF, Slides PDF, architectural documents)
    - `data/samples/` (Sample curated PGN files)
    - `data/annotations/test_ground_truth.csv` (Adjudicated benchmark)
    - `models/checkpoints/` (Trained PyTorch weights and ONNX model)
    - `requirements.txt`
    - `run_demo.bat` & `run_demo.sh`
    - `README.md`
  - **Exclusions**:
    - `.git/` directory
    - `venv/` and `.venv/`
    - `__pycache__/` and `*.pyc`
    - Huge raw PGN gigabyte files (only sample PGNs included)
  - Assert that final ZIP file size is $< 80 \text{ MB}$.

---

## 3. Packaging CLI Command

```bash
# Verify environment and run demo
python scripts/verify_install.py
./run_demo.bat

# Build clean submission ZIP
python scripts/package_submission.py
```

---

## 4. Claude Code Action Plan
1. Create `scripts/verify_install.py`.
2. Create `run_demo.bat` and `run_demo.sh`.
3. Create `docs/demo_rehearsal_checklist.md`.
4. Create `scripts/package_submission.py`.
5. Test unzipping the generated archive in a clean directory and verifying file completeness.

---

## 5. Verification & Acceptance Criteria
- Running `run_demo.bat` opens the working Streamlit UI immediately in default browser.
- Archive `MindCheck_Group29_Submission.zip` created containing all mandatory deliverables: Source Code, Report PDF, Slides PDF, Checkpoint, and Evidence of Evaluation.
- Submitted to Google Classroom before 06 Dec 2026 23:59 PKT.
