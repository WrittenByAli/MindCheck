# Subtask 2.9.1: Clean-Environment Installation & One-Click Launchers

**Subtask ID**: `2.9.1.CLEAN.ENVIRONMENT.VERIFIER`  
**Parent Task**: `Task 2.9`  
**Phase**: Phase 2  
**Milestone**: Milestone 15  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build installation verification script and one-click launch scripts for live demonstration.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Repository files and checkpoints.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `scripts/verify_install.py`, `run_demo.bat`, `run_demo.sh`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. `scripts/verify_install.py`:
   - Verifies Python version >= 3.11.
   - Imports `torch`, `chess`, `streamlit`, `numpy`, `pandas`, `sklearn`.
   - Asserts checkpoint `models/checkpoints/best_resnet.pt` exists and loads.
   - Asserts sample PGNs exist.
2. `run_demo.bat` (Windows) & `run_demo.sh` (Linux/macOS):
   - Single command to activate venv and launch Streamlit app.

---

## 5. Claude Code Action Plan
1. Implement `scripts/verify_install.py`.
2. Create `run_demo.bat` and `run_demo.sh`.
3. Test fresh virtual environment rebuild to confirm flawless startup.

---

## 6. Unit Testing & Acceptance Criteria
- `python scripts/verify_install.py` outputs all green checks.
- `run_demo.bat` launches the browser UI in < 5 seconds.
