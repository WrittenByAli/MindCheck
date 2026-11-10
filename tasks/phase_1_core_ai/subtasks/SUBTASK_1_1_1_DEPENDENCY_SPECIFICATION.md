# Subtask 1.1.1: Pinned Dependency Specification & Verification

**Subtask ID**: `1.1.1.DEPENDENCY.SPECIFICATION`  
**Parent Task**: `Task 1.1`  
**Phase**: Phase 1  
**Milestone**: Milestone 1  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Define all external library dependencies required by the system with exact compatible versions.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Python 3.11+ interpreter installed on the host system.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `requirements.txt` file at project root; successful pip installation log.

---

## 4. Detailed Technical Specification
### Technical Requirements:
1. File: `requirements.txt` at repository root.
2. Dependencies:
   - `torch>=2.1.0` (PyTorch core)
   - `torchvision>=0.16.0`
   - `python-chess>=1.9.4` (Chess logic, bitboards, PGN, SVG)
   - `streamlit>=1.30.0` (Interactive web UI)
   - `numpy>=1.24.0,<2.0.0` (Vector manipulation)
   - `pandas>=2.0.0` (DataFrames and metadata)
   - `scikit-learn>=1.3.0` (Metrics, F1, Cohen's kappa)
   - `matplotlib>=3.8.0` and `seaborn>=0.13.0` (Plots)
   - `pyyaml>=6.0.1` and `pydantic>=2.5.0` (Configuration)
   - `pytest>=7.4.0` (Testing framework)
   - `onnx>=1.15.0` and `onnxruntime>=1.16.0` (Fast CPU runtime)
   - `tqdm>=4.66.0` (Progress visualization)
3. Provide CPU and CUDA installation commands in documentation.

---

## 5. Claude Code Action Plan
1. Inspect current python environment.
2. Verify package compatibility and write `requirements.txt`.
3. Create `scripts/check_env.py` that verifies imports of all libraries and prints versions.
4. Run environment check script.

---

## 6. Unit Testing & Acceptance Criteria
- `python scripts/check_env.py` executes without `ImportError` or version conflicts.
- PyTorch correctly detects CPU or CUDA backend.
