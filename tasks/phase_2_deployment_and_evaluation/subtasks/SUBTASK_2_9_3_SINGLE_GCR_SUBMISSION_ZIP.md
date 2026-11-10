# Subtask 2.9.3: Single GCR Submission ZIP Packager & Verification

**Subtask ID**: `2.9.3.SINGLE.GCR.SUBMISSION.ZIP`  
**Parent Task**: `Task 2.9`  
**Phase**: Phase 2  
**Milestone**: Milestone 15  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build packaging script to generate single consolidated submission ZIP file for GCR.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Complete repository deliverables.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `scripts/package_submission.py` and `MindCheck_Group29_Submission.zip`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `scripts/package_submission.py`:
   - Package all required deliverables:
     - `src/` (All source code with comments)
     - `docs/` (Report PDF, Slides PDF, architectural docs)
     - `data/samples/` (Sample master PGNs)
     - `data/annotations/test_ground_truth.csv`
     - `models/checkpoints/` (Trained model weights)
     - `requirements.txt`, `config/`, `run_demo.bat`, `README.md`
   - Excludes `.git`, `venv`, `__pycache__`, and huge raw PGN files.
   - Verifies ZIP file integrity and file size < 80MB.

---

## 5. Claude Code Action Plan
1. Implement `scripts/package_submission.py`.
2. Generate submission ZIP.
3. Test extracting ZIP in an isolated test folder and running `run_demo.bat`.

---

## 6. Unit Testing & Acceptance Criteria
- Generated ZIP contains all mandatory course deliverables.
- Submits successfully to Google Classroom before 06 Dec 2026.
