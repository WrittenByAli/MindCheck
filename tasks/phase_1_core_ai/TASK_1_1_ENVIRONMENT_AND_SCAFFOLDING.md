# Task 1.1: Environment, Dependencies & Repository Scaffolding

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 1  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Establish a rock-solid, reproducible, production-grade engineering environment for MindCheck. This involves defining pinned dependencies, creating type-safe runtime configuration models via Pydantic and PyYAML, and verifying CPU/CUDA PyTorch acceleration.

---

## 2. Subtask Breakdown

### Subtask 1.1.1: Pinned Dependency Specification (`requirements.txt`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**: Create root `requirements.txt` with strictly pinned, compatible versions for Python 3.11+:
  - `torch>=2.1.0`
  - `torchvision>=0.16.0`
  - `python-chess>=1.9.4`
  - `streamlit>=1.30.0`
  - `numpy>=1.24.0,<2.0.0`
  - `pandas>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `matplotlib>=3.8.0`
  - `seaborn>=0.13.0`
  - `pyyaml>=6.0.1`
  - `pydantic>=2.5.0`
  - `pytest>=7.4.0`
  - `tqdm>=4.66.0`
  - `onnx>=1.15.0`
  - `onnxruntime>=1.16.0`

### Subtask 1.1.2: Base Python Package Layout (`src/` & `tests/`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**: Ensure every directory under `src/` and `tests/` has valid `__init__.py` files with appropriate version and export metadata.
- **Directory Verification**:
  ```
  src/
  ├── __init__.py
  ├── data/       # PGN parsing, tensor encoding, dataset splitting
  ├── rules/      # Bitboard fact extraction & rule templates
  ├── models/     # PyTorch ResNet, training loop, inference engine
  ├── tracking/   # Temporal state tracker, fusion, explanations
  ├── ui/         # Streamlit dashboard, SVG chessboard, charts
  └── utils/      # Pydantic configuration, logging, Stockfish client
  ```

### Subtask 1.1.3: Central Configuration Management (`config/config.yaml` & `src/utils/config.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**: Create `config/config.yaml` specifying all runtime hyperparameters:
  - Dataset paths, minimum game plies, Elo rating filter.
  - History length ($T=8$), plane dimensions ($14 \times 8 \times 8$).
  - ResNet channels (64), number of residual blocks (4), dropout rate (0.2).
  - Training batch size (64), learning rate (0.001), max epochs (30), early stopping patience (5).
  - Class weights, abstention confidence threshold (0.40).
- Create `src/utils/config.py` with Pydantic `BaseModel` classes validating the YAML configuration on startup.

---

## 3. Claude Code Action Plan
1. Write the root `requirements.txt` file.
2. Create `src/__init__.py` and subpackage `__init__.py` files.
3. Write `config/config.yaml` with documented parameters.
4. Implement `src/utils/config.py` using `pydantic` and `pyyaml` to load and validate settings.
5. Create `tests/test_config.py` to assert that configurations load cleanly with default values.

---

## 4. Verification & Acceptance Criteria
- Run `python -c "import torch, chess, streamlit; print('All core modules imported successfully!')"`
- Run `pytest tests/test_config.py` -> Must pass with 100% success.
