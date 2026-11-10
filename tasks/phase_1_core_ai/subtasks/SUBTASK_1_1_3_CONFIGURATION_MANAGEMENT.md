# Subtask 1.1.3: Type-Safe Configuration Schema with Pydantic & YAML

**Subtask ID**: `1.1.3.CONFIGURATION.MANAGEMENT`  
**Parent Task**: `Task 1.1`  
**Phase**: Phase 1  
**Milestone**: Milestone 1  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Create a centralized, type-safe runtime configuration system using PyYAML and Pydantic v2.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `config/config.yaml` file containing all system parameters.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/utils/config.py` defining Pydantic models `AppConfig`, `DatasetConfig`, `ModelConfig`, `TrainingConfig`, `TrackingConfig`, `FusionConfig`, `UIConfig`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. `src/utils/config.py`:
   - Class `DatasetConfig(BaseModel)`
   - Class `ModelConfig(BaseModel)`
   - Class `TrainingConfig(BaseModel)`
   - Class `TrackingConfig(BaseModel)`
   - Class `FusionConfig(BaseModel)`
   - Class `UIConfig(BaseModel)`
   - Class `AppConfig(BaseModel)` aggregating all sub-configs.
   - Function `load_config(yaml_path: str = 'config/config.yaml') -> AppConfig`.
2. Validation: Strict type validation for paths, learning rates, epochs, thresholds.

---

## 5. Claude Code Action Plan
1. Write `src/utils/config.py`.
2. Implement schema loading and error handling for missing keys.
3. Write `tests/test_config.py` testing config loading and default value fallbacks.
4. Run `pytest tests/test_config.py`.

---

## 6. Unit Testing & Acceptance Criteria
- `load_config()` loads `config/config.yaml` into strongly-typed Pydantic model.
- Invalid YAML structure or bad parameter types trigger clear validation errors.
