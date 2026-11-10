# Subtask 1.7.4: Checkpointing, TorchScript & ONNX Export Pipeline

**Subtask ID**: `1.7.4.ONNX.TORCHSCRIPT.EXPORT`  
**Parent Task**: `Task 1.7`  
**Phase**: Phase 1  
**Milestone**: Milestone 6  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build automated serialization pipeline exporting PyTorch models to TorchScript and ONNX formats for CPU inference.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Trained PyTorch model and checkpoint path.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `models/checkpoints/best_resnet.pt`, `model.torchscript.pt`, `model.onnx`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Script `src/models/export.py`:
   - `export_to_onnx(model, output_path)`: Uses `torch.onnx.export` with dynamic batching.
   - `export_to_torchscript(model, output_path)`: Uses `torch.jit.trace`.
2. Validation: Load exported ONNX model with `onnxruntime.InferenceSession` and assert output matches PyTorch within $1e-4$ tolerance.

---

## 5. Claude Code Action Plan
1. Implement `src/models/export.py`.
2. Add export tests in `tests/test_export.py`.
3. Verify ONNX CPU execution speed (< 10ms).

---

## 6. Unit Testing & Acceptance Criteria
- `model.onnx` passes `onnx.checker.check_model()`.
- Numeric output matches PyTorch model within numerical tolerance.
