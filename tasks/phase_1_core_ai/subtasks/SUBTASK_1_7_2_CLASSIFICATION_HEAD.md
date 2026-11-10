# Subtask 1.7.2: Classification Head & Softmax Probability Estimator

**Subtask ID**: `1.7.2.CLASSIFICATION.HEAD`  
**Parent Task**: `Task 1.7`  
**Phase**: Phase 1  
**Milestone**: Milestone 6  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Implement Global Average Pooling (GAP), LayerNorm, Dropout, and 6-class linear classification head.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Backbone feature map `(batch_size, 64, 8, 8)`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Raw logits `(batch_size, 6)` and softmax probabilities.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Head Architecture:
   - `nn.AdaptiveAvgPool2d((1, 1))` pooling to `(batch_size, 64)`
   - `nn.Flatten()`
   - `nn.LayerNorm(64)`
   - `nn.Dropout(p=0.20)`
   - `nn.Linear(64, num_classes=6)`
2. Methods:
   - `forward(x) -> logits`
   - `predict_proba(x) -> probabilities` via `torch.softmax(logits, dim=-1)`.

---

## 5. Claude Code Action Plan
1. Implement classification head in `src/models/residual_cnn.py`.
2. Write unit tests asserting output shape `(batch_size, 6)` and probabilities sum to 1.0.
3. Test dropout toggling between train() and eval() modes.

---

## 6. Unit Testing & Acceptance Criteria
- Output tensor has shape `(batch_size, 6)`.
- Sum of softmax probabilities equals 1.0 per sample.
