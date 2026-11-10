# Subtask 2.4.4: Low-Latency Inference Runner with Caching

**Subtask ID**: `2.4.4.INFERENCE.RUNNER.CACHING`  
**Parent Task**: `Task 2.4`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build high-speed inference runner cached with `@st.cache_resource` for zero UI lag.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Active ply and board history.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/inference_runner.py` returning prediction, state, and facts in < 20ms.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Model Caching:
   - `@st.cache_resource` loads ONNX runtime session or PyTorch checkpoint once.
2. Inference Flow:
   - Construct 8-ply window -> Tensor encoding -> ResNet forward pass -> Fact extractor -> Hybrid fusion -> State tracker update.
3. Latency: Measure and display execution time in sidebar diagnostic badge.

---

## 5. Claude Code Action Plan
1. Implement `src/ui/inference_runner.py`.
2. Benchmark inference latency inside Streamlit (< 20ms per ply).
3. Test switching models dynamically from sidebar.

---

## 6. Unit Testing & Acceptance Criteria
- Move navigation updates board and predictions instantly (< 20ms).
- Memory consumption remains stable during long interactive sessions.
