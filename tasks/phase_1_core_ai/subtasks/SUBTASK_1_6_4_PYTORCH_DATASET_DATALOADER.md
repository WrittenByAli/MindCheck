# Subtask 1.6.4: High-Performance PyTorch Dataset & DataLoader

**Subtask ID**: `1.6.4.PYTORCH.DATASET.DATALOADER`  
**Parent Task**: `Task 1.6`  
**Phase**: Phase 1  
**Milestone**: Milestone 5  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Build PyTorch Dataset and DataLoader supporting memory-mapped chunk caching.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Split record tables and PGN source.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/data/chess_dataset.py` with `ChessPlanDataset`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `ChessPlanDataset(torch.utils.data.Dataset)`:
   - Supports pre-encoded `.npy` / `.pt` memory-mapped arrays or dynamic encoding.
   - Returns: `(tensor, mask, label_id)` tuple.
2. Factory function `create_dataloaders(config)`:
   - Returns `train_loader`, `val_loader`, `test_loader` with pin_memory, multi-worker loading, and shuffling.

---

## 5. Claude Code Action Plan
1. Implement `src/data/chess_dataset.py`.
2. Write `tests/test_dataset.py` testing iteration, batch shapes, and worker stability.
3. Test DataLoader throughput (> 1,000 samples/sec on GPU/CPU).

---

## 6. Unit Testing & Acceptance Criteria
- Batch output shapes: `(batch_size, 112, 8, 8)`, `(batch_size, 8)`, `(batch_size,)`.
- Zero memory leaks during multi-epoch iteration.
