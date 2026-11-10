# Subtask 1.2.3: 8-Ply Sliding History Window Extractor

**Subtask ID**: `1.2.3.SLIDING.WINDOW.EXTRACTOR`  
**Parent Task**: `Task 1.2`  
**Phase**: Phase 1  
**Milestone**: Milestone 2  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Extract 8-half-move sliding windows across game plies for temporal feature encoding.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.pgn.Game` object.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/data/window_extractor.py` returning `Iterator[PositionWindow]`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Dataclass `PositionWindow`:
   - `game_id: str`
   - `ply_index: int`
   - `turn: chess.Color`
   - `board_history: List[chess.Board]` (length up to 8, past to current)
   - `last_move: Optional[chess.Move]`
   - `is_check: bool`
2. Logic:
   - Play through game ply-by-ply from ply 12 to termination.
   - At each ply $t$, extract past 8 board copies: $[B_{t-7}, \dots, B_t]$.
   - If ply $t < 7$, pad missing history slots with None or initial board states.
   - Yield `PositionWindow`.

---

## 5. Claude Code Action Plan
1. Implement `src/data/window_extractor.py`.
2. Ensure board states in history are independent clones (`board.copy()`).
3. Write `tests/test_window_extractor.py` verifying window length and correct ply sequencing.
4. Run `pytest tests/test_window_extractor.py`.

---

## 6. Unit Testing & Acceptance Criteria
- Extracted windows always contain up to 8 sequential boards.
- The final board in `board_history` matches the board state at `ply_index`.
