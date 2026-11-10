# Subtask 2.4.3: Interactive Move-by-Move Game Stepper & Timeline Navigation

**Subtask ID**: `2.4.3.MOVE.STEPPER.NAVIGATOR`  
**Parent Task**: `Task 2.4`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build move navigation controls: First, Previous, Play/Pause auto-play, Next, Last, and ply slider.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Active `chess.pgn.Game` in session state.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/navigator.py` rendering navigation bar and move notation display.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Controls:
   - `|<`, `< Prev`, `Play/Pause`, `Next >`, `>|` buttons.
   - Move slider: `st.slider('Move Slider', 0, total_plies, current_ply)`.
   - Current move box: Displays `Move 24. ... Nd5 (Black)`.
2. Auto-Play: Uses `st.empty()` or sleep loop when `is_playing` is active.

---

## 5. Claude Code Action Plan
1. Implement `src/ui/navigator.py`.
2. Bind button click callbacks to `st.session_state.current_ply`.
3. Test smooth slider scrubbing and boundary handling (ply 0 and max ply).

---

## 6. Unit Testing & Acceptance Criteria
- Buttons and slider update `current_ply` synchronously.
- Boundary checks prevent out-of-bounds indexing.
