# Subtask 2.4.1: Streamlit App Structure, Layout & Session State Manager

**Subtask ID**: `2.4.1.APP.STRUCTURE.SESSION.STATE`  
**Parent Task**: `Task 2.4`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build the Streamlit application shell, multi-column layout, and session state manager.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: None (app entry point).

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/app.py` with responsive 2-column layout and session state dictionary.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. App Entry: `src/ui/app.py`.
2. Session State:
   - `current_ply: int`
   - `game_moves: List[chess.Move]`
   - `board_positions: List[chess.Board]`
   - `selected_model_name: str`
   - `is_playing: bool`
   - `analysis_cache: Dict[int, any]`
3. Wide-screen layout: Sidebar controls, left column (board SVG), right column (predictions & charts).

---

## 5. Claude Code Action Plan
1. Implement `src/ui/app.py` with page config and session initialization.
2. Verify app launches with `streamlit run src/ui/app.py`.
3. Check state persistence across UI re-renders.

---

## 6. Unit Testing & Acceptance Criteria
- App launches cleanly without warnings or errors.
- Session state maintains game position across widget interactions.
