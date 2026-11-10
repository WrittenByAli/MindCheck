# Subtask 2.5.1: Interactive SVG Chessboard Component with Strategic Overlays

**Subtask ID**: `2.5.1.SVG.CHESSBOARD.OVERLAYS`  
**Parent Task**: `Task 2.5`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Render chessboard using `python-chess.svg` with move arrows and strategic square highlights.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Active `chess.Board`, last move, and detected strategic features.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/board_view.py` rendering SVG in Streamlit HTML component.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Overlays:
   - Last move: Solid green arrow (`#2ecc71`).
   - History moves: Translucent cyan arrows (`#3498db88`).
   - Outpost square: Gold circle highlight (`#f1c40f`).
   - Pawn storm target: Crimson square highlight (`#e74c3c`).
2. Board orientation: Automatically flip board when Black is attacking or via toggle.

---

## 5. Claude Code Action Plan
1. Implement SVG generation in `src/ui/board_view.py`.
2. Embed SVG in Streamlit using `st.components.v1.html(svg, height=450, width=450)`.
3. Verify arrow positioning and square highlights on sample positions.

---

## 6. Unit Testing & Acceptance Criteria
- Board renders cleanly at high resolution.
- Strategic arrows and highlights align precisely with target squares.
