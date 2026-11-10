# Task 2.5: Streamlit Board Visualization & Rich Analytical Views

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 12  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Develop rich, responsive visual components for the Streamlit interface: native SVG chessboard rendering with dynamic tactical arrows and strategic square highlights, multi-class probability charts, a whole-game temporal plan evolution timeline, and an interactive grounded evidence inspector.

---

## 2. Subtask Breakdown

### Subtask 2.5.1: Dynamic SVG Chessboard Component (`src/ui/board_view.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Render chessboard using `chess.svg.board()` embedded directly via `st.components.v1.html()`.
  - Dimensions: $450 \times 450 \text{ px}$, responsive scaling.
  - **Dynamic Visual Overlays**:
    - **Move Arrows**:
      - Last move: Solid green arrow (`#2ecc71`).
      - Previous 2–3 moves in history window: Translucent cyan arrows (`#3498db88`).
    - **Strategic Square Highlighting**:
      - Outpost square: Gold circle highlight (`#f1c40f`).
      - Pawn storm vector: Crimson highlight on target king quadrant (`#e74c3c`).
      - Open file penetration squares: Purple border highlight (`#9b59b6`).
    - Perspective: Automatically flip board if user chooses "Flip Board" or when Black is side to move.

### Subtask 2.5.2: Multi-Class Probability Bar Chart (`src/ui/charts.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Horizontal bar chart rendering probabilities for all 6 classes:
    - `K_STORM` (#e74c3c)
    - `Q_EXP` (#3498db)
    - `OUTPOST` (#f1c40f)
    - `ROOK_FILE` (#9b59b6)
    - `CHAIN_UNDERMINE` (#1abc9c)
    - `NO_PLAN` (#95a5a6)
  - Overlay vertical dashed line at the current **Abstention Gate Threshold** (e.g. $40\%$).
  - Display badge: `ACTIVE`, `IN TRANSITION`, or `ABSTAINED (UNCERTAIN)`.

### Subtask 2.5.3: Temporal Plan Evolution Timeline (`src/ui/timeline.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Whole-game timeline chart plotting probability of each plan class across all plies ($0 \dots T$).
  - Vertical scrub line indicating `current_ply`.
  - Markers on the timeline denoting:
    - **Plan Onset Ply**: The ply where the active plan first emerged.
    - **Plan Shift Plies**: The points where strategic transitions occurred.

### Subtask 2.5.4: Grounded Evidence Accordion (`src/ui/evidence_view.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Expandable UI sections presenting verified facts:
    - *Pawn Structures*: Detected pawn chains, base pawns, and advanced storm pawns.
    - *Piece Positioning*: Active outpost squares and defending pawn anchors.
    - *File Control*: Open and semi-open files, doubled rooks, heavy batteries.
  - Grounded explanation card formatted with clear pedagogical summaries.

---

## 3. Function Signatures & Data Model

```python
import chess
import chess.svg
import streamlit as st
from typing import List, Dict

def render_svg_board(
    board: chess.Board, 
    last_move: chess.Move = None, 
    highlight_squares: List[chess.Square] = None, 
    flipped: bool = False
):
    """Generates and renders raw SVG string inside Streamlit."""
    pass

def render_probability_bars(probabilities: Dict[str, float], threshold: float = 0.40):
    """Renders styled horizontal probability distribution."""
    pass

def render_game_timeline(game_history_probs: List[Dict[str, float]], current_ply: int):
    """Renders interactive multi-line chart of plan evolution."""
    pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/ui/board_view.py` with custom SVG arrow and highlight constructors.
2. Implement `src/ui/charts.py` using Streamlit native charts or Matplotlib.
3. Implement `src/ui/timeline.py` with historical probability plots.
4. Implement `src/ui/evidence_view.py` with styled markdown and callout alerts.

---

## 5. Verification & Acceptance Criteria
- Verify board SVG renders crisply without clipping or layout jumps.
- Verify that moving the slider instantly updates the board, probability bars, and explanation card synchronously.
