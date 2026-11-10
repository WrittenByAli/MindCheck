# Task 2.4: Streamlit Interactive Web Application - Core Engine

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 12  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Build the core application loop and navigation engine of the mandatory Streamlit web application (`src/ui/app.py`). Implement robust session state management, PGN file uploaders, curated master game libraries, smooth move-by-move ply navigation, and ultra-low-latency model inference caching.

---

## 2. Subtask Breakdown

### Subtask 2.4.1: App Architecture & Session State (`src/ui/app.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - `st.set_page_config(page_title="MindCheck - Chess Plan AI", layout="wide", page_icon="♟️")`.
  - Session State Initialization:
    - `st.session_state.current_ply`: Active ply index ($0 \dots T$)
    - `st.session_state.game_moves`: List of `chess.Move` in active game
    - `st.session_state.board_positions`: Reconstructed `chess.Board` objects
    - `st.session_state.selected_model`: Active model engine (`History ResNet`, `Single-Board Baseline`, `Hybrid`)
    - `st.session_state.game_metadata`: Players, Event, Result, ECO code
    - `st.session_state.is_playing`: Flag for auto-play animation

### Subtask 2.4.2: PGN Loader & Curated Game Library (`src/ui/game_loader.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Provide dual input options in sidebar:
    1. **Upload Custom PGN**: File uploader accepting `.pgn` files up to 10 MB.
    2. **Curated Historical Match Library**: Dropdown selector of 5 classic games representing each plan:
       - *Game 1*: Fischer vs Larsen (1958) -> `K_STORM`
       - *Game 2*: Petrosian vs Spassky (1966) -> `Q_EXP`
       - *Game 3*: Kasparov vs Karpov (1985, Game 16) -> `OUTPOST`
       - *Game 4*: Alekhine vs Nimzowitsch (1930) -> `ROOK_FILE`
       - *Game 5*: Botvinnik vs Capablanca (1938) -> `CHAIN_UNDERMINE`
  - Parse PGN headers and validate move sequences cleanly without throwing unhandled exceptions.

### Subtask 2.4.3: Interactive Move-by-Move Stepper (`src/ui/navigator.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Render 5 navigation buttons in a single row:
    - `|<` (Start of Game, Ply 0)
    - `< Prev` (Step backward 1 ply)
    - `▶ Play / ⏸ Pause` (Auto-advance 1 ply every 1.5 seconds)
    - `Next >` (Step forward 1 ply)
    - `>|` (End of Game, Final Ply)
  - Interactive slider allowing instant scrubbing to any ply.
  - Display current move in standard algebraic notation: e.g., `Move 24. ... Nd5 (Black)`.

### Subtask 2.4.4: Low-Latency Inference Runner with Caching (`src/ui/inference_runner.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Decorate model loading with `@st.cache_resource`:
    - Loads ONNX runtime session or PyTorch checkpoint once into memory.
  - Decorate game window parsing with `@st.cache_data`.
  - Execute end-to-end inference (Tensor encoding $\to$ ResNet forward pass $\to$ State tracking $\to$ Fact extraction) in $< 20 \text{ ms}$ per move.

---

## 3. Core Streamlit App Entry Point Skeleton

```python
import streamlit as st
import chess.pgn
from src.ui.game_loader import render_game_loader
from src.ui.navigator import render_move_navigator
from src.ui.inference_runner import get_inference_engine

def main():
    st.set_page_config(page_title="MindCheck - Chess Plan AI", layout="wide", page_icon="♟️")
    st.title("♟️ MindCheck: History-Aware Chess Plan Recognition")
    
    # Sidebar
    game = render_game_loader()
    engine = get_inference_engine()
    
    # Main Viewport
    col_board, col_analysis = st.columns([1, 1])
    # ... render interactive views ...

if __name__ == "__main__":
    main()
```

---

## 4. Claude Code Action Plan
1. Implement `src/ui/game_loader.py` with bundled PGNs.
2. Implement `src/ui/navigator.py` with button callbacks and slider binding.
3. Implement `src/ui/inference_runner.py` with model caching.
4. Implement `src/ui/app.py` coordinating the workflow.
5. Create sample games under `data/samples/*.pgn`.

---

## 5. Verification & Acceptance Criteria
- Run `streamlit run src/ui/app.py` -> Browser opens cleanly without errors.
- Test stepping through all 5 sample games from start to finish with zero exceptions.
