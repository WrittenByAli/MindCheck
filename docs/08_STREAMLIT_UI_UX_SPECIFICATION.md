# MindCheck: Streamlit UI/UX & Deployment Specification

This document details the interface layout, user interaction flows, visual styling, and component hierarchy for the **MindCheck** Streamlit Web Application (Mandatory Path A Deliverable).

---

## 1. User Interface Layout Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                    MINDCHECK: CHESS PLAN RECOGNITION                             |
|  History-Aware Neural Strategic Analysis | AI2002 Semester Project | Group 29 (Ali & Shahzad)    |
+--------------------------------------------------------------------------------------------------+
| SIDEBAR CONTROLS              | MAIN VIEWPORT                                                    |
|                               |                                                                  |
| [1] Game Input Mode           |  Game: G. Kasparov vs A. Karpov (1985) | Ply: 32 (White to move)   |
|   (•) Sample Library          |  [ |< ]  [ < Prev ]  [ Play/Pause ]  [ Next > ]  [ >| ]           |
|   ( ) Upload Custom PGN       |  Move Slider: ====================o====================== [Ply 32]|
|                               |                                                                  |
| [2] Select Match              | +------------------------------+ +------------------------------+ |
|   [Kasparov vs Karpov - G16 v]| |                              | | DETECTED STRATEGIC PLAN      | |
|                               | |    CHESSBOARD (SVG)          | | [badge: OUTPOST OCCUPATION]  | |
| [3] AI Model Engine           | |                              | | Confidence: 84.6%            | |
|   (•) History ResNet (8 plies)| |  8 [r][ ][b][q][k][b][ ][r]  | | Status: ACTIVE (onset: ply26)| |
|   ( ) Single Board CNN (1 ply)| |  7 [p][p][ ][ ][p][p][p][p]  | |                              | |
|   ( ) Neural-Rule Hybrid      | |  6 [ ][ ][p][ ][ ][n][ ][ ]  | | Class Probabilities:         | |
|                               | |  5 [ ][ ][ ][p][ ][ ][ ][ ]  | | OUTPOST     [========] 85%   | |
| [4] Analysis Thresholds       | |  4 [ ][ ][P][N][P][ ][ ][ ]  | | ROOK_FILE   [==      ] 08%   | |
|   Abstention Gate: [ 0.40 ]   | |  3 [ ][ ][ ][ ][ ][N][ ][ ]  | | K_STORM     [=       ] 04%   | |
|   Stockfish Screening: [ON ]  | |  2 [P][P][ ][ ][ ][P][P][P]  | | Q_EXP       [        ] 02%   | |
|                               | |  1 [R][ ][B][Q][K][B][ ][R]  | | UNDERMINE   [        ] 01%   | |
|                               | |     a  b  c  d  e  f  g  h   | | NO_PLAN     [        ] 00%   | |
|                               | +------------------------------+ +------------------------------+ |
|                               |                                                                  |
|                               | [TAB 1: Strategic Timeline] [TAB 2: Evidence] [TAB 3: Engine]   |
|                               | ---------------------------------------------------------------- |
|                               | TAB 1: Plan Probability Timeline (Plies 1 to 40)                 |
|                               |   Outpost probability spikes at ply 26 and stays sustained.      |
|                               |                                                                  |
|                               | TAB 2: Grounded Symbolic Evidence                                |
|                               |   • Hole identified on square d5 (uncontested by black pawns)    |
|                               |   • Knight maneuver: Nd2 -> Nf3 -> Nd4 -> d5                     |
|                               |   • Supported by friendly pawn on e4                             |
|                               |   "White has executed an Outpost Occupation on d5, establishing  |
|                               |    a dominant minor piece anchor."                               |
|                               |                                                                  |
|                               | TAB 3: Stockfish Defensive Counter-Measures                      |
|                               |   • Best counter: 32... Re8 (prevents secondary rook support)    |
+-------------------------------+------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Sidebar Controls (`src/ui/sidebar.py`)
- **Game Selection**:
  - Radio button between "Curated Master Game Library" and "Upload Custom PGN".
  - Curated library contains at least 5 famous historic games illustrating each of the 5 plans cleanly:
    1. *Kingside Pawn Storm*: Fischer vs Larsen (1958)
    2. *Queenside Expansion*: Petrosian vs Spassky (1966)
    3. *Outpost Occupation*: Kasparov vs Karpov (1985, Game 16)
    4. *Rook-File Buildup*: Alekhine vs Nimzowitsch (1930, San Remo)
    5. *Pawn-Chain Undermining*: French Defense Classic (Botvinnik vs Capablanca)
- **Model Engine Toggle**: Allows switching between History-Aware ResNet, Single-Board CNN Baseline, and Neural-Rule Hybrid to demonstrate ablations live.
- **Inference Diagnostics**: Displays model load status, checkpoint name, device (CPU/GPU), and average inference latency.

### 2.2 Interactive Move Navigation (`src/ui/navigator.py`)
- **Step Controls**: First Ply, Previous Ply, Play/Pause Auto-stepper, Next Ply, Last Ply.
- **Move Slider**: Smooth integer slider navigating between ply 0 and total game plies.
- **Move Notation Box**: Formatted algebraic notation showing current move (e.g., `32. Nd5!`) and side to move.

### 2.3 Board Visualization Engine (`src/ui/board_view.py`)
- Powered by `python-chess.svg.board()`.
- **Dynamic Overlays**:
  - Green arrow indicating the last move played.
  - Cyan arrows indicating historical moves from the 8-ply window.
  - Red / Gold square highlights displaying key strategic focus squares (e.g., the target outpost square, open file entry squares).
- Embedded directly into Streamlit via `st.components.v1.html(svg_data, height=450, width=450)`.

### 2.4 Multi-Class Probability Display (`src/ui/charts.py`)
- Interactive horizontal bar chart rendering the softmax probabilities across all 6 classes:
  - Red: `K_STORM`
  - Blue: `Q_EXP`
  - Gold: `OUTPOST`
  - Purple: `ROOK_FILE`
  - Teal: `CHAIN_UNDERMINE`
  - Gray: `NO_PLAN`
- Visual indicator showing whether the prediction is accepted or abstained based on the confidence threshold.

### 2.5 Temporal Strategic Timeline (`src/ui/timeline.py`)
- Area/Line chart across all plies of the game demonstrating:
  - Rise and fall of plan probabilities as the game develops.
  - Visual marker indicating the exact **Plan Onset Ply** detected by the temporal state tracker.
  - Plan switching points where strategic initiative shifted.

### 2.6 Grounded Explanation Card (`src/ui/explanation_card.py`)
- Controlled template text that explicitly cites verified bitboard facts:
  - *"Observed board features: Open d-file, Knight stationed on d5, undefended by enemy pawns."*
  - *"Learned model hypothesis: Outpost Occupation (85% confidence)."*
  - *"Strategic takeaway: The knight exerts central pressure, restricting enemy piece mobility."*

---

## 3. Performance & Usability Acceptance Criteria
1. **Zero-Lag Interaction**: Move transitions must complete inference and UI updates in $< 25 \text{ ms}$ on standard laptop CPU.
2. **Error Resilience**:
   - Uploading an invalid or corrupted PGN file displays a user-friendly error message without crashing the Streamlit server.
   - Opening moves (plies 0 to 7) smoothly render with the zero-padding validity mask.
3. **Single-Command Launch**:
   ```bash
   streamlit run src/ui/app.py
   ```
