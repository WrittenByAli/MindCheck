# Task 2.6: Stockfish Defensive Counterplay Integration

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 13  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Integrate an optional Stockfish UCI engine client (`chess.engine`) to screen defensive counterplay moves that neutralize or disrupt the opponent's detected plan. Ensure robust fallback: if the Stockfish binary is not present on the user's system, the application must disclose engine unavailability honestly without crashing or interrupting the core neural classifier.

---

## 2. Subtask Breakdown

### Subtask 2.6.1: Stockfish UCI Client Wrapper (`src/utils/stockfish_client.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Implement class `StockfishAnalyzer`.
  - Search engine binary in common paths (`stockfish/stockfish.exe`, PATH environment variable).
  - Configurable analysis limits:
    - Search depth: 10–12 plies
    - Time limit: 0.2 seconds per position
    - MultiPV: top 3 candidate moves
  - **Graceful Fallback**: If binary is missing, set `is_available = False` and log a clean message in Streamlit: *"Stockfish engine not detected. Displaying purely neural and symbolic analysis."*

### Subtask 2.6.2: Defensive Move Screening (`src/utils/stockfish_client.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - For the defending player, evaluate candidate moves:
    - Rank candidate moves by engine score (centipawns or mate).
    - Annotate whether candidate move directly disrupts the opponent's plan (e.g. trading the opponent's outpost knight, blocking the open rook file with a bishop, or playing a counter-strike in the center).

### Subtask 2.6.3: Threat Delta Simulation (`src/utils/stockfish_client.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Compute Threat Delta:
    $$\Delta_{\text{threat}} = \text{Eval}(\text{Proactive Defense}) - \text{Eval}(\text{Passive Move})$$
  - Disclose honestly in UI: *"Engine-assisted suggestion for post-game study; not a guaranteed counter-plan."*

---

## 3. Function Signatures & Data Model

```python
from dataclasses import dataclass
from typing import List, Optional
import chess
import chess.engine

@dataclass
class EngineCandidateMove:
    move_san: str
    move_uci: str
    eval_cp: Optional[int]       # Centipawns (+150 = +1.50)
    is_mate: bool
    mate_moves: Optional[int]
    strategic_disruption_note: str

class StockfishAnalyzer:
    def __init__(self, binary_path: Optional[str] = None):
        self.engine = None
        self.is_available = False

    def analyze_defense(
        self, 
        board: chess.Board, 
        detected_plan: str, 
        time_limit: float = 0.2
    ) -> List[EngineCandidateMove]:
        """Returns top screened defensive moves or empty list if unavailable."""
        pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/utils/stockfish_client.py` with safe `popen_uci` error handling.
2. Integrate defensive recommendation card in Streamlit under Tab 3.
3. Write `tests/test_stockfish_client.py` mocking the engine response to ensure zero failure when Stockfish is absent.

---

## 5. Verification & Acceptance Criteria
- App runs smoothly whether Stockfish is installed or not.
- When available, returns top 3 candidate moves with tactical evaluation in $< 250 \text{ ms}$.
