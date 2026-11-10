# Task 1.2: Dataset Acquisition & PGN Parsing Pipeline

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 2  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Acquire and parse high-quality chess games from public master archives (Lichess Elite Database, games with player Elo $\ge 2200$). Implement a memory-efficient streaming PGN parser and an 8-ply sliding history window generator that runs without loading the entire archive into RAM.

---

## 2. Subtask Breakdown

### Subtask 1.2.1: Dataset Acquisition & Manifest Generation (`data/raw/manifest.json`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Acquire public domain PGN archive (Lichess Elite Database or curated FIDE Grandmaster archive).
  - Store raw PGN under `data/raw/lichess_elite_subset.pgn`.
  - Create `data/raw/manifest.json` recording:
    - `source_url`: URL of source archive
    - `release_date`: Archive release stamp
    - `min_elo`: 2200
    - `total_games_target`: ~2,000–5,000 games (yielding 30,000–50,000 position windows)
    - `license`: Creative Commons Zero (CC0) / Public Domain
    - `checksum_sha256`: Hash for reproducibility

### Subtask 1.2.2: Memory-Safe Streaming PGN Parser (`src/data/pgn_parser.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Implement class `StreamingPGNParser`.
  - Generator method `stream_valid_games(pgn_path: str) -> Iterator[chess.pgn.Game]`.
  - Filter criteria:
    - Reject non-standard chess variants (only standard chess allowed).
    - Reject games with $< 20$ plies (too short for strategic formulation).
    - Reject games with illegal moves or termination without valid result.
    - Filter by player rating: White and Black Elo $\ge 2200$ (if Elo header present).

### Subtask 1.2.3: 8-Ply Sliding History Window Extractor (`src/data/window_extractor.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - For each valid game, iterate ply-by-ply from ply 12 to termination (avoiding trivial opening book moves plies 0–11).
  - At each ply $t$:
    - Extract recent 8 board states: $[B_{t-7}, B_{t-6}, \dots, B_{t-1}, B_t]$.
    - If $t < 7$, pad missing past states with `None` (handled by padding mask in tensor encoder).
    - Record metadata: `game_id`, `ply_index`, `turn` (White/Black), `last_move_san`, `is_check`.
    - Return structured object `PositionWindow`.

---

## 3. Data Structures & Signatures

```python
from dataclasses import dataclass
from typing import List, Optional
import chess

@dataclass
class PositionWindow:
    game_id: str
    ply_index: int
    turn: chess.Color
    board_history: List[chess.Board]   # Length up to 8 boards
    last_move: Optional[chess.Move]
    white_elo: Optional[int]
    black_elo: Optional[int]
    result: str
```

---

## 4. Claude Code Action Plan
1. Implement `src/data/pgn_parser.py` with `StreamingPGNParser`.
2. Implement `src/data/window_extractor.py` with sliding window logic.
3. Write a helper script `scripts/download_sample_dataset.py` that downloads or compiles a starter sample PGN of high-rated games.
4. Implement `tests/test_pgn_parser.py` using synthetic small PGN strings to verify filtering, move legality, and window length.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_pgn_parser.py` -> Must pass all tests.
- Parse a 1,000-game PGN: RAM usage must remain $< 300 \text{ MB}$ continuously throughout processing.
- Verify that every extracted `PositionWindow` contains legal chess board positions.
