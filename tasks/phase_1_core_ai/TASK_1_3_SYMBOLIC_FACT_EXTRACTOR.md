# Task 1.3: Symbolic Chess Fact & Predicate Extractor

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 3  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Build a deterministic, ultra-fast symbolic fact extractor using `python-chess` 64-bit unsigned integer bitboards. This module extracts grounded geometric predicates (pawn chains, open files, outposts, king safety, rook batteries) from any board state to support weak-labeling, rule-based baselines, and explainability.

---

## 2. Subtask Breakdown

### Subtask 1.3.1: Bitboard Board Facts & Attacks (`src/rules/fact_extractor.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Implement class `ChessFactExtractor`.
  - Extract 64-bit integer bitmasks for pieces of each color: `pawns`, `knights`, `bishops`, `rooks`, `queens`, `kings`.
  - Compute attack maps using bitwise OR over `board.attacks(square)` for all pieces of a given color.
  - Compute legal mobility counts per piece type.

### Subtask 1.3.2: Pawn Structure & King Quadrant Predicates (`src/rules/fact_extractor.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Specification**:
  - Open Files: Files with zero pawns of either color (`white_pawns | black_pawns == 0` on file bitmask).
  - Semi-Open Files: Files with opponent pawns but zero friendly pawns.
  - Pawn Chains: Detect diagonal pawn connections (pawn at square $s$ defended by friendly pawn at $s \pm 7$ or $s \pm 9$). Identify the base pawn (the undefended root of the chain).
  - Kingside Storm Vector: Count advancing flank pawns (f, g, h files) past rank 4 (White) / rank 5 (Black) directed toward the enemy king quadrant.

### Subtask 1.3.3: Positional Predicates: Outposts & Rook Batteries (`src/rules/fact_extractor.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - **Outpost Detection**:
    - A square $S$ on files c, d, e, or f.
    - $S$ is defended by at least one friendly pawn (`friendly_pawn_attacks & (1 << S) != 0`).
    - $S$ cannot be attacked by opponent pawns, neither now nor by forward pawn moves (enemy pawns on adjacent files are either traded or already past rank of $S$).
    - Check if a friendly minor piece (Knight or Bishop) occupies or targets $S$.
  - **Rook Battery Detection**:
    - Identify files containing $\ge 2$ friendly heavy pieces (Rook + Rook, or Rook + Queen).
    - Determine if the battery controls entrance to the 7th or 8th rank.

### Subtask 1.3.4: Automated Unit Test Suite (`tests/test_fact_extractor.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Test fact extraction against standard FEN benchmarks:
    - *Kasparov vs Karpov d5 outpost FEN*: Verify that `d5` is detected as a valid White outpost.
    - *Alekhine's Gun c-file FEN*: Verify that c-file is detected as an open file with heavy rook battery.
    - *French Defense FEN*: Verify that White's `e5-d4-c3` chain is detected with `c3` as base.

---

## 3. Function Signatures & Data Model

```python
from dataclasses import dataclass
from typing import List, Dict, Set
import chess

@dataclass
class ExtractedBoardFacts:
    ply_index: int
    turn: chess.Color
    white_king_square: chess.Square
    black_king_square: chess.Square
    open_files: List[int]             # File indices 0..7 (a..h)
    white_semi_open_files: List[int]
    black_semi_open_files: List[int]
    white_outpost_squares: List[chess.Square]
    black_outpost_squares: List[chess.Square]
    white_pawn_chains: List[List[chess.Square]]  # Chains with base at index 0
    black_pawn_chains: List[List[chess.Square]]
    white_rook_batteries: List[int]   # File indices with doubled heavy pieces
    black_rook_batteries: List[int]
    flank_pawns_advanced: Dict[str, int] # Count of advanced f,g,h / a,b,c pawns

class ChessFactExtractor:
    def extract_facts(self, board: chess.Board, ply_index: int = 0) -> ExtractedBoardFacts:
        """Extracts complete symbolic bitboard facts in under 1ms."""
        pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/rules/fact_extractor.py` containing `ChessFactExtractor`.
2. Implement bitwise helper functions (`file_mask`, `pawn_attacks_mask`, `king_zone_mask`).
3. Write `tests/test_fact_extractor.py` asserting correct predicate detection across classic FEN strings.
4. Verify execution speed: extracting facts for 1,000 boards must take $< 0.5 \text{ seconds}$.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_fact_extractor.py` -> 100% pass.
- Verified zero false positives on standard starting position FEN.
