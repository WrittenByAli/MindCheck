# Subtask 1.2.2: Memory-Efficient Streaming PGN Parser & Sanity Filter

**Subtask ID**: `1.2.2.STREAMING.PGN.PARSER`  
**Parent Task**: `Task 1.2`  
**Phase**: Phase 1  
**Milestone**: Milestone 2  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Build a streaming PGN reader that parses games iteratively with low memory consumption and filters out malformed games.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Path to PGN file (`data/raw/lichess_elite_subset.pgn`).

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/data/pgn_parser.py` containing `StreamingPGNParser`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `StreamingPGNParser`:
   - `stream_games(pgn_path: str) -> Iterator[chess.pgn.Game]`
   - Generator using `chess.pgn.read_game(handle)`.
2. Filtering Rules:
   - Variant: Only standard chess (reject Chess960, Atomic, Horde, etc.).
   - Length: Reject games with < 20 plies (too short for strategic planning).
   - Rating: If Elo headers present, require White Elo >= 2200 and Black Elo >= 2200.
   - Legality: Replay moves through `board.push()`; if any move is illegal, discard game cleanly.
3. Memory Safety: Never read entire file into list or memory. Maintain RAM usage < 300MB.

---

## 5. Claude Code Action Plan
1. Implement `src/data/pgn_parser.py`.
2. Implement sanity filter checks (variant, ply count, rating, move legality).
3. Write `tests/test_pgn_parser.py` with synthetic PGN strings (valid, malformed, non-standard, short).
4. Run `pytest tests/test_pgn_parser.py`.

---

## 6. Unit Testing & Acceptance Criteria
- `StreamingPGNParser` yields only valid standard chess games with >= 20 plies.
- Corrupted moves or non-standard variants are skipped with debug log warnings.
