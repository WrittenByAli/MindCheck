# Subtask 2.4.2: PGN File Uploader & Famous Master Game Library

**Subtask ID**: `2.4.2.PGN.LOADER.SAMPLE.LIBRARY`  
**Parent Task**: `Task 2.4`  
**Phase**: Phase 2  
**Milestone**: Milestone 12  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Implement PGN file uploader and pre-bundled library of 5 classic master games.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: User uploaded PGN file or library dropdown selection.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `src/ui/game_loader.py` returning parsed `chess.pgn.Game`.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Sidebar Options:
   - Mode radio: 'Curated Master Match Library' vs 'Upload Custom PGN'.
2. Bundled Library:
   - Game 1: Fischer vs Larsen (1958) -> `K_STORM`
   - Game 2: Petrosian vs Spassky (1966) -> `Q_EXP`
   - Game 3: Kasparov vs Karpov (1985, G16) -> `OUTPOST`
   - Game 4: Alekhine vs Nimzowitsch (1930) -> `ROOK_FILE`
   - Game 5: Botvinnik vs Capablanca (1938) -> `CHAIN_UNDERMINE`
3. Error Handling: Graceful warning alert if uploaded PGN is empty or corrupted.

---

## 5. Claude Code Action Plan
1. Implement `src/ui/game_loader.py`.
2. Create sample PGN files under `data/samples/`.
3. Test loading both bundled games and uploaded PGN files.

---

## 6. Unit Testing & Acceptance Criteria
- All 5 sample games load and parse without error.
- Custom PGN uploader handles edge cases cleanly.
