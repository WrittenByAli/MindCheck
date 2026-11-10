# Subtask 1.2.1: Dataset Source Acquisition & Selection Manifest

**Subtask ID**: `1.2.1.DATASET.ACQUISITION.MANIFEST`  
**Parent Task**: `Task 1.2`  
**Phase**: Phase 1  
**Milestone**: Milestone 2  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Document public PGN archive source provenance, download/prepare high-rated master games, and create manifest.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Public PGN source (Lichess Elite Database subset, games with White/Black Elo >= 2200).

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `data/raw/lichess_elite_subset.pgn` and `data/raw/manifest.json`.

---

## 4. Detailed Technical Specification
### Technical Specification:
1. PGN Requirements: Standard chess games, minimum 2200 Elo, clean PGN headers (Event, Site, Date, White, Black, Result, Elo, ECO).
2. Manifest Schema (`data/raw/manifest.json`):
   - `archive_name`: "Lichess Elite Database Subset"
   - `provider`: "Lichess.org / Public Domain"
   - `min_elo`: 2200
   - `total_games_target`: 2000-5000
   - `license`: "CC0 / Open Data"
   - `sha256_checksum`: String
   - `download_timestamp`: ISO8601 string.

---

## 5. Claude Code Action Plan
1. Create `scripts/download_sample_dataset.py` to fetch or assemble sample grandmaster PGN games.
2. Generate `data/raw/manifest.json` with archive provenance.
3. Validate that PGN file is readable by `chess.pgn`.

---

## 6. Unit Testing & Acceptance Criteria
- `data/raw/manifest.json` exists and validates against schema.
- Raw PGN contains >= 500 valid master games.
