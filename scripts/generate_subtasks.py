"""Script to generate all fine-grained subtask markdown specification files for MindCheck.
Generates 36 Phase 1 subtask files and 31 Phase 2 subtask files.
"""

import os
from pathlib import Path

BASE_DIR = Path("D:/AI_Project")
P1_SUBTASKS_DIR = BASE_DIR / "tasks" / "phase_1_core_ai" / "subtasks"
P2_SUBTASKS_DIR = BASE_DIR / "tasks" / "phase_2_deployment_and_evaluation" / "subtasks"

P1_SUBTASKS_DIR.mkdir(parents=True, exist_ok=True)
P2_SUBTASKS_DIR.mkdir(parents=True, exist_ok=True)


def write_subtask_file(directory: Path, filename: str, content: str):
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated: {filepath.name}")


# ==============================================================================
# PHASE 1 SUBTASKS DEFINITIONS
# ==============================================================================

phase_1_subtasks = [
    # Task 1.1
    (
        "SUBTASK_1_1_1_DEPENDENCY_SPECIFICATION.md",
        "Subtask 1.1.1: Pinned Dependency Specification & Verification",
        "1.1",
        "Phase 1",
        "Milestone 1",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Define all external library dependencies required by the system with exact compatible versions.",
        "Python 3.11+ interpreter installed on the host system.",
        "`requirements.txt` file at project root; successful pip installation log.",
        """### Technical Requirements:
1. File: `requirements.txt` at repository root.
2. Dependencies:
   - `torch>=2.1.0` (PyTorch core)
   - `torchvision>=0.16.0`
   - `python-chess>=1.9.4` (Chess logic, bitboards, PGN, SVG)
   - `streamlit>=1.30.0` (Interactive web UI)
   - `numpy>=1.24.0,<2.0.0` (Vector manipulation)
   - `pandas>=2.0.0` (DataFrames and metadata)
   - `scikit-learn>=1.3.0` (Metrics, F1, Cohen's kappa)
   - `matplotlib>=3.8.0` and `seaborn>=0.13.0` (Plots)
   - `pyyaml>=6.0.1` and `pydantic>=2.5.0` (Configuration)
   - `pytest>=7.4.0` (Testing framework)
   - `onnx>=1.15.0` and `onnxruntime>=1.16.0` (Fast CPU runtime)
   - `tqdm>=4.66.0` (Progress visualization)
3. Provide CPU and CUDA installation commands in documentation.""",
        """1. Inspect current python environment.
2. Verify package compatibility and write `requirements.txt`.
3. Create `scripts/check_env.py` that verifies imports of all libraries and prints versions.
4. Run environment check script.""",
        "- `python scripts/check_env.py` executes without `ImportError` or version conflicts.\n- PyTorch correctly detects CPU or CUDA backend.",
    ),
    (
        "SUBTASK_1_1_2_DIRECTORY_SCAFFOLDING.md",
        "Subtask 1.1.2: Repository Directory Scaffolding & Package Layout",
        "1.1",
        "Phase 1",
        "Milestone 1",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Establish modular Python package structure with proper `__init__.py` files across all modules.",
        "Root directory `d:/AI_Project`.",
        "`src/` subdirectories (`data`, `rules`, `models`, `tracking`, `ui`, `utils`), `tests/`, `config/`, `data/`.",
        """### Package Structure:
- `src/__init__.py`: Package root, defines `__version__ = '1.0.0'`.
- `src/data/__init__.py`: PGN parsing, tensor encoding, dataset splitting.
- `src/rules/__init__.py`: Bitboard fact extraction and rule templates.
- `src/models/__init__.py`: Neural architectures, training, and inference.
- `src/tracking/__init__.py`: Temporal tracker, fusion, and explanations.
- `src/ui/__init__.py`: Streamlit components and visualization.
- `src/utils/__init__.py`: Config models, logging, and Stockfish client.
- `tests/__init__.py`: Unit test package marker.""",
        """1. Verify existence of all subdirectories.
2. Create standard `__init__.py` files with descriptive docstrings.
3. Verify that `python -c "import src.data, src.rules, src.models, src.tracking, src.ui, src.utils"` succeeds.""",
        "- All packages import cleanly without syntax or import errors.\n- Directory tree conforms to project architectural guidelines.",
    ),
    (
        "SUBTASK_1_1_3_CONFIGURATION_MANAGEMENT.md",
        "Subtask 1.1.3: Type-Safe Configuration Schema with Pydantic & YAML",
        "1.1",
        "Phase 1",
        "Milestone 1",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Create a centralized, type-safe runtime configuration system using PyYAML and Pydantic v2.",
        "`config/config.yaml` file containing all system parameters.",
        "`src/utils/config.py` defining Pydantic models `AppConfig`, `DatasetConfig`, `ModelConfig`, `TrainingConfig`, `TrackingConfig`, `FusionConfig`, `UIConfig`.",
        """### Technical Implementation:
1. `src/utils/config.py`:
   - Class `DatasetConfig(BaseModel)`
   - Class `ModelConfig(BaseModel)`
   - Class `TrainingConfig(BaseModel)`
   - Class `TrackingConfig(BaseModel)`
   - Class `FusionConfig(BaseModel)`
   - Class `UIConfig(BaseModel)`
   - Class `AppConfig(BaseModel)` aggregating all sub-configs.
   - Function `load_config(yaml_path: str = 'config/config.yaml') -> AppConfig`.
2. Validation: Strict type validation for paths, learning rates, epochs, thresholds.""",
        """1. Write `src/utils/config.py`.
2. Implement schema loading and error handling for missing keys.
3. Write `tests/test_config.py` testing config loading and default value fallbacks.
4. Run `pytest tests/test_config.py`.""",
        "- `load_config()` loads `config/config.yaml` into strongly-typed Pydantic model.\n- Invalid YAML structure or bad parameter types trigger clear validation errors.",
    ),
    # Task 1.2
    (
        "SUBTASK_1_2_1_DATASET_ACQUISITION_MANIFEST.md",
        "Subtask 1.2.1: Dataset Source Acquisition & Selection Manifest",
        "1.2",
        "Phase 1",
        "Milestone 2",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Document public PGN archive source provenance, download/prepare high-rated master games, and create manifest.",
        "Public PGN source (Lichess Elite Database subset, games with White/Black Elo >= 2200).",
        "`data/raw/lichess_elite_subset.pgn` and `data/raw/manifest.json`.",
        """### Technical Specification:
1. PGN Requirements: Standard chess games, minimum 2200 Elo, clean PGN headers (Event, Site, Date, White, Black, Result, Elo, ECO).
2. Manifest Schema (`data/raw/manifest.json`):
   - `archive_name`: "Lichess Elite Database Subset"
   - `provider`: "Lichess.org / Public Domain"
   - `min_elo`: 2200
   - `total_games_target`: 2000-5000
   - `license`: "CC0 / Open Data"
   - `sha256_checksum`: String
   - `download_timestamp`: ISO8601 string.""",
        """1. Create `scripts/download_sample_dataset.py` to fetch or assemble sample grandmaster PGN games.
2. Generate `data/raw/manifest.json` with archive provenance.
3. Validate that PGN file is readable by `chess.pgn`.""",
        "- `data/raw/manifest.json` exists and validates against schema.\n- Raw PGN contains >= 500 valid master games.",
    ),
    (
        "SUBTASK_1_2_2_STREAMING_PGN_PARSER.md",
        "Subtask 1.2.2: Memory-Efficient Streaming PGN Parser & Sanity Filter",
        "1.2",
        "Phase 1",
        "Milestone 2",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Build a streaming PGN reader that parses games iteratively with low memory consumption and filters out malformed games.",
        "Path to PGN file (`data/raw/lichess_elite_subset.pgn`).",
        "`src/data/pgn_parser.py` containing `StreamingPGNParser`.",
        """### Technical Implementation:
1. Class `StreamingPGNParser`:
   - `stream_games(pgn_path: str) -> Iterator[chess.pgn.Game]`
   - Generator using `chess.pgn.read_game(handle)`.
2. Filtering Rules:
   - Variant: Only standard chess (reject Chess960, Atomic, Horde, etc.).
   - Length: Reject games with < 20 plies (too short for strategic planning).
   - Rating: If Elo headers present, require White Elo >= 2200 and Black Elo >= 2200.
   - Legality: Replay moves through `board.push()`; if any move is illegal, discard game cleanly.
3. Memory Safety: Never read entire file into list or memory. Maintain RAM usage < 300MB.""",
        """1. Implement `src/data/pgn_parser.py`.
2. Implement sanity filter checks (variant, ply count, rating, move legality).
3. Write `tests/test_pgn_parser.py` with synthetic PGN strings (valid, malformed, non-standard, short).
4. Run `pytest tests/test_pgn_parser.py`.""",
        "- `StreamingPGNParser` yields only valid standard chess games with >= 20 plies.\n- Corrupted moves or non-standard variants are skipped with debug log warnings.",
    ),
    (
        "SUBTASK_1_2_3_SLIDING_WINDOW_EXTRACTOR.md",
        "Subtask 1.2.3: 8-Ply Sliding History Window Extractor",
        "1.2",
        "Phase 1",
        "Milestone 2",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Extract 8-half-move sliding windows across game plies for temporal feature encoding.",
        "`chess.pgn.Game` object.",
        "`src/data/window_extractor.py` returning `Iterator[PositionWindow]`.",
        """### Technical Implementation:
1. Dataclass `PositionWindow`:
   - `game_id: str`
   - `ply_index: int`
   - `turn: chess.Color`
   - `board_history: List[chess.Board]` (length up to 8, past to current)
   - `last_move: Optional[chess.Move]`
   - `is_check: bool`
2. Logic:
   - Play through game ply-by-ply from ply 12 to termination.
   - At each ply $t$, extract past 8 board copies: $[B_{t-7}, \dots, B_t]$.
   - If ply $t < 7$, pad missing history slots with None or initial board states.
   - Yield `PositionWindow`.""",
        """1. Implement `src/data/window_extractor.py`.
2. Ensure board states in history are independent clones (`board.copy()`).
3. Write `tests/test_window_extractor.py` verifying window length and correct ply sequencing.
4. Run `pytest tests/test_window_extractor.py`.""",
        "- Extracted windows always contain up to 8 sequential boards.\n- The final board in `board_history` matches the board state at `ply_index`.",
    ),
    # Task 1.3
    (
        "SUBTASK_1_3_1_BITBOARD_FACT_EXTRACTOR.md",
        "Subtask 1.3.1: 64-Bit Bitboard Fact & Piece Mobility Extractor",
        "1.3",
        "Phase 1",
        "Milestone 3",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement high-speed bitboard extraction for piece locations, attack bitmasks, and mobility counts.",
        "`chess.Board` instance.",
        "`src/rules/fact_extractor.py` returning bitmask dictionaries and mobility counts.",
        """### Technical Implementation:
1. Class `ChessFactExtractor`:
   - Piece masks: `board.pieces_mask(piece_type, color)` for P, N, B, R, Q, K.
   - Attack bitboards: Compute `attack_mask(color)` using bitwise OR over `board.attacks(sq)`.
   - Mobility: Count legal moves per piece type using bitboard popcount.
   - Material balance: Compute standard centipawn material evaluation.""",
        """1. Implement bitboard extraction methods in `src/rules/fact_extractor.py`.
2. Write bitwise helper utilities (popcount, square mask, ray attacks).
3. Write unit tests in `tests/test_fact_extractor.py` verifying bitmask accuracy.
4. Benchmark fact extraction speed across 1,000 boards (< 0.5s total).""",
        "- Bitboard masks match piece positions exactly.\n- Extraction latency per board is < 0.5 ms.",
    ),
    (
        "SUBTASK_1_3_2_PAWN_STRUCTURE_PREDICATES.md",
        "Subtask 1.3.2: Pawn Structure & King Safety Predicate Engine",
        "1.3",
        "Phase 1",
        "Milestone 3",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Extract symbolic predicates for pawn chains, chain base/apex, open/semi-open files, and king pawn shield.",
        "`chess.Board` instance.",
        "Predicate outputs in `ExtractedBoardFacts` (`white_pawn_chains`, `open_files`, `king_shield_integrity`).",
        """### Technical Implementation:
1. Open Files: File mask & (white_pawns | black_pawns) == 0.
2. Semi-Open Files: File has opponent pawns but zero friendly pawns.
3. Pawn Chains: Detect diagonal mutual pawn defense ($s$ defended by $s \pm 7$ or $s \pm 9$). Identify the base pawn (root of chain with no pawn defending it).
4. King Safety / Flank Advance: King quadrant pawn shield status and count of advanced flank pawns (f, g, h / a, b, c).""",
        """1. Implement pawn structure predicates in `src/rules/fact_extractor.py`.
2. Add specific test cases in `tests/test_fact_extractor.py` for pawn chain detection and file classification.
3. Verify with Carlsbad and French Defense pawn structures.""",
        "- Accurately detects open and semi-open files for both colors.\n- Correctly identifies chain bases and flank advances.",
    ),
    (
        "SUBTASK_1_3_3_POSITIONAL_OUTPOST_PREDICATES.md",
        "Subtask 1.3.3: Positional Predicates: Outposts & Heavy Piece Batteries",
        "1.3",
        "Phase 1",
        "Milestone 3",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Extract symbolic predicates for minor piece outposts and heavy piece file doubling.",
        "`chess.Board` instance and recent move history.",
        "Predicate outputs: `outpost_squares`, `occupied_outposts`, `rook_batteries`.",
        """### Technical Implementation:
1. Outpost Detection:
   - Candidate square $S$ on ranks 4-6 (White) or ranks 3-5 (Black), files c, d, e, f.
   - $S$ is defended by at least one friendly pawn.
   - $S$ cannot be attacked by opponent pawns (enemy pawns on adjacent files are traded or past $S$).
   - Check if friendly Knight or Bishop occupies $S$ or can legally reach $S$ in 1 move.
2. Rook Battery:
   - Identify files containing $\ge 2$ friendly major pieces (Rook+Rook or Rook+Queen).
   - Check if battery controls 7th/8th rank entry squares.""",
        """1. Implement outpost and battery algorithms in `src/rules/fact_extractor.py`.
2. Write unit tests in `tests/test_fact_extractor.py` using Kasparov-Karpov 1985 G16 (d5 outpost) and Alekhine's Gun FENs.
3. Verify zero false positive outposts on empty or flank squares.""",
        "- Outpost squares correctly detected only when uncontested by enemy pawns.\n- Heavy batteries correctly detected along open/semi-open files.",
    ),
    (
        "SUBTASK_1_3_4_FACT_EXTRACTOR_UNIT_TESTS.md",
        "Subtask 1.3.4: Automated Unit Test Suite for Fact Extractor",
        "1.3",
        "Phase 1",
        "Milestone 3",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build a comprehensive test suite validating all bitboard predicates on standard chess positions.",
        "`src/rules/fact_extractor.py`.",
        "`tests/test_fact_extractor.py` with 100% predicate coverage.",
        """### Technical Implementation:
1. Test Positions:
   - Initial board (rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1): Verify 0 outposts, 0 open files, 0 storm pawns.
   - Outpost Benchmark: 8/8/3p4/4P3/3N4/8/8/4K2k w - - 0 1 (White knight on d4 supported by e5, no black c/e pawns).
   - Alekhine's Gun: 2r1r1k1/5ppp/8/8/2R5/2R5/2Q5/6K1 w - - 0 1 (Tripled heavy pieces on c-file).
   - Pawn Storm: 5rk1/5ppp/8/6P1/7P/8/8/6K1 w - - 0 1 (Advanced g- and h-pawns).""",
        """1. Write `tests/test_fact_extractor.py`.
2. Add assertions for every field in `ExtractedBoardFacts`.
3. Run `pytest tests/test_fact_extractor.py` and inspect coverage.""",
        "- All fact extractor tests pass without assertions failures.\n- Test suite executes in < 2 seconds.",
    ),
    # Task 1.4
    (
        "SUBTASK_1_4_1_KINGSIDE_STORM_RULE.md",
        "Subtask 1.4.1: Kingside Pawn Storm Operational Rule Template",
        "1.4",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement rule template evaluating heuristic score for Kingside Pawn Storm (`K_STORM`).",
        "`PositionWindow` and `ExtractedBoardFacts`.",
        "`evaluate_kingside_storm(window, facts) -> float` score in [0.0, 1.0].",
        """### Technical Implementation:
1. Operational Criteria:
   - Opponent king is castled kingside (f, g, h files).
   - Recent 8 plies contain $\ge 2$ flank pawn advances on f, g, or h files.
   - Flank pawns have advanced to rank 4+ (White) or rank 5- (Black).
   - Major piece behind advancing pawns adds bonus.
2. Formula:
   $$S = \\min(1.0, 0.4 \\times \\text{pawn\\_pushes} + 0.3 \\times \\text{rank\\_depth} + 0.3 \\times \\text{support})$$""",
        """1. Implement `evaluate_kingside_storm` in `src/rules/rule_engine.py`.
2. Write unit tests in `tests/test_rule_engine.py` with known storm game positions.
3. Test edge cases (queenside castling, quiet maneuvering).""",
        "- Scores > 0.70 on active kingside storm positions.\n- Scores < 0.20 when kings are castled on opposite wings without pawn advances.",
    ),
    (
        "SUBTASK_1_4_2_QUEENSIDE_EXPANSION_RULE.md",
        "Subtask 1.4.2: Queenside Expansion Operational Rule Template",
        "1.4",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement rule template evaluating heuristic score for Queenside Expansion (`Q_EXP`).",
        "`PositionWindow` and `ExtractedBoardFacts`.",
        "`evaluate_queenside_expansion(window, facts) -> float` score in [0.0, 1.0].",
        """### Technical Implementation:
1. Operational Criteria:
   - Recent 8 plies contain $\ge 2$ pawn advances on a, b, or c files.
   - Queenside pawns control space past 4th rank (White) or 5th rank (Black).
   - Minor piece or rook supports queenside pawn mass.
2. Formula:
   $$S = \\min(1.0, 0.4 \\times \\text{queenside\\_pushes} + 0.3 \\times \\text{space\\_advantage} + 0.3 \\times \\text{piece\\_support})$$""",
        """1. Implement `evaluate_queenside_expansion` in `src/rules/rule_engine.py`.
2. Add unit tests for minority attacks and Sicilian queenside pawn pushes.
3. Verify low scores on central or kingside games.""",
        "- Scores > 0.70 on positions with clear queenside expansion.\n- Disregards single isolated pawn pushes.",
    ),
    (
        "SUBTASK_1_4_3_OUTPOST_OCCUPATION_RULE.md",
        "Subtask 1.4.3: Outpost Occupation Operational Rule Template",
        "1.4",
        "Phase 1",
        "Milestone 4",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Implement rule template evaluating heuristic score for Outpost Occupation (`OUTPOST`).",
        "`PositionWindow` and `ExtractedBoardFacts`.",
        "`evaluate_outpost_occupation(window, facts) -> float` score in [0.0, 1.0].",
        """### Technical Implementation:
1. Operational Criteria:
   - Outpost square exists on central files (c, d, e, f) defended by friendly pawn and uncontested by enemy pawns.
   - Friendly Knight or Bishop entered the outpost square during recent 8 plies, or is positioned within 1 legal move.
2. Scoring:
   - 1.0 if minor piece occupies outpost currently.
   - 0.7 if piece maneuvered toward outpost in recent plies.
   - 0.0 if no outpost exists.""",
        """1. Implement `evaluate_outpost_occupation` in `src/rules/rule_engine.py`.
2. Add unit test with Kasparov Karpov Game 16 (d5 knight).
3. Verify score = 0 when outpost is easily attacked by enemy pawn.""",
        "- Returns 1.0 on established outposts occupied by knights/bishops.\n- Returns 0.0 when target square is vulnerable to enemy pawns.",
    ),
    (
        "SUBTASK_1_4_4_ROOK_FILE_BUILDUP_RULE.md",
        "Subtask 1.4.4: Rook-File Buildup Operational Rule Template",
        "1.4",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement rule template evaluating heuristic score for Rook-File Buildup (`ROOK_FILE`).",
        "`PositionWindow` and `ExtractedBoardFacts`.",
        "`evaluate_rook_file_buildup(window, facts) -> float` score in [0.0, 1.0].",
        """### Technical Implementation:
1. Operational Criteria:
   - File is open (no pawns) or semi-open (no friendly pawns).
   - $\ge 2$ heavy pieces (R+R or R+Q) occupy the file.
   - Moving rook to open file occurred in recent 8 plies.
2. Scoring:
   - 1.0 if doubled on open file.
   - 0.7 if doubled on semi-open file.
   - 0.4 if single rook occupies open file.""",
        """1. Implement `evaluate_rook_file_buildup` in `src/rules/rule_engine.py`.
2. Add unit tests for open-file doubling and tripling.
3. Test edge cases with closed files.""",
        "- Returns high score on doubled rooks on open files.\n- Returns 0.0 on closed files with pawns blocking rooks.",
    ),
    (
        "SUBTASK_1_4_5_PAWN_CHAIN_UNDERMINING_RULE.md",
        "Subtask 1.4.5: Pawn-Chain Undermining Operational Rule Template",
        "1.4",
        "Phase 1",
        "Milestone 4",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Implement rule template evaluating heuristic score for Pawn-Chain Undermining (`CHAIN_UNDERMINE`).",
        "`PositionWindow` and `ExtractedBoardFacts`.",
        "`evaluate_chain_undermining(window, facts) -> float` score in [0.0, 1.0].",
        """### Technical Implementation:
1. Operational Criteria:
   - Opponent maintains a locked or connected pawn chain ($\ge 2$ pawns).
   - Friendly pawn lever was pushed in recent 8 plies attacking the base or apex of the chain.
   - Pawn tension exists with capture possibility.
2. Scoring:
   - 1.0 if lever attacking base pawn.
   - 0.8 if lever attacking apex pawn.
   - 0.0 if no chain or no undermining moves.""",
        """1. Implement `evaluate_chain_undermining` in `src/rules/rule_engine.py`.
2. Add unit tests with French Defense (...c5 attacking d4).
3. Test closed structures without pawn levers.""",
        "- Accurately detects pawn levers targeting chain foundations.\n- Distinguishes structural levers from random pawn moves.",
    ),
    (
        "SUBTASK_1_4_6_RULE_ONLY_BASELINE_CLASSIFIER.md",
        "Subtask 1.4.6: Rule-Only Baseline Classifier & Ambiguity Resolver",
        "1.4",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Combine all 5 rule templates into a unified Rule-Only Classifier (Baseline M3) with ambiguity resolution.",
        "`PositionWindow`.",
        "`RuleEvaluationResult` containing predicted class, confidence, all scores, and evidence list.",
        """### Technical Implementation:
1. Class `SymbolicRuleEngine`:
   - Compute scores: $[S_0, S_1, S_2, S_3, S_4]$.
   - Let $S_{\\max} = \\max_k S_k$ and $S_{\\text{second}} = \\text{second\\_highest}(S)$.
   - Decision Logic:
     - If $S_{\\max} < 0.50$: assign `NO_PLAN` (Class 5).
     - If $S_{\\max} - S_{\\text{second}} < 0.15$: assign `NO_PLAN` (Ambiguous conflict).
     - Else: assign class $k = \\arg\\max_k S_k$.
2. Formats supporting evidence list for pedagogical display.""",
        """1. Implement `SymbolicRuleEngine.evaluate_window` in `src/rules/rule_engine.py`.
2. Write `tests/test_rule_engine.py` verifying multi-class resolution and ambiguity filtering.
3. Verify deterministic output across repeat evaluations.""",
        "- Predicts dominant class when evidence is strong.\n- Gracefully falls back to `NO_PLAN` when scores are low or conflicting.",
    ),
    # Task 1.5
    (
        "SUBTASK_1_5_1_LOOKAHEAD_WEAK_LABEL_ENGINE.md",
        "Subtask 1.5.1: Lookahead Retrospective Execution Heuristic Engine",
        "1.5",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Build the weak-label generator inspecting future plies to verify plan execution for training data.",
        "Full `chess.pgn.Game` and position ply index $t$.",
        "`src/data/weak_labeler.py` returning weak label class ID and confidence.",
        """### Technical Implementation:
1. Class `RetrospectiveWeakLabeler`:
   - Inspect future plies $t+1 \\dots t+8$:
     - For `OUTPOST`: Did minor piece occupy outpost and remain for $\\ge 2$ plies?
     - For `K_STORM`: Did flank pawns advance further or trade off opening king files?
     - For `ROOK_FILE`: Did heavy pieces maintain control or penetrate rank 7?
     - For `CHAIN_UNDERMINE`: Did pawn structure alter through captures or tension resolution?
   - If future confirms execution: assign plan label.
   - If future reveals tactical diversion or immediate blunder: assign `NO_PLAN` or reject.""",
        """1. Implement `src/data/weak_labeler.py`.
2. Implement future lookahead checkers for all 5 classes.
3. Write `tests/test_weak_labeler.py` verifying retrospective confirmation logic.
4. Benchmark weak-labeling speed on 100 games.""",
        "- Confirmed executed plans receive high-confidence weak labels.\n- Unfinished or abandoned plans are rejected or labeled `NO_PLAN`.",
    ),
    (
        "SUBTASK_1_5_2_NOISE_FILTER_AND_BALANCING.md",
        "Subtask 1.5.2: Label Noise Filtering, Ambiguity Rejection & Class Balancing",
        "1.5",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Filter noisy weak labels, enforce class balancing, and cap samples per game.",
        "Raw extracted weak-labeled records.",
        "Filtered and balanced record collection (target 20,000-50,000 samples).",
        """### Technical Implementation:
1. Noise Filter:
   - Discard records where weak confidence $< 0.55$.
   - Discard records with extreme material imbalance ($|\\Delta| > 800$ centipawns, e.g. queen down).
2. Per-Game Capping:
   - Cap maximum extracted samples from any single game to $\\le 25$ positions.
3. Stratified Balancing:
   - Target roughly 15-20% representation per active plan class and ~20% `NO_PLAN`.""",
        """1. Implement filtering and balancing routines in `src/data/weak_labeler.py`.
2. Write tests asserting sample cap per game and confidence thresholding.
3. Generate distribution audit plots.""",
        "- No single game contributes > 25 positions.\n- No class occupies > 30% of total dataset (balanced distribution).",
    ),
    (
        "SUBTASK_1_5_3_GAME_LEVEL_SPLITTER.md",
        "Subtask 1.5.3: Game-Level Train/Validation/Test Splitter (70/15/15)",
        "1.5",
        "Phase 1",
        "Milestone 4",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Partition dataset strictly at the game level (70% train, 15% val, 15% test) to prevent window leakage.",
        "List of labeled position records.",
        "`train_records.parquet`, `val_records.parquet`, `test_records.parquet`.",
        """### Technical Implementation:
1. Class `GameLevelSplitter`:
   - Extract unique `game_id` set.
   - Deterministic hash partitioning: `hash(game_id + seed) % 100`:
     - 0..69: Train (70%)
     - 70..84: Validation (15%)
     - 85..99: Test (15%)
   - Group all position windows by their parent `game_id`.
2. Assertion: $\\text{Games}_{\\text{train}} \\cap \\text{Games}_{\\text{val}} \\cap \\text{Games}_{\\text{test}} = \\emptyset$.""",
        """1. Implement `src/data/dataset_splitter.py`.
2. Write `tests/test_dataset_splitter.py` verifying zero game overlap across splits.
3. Save split record tables to `data/processed/`.""",
        "- Game overlap across splits is strictly 0%.\n- Class proportions are preserved across train, val, and test splits.",
    ),
    (
        "SUBTASK_1_5_4_DATASET_AUDIT_MANIFEST.md",
        "Subtask 1.5.4: Dataset Manifest & Statistical Distribution Audit",
        "1.5",
        "Phase 1",
        "Milestone 4",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Generate automated dataset summary report with class counts, rating statistics, and ply distributions.",
        "Split parquet/csv files in `data/processed/`.",
        "`data/processed/dataset_summary.json` and statistical bar charts.",
        """### Technical Implementation:
1. Script `src/data/audit.py`:
   - Compute: total games, total windows, per-class counts in train/val/test splits.
   - Compute mean player rating, mean ply index, and result breakdown (1-0, 0-1, 1/2-1/2).
   - Export summary to `data/processed/dataset_summary.json`.
   - Generate distribution plots in `experiments/plots/dataset_class_distribution.png`.""",
        """1. Implement `src/data/audit.py`.
2. Run audit over processed dataset.
3. Verify JSON summary output.""",
        "- `data/processed/dataset_summary.json` exists with valid metrics.\n- Class distribution charts generated.",
    ),
    # Task 1.6
    (
        "SUBTASK_1_6_1_BOARD_CANONICAL_ENCODER.md",
        "Subtask 1.6.1: Single-Ply Canonical 14-Plane Board State Encoder",
        "1.6",
        "Phase 1",
        "Milestone 5",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Encode a single chess board state into 14 binary $8 \\times 8$ spatial planes.",
        "`chess.Board` instance and attacking side color.",
        "NumPy array of shape `(14, 8, 8)`, dtype `float32`.",
        """### Technical Implementation:
1. Class `BoardPlaneEncoder`:
   - Planes 0-5: Friendly P, N, B, R, Q, K
   - Planes 6-11: Opponent p, n, b, r, q, k
   - Plane 12: Castling rights (4 bits replicated across 8x8)
   - Plane 13: En-passant square (1 on target square, else 0)
2. Bitboard conversion: Use `board.pieces_mask()` and bit shifts for fast array filling.""",
        """1. Implement `BoardPlaneEncoder` in `src/data/tensor_encoder.py`.
2. Write unit tests in `tests/test_tensor_encoder.py` asserting shape, dtypes, and piece counts.
3. Benchmark encoding speed (< 0.1ms per board).""",
        "- Output shape is strictly `(14, 8, 8)` with dtype `float32`.\n- Piece counts match board representation.",
    ),
    (
        "SUBTASK_1_6_2_HISTORY_STACKING_NORMALIZATION.md",
        "Subtask 1.6.2: 8-Ply History Stacking & Attacker Normalization",
        "1.6",
        "Phase 1",
        "Milestone 5",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Stack 8 consecutive board states into shape `(112, 8, 8)` and normalize board orientation relative to attacking side.",
        "List of 8 `chess.Board` states and attacking player color.",
        "NumPy array of shape `(112, 8, 8)`.",
        """### Technical Implementation:
1. Normalization Logic:
   - If attacking side is White: keep standard orientation.
   - If attacking side is Black: flip array along row axis (`np.flip(arr, axis=-2)`), swap Friendly (0-5) and Opponent (6-11) piece planes.
   - Result: Attacking player is always advancing toward row 0 / Rank 8.
2. Channel Stacking:
   - $8 \\times 14 = 112$ channels ordered chronologically from $t-7$ to $t$.""",
        """1. Implement stacking and normalization in `src/data/tensor_encoder.py`.
2. Write unit test with identical pawn storm played by White and Black; verify identical normalized tensors.
3. Test channel alignment.""",
        "- Output shape is strictly `(112, 8, 8)`.\n- Rotational invariance confirmed between White and Black maneuvers.",
    ),
    (
        "SUBTASK_1_6_3_HISTORY_PADDING_VALIDITY_MASK.md",
        "Subtask 1.6.3: History Padding Mask & Sequence Validity Encoding",
        "1.6",
        "Phase 1",
        "Milestone 5",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Handle early game plies (< 8 plies) with zero-padding and produce 1D validity masks.",
        "Position history with length $< 8$.",
        "Padded tensor `(112, 8, 8)` and 1D validity mask `(8,)`.",
        """### Technical Implementation:
1. Zero-Padding:
   - For missing plies $t-7 \\dots t-k$, zero out the corresponding $14 \\times 8 \\times 8$ channels.
2. Validity Mask:
   - Mask $M \\in \\mathbb{R}^8$ where $M_i = 1.0$ if ply exists, $0.0$ if padded.
3. Guarantees model can analyze opening positions without crashing.""",
        """1. Implement padding and mask logic in `src/data/tensor_encoder.py`.
2. Write unit tests for plies 0, 1, 4, 7, and 20.
3. Verify mask values match valid ply counts.""",
        "- Padded positions produce valid tensors without NaN.\n- Validity mask accurately reflects available history depth.",
    ),
    (
        "SUBTASK_1_6_4_PYTORCH_DATASET_DATALOADER.md",
        "Subtask 1.6.4: High-Performance PyTorch Dataset & DataLoader",
        "1.6",
        "Phase 1",
        "Milestone 5",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Build PyTorch Dataset and DataLoader supporting memory-mapped chunk caching.",
        "Split record tables and PGN source.",
        "`src/data/chess_dataset.py` with `ChessPlanDataset`.",
        """### Technical Implementation:
1. Class `ChessPlanDataset(torch.utils.data.Dataset)`:
   - Supports pre-encoded `.npy` / `.pt` memory-mapped arrays or dynamic encoding.
   - Returns: `(tensor, mask, label_id)` tuple.
2. Factory function `create_dataloaders(config)`:
   - Returns `train_loader`, `val_loader`, `test_loader` with pin_memory, multi-worker loading, and shuffling.""",
        """1. Implement `src/data/chess_dataset.py`.
2. Write `tests/test_dataset.py` testing iteration, batch shapes, and worker stability.
3. Test DataLoader throughput (> 1,000 samples/sec on GPU/CPU).""",
        "- Batch output shapes: `(batch_size, 112, 8, 8)`, `(batch_size, 8)`, `(batch_size,)`.\n- Zero memory leaks during multi-epoch iteration.",
    ),
    # Task 1.7
    (
        "SUBTASK_1_7_1_RESIDUAL_CNN_BACKBONE.md",
        "Subtask 1.7.1: Residual Convolutional Backbone Implementation",
        "1.7",
        "Phase 1",
        "Milestone 6",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement the custom PyTorch Residual CNN backbone with 4 residual blocks and 64 channels.",
        "Input tensor `(batch_size, 112, 8, 8)`.",
        "`src/models/residual_cnn.py` defining `ResidualBlock` and `HistoryChessResNet`.",
        """### Technical Implementation:
1. `ResidualBlock(nn.Module)`:
   - Conv2D(64, 64, kernel=3, padding=1, bias=False) -> BatchNorm2D(64) -> GELU()
   - Conv2D(64, 64, kernel=3, padding=1, bias=False) -> BatchNorm2D(64)
   - Skip connection: $y = \\text{GELU}(x + \\mathcal{F}(x))$
2. Stem:
   - Conv2D(112, 64, kernel=3, padding=1, bias=False) -> BatchNorm2D(64) -> GELU()
3. Backbone: 4 sequential Residual Blocks.""",
        """1. Implement `ResidualBlock` and backbone in `src/models/residual_cnn.py`.
2. Write unit tests in `tests/test_models.py` verifying forward pass and residual addition.
3. Assert parameter count is ~250k-300k.""",
        "- Forward pass succeeds without shape errors.\n- Gradients backpropagate through all residual blocks.",
    ),
    (
        "SUBTASK_1_7_2_CLASSIFICATION_HEAD.md",
        "Subtask 1.7.2: Classification Head & Softmax Probability Estimator",
        "1.7",
        "Phase 1",
        "Milestone 6",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement Global Average Pooling (GAP), LayerNorm, Dropout, and 6-class linear classification head.",
        "Backbone feature map `(batch_size, 64, 8, 8)`.",
        "Raw logits `(batch_size, 6)` and softmax probabilities.",
        """### Technical Implementation:
1. Head Architecture:
   - `nn.AdaptiveAvgPool2d((1, 1))` pooling to `(batch_size, 64)`
   - `nn.Flatten()`
   - `nn.LayerNorm(64)`
   - `nn.Dropout(p=0.20)`
   - `nn.Linear(64, num_classes=6)`
2. Methods:
   - `forward(x) -> logits`
   - `predict_proba(x) -> probabilities` via `torch.softmax(logits, dim=-1)`.""",
        """1. Implement classification head in `src/models/residual_cnn.py`.
2. Write unit tests asserting output shape `(batch_size, 6)` and probabilities sum to 1.0.
3. Test dropout toggling between train() and eval() modes.""",
        "- Output tensor has shape `(batch_size, 6)`.\n- Sum of softmax probabilities equals 1.0 per sample.",
    ),
    (
        "SUBTASK_1_7_3_SINGLE_BOARD_BASELINE_CNN.md",
        "Subtask 1.7.3: Single-Board Current-Position Baseline CNN",
        "1.7",
        "Phase 1",
        "Milestone 6",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement matched single-board CNN architecture (14 input channels) for scientific history ablation.",
        "Current board tensor `(batch_size, 14, 8, 8)`.",
        "`src/models/single_board_cnn.py` defining `CurrentPositionResNet`.",
        """### Technical Implementation:
1. Matched Capacity:
   - Class `CurrentPositionResNet(nn.Module)`.
   - In-channels: 14 (only current position, zero history).
   - Identical stem, 4 residual blocks, 64 channels, and 6-class classification head.
2. Directly answers the primary research question by providing a controlled ablation baseline.""",
        """1. Implement `src/models/single_board_cnn.py`.
2. Write unit test verifying forward pass on `(batch_size, 14, 8, 8)`.
3. Verify parameter parity with history-aware network.""",
        "- Single-board baseline runs forward passes cleanly.\n- Architecture matches history model in all aspects except input channels.",
    ),
    (
        "SUBTASK_1_7_4_ONNX_TORCHSCRIPT_EXPORT.md",
        "Subtask 1.7.4: Checkpointing, TorchScript & ONNX Export Pipeline",
        "1.7",
        "Phase 1",
        "Milestone 6",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build automated serialization pipeline exporting PyTorch models to TorchScript and ONNX formats for CPU inference.",
        "Trained PyTorch model and checkpoint path.",
        "`models/checkpoints/best_resnet.pt`, `model.torchscript.pt`, `model.onnx`.",
        """### Technical Implementation:
1. Script `src/models/export.py`:
   - `export_to_onnx(model, output_path)`: Uses `torch.onnx.export` with dynamic batching.
   - `export_to_torchscript(model, output_path)`: Uses `torch.jit.trace`.
2. Validation: Load exported ONNX model with `onnxruntime.InferenceSession` and assert output matches PyTorch within $1e-4$ tolerance.""",
        """1. Implement `src/models/export.py`.
2. Add export tests in `tests/test_export.py`.
3. Verify ONNX CPU execution speed (< 10ms).""",
        "- `model.onnx` passes `onnx.checker.check_model()`.\n- Numeric output matches PyTorch model within numerical tolerance.",
    ),
    # Task 1.8
    (
        "SUBTASK_1_8_1_TRAINING_LOSS_CLASS_WEIGHTS.md",
        "Subtask 1.8.1: PyTorch Training Loop, Loss & Class Weighting",
        "1.8",
        "Phase 1",
        "Milestone 7",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Build the PyTorch training loop with class-weighted Cross-Entropy loss and label smoothing.",
        "DataLoader instances and configuration parameters.",
        "`src/models/train.py` with complete optimization step.",
        """### Technical Implementation:
1. Class `ModelTrainer`:
   - Loss: `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`.
   - Weights: $w_c = \\frac{N}{K \\cdot N_c}$ to prevent majority class dominance.
   - Optimizer: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`.
   - Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.""",
        """1. Implement training loop in `src/models/train.py`.
2. Write unit tests in `tests/test_training_step.py` verifying loss decreases across 5 iterations on synthetic batch.
3. Test class weight calculation.""",
        "- Single training step executes without NaN or exploding gradients.\n- Class weights correctly penalize rare class errors.",
    ),
    (
        "SUBTASK_1_8_2_VALIDATION_EARLY_STOPPING_SCHEDULER.md",
        "Subtask 1.8.2: Validation Early Stopping, Learning Rate Scheduler & Seeds",
        "1.8",
        "Phase 1",
        "Milestone 7",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Implement validation Macro-F1 early stopping, cosine annealing schedule, and deterministic seeding.",
        "Validation DataLoader and checkpoint directory.",
        "Early stopping handler and learning rate updates.",
        """### Technical Implementation:
1. Scheduler: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)`.
2. Early Stopping: Monitor **Validation Macro-F1**; if no improvement for 5 consecutive epochs, terminate and reload best weights.
3. Deterministic Seed: Set seed 42 across PyTorch, NumPy, and Python random.""",
        """1. Implement early stopping and scheduling in `src/models/train.py`.
2. Write test verifying early stopping triggers when validation metrics plateau.
3. Verify deterministic repeatability across duplicate seeded runs.""",
        "- Training halts automatically when validation metric stops improving.\n- Best checkpoint restored and saved to disk.",
    ),
    (
        "SUBTASK_1_8_3_HISTORY_LENGTH_ABLATION.md",
        "Subtask 1.8.3: History Length Ablation Engine (1 vs 4 vs 8 Plies)",
        "1.8",
        "Phase 1",
        "Milestone 7",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Execute controlled ablation comparing 1-ply, 4-ply, and 8-ply history models under matched hyperparameter settings.",
        "Identical train/val splits and random seeds.",
        "`src/models/train_ablation.py` and `experiments/ablation_results.json`.",
        """### Technical Implementation:
1. Script `src/models/train_ablation.py`:
   - Trains 3 variants:
     1. 1-ply: Current board only ($14 \\times 8 \\times 8$)
     2. 4-ply: Recent 4 half-moves ($56 \\times 8 \\times 8$)
     3. 8-ply: Full MindCheck model ($112 \\times 8 \\times 8$)
   - Uses matched batch sizes, optimizer, and seeds.
   - Evaluates on validation set and logs Macro-F1 deltas.""",
        """1. Implement `src/models/train_ablation.py`.
2. Run ablation benchmark.
3. Output comparative summary table and bar chart in `experiments/plots/ablation_history.png`.""",
        "- Successfully trains all 3 ablation models.\n- Produces empirical F1 comparison answering the primary research question.",
    ),
    (
        "SUBTASK_1_8_4_METRICS_LOGGING_CHECKPOINTING.md",
        "Subtask 1.8.4: Training Metrics Logging, TensorBoard & Checkpoint Selection",
        "1.8",
        "Phase 1",
        "Milestone 7",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Log epoch losses, accuracies, and per-class F1 metrics; manage checkpoint files.",
        "Training and validation statistics per epoch.",
        "`models/checkpoints/best_resnet.pt`, `training_metrics.json`, and learning curve plots.",
        """### Technical Implementation:
1. Class `MetricsLogger`:
   - Logs: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1`, per-class F1.
   - Exports `experiments/training_history.json`.
   - Plots training vs validation loss and Macro-F1 curves to `experiments/plots/learning_curves.png`.
   - Saves checkpoint dict with `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_f1`.""",
        """1. Implement `src/models/logger.py`.
2. Integrate logger into `src/models/train.py`.
3. Verify plot generation and checkpoint file persistence.""",
        "- `best_resnet.pt` saved with metadata and retrievable weights.\n- Training curve plots generated cleanly.",
    ),
    # Task 1.9
    (
        "SUBTASK_1_9_1_ANNOTATION_RUBRIC_GUIDELINE.md",
        "Subtask 1.9.1: Written Annotation Rubric & Guideline Documentation",
        "1.9",
        "Phase 1",
        "Milestone 8",
        "Shared (Both Members)",
        "Shahzad Ahmad & Muhammad Ali",
        "Formalize the human annotation rubric and rating instructions for the 250 held-out test windows.",
        "`docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md`.",
        "`data/annotations/ANNOTATION_GUIDELINES.md` signed off by both members.",
        """### Technical Implementation:
1. Operational guidelines for annotators:
   - History-only constraint: Inspect ONLY plies $t-7 \\dots t$. Never peek forward.
   - Independence constraint: No communication between annotators during scoring.
   - Record: Primary plan, confidence score (0.4-1.0), secondary plausible plan, and rationale.
   - Disclose reviewer chess ratings in final submission.""",
        """1. Draft and finalize `data/annotations/ANNOTATION_GUIDELINES.md`.
2. Both members review and confirm operational consensus.
3. Commit guideline document to repository.""",
        "- Guidelines clearly specify boundary criteria for all 5 plans and `NO_PLAN`.\n- Both team members understand and follow the identical rubric.",
    ),
    (
        "SUBTASK_1_9_2_TEST_WINDOW_SAMPLER.md",
        "Subtask 1.9.2: 250 Held-Out Position Window Extraction Tool",
        "1.9",
        "Phase 1",
        "Milestone 8",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Extract exactly 250 stratified held-out position windows from the test split games for human annotation.",
        "`test_records.parquet` from 15% test split.",
        "`data/annotations/raw_annotations_member1.csv` and `raw_annotations_member2.csv`.",
        """### Technical Implementation:
1. Script `src/data/sample_test_set.py`:
   - Sample 250 distinct position windows from test games.
   - Stratify across game phases: Opening (plies 15-25), Middlegame (26-55), Late (56+).
   - Export blank annotation CSV templates for Shahzad Ahmad and Muhammad Ali.
   - Mask out all weak labels and model predictions to prevent bias.""",
        """1. Implement `src/data/sample_test_set.py`.
2. Run sampling script.
3. Verify that sampled games have 0% overlap with training/validation splits.""",
        "- Exactly 250 unique test position rows generated.\n- No labels or model hints present in annotation sheets.",
    ),
    (
        "SUBTASK_1_9_3_INDEPENDENT_DOUBLE_BLIND_ANNOTATION.md",
        "Subtask 1.9.3: Double-Blind Independent Annotation Execution",
        "1.9",
        "Phase 1",
        "Milestone 8",
        "Shared (Both Members)",
        "Muhammad Ali & Shahzad Ahmad",
        "Execute independent double-blind manual annotation of 250 test windows by both team members.",
        "Blank CSV files `raw_annotations_member1.csv` and `raw_annotations_member2.csv`.",
        "Completed independent annotation CSVs with 250 labeled rows each.",
        """### Protocol Execution:
1. Shahzad Ahmad completes `data/annotations/raw_annotations_member1.csv` independently.
2. Muhammad Ali completes `data/annotations/raw_annotations_member2.csv` independently.
3. Each record contains: `window_id`, `primary_label`, `confidence`, `secondary_label`, `rationale`.""",
        """1. Both members label positions independently.
2. Commit completed CSV files.
3. Verify no missing cells or invalid class strings.""",
        "- Both CSV files contain exactly 250 valid entries.\n- Class names conform to taxonomy strings.",
    ),
    (
        "SUBTASK_1_9_4_COHEN_KAPPA_ADJUDICATION.md",
        "Subtask 1.9.4: Cohen's Kappa Inter-Rater Reliability & Adjudication Tool",
        "1.9",
        "Phase 1",
        "Milestone 8",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Calculate Cohen's Kappa score (\\kappa), generate conflict report, and produce frozen consensus ground truth.",
        "Completed annotation files from both members.",
        "`data/annotations/test_ground_truth.csv` and `experiments/inter_rater_reliability.json`.",
        """### Technical Implementation:
1. Script `src/models/adjudicate.py`:
   - Load both member annotation CSVs.
   - Compute observed agreement $p_o$ and Cohen's Kappa:
     $$\\kappa = \\frac{p_o - p_e}{1 - p_e}$$
   - Target $\\kappa \\ge 0.65$ (substantial agreement).
   - Export disputed positions to `data/annotations/disagreements.csv`.
2. Joint Adjudication: Resolve disputes through discussion and export final frozen `test_ground_truth.csv`.""",
        """1. Implement `src/models/adjudicate.py`.
2. Run script to calculate Kappa and extract disagreements.
3. Conduct consensus meeting to adjudicate disputed rows.
4. Freeze `test_ground_truth.csv`.""",
        "- `test_ground_truth.csv` finalized with 250 adjudicated labels.\n- Inter-rater Kappa score documented and reported.",
    ),
]

# ==============================================================================
# PHASE 2 SUBTASKS DEFINITIONS
# ==============================================================================

phase_2_subtasks = [
    # Task 2.1
    (
        "SUBTASK_2_1_1_STATE_MACHINE_HYSTERESIS.md",
        "Subtask 2.1.1: Per-Player State Machine & Hysteresis Logic",
        "2.1",
        "Phase 2",
        "Milestone 9",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Implement per-player plan state tracking with 2-ply hysteresis to eliminate prediction flickering.",
        "Stream of per-ply model predictions and active player turn.",
        "`src/tracking/state_tracker.py` with `TemporalPlanTracker`.",
        """### Technical Implementation:
1. Maintain separate state for White and Black.
2. Hysteresis Transition Rules:
   - Require candidate plan $C$ to be predicted with confidence $\\ge 0.45$ for 2 consecutive plies before activating.
   - Maintain active plan even if confidence dips for 1 ply.
   - Require 2 consecutive contradictory updates to switch active plans.
   - If `NO_PLAN` persists for $\\ge 3$ plies, decay active state to `NO_PLAN`.""",
        """1. Implement state machine in `src/tracking/state_tracker.py`.
2. Write unit tests in `tests/test_state_tracker.py` simulating move sequences.
3. Verify that isolated 1-ply noise spikes do not trigger plan flips.""",
        "- State tracker maintains persistent active plan across consecutive plies.\n- Hysteresis successfully filters out transient tactical spikes.",
    ),
    (
        "SUBTASK_2_1_2_PLAN_ONSET_BACKTRACKING.md",
        "Subtask 2.1.2: Earliest Detected Onset Estimation & Timeline History",
        "2.1",
        "Phase 2",
        "Milestone 9",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Detect the earliest supporting ply of an activated plan by backtracking through the history window.",
        "Candidate activation event in `TemporalPlanTracker`.",
        "`onset_ply` integer attribute and timeline record in `PlayerPlanState`.",
        """### Technical Implementation:
1. Onset Backtracking:
   - When candidate plan $C$ activates at ply $t$:
     - Scan backward through the 8-ply window to identify the earliest ply $t_{\\text{onset}}$ where supporting moves or candidate confidence $> 0.35$ appeared.
   - Record `onset_ply = t_onset`.
   - Store `consecutive_plies_active = t - t_onset + 1`.""",
        """1. Implement onset backtracking in `src/tracking/state_tracker.py`.
2. Add unit test verifying onset detection on known games.
3. Test edge case where onset occurs at ply 0.""",
        "- Onset ply correctly identifies the first supporting ply of the plan.\n- Timeline accurately logs plan duration.",
    ),
    (
        "SUBTASK_2_1_3_PROGRESS_VS_THREAT_METRIC.md",
        "Subtask 2.1.3: Plan Progress vs Threat Severity Metric Formulation",
        "2.1",
        "Phase 2",
        "Milestone 9",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Formulate independent metrics for plan completion progress versus immediate tactical threat severity.",
        "`PlayerPlanState` and `ExtractedBoardFacts`.",
        "`progress_score` and `threat_score` floats in [0.0, 1.0].",
        """### Technical Implementation:
1. Plan Progress ($0.0 \\dots 1.0$):
   - Measures completion of strategic preconditions (e.g. flank pawns advanced to target rank, pieces placed on outposts).
2. Threat Severity ($0.0 \\dots 1.0$):
   - Measures danger to defender (king proximity, attackers vs defenders, open file entry threats).
3. Displayed as separate meters in UI to avoid confusing progress with danger.""",
        """1. Implement metric evaluators in `src/tracking/state_tracker.py`.
2. Write unit tests asserting bounds [0.0, 1.0].
3. Verify progress increases as pieces advance toward target squares.""",
        "- Progress score correlates with geometric advancement.\n- Threat score reflects positional pressure on defending king.",
    ),
    (
        "SUBTASK_2_1_4_STATE_TRACKER_SIMULATION_TESTS.md",
        "Subtask 2.1.4: Unit Tests & Move-Sequence Simulation for State Tracker",
        "2.1",
        "Phase 2",
        "Milestone 9",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build a comprehensive simulation test suite for temporal plan tracking across full game sequences.",
        "`src/tracking/state_tracker.py`.",
        "`tests/test_state_tracker.py` with multi-ply simulation scenarios.",
        """### Technical Implementation:
1. Test Scenarios:
   - Sustained plan: 10 plies of `OUTPOST` -> remains active throughout.
   - Alternating noise: `[K_STORM, NO_PLAN, K_STORM, NO_PLAN]` -> filters oscillation.
   - Strategic shift: 5 plies of `K_STORM` followed by 3 plies of `Q_EXP` -> clean transition.
   - Decay: 5 plies of `NO_PLAN` -> active plan gracefully decays to `NO_PLAN`.""",
        """1. Implement simulation scenarios in `tests/test_state_tracker.py`.
2. Run `pytest tests/test_state_tracker.py`.
3. Check 100% test coverage of state machine branches.""",
        "- All simulated game sequences pass expected state assertions.\n- Tracker correctly handles game resets.",
    ),
    # Task 2.2
    (
        "SUBTASK_2_2_1_SCORE_CALIBRATION_NORMALIZATION.md",
        "Subtask 2.2.1: Score Calibration & Symbolic Rule Normalization",
        "2.2",
        "Phase 2",
        "Milestone 10",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Calibrate neural logits with temperature scaling and normalize symbolic rule scores.",
        "Raw model logits and symbolic rule scores.",
        "Calibrated probabilities and normalized rule vectors.",
        """### Technical Implementation:
1. Temperature Scaling:
   - $P_{\\text{neural}}(c) = \\frac{\\exp(z_c / T)}{\\sum_j \\exp(z_j / T)}$ with $T \\approx 1.2$ tuned on validation data.
2. Rule Normalization:
   - Normalize rule scores $S_{\\text{rule}}(c) \\in [0.0, 1.0]$ across classes: $\\hat{S}_{\\text{rule}}(c) = \\frac{S_{\\text{rule}}(c)}{\\sum_k S_{\\text{rule}}(k) + \\epsilon}$.""",
        """1. Implement calibration functions in `src/tracking/fusion.py`.
2. Write unit tests in `tests/test_fusion.py` asserting probability sums and positive bounds.
3. Test temperature scaling effect on entropy.""",
        "- Output vectors strictly sum to 1.0.\n- Overconfident extreme probabilities are softened appropriately.",
    ),
    (
        "SUBTASK_2_2_2_WEIGHTED_SCORE_FUSION.md",
        "Subtask 2.2.2: Validation-Weighted Neural-Rule Score Fusion Engine",
        "2.2",
        "Phase 2",
        "Milestone 10",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Blend neural probabilities with symbolic rule scores using validation-tuned weight \\alpha.",
        "Calibrated neural probabilities and normalized rule scores.",
        "`FusedPrediction` object with hybrid score vector.",
        """### Technical Implementation:
1. Fusion Formula:
   $$S_{\\text{hybrid}}(c) = \\alpha \\cdot P_{\\text{neural}}(c) + (1 - \\alpha) \\cdot S_{\\text{rule}}(c)$$
2. Weight Parameter: $\\alpha = 0.70$ (default, configurable via YAML).
3. Script `scripts/tune_fusion_alpha.py` running grid search $\\alpha \\in [0.0, 1.0]$ on validation set.""",
        """1. Implement `HybridFusionEngine.fuse` in `src/tracking/fusion.py`.
2. Implement `scripts/tune_fusion_alpha.py`.
3. Add unit tests verifying fusion arithmetic and bounds.""",
        "- Hybrid score combines both modalities smoothly.\n- Yields higher Macro-F1 than rule baseline or neural model alone.",
    ),
    (
        "SUBTASK_2_2_3_UNCERTAINTY_ABSTENTION_GATING.md",
        "Subtask 2.2.3: Prediction Abstention & Ambiguity Threshold Filter",
        "2.2",
        "Phase 2",
        "Milestone 10",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Implement confidence and margin abstention gates to flag ambiguous or low-confidence positions.",
        "Hybrid score vector $S_{\\text{hybrid}}$.",
        "`is_abstained: bool`, `abstention_reason: str`, and final class assignment.",
        """### Technical Implementation:
1. Confidence Gate: $\\max_c S_{\\text{hybrid}}(c) < \\theta_{\\text{conf}}$ (default: 0.40).
2. Margin Gate: $S_{\\text{hybrid}}(c_{(1)}) - S_{\\text{hybrid}}(c_{(2)}) < \\theta_{\\text{margin}}$ (default: 0.10).
3. If either gate triggers:
   - `is_abstained = True`
   - Assign class `NO_PLAN` (Class 5) with status 'Strategically Ambiguous'.""",
        """1. Implement abstention gates in `src/tracking/fusion.py`.
2. Write unit tests in `tests/test_fusion.py` testing uniform and close-margin distributions.
3. Compute selective accuracy on accepted predictions.""",
        "- Successfully flags ambiguous and uniform probability profiles.\n- Accuracy on non-abstained predictions is significantly higher than full test set.",
    ),
    # Task 2.3
    (
        "SUBTASK_2_3_1_CONTROLLED_EXPLANATION_TEMPLATES.md",
        "Subtask 2.3.1: Controlled Explanation Template Library",
        "2.3",
        "Phase 2",
        "Milestone 11",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build deterministic template library separating observed board facts from model hypotheses.",
        "Detected plan, confidence, and attacking side.",
        "`src/tracking/explanations.py` defining template formats for all 6 classes.",
        """### Technical Implementation:
1. Strict Fact vs Hypothesis Separation:
   - Clear distinction: 'Observed Evidence' (verified pieces/moves) vs 'Model Hypothesis' (learned strategic intent).
2. Templates:
   - `K_STORM`: Flank pawn depth, king sector target, rook alignment.
   - `Q_EXP`: Queenside pawn space, flank control.
   - `OUTPOST`: Minor piece on hole, defending pawn anchor.
   - `ROOK_FILE`: Heavy piece alignment, open file entry squares.
   - `CHAIN_UNDERMINE`: Pawn lever move, target chain base/apex.
   - `NO_PLAN`: Balanced maneuvering or transitional state.""",
        """1. Implement template definitions in `src/tracking/explanations.py`.
2. Write unit tests checking that templates render valid sentences without syntax errors.
3. Verify no unformatted template placeholders `{}` remain in rendered outputs.""",
        "- All templates produce grammatically correct, pedagogical sentences.\n- Explicitly labels predictions as hypotheses, not telepathic intentions.",
    ),
    (
        "SUBTASK_2_3_2_STRATEGIC_RATIONALE_FORMATTER.md",
        "Subtask 2.3.2: Strategic Plan Rationale & Supporting Evidence Formatter",
        "2.3",
        "Phase 2",
        "Milestone 11",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Format concrete chess moves, square names, and tactical facts into pedagogical bullet points.",
        "`PlayerPlanState`, `ExtractedBoardFacts`, and recent SAN move list.",
        "`ExplanationCard` with summary sentence and grounded factual bullet points.",
        """### Technical Implementation:
1. Class `DeterministicExplanationGenerator`:
   - Interpolates verified square coordinates (e.g. 'd5', 'g5', 'c-file').
   - Constructs 3-4 bullet points highlighting supporting evidence.
   - Summarizes tactical context (e.g. 'Opponent king castled on g8, White pawn on g5').""",
        """1. Implement formatting methods in `src/tracking/explanations.py`.
2. Test formatting across all 5 sample master games.
3. Assert that rendered bullet points contain valid algebraic notation.""",
        "- Bullet points accurately reflect the verified move and board state.\n- Pedagogical tone suitable for chess learners.",
    ),
    (
        "SUBTASK_2_3_3_STRUCTURED_CLAIM_VERIFIER.md",
        "Subtask 2.3.3: Strict Structured Claim Schema & Validation Filter",
        "2.3",
        "Phase 2",
        "Milestone 11",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Verify that every square and piece claim made in an explanation actually exists on the board.",
        "`chess.Board` and generated `ExplanationCard`.",
        "`verified_claim_status: bool` and safe fallback if verification fails.",
        """### Technical Implementation:
1. Claim Verifier:
   - Parse all mentioned squares (e.g. 'd5', 'e4') from explanation text.
   - Assert that claimed piece is currently on that square or visited it in the 8-ply window.
   - If any claim fails verification: discard card and fall back to safe generic summary:
     'Model predicts [Plan] with [X]% confidence based on recent move history.'""",
        """1. Implement verification logic in `src/tracking/explanations.py`.
2. Write unit tests injecting intentional corrupted square claims; verify fallback triggers.
3. Test with genuine board states; verify 100% pass rate.""",
        "- Zero hallucination: no non-existent pieces or squares are presented as facts.\n- Automatic fallback on unexpected board states.",
    ),
    # Task 2.4
    (
        "SUBTASK_2_4_1_APP_STRUCTURE_SESSION_STATE.md",
        "Subtask 2.4.1: Streamlit App Structure, Layout & Session State Manager",
        "2.4",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build the Streamlit application shell, multi-column layout, and session state manager.",
        "None (app entry point).",
        "`src/ui/app.py` with responsive 2-column layout and session state dictionary.",
        """### Technical Implementation:
1. App Entry: `src/ui/app.py`.
2. Session State:
   - `current_ply: int`
   - `game_moves: List[chess.Move]`
   - `board_positions: List[chess.Board]`
   - `selected_model_name: str`
   - `is_playing: bool`
   - `analysis_cache: Dict[int, any]`
3. Wide-screen layout: Sidebar controls, left column (board SVG), right column (predictions & charts).""",
        """1. Implement `src/ui/app.py` with page config and session initialization.
2. Verify app launches with `streamlit run src/ui/app.py`.
3. Check state persistence across UI re-renders.""",
        "- App launches cleanly without warnings or errors.\n- Session state maintains game position across widget interactions.",
    ),
    (
        "SUBTASK_2_4_2_PGN_LOADER_SAMPLE_LIBRARY.md",
        "Subtask 2.4.2: PGN File Uploader & Famous Master Game Library",
        "2.4",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Implement PGN file uploader and pre-bundled library of 5 classic master games.",
        "User uploaded PGN file or library dropdown selection.",
        "`src/ui/game_loader.py` returning parsed `chess.pgn.Game`.",
        """### Technical Implementation:
1. Sidebar Options:
   - Mode radio: 'Curated Master Match Library' vs 'Upload Custom PGN'.
2. Bundled Library:
   - Game 1: Fischer vs Larsen (1958) -> `K_STORM`
   - Game 2: Petrosian vs Spassky (1966) -> `Q_EXP`
   - Game 3: Kasparov vs Karpov (1985, G16) -> `OUTPOST`
   - Game 4: Alekhine vs Nimzowitsch (1930) -> `ROOK_FILE`
   - Game 5: Botvinnik vs Capablanca (1938) -> `CHAIN_UNDERMINE`
3. Error Handling: Graceful warning alert if uploaded PGN is empty or corrupted.""",
        """1. Implement `src/ui/game_loader.py`.
2. Create sample PGN files under `data/samples/`.
3. Test loading both bundled games and uploaded PGN files.""",
        "- All 5 sample games load and parse without error.\n- Custom PGN uploader handles edge cases cleanly.",
    ),
    (
        "SUBTASK_2_4_3_MOVE_STEPPER_NAVIGATOR.md",
        "Subtask 2.4.3: Interactive Move-by-Move Game Stepper & Timeline Navigation",
        "2.4",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build move navigation controls: First, Previous, Play/Pause auto-play, Next, Last, and ply slider.",
        "Active `chess.pgn.Game` in session state.",
        "`src/ui/navigator.py` rendering navigation bar and move notation display.",
        """### Technical Implementation:
1. Controls:
   - `|<`, `< Prev`, `Play/Pause`, `Next >`, `>|` buttons.
   - Move slider: `st.slider('Move Slider', 0, total_plies, current_ply)`.
   - Current move box: Displays `Move 24. ... Nd5 (Black)`.
2. Auto-Play: Uses `st.empty()` or sleep loop when `is_playing` is active.""",
        """1. Implement `src/ui/navigator.py`.
2. Bind button click callbacks to `st.session_state.current_ply`.
3. Test smooth slider scrubbing and boundary handling (ply 0 and max ply).""",
        "- Buttons and slider update `current_ply` synchronously.\n- Boundary checks prevent out-of-bounds indexing.",
    ),
    (
        "SUBTASK_2_4_4_INFERENCE_RUNNER_CACHING.md",
        "Subtask 2.4.4: Low-Latency Inference Runner with Caching",
        "2.4",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build high-speed inference runner cached with `@st.cache_resource` for zero UI lag.",
        "Active ply and board history.",
        "`src/ui/inference_runner.py` returning prediction, state, and facts in < 20ms.",
        """### Technical Implementation:
1. Model Caching:
   - `@st.cache_resource` loads ONNX runtime session or PyTorch checkpoint once.
2. Inference Flow:
   - Construct 8-ply window -> Tensor encoding -> ResNet forward pass -> Fact extractor -> Hybrid fusion -> State tracker update.
3. Latency: Measure and display execution time in sidebar diagnostic badge.""",
        """1. Implement `src/ui/inference_runner.py`.
2. Benchmark inference latency inside Streamlit (< 20ms per ply).
3. Test switching models dynamically from sidebar.""",
        "- Move navigation updates board and predictions instantly (< 20ms).\n- Memory consumption remains stable during long interactive sessions.",
    ),
    # Task 2.5
    (
        "SUBTASK_2_5_1_SVG_CHESSBOARD_OVERLAYS.md",
        "Subtask 2.5.1: Interactive SVG Chessboard Component with Strategic Overlays",
        "2.5",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Render chessboard using `python-chess.svg` with move arrows and strategic square highlights.",
        "Active `chess.Board`, last move, and detected strategic features.",
        "`src/ui/board_view.py` rendering SVG in Streamlit HTML component.",
        """### Technical Implementation:
1. Overlays:
   - Last move: Solid green arrow (`#2ecc71`).
   - History moves: Translucent cyan arrows (`#3498db88`).
   - Outpost square: Gold circle highlight (`#f1c40f`).
   - Pawn storm target: Crimson square highlight (`#e74c3c`).
2. Board orientation: Automatically flip board when Black is attacking or via toggle.""",
        """1. Implement SVG generation in `src/ui/board_view.py`.
2. Embed SVG in Streamlit using `st.components.v1.html(svg, height=450, width=450)`.
3. Verify arrow positioning and square highlights on sample positions.""",
        "- Board renders cleanly at high resolution.\n- Strategic arrows and highlights align precisely with target squares.",
    ),
    (
        "SUBTASK_2_5_2_PROBABILITY_BAR_CHARTS.md",
        "Subtask 2.5.2: Multi-Class Probability Bar Chart & Confidence Breakdown",
        "2.5",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Render horizontal probability bar chart across all 6 classes with abstention threshold line.",
        "Probability dictionary from inference runner.",
        "`src/ui/charts.py` rendering styled probability bars.",
        """### Technical Implementation:
1. Chart: Horizontal bars for `K_STORM`, `Q_EXP`, `OUTPOST`, `ROOK_FILE`, `CHAIN_UNDERMINE`, `NO_PLAN`.
2. Features:
   - Color coding matching each plan class.
   - Vertical dashed threshold line at abstention cut-off (e.g. 40%).
   - Status badge: Active / Uncertain.""",
        """1. Implement `render_probability_bars` in `src/ui/charts.py`.
2. Test chart rendering across diverse probability profiles.
3. Verify responsive layout in Streamlit right column.""",
        "- Probabilities render cleanly with percentage labels.\n- Status badge updates dynamically based on confidence.",
    ),
    (
        "SUBTASK_2_5_3_TEMPORAL_PLAN_TIMELINE.md",
        "Subtask 2.5.3: Whole-Game Temporal Plan Evolution Chart",
        "2.5",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build interactive line/area chart plotting plan probabilities across all game plies with onset markers.",
        "Precomputed game history probabilities.",
        "`src/ui/timeline.py` rendering multi-line temporal chart.",
        """### Technical Implementation:
1. Chart:
   - X-axis: Game ply ($0 \\dots T$).
   - Y-axis: Softmax probability ($0.0 \\dots 1.0$).
   - 6 lines/curves for all classes.
   - Vertical cursor line at `current_ply`.
   - Star/circle marker at the detected **Plan Onset Ply**.""",
        """1. Implement `render_game_timeline` in `src/ui/timeline.py`.
2. Precompute whole-game predictions with progress indicator.
3. Verify cursor line syncs with navigation stepper.""",
        "- Visualizes the birth, dominance, and decline of strategic plans.\n- Cursor follows move stepper smoothly.",
    ),
    (
        "SUBTASK_2_5_4_GROUNDED_EVIDENCE_ACCORDION.md",
        "Subtask 2.5.4: Grounded Evidence Fact Inspector & Accordion",
        "2.5",
        "Phase 2",
        "Milestone 12",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build expandable accordion presenting pawn structures, outposts, file control, and explanation card.",
        "`ExtractedBoardFacts` and `ExplanationCard`.",
        "`src/ui/evidence_view.py` rendering formatted evidence blocks.",
        """### Technical Implementation:
1. Accordion Sections:
   - *Strategic Summary*: Active plan, onset ply, confidence, explanation text.
   - *Pawn Structures*: Chains, base pawns, advanced storm pawns.
   - *Piece Coordinates*: Outpost squares and piece maneuvers.
   - *File Control*: Open/semi-open files and rook batteries.""",
        """1. Implement `src/ui/evidence_view.py`.
2. Style with markdown callout boxes and metric widgets.
3. Verify evidence updates correctly at each move step.""",
        "- Clear, human-readable breakdown of underlying chess facts.\n- Allows learners to inspect why the AI made its prediction.",
    ),
    # Task 2.6
    (
        "SUBTASK_2_6_1_STOCKFISH_UCI_CLIENT.md",
        "Subtask 2.6.1: Stockfish Engine UCI Client Wrapper",
        "2.6",
        "Phase 2",
        "Milestone 13",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build UCI client wrapper using `chess.engine` with configurable depth and graceful fallback if missing.",
        "Path to Stockfish binary (optional).",
        "`src/utils/stockfish_client.py` with `StockfishAnalyzer`.",
        """### Technical Implementation:
1. Class `StockfishAnalyzer`:
   - Search paths: `stockfish/stockfish.exe`, PATH env.
   - Limits: depth 10-12, time 0.2s, MultiPV 3.
2. Fallback: If binary missing, set `is_available = False` and log clean notification without crashing.""",
        """1. Implement `src/utils/stockfish_client.py`.
2. Write unit tests in `tests/test_stockfish_client.py` with mock engine.
3. Verify zero crash when binary does not exist.""",
        "- Wrapper initializes cleanly whether Stockfish is present or absent.\n- Never blocks or crashes UI execution.",
    ),
    (
        "SUBTASK_2_6_2_DEFENSIVE_COUNTERPLAY_SCREENING.md",
        "Subtask 2.6.2: Defender Counter-Move Screening & Plan Disruption",
        "2.6",
        "Phase 2",
        "Milestone 13",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Screen defender candidate moves to identify counter-moves that disrupt the opponent's detected plan.",
        "`chess.Board` and detected opponent plan.",
        "List of `EngineCandidateMove` with strategic disruption notes.",
        """### Technical Implementation:
1. Method `screen_counterplay(board, detected_plan)`:
   - Identify candidate defender moves.
   - Check if move attacks opponent's outpost piece, blocks open file, or counter-attacks center.
   - Return top 3 moves with centipawn evaluation and strategic annotation.""",
        """1. Implement defensive screening in `src/utils/stockfish_client.py`.
2. Test on tactical positions with obvious defensive replies.
3. Display suggestions in Streamlit Tab 3.""",
        "- Produces sensible defensive recommendations.\n- Explicitly disclaims engine output as study suggestions.",
    ),
    (
        "SUBTASK_2_6_3_THREAT_DELTA_SIMULATION.md",
        "Subtask 2.6.3: Threat Delta Simulation (Proactive vs Passive)",
        "2.6",
        "Phase 2",
        "Milestone 13",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Compute evaluation delta between proactive defensive moves and ignoring the opponent's plan.",
        "Current position and candidate moves.",
        "Threat delta numerical value $\\Delta_{\\text{threat}}$ and interpretation note.",
        """### Technical Implementation:
1. Formula:
   $$\\Delta_{\\text{threat}} = \\text{Eval}(\\text{Proactive Defense}) - \\text{Eval}(\\text{Passive Move})$$
2. Illustrates positional consequence of allowing opponent to execute their plan unimpeded.""",
        """1. Implement threat delta calculation in `src/utils/stockfish_client.py`.
2. Integrate delta display in Streamlit engine tab.
3. Verify proper centipawn formatting (+1.5, -0.8).""",
        "- Accurately reports engine evaluation difference.\n- Helps learners appreciate the severity of positional threats.",
    ),
    # Task 2.7
    (
        "SUBTASK_2_7_1_MULTI_MODEL_BENCHMARK_RUNNER.md",
        "Subtask 2.7.1: Multi-Model Benchmark Runner on Test Ground Truth",
        "2.7",
        "Phase 2",
        "Milestone 14",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Execute automated benchmarking of 4 systems on the frozen 250-position test set.",
        "`data/annotations/test_ground_truth.csv` and trained checkpoints.",
        "`experiments/evaluation_results.csv` containing side-by-side predictions.",
        """### Technical Implementation:
1. Script `src/models/evaluate.py`:
   - Evaluates:
     1. MindCheck History-Aware ResNet (8 plies)
     2. Current-Position ResNet Baseline (1 ply)
     3. Rule-Only Symbolic Detector
     4. Neural-Rule Hybrid Model
   - Logs prediction class, probabilities, and latency per sample.""",
        """1. Implement `src/models/evaluate.py`.
2. Execute full benchmark on 250 test positions.
3. Save raw prediction outputs to CSV.""",
        "- All 4 systems evaluated on identical test examples.\n- Output table saved with complete prediction records.",
    ),
    (
        "SUBTASK_2_7_2_METRICS_CONFUSION_MATRICES.md",
        "Subtask 2.7.2: Confusion Matrix, Macro-F1 & Per-Class Metrics Generator",
        "2.7",
        "Phase 2",
        "Milestone 14",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Compute classification metrics and generate normalized confusion matrix heatmaps.",
        "`experiments/evaluation_results.csv`.",
        "`experiments/plots/confusion_matrix_*.png` and `experiments/metrics_summary.json`.",
        """### Technical Implementation:
1. Script `src/models/metrics.py`:
   - Compute: Overall Accuracy, Macro-F1, per-class Precision/Recall/F1 for all 6 classes.
   - Plot normalized confusion matrices using Seaborn heatmap.
   - Format publication-ready LaTeX / Markdown tables.""",
        """1. Implement `src/models/metrics.py`.
2. Generate confusion matrix plots for each model.
3. Verify metric calculation with `scikit-learn.metrics.classification_report`.""",
        "- Confusion matrix heatmaps saved to `experiments/plots/`.\n- Macro-F1 and per-class metrics exported to JSON.",
    ),
    (
        "SUBTASK_2_7_3_GAME_CLUSTER_BOOTSTRAP_CI.md",
        "Subtask 2.7.3: Game-Cluster Bootstrap Confidence Interval Calculator",
        "2.7",
        "Phase 2",
        "Milestone 14",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Compute 95% Confidence Intervals for Macro-F1 using 1,000 game-cluster bootstrap iterations.",
        "`experiments/evaluation_results.csv` with `game_id` column.",
        "`experiments/bootstrap_confidence_intervals.json`.",
        """### Technical Implementation:
1. Script `src/models/bootstrap.py`:
   - Resample unique game IDs with replacement $B = 1,000$ times.
   - Compute Macro-F1 on each bootstrap sample.
   - Extract empirical 2.5th and 97.5th percentiles.
   - Report: $\\text{Macro-F1} = \\hat{\\mu} \\; [95\\% \\text{ CI}: L, U]$.""",
        """1. Implement `src/models/bootstrap.py`.
2. Run 1,000 bootstrap iterations.
3. Log confidence intervals for inclusion in the final report.""",
        "- Statistically sound confidence intervals accounting for clustered game data.\n- Confirms whether History CNN superiority is statistically significant.",
    ),
    (
        "SUBTASK_2_7_4_QUALITATIVE_FAILURE_ANALYSIS.md",
        "Subtask 2.7.4: Systematic Qualitative Failure Analysis & Edge-Case Audit",
        "2.7",
        "Phase 2",
        "Milestone 14",
        "Shahzad Ahmad (24I-3090)",
        "Muhammad Ali (24I-3158)",
        "Document 10 specific failure positions (5 false positives, 5 false negatives) with chess analysis.",
        "Misclassified test records in `evaluation_results.csv`.",
        "`docs/failure_analysis.md` with FENs, move sequences, and chess explanations.",
        """### Technical Implementation:
1. Audit File: `docs/failure_analysis.md`.
2. Structure for each failure:
   - Position FEN and move history.
   - True Label vs Predicted Label.
   - Model confidence probabilities.
   - Deep chess explanation of why the neural network failed.
   - Actionable takeaways for future research.""",
        """1. Identify 5 representative false positives and 5 false negatives.
2. Draft `docs/failure_analysis.md`.
3. Review chess analysis with both group members.""",
        "- Candid, thorough analysis of model limitations.\n- Demonstrates genuine scientific understanding for course evaluation.",
    ),
    # Task 2.8
    (
        "SUBTASK_2_8_1_ACADEMIC_REPORT_AUTHORING.md",
        "Subtask 2.8.1: Formal Academic Project Report Authoring",
        "2.8",
        "Phase 2",
        "Milestone 15",
        "Shared (Both Members)",
        "Shahzad Ahmad & Muhammad Ali",
        "Author comprehensive publication-grade academic project report conforming to all AI2002 guidelines.",
        "Experimental results, architectures, and evaluation data.",
        "`docs/report.pdf` (or `docs/REPORT.md`).",
        """### Technical Structure:
1. Problem Statement & Real-World Justification
2. Selected Path & Objectives (Path A)
3. Methodology & Spatial-Temporal Tensor Formulation
4. Deep Residual CNN Architecture & Baseline Design
5. Dataset Description, Weak Labeling & Game-Level Splitting
6. Implementation Details & Streamlit Deployment
7. Testing & Evaluation (Macro-F1, Cohen's Kappa, Confusion Matrices, Bootstrap CIs)
8. Ablation Findings (History Length Comparison)
9. Qualitative Failure Analysis & Limitations
10. Conclusion & References.""",
        """1. Draft report in Markdown (`docs/REPORT.md`).
2. Integrate figures, metric tables, and bibliography.
3. Compile to PDF (`docs/report.pdf`).
4. Both members review for accuracy and completeness.""",
        "- Report contains complete measured results (no placeholder text).\n- Fully complies with all AI2002 Course Learning Outcomes (CLO 1, 2, 3).",
    ),
    (
        "SUBTASK_2_8_2_SYSTEM_PIPELINE_DIAGRAM.md",
        "Subtask 2.8.2: Publication-Grade System Architecture & Pipeline Diagram",
        "2.8",
        "Phase 2",
        "Milestone 15",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Generate high-resolution system architecture and software workflow diagram.",
        "System architecture specification.",
        "`docs/figures/system_pipeline.png`.",
        """### Technical Implementation:
1. Diagram Components:
   - Raw PGN -> Streaming Parser -> 8-Ply Sliding Window -> 112x8x8 Tensor Encoder.
   - PyTorch Residual CNN -> Probability Vector.
   - Bitboard Fact Extractor -> Symbolic Rules -> Hybrid Score Fusion.
   - Temporal State Tracker (Hysteresis) -> Streamlit Interactive UI.
2. Format: Clean vector/high-res PNG with distinct colored pipeline blocks.""",
        """1. Design diagram using Mermaid / SVG / Matplotlib / Draw.io.
2. Export high-res PNG to `docs/figures/system_pipeline.png`.
3. Embed diagram into report and presentation slides.""",
        "- Clean, professional visual representation of end-to-end pipeline.\n- Matches actual codebase implementation exactly.",
    ),
    (
        "SUBTASK_2_8_3_PRESENTATION_SLIDE_DECK.md",
        "Subtask 2.8.3: Project Presentation Slide Deck Preparation",
        "2.8",
        "Phase 2",
        "Milestone 15",
        "Shared (Both Members)",
        "Muhammad Ali & Shahzad Ahmad",
        "Prepare 12-15 slide presentation deck for the live defense (07-09 Dec 2026).",
        "Report contents, pipeline diagrams, and benchmark charts.",
        "`docs/slides/presentation_deck.pptx` or `docs/slides.pdf`.",
        """### Technical Implementation:
1. Slide Structure:
   - S1: Title & Group 29 Introduction
   - S2: Real-World Motivation & Problem Formulation
   - S3: Scope & Research Questions (Path A)
   - S4: 5-Plan Taxonomy & Operational Rubric
   - S5: Data Pipeline & Lookahead Weak-Labeling
   - S6: Tensor Representation ($112 \\times 8 \\times 8$)
   - S7: Residual CNN Architecture & Capacity
   - S8: Temporal Plan State Tracker (Hysteresis & Onset)
   - S9: Streamlit App Architecture & Feature Walkthrough
   - S10: Ground Truth Test Benchmark & Cohen's Kappa
   - S11: Quantitative Results: History ResNet vs Baselines
   - S12: History Length Ablation Findings
   - S13: Failure Analysis & Limitations
   - S14: Work Distribution & Team Contributions
   - S15: Conclusion & Live Demonstration.""",
        """1. Build slide deck using Markdown/Marp, PowerPoint, or Keynote.
2. Embed high-resolution plots and board screenshots.
3. Export final deck to `docs/slides.pdf`.""",
        "- Visually polished deck covering all mandatory presentation points.\n- Both team members prepared to present their respective sections.",
    ),
    # Task 2.9
    (
        "SUBTASK_2_9_1_CLEAN_ENVIRONMENT_VERIFIER.md",
        "Subtask 2.9.1: Clean-Environment Installation & One-Click Launchers",
        "2.9",
        "Phase 2",
        "Milestone 15",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build installation verification script and one-click launch scripts for live demonstration.",
        "Repository files and checkpoints.",
        "`scripts/verify_install.py`, `run_demo.bat`, `run_demo.sh`.",
        """### Technical Implementation:
1. `scripts/verify_install.py`:
   - Verifies Python version >= 3.11.
   - Imports `torch`, `chess`, `streamlit`, `numpy`, `pandas`, `sklearn`.
   - Asserts checkpoint `models/checkpoints/best_resnet.pt` exists and loads.
   - Asserts sample PGNs exist.
2. `run_demo.bat` (Windows) & `run_demo.sh` (Linux/macOS):
   - Single command to activate venv and launch Streamlit app.""",
        """1. Implement `scripts/verify_install.py`.
2. Create `run_demo.bat` and `run_demo.sh`.
3. Test fresh virtual environment rebuild to confirm flawless startup.""",
        "- `python scripts/verify_install.py` outputs all green checks.\n- `run_demo.bat` launches the browser UI in < 5 seconds.",
    ),
    (
        "SUBTASK_2_9_2_LIVE_DEMO_REHEARSAL_CHECKLIST.md",
        "Subtask 2.9.2: Streamlit Live Demo Rehearsal Checklist & Script",
        "2.9",
        "Phase 2",
        "Milestone 15",
        "Shared (Both Members)",
        "Muhammad Ali & Shahzad Ahmad",
        "Prepare step-by-step demonstration script and rehearsal checklist for the 5-7 minute defense.",
        "Running Streamlit app and curated master matches.",
        "`docs/demo_rehearsal_checklist.md` with timed cues.",
        """### Rehearsal Protocol:
- Minute 1: Introductions, problem statement, 5 strategic classes.
- Minute 2: Data pipeline, tensor encoding, and ResNet architecture.
- Minute 3: Demo Kasparov-Karpov (Outpost on d5 detected at ply 26).
- Minute 4: Demo Fischer-Larsen (Kingside Pawn Storm with crimson vector overlays).
- Minute 5: Model comparison live toggle (History ResNet vs Single-Board baseline).
- Minute 6: Evaluation metrics, confusion matrix, Cohen's Kappa, and failure analysis.
- Minute 7: Live Q&A defense.""",
        """1. Draft `docs/demo_rehearsal_checklist.md`.
2. Conduct timed live trial runs with both members.
3. Verify backup sample PGNs in case of internet outage.""",
        "- Both members execute seamless handoffs during the demonstration.\n- Demo completes within the allotted 5-7 minute defense window.",
    ),
    (
        "SUBTASK_2_9_3_SINGLE_GCR_SUBMISSION_ZIP.md",
        "Subtask 2.9.3: Single GCR Submission ZIP Packager & Verification",
        "2.9",
        "Phase 2",
        "Milestone 15",
        "Muhammad Ali (24I-3158)",
        "Shahzad Ahmad (24I-3090)",
        "Build packaging script to generate single consolidated submission ZIP file for GCR.",
        "Complete repository deliverables.",
        "`scripts/package_submission.py` and `MindCheck_Group29_Submission.zip`.",
        """### Technical Implementation:
1. Script `scripts/package_submission.py`:
   - Package all required deliverables:
     - `src/` (All source code with comments)
     - `docs/` (Report PDF, Slides PDF, architectural docs)
     - `data/samples/` (Sample master PGNs)
     - `data/annotations/test_ground_truth.csv`
     - `models/checkpoints/` (Trained model weights)
     - `requirements.txt`, `config/`, `run_demo.bat`, `README.md`
   - Excludes `.git`, `venv`, `__pycache__`, and huge raw PGN files.
   - Verifies ZIP file integrity and file size < 80MB.""",
        """1. Implement `scripts/package_submission.py`.
2. Generate submission ZIP.
3. Test extracting ZIP in an isolated test folder and running `run_demo.bat`.""",
        "- Generated ZIP contains all mandatory course deliverables.\n- Submits successfully to Google Classroom before 06 Dec 2026.",
    ),
]

# ==============================================================================
# TEMPLATE RENDERER
# ==============================================================================

SUBTASK_TEMPLATE = """# {title}

**Subtask ID**: `{subtask_id}`  
**Parent Task**: `Task {task_id}`  
**Phase**: {phase}  
**Milestone**: {milestone}  
**Primary Owner**: **{owner}**  
**Collaborator**: **{collaborator}**  

---

## 1. Context & Objective
{context}

---

## 2. Input Requirements
- **Prerequisites / Input Files**: {inputs}

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: {outputs}

---

## 4. Detailed Technical Specification
{tech_spec}

---

## 5. Claude Code Action Plan
{action_plan}

---

## 6. Unit Testing & Acceptance Criteria
{acceptance_criteria}
"""


def generate_all():
    print(f"Generating {len(phase_1_subtasks)} Phase 1 subtask files...")
    for (
        filename,
        title,
        task_id,
        phase,
        milestone,
        owner,
        collaborator,
        context,
        inputs,
        outputs,
        tech_spec,
        action_plan,
        acceptance,
    ) in phase_1_subtasks:
        subtask_id = filename.replace(".md", "").replace("SUBTASK_", "").replace("_", ".")
        content = SUBTASK_TEMPLATE.format(
            title=title,
            subtask_id=subtask_id,
            task_id=task_id,
            phase=phase,
            milestone=milestone,
            owner=owner,
            collaborator=collaborator,
            context=context,
            inputs=inputs,
            outputs=outputs,
            tech_spec=tech_spec,
            action_plan=action_plan,
            acceptance_criteria=acceptance,
        )
        write_subtask_file(P1_SUBTASKS_DIR, filename, content)

    print(f"Generating {len(phase_2_subtasks)} Phase 2 subtask files...")
    for (
        filename,
        title,
        task_id,
        phase,
        milestone,
        owner,
        collaborator,
        context,
        inputs,
        outputs,
        tech_spec,
        action_plan,
        acceptance,
    ) in phase_2_subtasks:
        subtask_id = filename.replace(".md", "").replace("SUBTASK_", "").replace("_", ".")
        content = SUBTASK_TEMPLATE.format(
            title=title,
            subtask_id=subtask_id,
            task_id=task_id,
            phase=phase,
            milestone=milestone,
            owner=owner,
            collaborator=collaborator,
            context=context,
            inputs=inputs,
            outputs=outputs,
            tech_spec=tech_spec,
            action_plan=action_plan,
            acceptance_criteria=acceptance,
        )
        write_subtask_file(P2_SUBTASKS_DIR, filename, content)

    print(f"Total subtasks generated: {len(phase_1_subtasks) + len(phase_2_subtasks)}")


if __name__ == "__main__":
    generate_all()
