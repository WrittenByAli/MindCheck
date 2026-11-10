# Subtask 1.3.4: Automated Unit Test Suite for Fact Extractor

**Subtask ID**: `1.3.4.FACT.EXTRACTOR.UNIT.TESTS`  
**Parent Task**: `Task 1.3`  
**Phase**: Phase 1  
**Milestone**: Milestone 3  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Build a comprehensive test suite validating all bitboard predicates on standard chess positions.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `src/rules/fact_extractor.py`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `tests/test_fact_extractor.py` with 100% predicate coverage.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Test Positions:
   - Initial board (rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1): Verify 0 outposts, 0 open files, 0 storm pawns.
   - Outpost Benchmark: 8/8/3p4/4P3/3N4/8/8/4K2k w - - 0 1 (White knight on d4 supported by e5, no black c/e pawns).
   - Alekhine's Gun: 2r1r1k1/5ppp/8/8/2R5/2R5/2Q5/6K1 w - - 0 1 (Tripled heavy pieces on c-file).
   - Pawn Storm: 5rk1/5ppp/8/6P1/7P/8/8/6K1 w - - 0 1 (Advanced g- and h-pawns).

---

## 5. Claude Code Action Plan
1. Write `tests/test_fact_extractor.py`.
2. Add assertions for every field in `ExtractedBoardFacts`.
3. Run `pytest tests/test_fact_extractor.py` and inspect coverage.

---

## 6. Unit Testing & Acceptance Criteria
- All fact extractor tests pass without assertions failures.
- Test suite executes in < 2 seconds.
