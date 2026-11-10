# Task 1.4: Five-Plan Symbolic Rule Engine & Baseline

**Phase**: Phase 1 (Core AI & Machine Learning Pipeline)  
**Milestone**: Milestone 4  
**Primary Owner**: Shahzad Ahmad (`24I-3090`)  
**Collaborator**: Muhammad Ali (`24I-3158`)  

---

## 1. Objective & Scope
Implement the deterministic 5-Plan Symbolic Rule Engine. This module operates over the 8-ply history window and extracted board facts, producing normalized heuristic scores $S_{\text{rule}}(c) \in [0.0, 1.0]$ for each of the 5 strategic classes and `NO_PLAN`. This serves as both the **Rule-Only Baseline (M3)** for course evaluation and the symbolic evidence provider for hybrid fusion.

---

## 2. Subtask Breakdown

### Subtask 1.4.1: Kingside Pawn Storm Template (`src/rules/rule_engine.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Logic**:
  - Check if opponent king is castled kingside (files f, g, h).
  - Inspect last 8 plies: count moves advancing f-, g-, or h-pawns toward opponent side.
  - Check presence of friendly rooks or queen aligned on f-, g-, or h-files.
  - Formula:
    $$S_{\text{k\_storm}} = \min(1.0, 0.4 \times \text{pawn\_advances} + 0.3 \times \text{piece\_support} + 0.3 \times \text{king\_exposure})$$

### Subtask 1.4.2: Queenside Expansion Template (`src/rules/rule_engine.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Logic**:
  - Inspect last 8 plies: count pawn pushes on a-, b-, or c-files.
  - Check spatial gain past rank 4 (White) / rank 5 (Black).
  - Formula:
    $$S_{\text{q\_exp}} = \min(1.0, 0.4 \times \text{queenside\_pushes} + 0.3 \times \text{space\_advantage} + 0.3 \times \text{file\_control})$$

### Subtask 1.4.3: Outpost Occupation Template (`src/rules/rule_engine.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Logic**:
  - Identify whether an active outpost hole exists on files c, d, e, f.
  - Check if friendly Knight or Bishop entered the outpost square during the last 8 plies, or is positioned within 1 legal move of it.
  - Formula:
    $$S_{\text{outpost}} = 1.0 \text{ if piece occupied}, 0.7 \text{ if maneuvering toward outpost}, 0.0 \text{ otherwise}$$

### Subtask 1.4.4: Rook-File Buildup Template (`src/rules/rule_engine.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Logic**:
  - Identify open or semi-open files.
  - Check if two heavy pieces (Rook+Rook or Rook+Queen) occupy the file.
  - Check if recent plies included moving a rook to that open file.
  - Formula:
    $$S_{\text{rook\_file}} = 1.0 \text{ if doubled on open file}, 0.7 \text{ if doubled on semi-open file}, 0.4 \text{ if single rook on open file}$$

### Subtask 1.4.5: Pawn-Chain Undermining Template (`src/rules/rule_engine.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Logic**:
  - Identify opponent pawn chains ($\ge 2$ pawns).
  - Check if a friendly pawn lever was played in the last 8 plies attacking the base or apex of the chain.
  - Formula:
    $$S_{\text{undermine}} = 1.0 \text{ if lever attacking base pawn}, 0.8 \text{ if attacking apex}, 0.0 \text{ otherwise}$$

### Subtask 1.4.6: Rule-Only Classifier & Ambiguity Resolver (`src/rules/rule_engine.py`)
- **Owner**: Shahzad Ahmad (`24I-3090`)
- **Logic**:
  - Compile scores: $[S_0, S_1, S_2, S_3, S_4]$.
  - Assign class label = $\arg\max_k S_k$ if $\max_k S_k \ge \theta_{\text{rule}}$ (e.g. 0.50).
  - If $\max_k S_k < 0.50$ or if the top two scores differ by $< 0.15$ (ambiguous conflict), assign class = `NO_PLAN` (Class 5).

---

## 3. Function Signatures & Data Model

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple
import chess
from src.rules.fact_extractor import ExtractedBoardFacts

@dataclass
class RuleEvaluationResult:
    predicted_class: str             # "K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"
    class_id: int                    # 0 to 5
    confidence: float
    all_scores: Dict[str, float]     # {"K_STORM": 0.8, ...}
    supporting_evidence: List[str]   # Human-readable evidence strings

class SymbolicRuleEngine:
    def __init__(self, activation_threshold: float = 0.50):
        self.threshold = activation_threshold

    def evaluate_window(
        self, 
        history: List[chess.Board], 
        facts: ExtractedBoardFacts
    ) -> RuleEvaluationResult:
        """Evaluates 8-ply history and returns deterministic rule predictions."""
        pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/rules/rule_engine.py` with individual template evaluator functions.
2. Implement `SymbolicRuleEngine.evaluate_window()`.
3. Create `tests/test_rule_engine.py` with known master test positions testing each rule template individually.
4. Verify that non-strategic quiet positions evaluate correctly to `NO_PLAN`.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_rule_engine.py` -> 100% pass.
- Verify deterministic behavior: same position sequence produces identical score output every time.
