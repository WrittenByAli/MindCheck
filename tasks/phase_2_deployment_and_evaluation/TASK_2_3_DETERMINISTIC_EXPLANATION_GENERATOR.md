# Task 2.3: Deterministic Natural Language Explanation Generator

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 11  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Generate grounded, natural-language pedagogical explanations that interpret the detected strategic plan for learners. To guarantee 100% academic integrity and zero hallucination, explanations are generated using **strictly controlled deterministic templates** filled with verified bitboard predicates and recent move notations.

---

## 2. Subtask Breakdown

### Subtask 2.3.1: Controlled Template Library (`src/tracking/explanations.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Core Principle**: Clear division between **Observed Board Facts** and **Model Hypotheses**:
  - *Bad/Hallucinatory*: "White intends to sacrifice the knight on f7 next turn to checkmate."
  - *Good/Grounded*: "Recent moves suggest an Outpost Occupation on d5 (Model Confidence: 84%). Verified supporting evidence: Knight maneuvered via f3 to d5; square d5 is defended by friendly e4-pawn and cannot be attacked by Black pawns."

### Subtask 2.3.2: Template Slots & Strategic Phrasing (`src/tracking/explanations.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Templates per Class**:
  - `K_STORM`:
    - *"Observed moves indicate a Kingside Pawn Storm by {side}. Advanced pawns on {files} toward enemy king on {king_square}. Supporting piece pressure observed on {battery_file}."*
  - `Q_EXP`:
    - *"{side} is pursuing Queenside Expansion. Recent pawn advancements on {files} establish spatial control over the queenside sector ({squares})."*
  - `OUTPOST`:
    - *"{side} has executed an Outpost Occupation on square {square}. The {piece_type} occupies a hole defended by pawn on {pawn_square} that cannot be challenged by enemy pawns."*
  - `ROOK_FILE`:
    - *"Heavy piece buildup detected on the {file}-file. {side} has aligned {pieces} along the open/semi-open file, pressuring the entry square {entry_square}."*
  - `CHAIN_UNDERMINE`:
    - *"{side} has initiated Pawn-Chain Undermining with the lever move {lever_move}, targeting the {base_or_apex} pawn of opponent's chain on {target_square}."*
  - `NO_PLAN`:
    - *"No dominant single strategic plan identified. The position is characterized by balanced maneuvering, transitional play, or short-term tactical tension."*

### Subtask 2.3.3: Structured Claim Schema & Verifier (`src/tracking/explanations.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Every generated explanation passes through a programmatic claim verifier:
    - Asserts that all square coordinates mentioned actually match current piece positions or legal moves.
    - If any predicate fails verification, fallback to a safe generic factual summary.

---

## 3. Function Signatures & Data Model

```python
from dataclasses import dataclass
from typing import List, Dict
import chess
from src.rules.fact_extractor import ExtractedBoardFacts
from src.tracking.state_tracker import PlayerPlanState

@dataclass
class ExplanationCard:
    plan_name: str
    confidence_display: str
    summary_sentence: str
    factual_bullet_points: List[str]
    strategic_takeaway: str
    verified_claim_status: bool

class DeterministicExplanationGenerator:
    def generate_explanation(
        self, 
        state: PlayerPlanState, 
        facts: ExtractedBoardFacts, 
        recent_moves: List[str]
    ) -> ExplanationCard:
        """Constructs and verifies deterministic explanation card."""
        pass
```

---

## 4. Claude Code Action Plan
1. Implement `src/tracking/explanations.py` with `DeterministicExplanationGenerator`.
2. Populate template dictionaries for all 6 classes.
3. Implement `tests/test_explanations.py` asserting that rendered strings correctly interpolate coordinates and contain no unformatted template braces `{}`.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_explanations.py` -> 100% pass.
- Verified that all explanations clearly distinguish between observed board facts and model hypotheses.
