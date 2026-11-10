# Task 2.1: Temporal Plan State Tracker Engine

**Phase**: Phase 2 (System Integration, Streamlit Deployment & Evaluation)  
**Milestone**: Milestone 9  
**Primary Owner**: Muhammad Ali (`24I-3158`)  
**Collaborator**: Shahzad Ahmad (`24I-3090`)  

---

## 1. Objective & Scope
Implement the stateful Temporal Plan Tracker. In real chess games, strategic plans persist over several plies and do not flicker sporadically. This module implements a per-player finite state machine with hysteresis logic (2 consecutive supporting updates to activate, 2 contradictory updates to switch), detects the earliest onset ply of a plan, and tracks plan progress versus threat severity.

---

## 2. Subtask Breakdown

### Subtask 2.1.1: Per-Player State Machine & Hysteresis (`src/tracking/state_tracker.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Maintain independent state machines for White and Black.
  - State attributes per player:
    - `active_plan`: Currently committed plan (`str`, defaults to `"NO_PLAN"`)
    - `pending_candidate`: Candidate plan under observation (`Optional[str]`)
    - `candidate_consecutive_count`: Number of consecutive plies candidate has been predicted (`int`)
    - `active_plan_duration`: Number of plies current plan has remained active (`int`)
  - **Hysteresis Transition Rules**:
    - If model predicts plan $C \ne \text{active\_plan}$ with confidence $\ge 0.45$:
      - If $C == \text{pending\_candidate}$: increment count.
        - If count $\ge 2$: transition `active_plan = C`, reset candidate count.
      - Else: set `pending_candidate = C`, count = 1.
    - If model predicts `NO_PLAN` or low confidence for $\ge 3$ consecutive plies: decay `active_plan` to `"NO_PLAN"`.

### Subtask 2.1.2: Plan Onset Estimation (`src/tracking/state_tracker.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - When `active_plan` switches to $C$ at ply $t$:
    - The onset ply is estimated by scanning back to the first ply where candidate $C$ was first detected in the supporting window ($t - \text{candidate\_consecutive\_count} + 1$).
    - Store `onset_ply` in the plan state record.

### Subtask 2.1.3: Plan Progress vs Threat Severity Metrics (`src/tracking/state_tracker.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - **Plan Progress ($0.0 \dots 1.0$)**:
    - Quantitative measure of plan completion based on geometric milestones (e.g. for Kingside Storm: pawns on rank 5 = 0.5 progress, pawn trade opening file = 0.8 progress).
  - **Threat Severity ($0.0 \dots 1.0$)**:
    - Danger level posed to the defending player (proximity to defender's king, number of attackers vs defenders).
  - Both metrics are computed and displayed separately in the UI to avoid conflating progress with immediate tactical danger.

### Subtask 2.1.4: Unit Tests & Move Simulation (`tests/test_state_tracker.py`)
- **Owner**: Muhammad Ali (`24I-3158`)
- **Specification**:
  - Simulate move sequences:
    - Feed sequence: `[K_STORM, K_STORM]` $\to$ state switches to `K_STORM`.
    - Feed sequence: `[K_STORM, NO_PLAN, K_STORM]` $\to$ state remains `K_STORM` (hysteresis absorbs temporary tactical pause).
    - Feed sequence: `[Q_EXP, Q_EXP]` $\to$ state switches smoothly to `Q_EXP`.

---

## 3. Class Schema & Function Signatures

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import chess

@dataclass
class PlayerPlanState:
    color: chess.Color
    active_plan: str = "NO_PLAN"
    active_confidence: float = 0.0
    onset_ply: Optional[int] = None
    consecutive_plies: int = 0
    pending_plan: Optional[str] = None
    pending_count: int = 0
    progress_score: float = 0.0
    threat_score: float = 0.0
    history_plies: List[Dict[str, any]] = field(default_factory=list)

class TemporalPlanTracker:
    def __init__(self, activation_count: int = 2, decay_count: int = 3, min_confidence: float = 0.45):
        self.white_state = PlayerPlanState(color=chess.WHITE)
        self.black_state = PlayerPlanState(color=chess.BLACK)
        self.activation_threshold = activation_count
        self.decay_threshold = decay_count
        self.min_confidence = min_confidence

    def update(
        self, 
        ply: int, 
        turn: chess.Color, 
        predicted_plan: str, 
        confidence: float, 
        facts: any
    ) -> PlayerPlanState:
        """Updates tracker state and returns current plan state for the active player."""
        pass

    def reset(self):
        """Resets tracker for a new game."""
        self.white_state = PlayerPlanState(color=chess.WHITE)
        self.black_state = PlayerPlanState(color=chess.BLACK)
```

---

## 4. Claude Code Action Plan
1. Implement `src/tracking/state_tracker.py` with `TemporalPlanTracker`.
2. Implement progress and threat heuristic evaluators.
3. Implement `tests/test_state_tracker.py` asserting state transitions and hysteresis robustness.

---

## 5. Verification & Acceptance Criteria
- Run `pytest tests/test_state_tracker.py` -> 100% pass.
- Single isolated prediction noise spikes do not cause erratic plan switching.
