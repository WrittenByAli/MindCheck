"""Retrospective weak-label generator and label cleaning (Task 1.5, Milestone 4).

The neural network only ever sees *past* history ``[B_{t-7} ... B_t]``. The
**label**, however, is allowed to peek into the future ``[B_{t+1} ... B_{t+L}]``
to check whether a strategic plan was actually carried out — this is the whole
point of weak supervision and is strictly a *training-time* signal that never
enters the model's input tensor.

:class:`RetrospectiveWeakLabeler` implements this by aggregating the symbolic
rule engine's per-plan scores across the present ply *and* the look-ahead span.
A plan that genuinely executes keeps its rule score high across those plies and
wins the aggregate; a one-ply flicker that fizzles out averages down to
``NO_PLAN``. The same ambiguity/threshold gating as the rule engine is applied
to the aggregate, so the label is deterministic.

:func:`filter_and_balance` then drops low-confidence / lopsided-material records,
caps positions per game, and downsamples over-represented classes.

Owner: Shahzad Ahmad (24I-3090). Subtasks 1.5.1 and 1.5.2.
"""

from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import chess.pgn

from src.data.pgn_parser import parse_elo
from src.data.window_extractor import DEFAULT_HISTORY_LENGTH, DEFAULT_START_PLY, default_game_id
from src.rules.rule_engine import SymbolicRuleEngine

__all__ = [
    "LabeledPositionRecord",
    "RetrospectiveWeakLabeler",
    "filter_and_balance",
    "material_balance",
    "PLAN_ORDER",
    "PLAN_IDS",
]

PLAN_ORDER = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE"]
PLAN_IDS: Dict[str, int] = {name: idx for idx, name in enumerate(PLAN_ORDER)}
PLAN_IDS["NO_PLAN"] = 5

DEFAULT_LOOKAHEAD = 8

_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


@dataclass
class LabeledPositionRecord:
    """A weakly-labeled position window, the atomic unit of the dataset."""

    window_id: str
    game_id: str
    ply_index: int
    turn: str  # "white" / "black"
    label_class: str
    label_id: int
    weak_confidence: float
    fen: str
    material_balance: int = 0
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: str = "*"
    split: str = ""


def material_balance(board: chess.Board) -> int:
    """White-minus-black material in centipawns (kings excluded)."""
    total = 0
    for piece_type, value in _PIECE_VALUES.items():
        total += value * chess.popcount(board.pieces_mask(piece_type, chess.WHITE))
        total -= value * chess.popcount(board.pieces_mask(piece_type, chess.BLACK))
    return total


class RetrospectiveWeakLabeler:
    """Assign weak plan labels by confirming execution over a look-ahead span."""

    def __init__(
        self,
        engine: Optional[SymbolicRuleEngine] = None,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        start_ply: int = DEFAULT_START_PLY,
        lookahead: int = DEFAULT_LOOKAHEAD,
    ) -> None:
        self.engine = engine or SymbolicRuleEngine()
        self.history_length = history_length
        self.start_ply = start_ply
        self.lookahead = lookahead
        self.threshold = self.engine.threshold
        self.margin = self.engine.ambiguity_margin

    def label_game(
        self, game: chess.pgn.Game, game_id: Optional[str] = None
    ) -> List[LabeledPositionRecord]:
        """Produce one :class:`LabeledPositionRecord` per window in ``game``."""
        gid = game_id if game_id is not None else default_game_id(game)
        white_elo = parse_elo(game.headers.get("WhiteElo"))
        black_elo = parse_elo(game.headers.get("BlackElo"))
        result = game.headers.get("Result", "*")

        # Replay the game into per-ply board snapshots (no undo stack needed).
        board = game.board()
        boards: List[chess.Board] = [board.copy(stack=False)]
        for move in game.mainline_moves():
            board.push(move)
            boards.append(board.copy(stack=False))

        total_plies = len(boards) - 1
        if total_plies < self.start_ply:
            return []

        # Score every ply once (reused by all overlapping look-ahead windows).
        per_ply_scores: Dict[int, Dict[str, float]] = {}
        for t in range(self.start_ply, total_plies + 1):
            lo = max(0, t - (self.history_length - 1))
            per_ply_scores[t] = self.engine.evaluate_window(boards[lo : t + 1]).all_scores

        records: List[LabeledPositionRecord] = []
        for t in range(self.start_ply, total_plies + 1):
            windows = [per_ply_scores[t]]
            for k in range(1, self.lookahead + 1):
                if t + k in per_ply_scores:
                    windows.append(per_ply_scores[t + k])

            agg = {p: sum(w[p] for w in windows) / len(windows) for p in PLAN_ORDER}
            label_class, label_id, confidence = self._decide(agg)

            current = boards[t]
            records.append(
                LabeledPositionRecord(
                    window_id=f"{gid}#{t}",
                    game_id=gid,
                    ply_index=t,
                    turn="white" if current.turn == chess.WHITE else "black",
                    label_class=label_class,
                    label_id=label_id,
                    weak_confidence=confidence,
                    fen=current.fen(),
                    material_balance=material_balance(current),
                    white_elo=white_elo,
                    black_elo=black_elo,
                    result=result,
                )
            )
        return records

    def _decide(self, agg: Dict[str, float]):
        """Apply threshold + ambiguity gating to the aggregated scores."""
        best_plan = min(PLAN_ORDER, key=lambda p: (-agg[p], PLAN_ORDER.index(p)))
        ordered = sorted(agg.values(), reverse=True)
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0

        if best >= self.threshold and (best - second) >= self.margin:
            return best_plan, PLAN_IDS[best_plan], round(best, 4)
        return "NO_PLAN", PLAN_IDS["NO_PLAN"], round(1.0 - best, 4)


def filter_and_balance(
    records: List[LabeledPositionRecord],
    min_confidence: float = 0.55,
    max_material: int = 800,
    max_per_game: int = 25,
    max_class_fraction: float = 0.30,
    seed: int = 42,
) -> List[LabeledPositionRecord]:
    """Clean and balance raw weak-labeled records (Subtask 1.5.2).

    Steps: drop low-confidence / lopsided-material records, cap each game's
    contribution, then downsample any class exceeding ``max_class_fraction`` of
    the total. Deterministic for a given ``seed``.
    """
    # 1. Confidence and material-sanity filter.
    kept = [
        r
        for r in records
        if r.weak_confidence >= min_confidence and abs(r.material_balance) <= max_material
    ]

    # 2. Per-game cap: keep the highest-confidence positions from each game.
    by_game: Dict[str, List[LabeledPositionRecord]] = collections.defaultdict(list)
    for r in kept:
        by_game[r.game_id].append(r)
    capped: List[LabeledPositionRecord] = []
    for recs in by_game.values():
        recs.sort(key=lambda r: (-r.weak_confidence, r.ply_index))
        capped.extend(recs[:max_per_game])

    # 3. Class balancing: iteratively cap any over-represented class.
    by_class: Dict[str, List[LabeledPositionRecord]] = collections.defaultdict(list)
    for r in capped:
        by_class[r.label_class].append(r)
    counts = {c: len(v) for c, v in by_class.items()}

    # A single class can never be below max_class_fraction of a one-class set, so
    # the target can't drop under the uniform floor (1 / number of classes);
    # otherwise balancing would delete everything.
    num_classes = sum(1 for v in counts.values() if v > 0)
    effective_fraction = max(max_class_fraction, 1.0 / num_classes) if num_classes else 1.0

    # Each non-breaking pass strictly reduces the worst class's count by at
    # least 1 (it is only reassigned when its current value exceeds the
    # target), so the initial total is a safe bound on passes needed to reach
    # a fixed point. `len(counts) + 1` previously assumed one pass per class,
    # which only holds when one class dominates; with several comparably-sized
    # classes it stopped early and left the cap violated.
    for _ in range(sum(counts.values()) + 1):
        total = sum(counts.values())
        if total == 0:
            break
        worst = max(counts, key=lambda c: counts[c])
        if counts[worst] <= effective_fraction * total + 1e-9:
            break
        others = total - counts[worst]
        # `+ 1e-9` guards against float error landing an exact-integer target
        # just under its true value (e.g. 6.999999999999999 instead of 7),
        # which would floor it one short and, compounded over iterations,
        # decay every class toward zero instead of settling at the target.
        allowed = int(math.floor(effective_fraction / (1.0 - effective_fraction) * others + 1e-9))
        counts[worst] = max(allowed, 0)

    rng = random.Random(seed)
    result: List[LabeledPositionRecord] = []
    for cls, recs in by_class.items():
        if counts[cls] >= len(recs):
            result.extend(recs)
        else:
            # Keep the most confident, breaking ties deterministically.
            recs_sorted = sorted(recs, key=lambda r: (-r.weak_confidence, r.window_id))
            result.extend(recs_sorted[: counts[cls]])
    rng.shuffle(result)
    return result
