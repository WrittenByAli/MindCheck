"""Deterministic five-plan symbolic rule engine (Task 1.4, Milestone 4).

:class:`SymbolicRuleEngine` scores an 8-ply history window against each of the
five strategic plans and resolves a single label (or ``NO_PLAN``). It serves two
roles in MindCheck:

1. the **rule-only baseline** the neural network is benchmarked against, and
2. the symbolic evidence provider blended into the neural prediction by the
   fusion layer (Task 2.2).

Each rule produces a normalized score in ``[0.0, 1.0]`` from grounded board
facts (:class:`~src.rules.fact_extractor.ExtractedBoardFacts`) and the moves
played inside the window. Because the six plan classes do not encode which side
is acting, every rule is evaluated for *both* colours and the stronger score is
kept; the acting colour is recorded in the evidence. Output is fully
deterministic: identical inputs always yield identical scores.

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import chess

from src.rules.fact_extractor import ChessFactExtractor, ExtractedBoardFacts

__all__ = ["RuleEvaluationResult", "SymbolicRuleEngine"]

_PLAN_IDS: Dict[str, int] = {
    "K_STORM": 0,
    "Q_EXP": 1,
    "OUTPOST": 2,
    "ROOK_FILE": 3,
    "CHAIN_UNDERMINE": 4,
    "NO_PLAN": 5,
}
_PLAN_ORDER = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE"]

_KINGSIDE_FILES = {5, 6, 7}  # f, g, h
_QUEENSIDE_FILES = {0, 1, 2}  # a, b, c

Evidence = List[str]


@dataclass
class RuleEvaluationResult:
    """Outcome of evaluating one window with the symbolic rule engine."""

    predicted_class: str
    class_id: int
    confidence: float
    all_scores: Dict[str, float]
    supporting_evidence: List[str] = field(default_factory=list)


def _color_name(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


class SymbolicRuleEngine:
    """Evaluate the five strategic plans over an 8-ply board history window."""

    def __init__(self, activation_threshold: float = 0.50, ambiguity_margin: float = 0.15) -> None:
        self.threshold = activation_threshold
        self.ambiguity_margin = ambiguity_margin
        self._extractor = ChessFactExtractor()

    # ------------------------------------------------------------------ #
    # Top-level classifier (Subtask 1.4.6)                               #
    # ------------------------------------------------------------------ #
    def evaluate_window(
        self, history: List[chess.Board], facts: ExtractedBoardFacts | None = None
    ) -> RuleEvaluationResult:
        """Score every plan and resolve a single class with ambiguity gating."""
        if not history:
            raise ValueError("history must contain at least one board.")
        if facts is None:
            facts = self._extractor.extract_facts(history[-1])

        rule_fns = {
            "K_STORM": self._kingside_storm,
            "Q_EXP": self._queenside_expansion,
            "OUTPOST": self._outpost_occupation,
            "ROOK_FILE": self._rook_file_buildup,
            "CHAIN_UNDERMINE": self._chain_undermining,
        }

        scores: Dict[str, float] = {}
        evidence: Dict[str, Evidence] = {}
        for name in _PLAN_ORDER:
            best_score, best_evidence = 0.0, []
            for color in (chess.WHITE, chess.BLACK):
                score, ev = rule_fns[name](history, facts, color)
                if score > best_score:
                    best_score, best_evidence = score, ev
            scores[name] = round(best_score, 4)
            evidence[name] = best_evidence

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], _PLAN_ORDER.index(kv[0])))
        top_name, top_score = ordered[0]
        second_score = ordered[1][1]

        if top_score < self.threshold:
            return self._no_plan(
                scores, f"Top score {top_score:.2f} below threshold {self.threshold:.2f}."
            )
        if (top_score - second_score) < self.ambiguity_margin:
            return self._no_plan(
                scores,
                f"Ambiguous: {top_name} ({top_score:.2f}) vs {ordered[1][0]} "
                f"({second_score:.2f}) within margin {self.ambiguity_margin:.2f}.",
            )

        return RuleEvaluationResult(
            predicted_class=top_name,
            class_id=_PLAN_IDS[top_name],
            confidence=top_score,
            all_scores=scores,
            supporting_evidence=evidence[top_name],
        )

    def _no_plan(self, scores: Dict[str, float], reason: str) -> RuleEvaluationResult:
        top_score = max(scores.values()) if scores else 0.0
        return RuleEvaluationResult(
            predicted_class="NO_PLAN",
            class_id=_PLAN_IDS["NO_PLAN"],
            confidence=round(1.0 - top_score, 4),
            all_scores=scores,
            supporting_evidence=[reason],
        )

    # ------------------------------------------------------------------ #
    # Public per-rule scorers (spec signatures; max over both colours)   #
    # ------------------------------------------------------------------ #
    def evaluate_kingside_storm(
        self, history: List[chess.Board], facts: ExtractedBoardFacts
    ) -> float:
        return max(self._kingside_storm(history, facts, c)[0] for c in (chess.WHITE, chess.BLACK))

    def evaluate_queenside_expansion(
        self, history: List[chess.Board], facts: ExtractedBoardFacts
    ) -> float:
        return max(
            self._queenside_expansion(history, facts, c)[0] for c in (chess.WHITE, chess.BLACK)
        )

    def evaluate_outpost_occupation(
        self, history: List[chess.Board], facts: ExtractedBoardFacts
    ) -> float:
        return max(
            self._outpost_occupation(history, facts, c)[0] for c in (chess.WHITE, chess.BLACK)
        )

    def evaluate_rook_file_buildup(
        self, history: List[chess.Board], facts: ExtractedBoardFacts
    ) -> float:
        return max(
            self._rook_file_buildup(history, facts, c)[0] for c in (chess.WHITE, chess.BLACK)
        )

    def evaluate_chain_undermining(
        self, history: List[chess.Board], facts: ExtractedBoardFacts
    ) -> float:
        return max(
            self._chain_undermining(history, facts, c)[0] for c in (chess.WHITE, chess.BLACK)
        )

    # ------------------------------------------------------------------ #
    # Window-move helpers                                                 #
    # ------------------------------------------------------------------ #
    def _window_moves(self, history: List[chess.Board]):
        """Yield (board_after, move, mover_color, piece_type) for each window ply."""
        out = []
        for board in history:
            if board.move_stack:
                move = board.peek()
                piece = board.piece_at(move.to_square)
                if piece is not None:
                    out.append((board, move, not board.turn, piece.piece_type))
        return out

    def _pawn_pushes_on_files(self, history: List[chess.Board], color: chess.Color, files: set):
        """Count forward pawn pushes by ``color`` on ``files`` within the window."""
        count = 0
        to_mask = 0
        for _, move, mover, piece_type in self._window_moves(history):
            if mover != color or piece_type != chess.PAWN:
                continue
            if chess.square_file(move.to_square) not in files:
                continue
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            forward = to_rank > from_rank if color == chess.WHITE else to_rank < from_rank
            if forward:
                count += 1
                to_mask |= chess.BB_SQUARES[move.to_square]
        return count, to_mask

    def _count_pieces_on_files(
        self, board: chess.Board, color: chess.Color, files: set, piece_types
    ) -> int:
        total = 0
        for piece_type in piece_types:
            mask = board.pieces_mask(piece_type, color)
            for f in files:
                total += chess.popcount(mask & chess.BB_FILES[f])
        return total

    # ------------------------------------------------------------------ #
    # Rule 1.4.1: Kingside pawn storm                                     #
    # ------------------------------------------------------------------ #
    def _kingside_storm(self, history, facts, color) -> Tuple[float, Evidence]:
        pushes, _ = self._pawn_pushes_on_files(history, color, _KINGSIDE_FILES)
        advanced = facts.flank_pawns_advanced[
            "white_kingside" if color == chess.WHITE else "black_kingside"
        ]
        if pushes == 0 and advanced == 0:
            return 0.0, []

        opp_king = facts.black_king_square if color == chess.WHITE else facts.white_king_square
        king_exposed = opp_king is not None and chess.square_file(opp_king) in _KINGSIDE_FILES
        support = self._count_pieces_on_files(
            history[-1], color, _KINGSIDE_FILES, (chess.ROOK, chess.QUEEN)
        )

        # Piece-support and king-exposure only count once the pawn advance is a
        # committed storm (>= 2 pushes or >= 2 flank pawns advanced), so a lone
        # push is not mistaken for a storm.
        committed = pushes >= 2 or advanced >= 2
        pawn_component = min(1.0, pushes / 2.0)
        depth_component = min(1.0, advanced / 2.0)
        support_component = min(1.0, float(support)) if committed else 0.0
        king_component = (1.0 if king_exposed else 0.0) if committed else 0.0

        score = min(
            1.0,
            0.4 * pawn_component
            + 0.2 * depth_component
            + 0.2 * support_component
            + 0.2 * king_component,
        )
        evidence = [
            f"{_color_name(color)} pushed {pushes} kingside flank pawn(s) in the window",
            f"{advanced} {_color_name(color)} f/g/h pawn(s) advanced",
        ]
        if support:
            evidence.append(f"{support} heavy piece(s) supporting the f/g/h files")
        if king_exposed:
            evidence.append("opponent king is on the kingside")
        return score, evidence

    # ------------------------------------------------------------------ #
    # Rule 1.4.2: Queenside expansion                                     #
    # ------------------------------------------------------------------ #
    def _queenside_expansion(self, history, facts, color) -> Tuple[float, Evidence]:
        pushes, _ = self._pawn_pushes_on_files(history, color, _QUEENSIDE_FILES)
        advanced = facts.flank_pawns_advanced[
            "white_queenside" if color == chess.WHITE else "black_queenside"
        ]
        if pushes == 0 and advanced == 0:
            return 0.0, []

        support = self._count_pieces_on_files(
            history[-1],
            color,
            _QUEENSIDE_FILES,
            (chess.ROOK, chess.QUEEN, chess.KNIGHT, chess.BISHOP),
        )
        # As with the storm, piece-support only counts for a committed advance.
        committed = pushes >= 2 or advanced >= 2
        push_component = min(1.0, pushes / 2.0)
        space_component = min(1.0, advanced / 2.0)
        support_component = min(1.0, float(support)) if committed else 0.0

        score = min(1.0, 0.4 * push_component + 0.3 * space_component + 0.3 * support_component)
        evidence = [
            f"{_color_name(color)} pushed {pushes} queenside pawn(s) in the window",
            f"{advanced} {_color_name(color)} a/b/c pawn(s) advanced for space",
        ]
        if support:
            evidence.append(f"{support} piece(s) supporting the queenside pawn mass")
        return score, evidence

    # ------------------------------------------------------------------ #
    # Rule 1.4.3: Outpost occupation                                      #
    # ------------------------------------------------------------------ #
    def _outpost_occupation(self, history, facts, color) -> Tuple[float, Evidence]:
        occupied = (
            facts.white_occupied_outposts if color == chess.WHITE else facts.black_occupied_outposts
        )
        candidates = (
            facts.white_outpost_squares if color == chess.WHITE else facts.black_outpost_squares
        )

        if occupied:
            squares = ", ".join(chess.square_name(s) for s in occupied)
            return 1.0, [f"{_color_name(color)} minor piece occupies outpost(s): {squares}"]
        if candidates:
            squares = ", ".join(chess.square_name(s) for s in candidates)
            return 0.7, [f"{_color_name(color)} can maneuver a minor to outpost(s): {squares}"]
        return 0.0, []

    # ------------------------------------------------------------------ #
    # Rule 1.4.4: Rook-file buildup                                       #
    # ------------------------------------------------------------------ #
    def _rook_file_buildup(self, history, facts, color) -> Tuple[float, Evidence]:
        batteries = (
            facts.white_rook_batteries if color == chess.WHITE else facts.black_rook_batteries
        )
        semi_open = (
            facts.white_semi_open_files if color == chess.WHITE else facts.black_semi_open_files
        )

        best = 0.0
        evidence: Evidence = []
        for f in batteries:
            if f in facts.open_files and best < 1.0:
                best, evidence = 1.0, [
                    f"{_color_name(color)} heavy battery on the open {chr(97 + f)}-file"
                ]
            elif f in semi_open and best < 0.7:
                best, evidence = 0.7, [
                    f"{_color_name(color)} heavy battery on the semi-open {chr(97 + f)}-file"
                ]

        if best == 0.0:
            rooks = history[-1].pieces_mask(chess.ROOK, color)
            for f in facts.open_files:
                if chess.popcount(rooks & chess.BB_FILES[f]) >= 1:
                    best, evidence = 0.4, [
                        f"{_color_name(color)} rook on the open {chr(97 + f)}-file"
                    ]
                    break
        return best, evidence

    # ------------------------------------------------------------------ #
    # Rule 1.4.5: Pawn-chain undermining                                  #
    # ------------------------------------------------------------------ #
    def _chain_undermining(self, history, facts, color) -> Tuple[float, Evidence]:
        opponent = not color
        opp_chains = facts.black_pawn_chains if opponent == chess.BLACK else facts.white_pawn_chains
        if not opp_chains:
            return 0.0, []

        board = history[-1]
        friendly_pawns = board.pieces_mask(chess.PAWN, color)
        _, recent_push_mask = self._pawn_pushes_on_files(history, color, set(range(8)))
        if not recent_push_mask:
            return 0.0, []

        best = 0.0
        evidence: Evidence = []
        for chain in opp_chains:
            base, apex = chain[0], chain[-1]
            base_attackers = friendly_pawns & chess.BB_PAWN_ATTACKS[opponent][base]
            apex_attackers = friendly_pawns & chess.BB_PAWN_ATTACKS[opponent][apex]
            if base_attackers & recent_push_mask and best < 1.0:
                best = 1.0
                evidence = [
                    f"{_color_name(color)} lever attacks the chain base on {chess.square_name(base)}"
                ]
            elif apex_attackers & recent_push_mask and best < 0.8:
                best = 0.8
                evidence = [
                    f"{_color_name(color)} lever attacks the chain apex on {chess.square_name(apex)}"
                ]
        return best, evidence
