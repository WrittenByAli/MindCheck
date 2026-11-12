"""Deterministic symbolic chess fact extractor (Task 1.3, Milestone 3).

:class:`ChessFactExtractor` turns any ``chess.Board`` into a bundle of grounded,
geometric facts — open/semi-open files, pawn chains and their bases, minor-piece
outposts, heavy-piece batteries, flank-pawn advances, king-shield integrity,
mobility and material — computed with ``python-chess`` 64-bit bitboards.

These facts are *verifiable truths about the position* (never guesses about a
player's intention). They feed three downstream consumers:
  * the weak-labeler and rule engine (Tasks 1.4/1.5),
  * the rule-only baseline classifier, and
  * the deterministic explanation generator (Task 2.3).

All predicates are side-effect free and independently testable against FEN
fixtures. Extraction is designed to run in well under 1 ms per board.

Owner: Shahzad Ahmad (24I-3090).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import chess

__all__ = ["ExtractedBoardFacts", "ChessFactExtractor"]

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_PIECE_NAMES: Dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
# Standard centipawn material values (king excluded from the balance).
_PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
# Outposts are meaningful only in the central/kingside files c, d, e, f.
_OUTPOST_FILES = {2, 3, 4, 5}
# Candidate outpost ranks (0-indexed): white ranks 4-6, black ranks 3-5.
_OUTPOST_RANKS = {chess.WHITE: {3, 4, 5}, chess.BLACK: {2, 3, 4}}
_KINGSIDE_FILES = {5, 6, 7}  # f, g, h
_QUEENSIDE_FILES = {0, 1, 2}  # a, b, c


@dataclass
class ExtractedBoardFacts:
    """Immutable-ish bundle of symbolic facts for a single board state."""

    ply_index: int
    turn: chess.Color
    white_king_square: chess.Square
    black_king_square: chess.Square
    open_files: List[int]
    white_semi_open_files: List[int]
    black_semi_open_files: List[int]
    white_outpost_squares: List[chess.Square]
    black_outpost_squares: List[chess.Square]
    white_occupied_outposts: List[chess.Square]
    black_occupied_outposts: List[chess.Square]
    white_pawn_chains: List[List[chess.Square]]
    black_pawn_chains: List[List[chess.Square]]
    white_rook_batteries: List[int]
    black_rook_batteries: List[int]
    flank_pawns_advanced: Dict[str, int]
    king_shield: Dict[str, int]
    white_mobility: Dict[str, int]
    black_mobility: Dict[str, int]
    material: Dict[str, int]
    metadata: dict = field(default_factory=dict)


class ChessFactExtractor:
    """Extract grounded bitboard predicates from a chess position."""

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def extract_facts(self, board: chess.Board, ply_index: int = 0) -> ExtractedBoardFacts:
        """Return the complete :class:`ExtractedBoardFacts` for ``board``."""
        open_files, white_semi, black_semi = self._file_classification(board)
        white_outposts, white_occ = self._outposts(board, chess.WHITE)
        black_outposts, black_occ = self._outposts(board, chess.BLACK)

        return ExtractedBoardFacts(
            ply_index=ply_index,
            turn=board.turn,
            white_king_square=board.king(chess.WHITE),
            black_king_square=board.king(chess.BLACK),
            open_files=open_files,
            white_semi_open_files=white_semi,
            black_semi_open_files=black_semi,
            white_outpost_squares=white_outposts,
            black_outpost_squares=black_outposts,
            white_occupied_outposts=white_occ,
            black_occupied_outposts=black_occ,
            white_pawn_chains=self._pawn_chains(board, chess.WHITE),
            black_pawn_chains=self._pawn_chains(board, chess.BLACK),
            white_rook_batteries=self._rook_batteries(board, chess.WHITE),
            black_rook_batteries=self._rook_batteries(board, chess.BLACK),
            flank_pawns_advanced=self._flank_advanced(board),
            king_shield=self._king_shield(board),
            white_mobility=self._mobility(board, chess.WHITE),
            black_mobility=self._mobility(board, chess.BLACK),
            material=self._material(board),
        )

    def piece_masks(self, board: chess.Board, color: chess.Color) -> Dict[str, int]:
        """Return the raw 64-bit bitmask for each piece type of ``color``."""
        return {_PIECE_NAMES[pt]: board.pieces_mask(pt, color) for pt in _PIECE_TYPES}

    def attack_mask(self, board: chess.Board, color: chess.Color) -> int:
        """Union of all squares attacked by any piece of ``color``."""
        mask = 0
        for sq in chess.scan_forward(board.occupied_co[color]):
            mask |= board.attacks_mask(sq)
        return mask

    # ------------------------------------------------------------------ #
    # Files                                                              #
    # ------------------------------------------------------------------ #
    def _file_classification(self, board: chess.Board):
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
        open_files: List[int] = []
        white_semi: List[int] = []
        black_semi: List[int] = []
        for f in range(8):
            file_mask = chess.BB_FILES[f]
            has_white = bool(white_pawns & file_mask)
            has_black = bool(black_pawns & file_mask)
            if not has_white and not has_black:
                open_files.append(f)
            elif not has_white and has_black:
                white_semi.append(f)  # no friendly (white) pawn, enemy present
            elif has_white and not has_black:
                black_semi.append(f)
        return open_files, white_semi, black_semi

    # ------------------------------------------------------------------ #
    # Pawn chains                                                        #
    # ------------------------------------------------------------------ #
    def _pawn_attack_mask(self, board: chess.Board, color: chess.Color) -> int:
        mask = 0
        for sq in chess.scan_forward(board.pieces_mask(chess.PAWN, color)):
            mask |= chess.BB_PAWN_ATTACKS[color][sq]
        return mask

    def _pawn_chains(self, board: chess.Board, color: chess.Color) -> List[List[chess.Square]]:
        pawns = board.pieces_mask(chess.PAWN, color)
        squares = list(chess.scan_forward(pawns))
        if not squares:
            return []

        enemy = not color
        adjacency: Dict[int, set] = {s: set() for s in squares}
        for s in squares:
            # Friendly pawns that defend s sit on the reverse pawn-attack squares.
            supporters = pawns & chess.BB_PAWN_ATTACKS[enemy][s]
            for q in chess.scan_forward(supporters):
                adjacency[s].add(q)
                adjacency[q].add(s)

        seen: set = set()
        chains: List[List[chess.Square]] = []
        for start in squares:
            if start in seen:
                continue
            component: List[int] = []
            stack = [start]
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in adjacency[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            if len(component) >= 2:
                # Order so the base (rearmost pawn) is at index 0.
                component.sort(key=chess.square_rank, reverse=(color == chess.BLACK))
                chains.append(component)
        return chains

    # ------------------------------------------------------------------ #
    # Outposts                                                           #
    # ------------------------------------------------------------------ #
    def _enemy_pawn_can_attack(self, square: int, color: chess.Color, enemy_pawns: int) -> bool:
        """True if an enemy pawn attacks ``square`` now or after advancing."""
        f = chess.square_file(square)
        r = chess.square_rank(square)
        for adjacent_file in (f - 1, f + 1):
            if not 0 <= adjacent_file <= 7:
                continue
            on_file = enemy_pawns & chess.BB_FILES[adjacent_file]
            for q in chess.scan_forward(on_file):
                qr = chess.square_rank(q)
                if color == chess.WHITE and qr > r:
                    return True
                if color == chess.BLACK and qr < r:
                    return True
        return False

    def _outposts(self, board: chess.Board, color: chess.Color):
        friendly_pawn_attacks = self._pawn_attack_mask(board, color)
        enemy_pawns = board.pieces_mask(chess.PAWN, not color)
        own_occupied = board.occupied_co[color]
        minors = board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color)

        minor_attacks = 0
        for sq in chess.scan_forward(minors):
            minor_attacks |= board.attacks_mask(sq)

        rank_set = _OUTPOST_RANKS[color]
        outposts: List[int] = []
        occupied: List[int] = []
        for s in range(64):
            if chess.square_file(s) not in _OUTPOST_FILES:
                continue
            if chess.square_rank(s) not in rank_set:
                continue
            bb = chess.BB_SQUARES[s]
            if not (friendly_pawn_attacks & bb):
                continue  # a hole must be defended by a friendly pawn
            if self._enemy_pawn_can_attack(s, color, enemy_pawns):
                continue  # contestable by an enemy pawn -> not a true outpost
            occupied_by_minor = bool(minors & bb)
            enemy_occupied = bool(board.occupied_co[not color] & bb)
            reachable = bool(minor_attacks & bb) and not (own_occupied & bb) and not enemy_occupied
            if occupied_by_minor or reachable:
                outposts.append(s)
                if occupied_by_minor:
                    occupied.append(s)
        return outposts, occupied

    # ------------------------------------------------------------------ #
    # Heavy-piece batteries                                             #
    # ------------------------------------------------------------------ #
    def _rook_batteries(self, board: chess.Board, color: chess.Color) -> List[int]:
        heavy = board.pieces_mask(chess.ROOK, color) | board.pieces_mask(chess.QUEEN, color)
        entry_ranks = (
            (chess.BB_RANK_7 | chess.BB_RANK_8)
            if color == chess.WHITE
            else (chess.BB_RANK_1 | chess.BB_RANK_2)
        )
        files: List[int] = []
        for f in range(8):
            file_mask = chess.BB_FILES[f]
            if chess.popcount(heavy & file_mask) < 2:
                continue
            has_entry_control = False
            for sq in chess.scan_forward(heavy & file_mask):
                if board.attacks_mask(sq) & entry_ranks & file_mask:
                    has_entry_control = True
                    break
            if has_entry_control:
                files.append(f)
        return files

    # ------------------------------------------------------------------ #
    # Flank advances & king shield                                      #
    # ------------------------------------------------------------------ #
    def _flank_advanced(self, board: chess.Board) -> Dict[str, int]:
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

        def count(pawns: int, files: set, white: bool) -> int:
            # "Advanced" = at least two squares forward of the home rank: white
            # pawns on rank 4+ (index >= 3), the mirror being black on rank 5-
            # (index <= 4).
            total = 0
            for sq in chess.scan_forward(pawns):
                if chess.square_file(sq) not in files:
                    continue
                r = chess.square_rank(sq)
                if white and r >= 3:
                    total += 1
                elif not white and r <= 4:
                    total += 1
            return total

        return {
            "white_kingside": count(white_pawns, _KINGSIDE_FILES, True),
            "white_queenside": count(white_pawns, _QUEENSIDE_FILES, True),
            "black_kingside": count(black_pawns, _KINGSIDE_FILES, False),
            "black_queenside": count(black_pawns, _QUEENSIDE_FILES, False),
        }

    def _king_shield(self, board: chess.Board) -> Dict[str, int]:
        return {
            "white": self._king_shield_for(board, chess.WHITE),
            "black": self._king_shield_for(board, chess.BLACK),
        }

    def _king_shield_for(self, board: chess.Board, color: chess.Color) -> int:
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        pawns = board.pieces_mask(chess.PAWN, color)
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        files = [f for f in (kf - 1, kf, kf + 1) if 0 <= f <= 7]
        ranks = [kr + 1, kr + 2] if color == chess.WHITE else [kr - 1, kr - 2]
        count = 0
        for f in files:
            for r in ranks:
                if 0 <= r <= 7 and pawns & chess.BB_SQUARES[chess.square(f, r)]:
                    count += 1
        return count

    # ------------------------------------------------------------------ #
    # Mobility & material                                               #
    # ------------------------------------------------------------------ #
    def _mobility(self, board: chess.Board, color: chess.Color) -> Dict[str, int]:
        own = board.occupied_co[color]
        out: Dict[str, int] = {}
        for pt in _PIECE_TYPES:
            moves = 0
            for sq in chess.scan_forward(board.pieces_mask(pt, color)):
                moves += chess.popcount(board.attacks_mask(sq) & ~own & chess.BB_ALL)
            out[_PIECE_NAMES[pt]] = moves
        return out

    def _material(self, board: chess.Board) -> Dict[str, int]:
        white = sum(
            _PIECE_VALUES[pt] * chess.popcount(board.pieces_mask(pt, chess.WHITE))
            for pt in _PIECE_TYPES
        )
        black = sum(
            _PIECE_VALUES[pt] * chess.popcount(board.pieces_mask(pt, chess.BLACK))
            for pt in _PIECE_TYPES
        )
        return {"white": white, "black": black, "balance": white - black}
