"""Unit tests for the symbolic fact extractor (Task 1.3)."""

from __future__ import annotations

import time

import chess

from src.rules.fact_extractor import ChessFactExtractor, ExtractedBoardFacts

EXTRACTOR = ChessFactExtractor()


def facts(fen: str, ply: int = 0) -> ExtractedBoardFacts:
    return EXTRACTOR.extract_facts(chess.Board(fen), ply_index=ply)


# --------------------------------------------------------------------------- #
# Starting position: zero false positives                                     #
# --------------------------------------------------------------------------- #
def test_starting_position_is_quiet() -> None:
    f = facts(chess.STARTING_FEN)
    assert f.open_files == []
    assert f.white_semi_open_files == [] and f.black_semi_open_files == []
    assert f.white_outpost_squares == [] and f.black_outpost_squares == []
    assert f.white_pawn_chains == [] and f.black_pawn_chains == []
    assert f.white_rook_batteries == [] and f.black_rook_batteries == []
    assert sum(f.flank_pawns_advanced.values()) == 0
    assert f.material["balance"] == 0
    assert f.turn == chess.WHITE
    assert f.white_king_square == chess.E1 and f.black_king_square == chess.E8


def test_piece_masks_match_board() -> None:
    board = chess.Board()
    masks = EXTRACTOR.piece_masks(board, chess.WHITE)
    assert masks["pawn"] == chess.BB_RANK_2
    assert masks["rook"] == (chess.BB_A1 | chess.BB_H1)
    assert masks["king"] == chess.BB_E1


def test_material_values() -> None:
    f = facts(chess.STARTING_FEN)
    # 8*100 + 2*320 + 2*330 + 2*500 + 900 = 4000 per side.
    assert f.material["white"] == 4000
    assert f.material["black"] == 4000


# --------------------------------------------------------------------------- #
# Open / semi-open files                                                       #
# --------------------------------------------------------------------------- #
def test_open_and_semi_open_files() -> None:
    # White pawn on d4, black pawn on e5. c-file fully open; d semi-open for
    # black (only white pawn), e semi-open for white (only black pawn).
    f = facts("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1")
    assert chess.square_file(chess.C1) in f.open_files  # file index 2 open
    assert 3 in f.black_semi_open_files  # d-file: white pawn only
    assert 4 in f.white_semi_open_files  # e-file: black pawn only


# --------------------------------------------------------------------------- #
# Outposts (Kasparov-style d5 outpost, geometrically correct)                 #
# --------------------------------------------------------------------------- #
def test_white_d5_outpost_detected() -> None:
    # White Nd5 defended by pawn e4; black has only a d6 pawn (cannot contest d5).
    f = facts("7k/8/3p4/3N4/4P3/8/8/4K3 w - - 0 1")
    assert chess.D5 in f.white_outpost_squares
    assert chess.D5 in f.white_occupied_outposts
    assert f.black_outpost_squares == []


def test_outpost_rejected_when_contestable_by_enemy_pawn() -> None:
    # Same as above but add a black c7 pawn, which can play c6 to hit d5.
    f = facts("7k/2p5/3p4/3N4/4P3/8/8/4K3 w - - 0 1")
    assert chess.D5 not in f.white_outpost_squares


def test_no_outpost_without_pawn_support() -> None:
    # Knight on d5 but no friendly pawn defends it -> not an outpost.
    f = facts("7k/8/3p4/3N4/8/8/8/4K3 w - - 0 1")
    assert chess.D5 not in f.white_outpost_squares


# --------------------------------------------------------------------------- #
# Heavy-piece batteries (Alekhine's Gun)                                       #
# --------------------------------------------------------------------------- #
def test_alekhines_gun_c_file_battery_and_open_file() -> None:
    f = facts("2r1r1k1/5ppp/8/8/2R5/2R5/2Q5/6K1 w - - 0 1")
    assert 2 in f.white_rook_batteries  # c-file: Rc4 + Rc3 + Qc2
    assert 2 in f.open_files  # no pawns on the c-file
    assert f.black_rook_batteries == []  # black rooks are on c8 and e8


# --------------------------------------------------------------------------- #
# Pawn chains                                                                  #
# --------------------------------------------------------------------------- #
def test_french_chain_with_c3_base() -> None:
    # White chain c3 -> d4 -> e5; base (rearmost) is c3.
    f = facts("4k3/8/4p3/3pP3/3P4/2P5/8/4K3 w - - 0 1")
    chains = f.white_pawn_chains
    assert len(chains) == 1
    chain = chains[0]
    assert chain[0] == chess.C3  # base at index 0
    assert set(chain) == {chess.C3, chess.D4, chess.E5}


# --------------------------------------------------------------------------- #
# Kingside pawn storm (flank advance counting)                                 #
# --------------------------------------------------------------------------- #
def test_kingside_storm_pawns_counted() -> None:
    f = facts("5rk1/5ppp/8/6P1/7P/8/8/6K1 w - - 0 1")  # white g5 + h4
    assert f.flank_pawns_advanced["white_kingside"] == 2
    assert f.flank_pawns_advanced["white_queenside"] == 0
    assert f.flank_pawns_advanced["black_kingside"] == 0


# --------------------------------------------------------------------------- #
# King shield                                                                  #
# --------------------------------------------------------------------------- #
def test_king_shield_intact_after_castling() -> None:
    # White king g1 with f2/g2/h2 pawns in front -> shield of 3.
    f = facts("6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1")
    assert f.king_shield["white"] == 3


# --------------------------------------------------------------------------- #
# Mobility, ply_index passthrough                                              #
# --------------------------------------------------------------------------- #
def test_mobility_and_ply_index() -> None:
    f = facts(chess.STARTING_FEN, ply=12)
    assert f.ply_index == 12
    # Both knights have exactly two legal squares each at the start (a3/c3, f3/h3).
    assert f.white_mobility["knight"] == 4
    # The opening position is colour-symmetric, so mobility dicts match.
    assert f.white_mobility == f.black_mobility


# --------------------------------------------------------------------------- #
# Black-side mirrors (every field works for both colours)                      #
# --------------------------------------------------------------------------- #
def test_black_outpost_and_shield() -> None:
    # Black Ne4 defended by pawn d5; black king g8 with f7/g7/h7 shield.
    f = facts("6k1/5ppp/8/3p4/4n3/8/8/6K1 w - - 0 1")
    assert chess.E4 in f.black_outpost_squares
    assert chess.E4 in f.black_occupied_outposts
    assert f.king_shield["black"] == 3
    assert f.white_outpost_squares == [] and f.white_rook_batteries == []


def test_black_pawn_chain_base_at_top() -> None:
    # Black chain c4<-d5<-e6; base (rearmost for black) is e6 at index 0.
    f = facts("6k1/8/4p3/3p4/2p5/8/8/6K1 w - - 0 1")
    assert len(f.black_pawn_chains) == 1
    chain = f.black_pawn_chains[0]
    assert chain[0] == chess.E6
    assert set(chain) == {chess.C4, chess.D5, chess.E6}


# --------------------------------------------------------------------------- #
# Performance                                                                  #
# --------------------------------------------------------------------------- #
def test_extraction_is_fast() -> None:
    board = chess.Board("2r1r1k1/5ppp/8/8/2R5/2R5/2Q5/6K1 w - - 0 1")
    start = time.perf_counter()
    for _ in range(1000):
        EXTRACTOR.extract_facts(board)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"1000 extractions took {elapsed:.3f}s (budget 0.5s)"
