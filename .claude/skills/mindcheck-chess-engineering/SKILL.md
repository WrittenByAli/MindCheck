---
name: mindcheck-chess-engineering
description: Specialized chess-domain engineering for PGN/FEN processing, python-chess bitboard usage, strategic evidence extraction and temporal plan tracking. Load when working on src/data, src/rules, or src/tracking chess logic.
---

# MindCheck Chess Engineering

## python-chess fundamentals
- Use 64-bit bitboards (`board.pieces_mask`), bitwise `&`/`|`/`^` over attack masks, and `bin(mask).count('1')` for popcount. Never loop square-by-square when a bitwise op suffices.
- Validate legality through `board.legal_moves`; never assume a move or square is reachable without checking.
- Track castling rights and en-passant squares as explicit metadata channels, not inferred from history.

## Orientation & representation
- Normalize to attacker-perspective: if Black is to move, flip the board array on the row axis and swap friendly/opponent planes. The network must always see the active side attacking toward rank 8 (row 0).
- Keep the most recent N half-moves available as discrete board snapshots, not merged/aggregated state.

## Six-class strategic taxonomy
`K_STORM`, `Q_EXP`, `OUTPOST`, `ROOK_FILE`, `CHAIN_UNDERMINE`, `NO_PLAN` — see root `CLAUDE.md` for full preconditions per class.

## Symbolic evidence & temporal tracking
- Every strategic predicate (e.g. "pawn on g5 supported by heavy piece") must be independently testable against a `chess.Board` fixture.
- Plan activation requires 2 consecutive plies of confidence ≥ 0.45; switching requires 2 consecutive contradicting plies; decay to `NO_PLAN` after ≥3 ambiguous plies.
- Onset backtracking: walk back through the 8-ply window to the earliest ply where supporting evidence appears, and record it for the explanation.
- Explanations cite only verified bitboard facts (squares, pieces, files) — never a player's inferred intention.

## Hard restrictions
- Distinguish "square is attacked" from "move is legal" — these are different predicates, don't conflate them.
- Never claim to know a player's actual plan/intent; only report observed facts and the model's hypothesis, labeled as such.
- Symbolic rules (`src/rules/`) feed the fusion/evidence layer — they must never silently replace the neural classifier's output.
- Keep Stockfish integration (`src/utils/stockfish_client.py`) strictly optional; the pipeline must work without it.
