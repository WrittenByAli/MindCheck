# MindCheck — Independent Annotation Guidelines (Task 1.9.1)

> Written for the two annotators (Shahzad Ahmad, Muhammad Ali). Read this fully
> and sign off at the bottom **before** opening your annotation sheet. The
> authoritative plan definitions live in
> [docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md](../../docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md);
> this file is the operational procedure that sits on top of it.

## 1. What you are doing
You are independently assigning the single best **strategic plan** to each of
250 held-out chess position windows, to build the frozen human gold-standard the
models are judged against. Your labels — not the weak labels, not the model —
are the ground truth.

## 2. Hard rules (these protect the benchmark's validity)
1. **History-only.** For each row you may inspect **only the last 8 half-moves**
   leading to the position (the `history_moves_san` column) and the current
   board (`fen_current`). **Never look at what happens next** in the game.
2. **Independence.** Do **not** communicate about any position, and do **not**
   open the other annotator's file, until both sheets are complete and
   submitted. No weak labels or model predictions appear in your sheet by
   design — do not seek them out.
3. **One primary label per row**, drawn exactly from the taxonomy strings below.
4. Fill **every** row. No blanks, no made-up class names.

## 3. The six labels
Assign the plan of the **side to move** (`turn`): the player whose move it is in
the current position and who is orchestrating the plan.

| Label | Plan | Assign when… |
| :-- | :-- | :-- |
| `K_STORM` | Kingside Pawn Storm | f/g/h pawns are being advanced toward the opponent's castled king, usually with heavy-piece support. |
| `Q_EXP` | Queenside Expansion | a/b/c pawns advance to grab queenside space or launch a minority attack. |
| `OUTPOST` | Outpost Occupation | a knight/bishop is being placed (or maneuvered) onto a protected hole on c–f that enemy pawns cannot challenge. |
| `ROOK_FILE` | Rook-File Buildup | rooks/queen are doubling on an open or semi-open file aiming at the 7th/8th rank. |
| `CHAIN_UNDERMINE` | Pawn-Chain Undermining | a pawn lever is struck at the base/apex of the opponent's pawn chain. |
| `NO_PLAN` | No Dominant Plan | balanced maneuvering, pure development, tactics, or two weak signals that conflict — when nothing above clearly dominates. |

**Tie-breaking:** if two plans are both plausible, pick the one with the stronger
board evidence as `primary_label` and record the other as `secondary_label`.
If genuinely nothing dominates, the primary label is `NO_PLAN`.

## 4. The columns you fill
- `primary_label` — one taxonomy string from the table above. **Required.**
- `confidence` — your certainty, a number in **0.4–1.0** (0.4 = barely leaning,
  1.0 = textbook-clear). **Required.**
- `secondary_label` — the next most plausible plan, or `NO_PLAN` if none.
- `notes` — one short line of grounded rationale citing squares/pieces
  (e.g. "knight to d5 defended by e4, no black c/e pawn"). No guesses about the
  opponent's psychology.

Do **not** edit `window_id`, `game_id`, `ply_index`, `turn`, `fen_current`, or
`history_moves_san`.

## 5. Procedure
1. Read this file and [docs/06](../../docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md).
2. Open only your own sheet: Shahzad → `raw_annotations_member1.csv`,
   Ali → `raw_annotations_member2.csv`.
3. Work top to bottom, filling the four columns in §4 for all 250 rows.
4. Submit. Only then do we run `src/models/adjudicate.py` to compute Cohen's
   Kappa and list disagreements for a joint adjudication session.

## 6. Report disclosure
Both annotators disclose their chess rating / experience level in the final
report (required by Subtask 1.9.3).

- Shahzad Ahmad — rating/experience: ______________
- Muhammad Ali — rating/experience: ______________

## 7. Sign-off (required before annotating)
By signing, each member confirms they understand the identical rubric and will
follow the independence and history-only rules.

- [ ] Shahzad Ahmad (24I-3090) — date: __________
- [ ] Muhammad Ali (24I-3158) — date: __________
