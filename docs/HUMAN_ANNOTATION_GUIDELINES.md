# MindCheck — Human Annotation Guidelines (Revised)

> Written for the two annotators: **Shahzad Ahmad** (24I-3090) and
> **Muhammad Ali** (24I-3158). Read this document and the authoritative taxonomy
> in [docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md](06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md)
> in full before opening your annotation sheet or signing off.
>
> This document is the operational procedure for the human annotation benchmark.
> The authoritative class definitions live in `docs/06`; this file does not
> override or extend them.
>
> **Supersedes**: `data/annotations/ANNOTATION_GUIDELINES.md` (the original
> version, preserved unchanged for the record).

---

## 1. What you are doing

You are independently assigning strategic-plan labels to 250 held-out chess
position windows drawn from the **test** split — games the model has never seen
during training or validation.

Your labels serve as **independent human-reference judgments** against which the
neural model, the rule engine, and the fusion system will be evaluated. They
measure how consistently trained chess players interpret strategic intent from
the same historical board information the model receives.

Human annotations are not automatically objective truth. Inter-annotator
agreement (Cohen's Kappa) measures how consistently two independent reviewers
apply the same rubric. Adjudicated labels represent a negotiated consensus, not
a ground-truth fact about a player's private intention. Report your findings
with this epistemic framing — do not overstate what this benchmark can
establish.

---

## 2. Hard rules

These rules protect the benchmark's scientific validity. Violations invalidate
the affected annotations.

1. **History-only.** For each row you may inspect **only** the information
   provided in the annotation sheet:
   - `fen_current` — the board position at the evaluated ply.
   - `history_moves_san` — the preceding moves (up to 8 half-moves) in
     standard algebraic notation, listed in chronological order (earliest move
     first, most recent move last).
   - `turn` — which player's move it is (the evaluated player; see §4).
   - `game_id` and `ply_index` — for reference only.

   **You must not:**
   - Open the full game to see what happens after the current position.
   - Run chess engine analysis (Stockfish, Leela, etc.) on the position.
   - Look up the game in a database to see the continuation.
   - Access the weak labels, model predictions, or rule-engine scores for any
     row.
   - Access the other annotator's file or discuss any position before both
     sheets are submitted.

2. **Independence.** Each annotator works alone. Do not communicate about any
   position, compare partial results, or coordinate labels in any way until
   both complete sheets are submitted and inter-annotator agreement is
   calculated.

3. **One primary label per row**, drawn exactly from the six taxonomy strings
   in §3. No blanks, no invented class names.

4. **Complete every row.** All 250 positions must have a `primary_label` and a
   `confidence` value. No skipped rows.

---

## 3. The six labels

The authoritative definitions are in
[docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md](06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md).
The table below is a condensed operational reference — when in doubt, consult
`docs/06`.

| Label | Plan | Assign when the board evidence shows… |
| :-- | :-- | :-- |
| `K_STORM` | Kingside Pawn Storm | f/g/h pawns advanced toward the opponent's castled king, with supporting context (heavy-piece backing, multiple recent pawn pushes on the kingside). See `docs/06` §1.2 for the full predicate set. |
| `Q_EXP` | Queenside Expansion | a/b/c pawns systematically advanced to gain queenside space, launch a minority attack, or fix pawn weaknesses. See `docs/06` §2.2. |
| `OUTPOST` | Outpost Occupation | A knight or bishop occupying, or being maneuvered toward, a protected advanced square on files c–f that enemy pawns cannot challenge. See `docs/06` §3.2. |
| `ROOK_FILE` | Rook-File Buildup | Heavy pieces (rooks, queen) doubling on an open or semi-open file, with the aim of penetrating to the 7th/8th rank. See `docs/06` §4.2. |
| `CHAIN_UNDERMINE` | Pawn-Chain Undermining | A pawn lever struck at the base or apex of the opponent's pawn chain to destabilize their structure. See `docs/06` §5.2. |
| `NO_PLAN` | No Dominant Plan | No single plan from the five active classes clearly dominates. This includes: balanced maneuvering, early development without a crystallized direction, tactical skirmishes unrelated to a long-term plan, or two weak/conflicting plan signals where neither achieves dominance. See `docs/06` §6. |

### Distinguishing `NO_PLAN` from ambiguity

- **`NO_PLAN` as primary label**: Use when the board genuinely lacks sufficient
  evidence for any of the five active plans — not because you are uncertain, but
  because the position itself does not commit to a recognizable strategic theme.
- **Genuine tie between two plans**: If two plans are both plausible with
  comparable board evidence, assign the one with marginally stronger evidence as
  `primary_label` and the other as `secondary_label`. Record a lower
  `confidence` value to reflect the ambiguity.
- **Low certainty**: If you can identify a likely plan but feel unsure,
  assign that plan as `primary_label` with a low `confidence` value
  (e.g., 0.3–0.4). Do not use `NO_PLAN` as a substitute for "I'm not sure" —
  `NO_PLAN` means "no plan is evident," not "I cannot decide."

---

## 4. Evaluated player

The **evaluated player** is the player indicated by the `turn` column — the
side to move in the current FEN position.

You are identifying the strategic plan being pursued by this player, as
evidenced by the board state and the preceding move history.

Important clarifications:

- **Side-to-move alone is not evidence of a plan.** The fact that it is White's
  turn does not mean White is executing a plan. Assess the board features and
  recent move history.
- **Attribute plans to the evaluated player only.** If the opponent has a
  visible strategic theme (e.g., the opponent's pawns form a storm pattern),
  that is the opponent's plan. You are labeling the plan of the side to move.
- **History is shared.** The `history_moves_san` column contains moves by both
  players in alternation. Use the full history to understand the position, but
  assign the label based on the evaluated player's strategic activity.
- **Not every visible feature is the evaluated player's doing.** Pawn
  structures, piece placements, and positional features may result from both
  players' moves. Focus on what the evaluated player has been actively pursuing
  in the recent history.

This definition is consistent with the model's prediction target: the model
receives the same 8-ply history (from both players' moves) and predicts the
plan of the side to move at the current position, using attacker-perspective
tensor encoding.

---

## 5. Primary and secondary label rules

- **`primary_label`** — exactly one of the six taxonomy strings from §3.
  **Required for every row.**
- **`secondary_label`** — the next most plausible plan if one exists.
  - Must be a different label from `primary_label` (do not duplicate).
  - If no plausible alternative exists, enter `NO_PLAN`.
  - Use only the six taxonomy strings from §3.

The adjudication script (`src/models/adjudicate.py`) validates `primary_label`
against the six permitted strings and will reject any value not in that set.
`secondary_label` is recorded for analysis but is not used in the adjudication
computation.

---

## 6. Confidence scale

Record your subjective certainty as a number in the `confidence` column.

**Scale: 0.0 to 1.0** (continuous).

| Range | Interpretation |
| :-- | :-- |
| 0.8–1.0 | High confidence — the plan is clearly supported by multiple board features and recent moves. Textbook example. |
| 0.5–0.7 | Moderate confidence — the plan is the most likely interpretation but some ambiguity exists, or the evidence is partial. |
| 0.3–0.5 | Low confidence — you are leaning toward this label but the position is genuinely ambiguous or the evidence is thin. |
| 0.0–0.2 | Very low confidence — essentially a guess; the position offers minimal distinguishing evidence. |

Confidence reflects your subjective certainty about the label assignment. It
is not a calibrated probability and will not be treated as one. It serves two
purposes: (1) flagging positions where the rubric may be ambiguous for
post-annotation analysis, and (2) weighting potential disagreements during
adjudication discussion.

Do not feel pressured to report high confidence. Honest low-confidence
annotations are more valuable than inflated certainty.

---

## 7. Annotation columns

| Column | You fill? | Description |
| :-- | :-- | :-- |
| `window_id` | No | Unique position identifier. Do not edit. |
| `game_id` | No | Source game identifier. Do not edit. |
| `ply_index` | No | Current half-move number. Do not edit. |
| `turn` | No | Evaluated player (`white` or `black`). Do not edit. |
| `fen_current` | No | Board position in FEN notation. Do not edit. |
| `history_moves_san` | No | Preceding moves in SAN, chronological order. Do not edit. |
| `primary_label` | **Yes** | One taxonomy string from §3. **Required.** |
| `confidence` | **Yes** | Your certainty, 0.0–1.0. **Required.** |
| `secondary_label` | **Yes** | Next most plausible plan, or `NO_PLAN` if none. |
| `notes` | **Yes** | One short line of grounded rationale citing specific squares, pieces, or moves (e.g., "Nd5 defended by e4-pawn, no Black c/e pawn to challenge"). No speculation about the opponent's psychology or the game's future. |

---

## 8. Annotation procedure

1. Read this document in full.
2. Read the authoritative taxonomy:
   [docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md](06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md).
3. Open **only your own** annotation sheet:
   - Shahzad Ahmad → `data/annotations/raw_annotations_member1.csv`
   - Muhammad Ali → `data/annotations/raw_annotations_member2.csv`
4. For each of the 250 rows, working top to bottom:
   - Set up the board from `fen_current`. Replay `history_moves_san`
     mentally or on a board to understand how the position was reached.
   - Identify the evaluated player from `turn`.
   - Assess which strategic plan (if any) the evaluated player is pursuing,
     using only the board state and move history.
   - Fill `primary_label`, `confidence`, `secondary_label`, and `notes`.
5. When all 250 rows are complete, submit your sheet. Do not revisit rows
   after submission.

---

## 9. Independence and adjudication

### 9.1 Original independent annotations

Each annotator's submitted CSV is an **immutable original record**:

- Annotator A's responses (`raw_annotations_member1.csv`).
- Annotator B's responses (`raw_annotations_member2.csv`).
- Once submitted, these files are **never modified**. They are the basis for
  calculating raw agreement and Cohen's Kappa.

### 9.2 Inter-annotator agreement

After both sheets are submitted, `src/models/adjudicate.py` computes:

- **Observed agreement** — fraction of positions where both annotators chose
  the same `primary_label`.
- **Cohen's Kappa (κ)** — chance-corrected agreement, using the formula and
  interpretation bands defined in
  [docs/07_INDEPENDENT_EVALUATION_PROTOCOL.md](07_INDEPENDENT_EVALUATION_PROTOCOL.md) §2:
  - κ > 0.75: Excellent agreement.
  - 0.60 ≤ κ ≤ 0.75: Substantial agreement.
  - κ < 0.60: Requires rubric refinement before adjudication.

Agreement statistics are computed **before** any adjudication takes place and
are reported as-is, including all disagreements.

### 9.3 Adjudication

After agreement is computed, disagreements are listed in
`data/annotations/disagreements.csv` (generated by `adjudicate.py`).

Adjudication rules:

- Both annotators meet to review each disagreement, inspecting only the same
  `fen_current` and `history_moves_san` available during annotation.
- **Model predictions, weak labels, and rule-engine scores must not be used to
  settle human disagreements.** The purpose of this benchmark is to provide an
  independent human reference — using model output to adjudicate defeats that
  purpose.
- If consensus is reached, record the agreed label.
- If genuine disagreement persists after discussion, the position must be
  explicitly resolved — `build_ground_truth()` in `adjudicate.py` raises an
  error on unresolved disagreements rather than silently defaulting. Per the
  evaluation protocol (`docs/07` §2), genuinely unresolvable positions are
  adjudicated as `NO_PLAN`.
- Adjudicated labels are stored in a **separate** file
  (`data/annotations/test_ground_truth.csv`), never overwriting the original
  independent annotations.

### 9.4 What the annotations preserve

| Artifact | Contents | Mutable? |
| :-- | :-- | :-- |
| `raw_annotations_member1.csv` | Shahzad's original independent labels | No — frozen after submission |
| `raw_annotations_member2.csv` | Ali's original independent labels | No — frozen after submission |
| `experiments/inter_rater_reliability.json` | Observed agreement, κ, disagreement count | No — computed once from originals |
| `disagreements.csv` | List of positions where annotators disagreed | No — derived from originals |
| `test_ground_truth.csv` | Adjudicated consensus labels | Created after adjudication only |

---

## 10. Sampling and benchmark considerations

The 250 benchmark positions were sampled by `src/data/sample_test_set.py` with
the following properties:

- **Source**: Test-split games only (the same deterministic game-level split
  used for model evaluation, seed=42).
- **Phase stratification**: Balanced across opening (plies 15–25), middlegame
  (plies 26–55), and late middlegame (56+).
- **Per-game cap**: At most 3 positions per game, to limit within-game
  correlation.
- **No class stratification**: Positions were sampled by phase and game
  diversity, not by weak-label class. The class distribution of the 250
  positions reflects the natural test-set distribution, not a balanced design.

Annotators and report authors should be aware of:

- **Within-game correlation**: Multiple positions from the same game share
  context (the same players, opening, and structure). The number of unique
  games and the positions-per-game distribution must be reported alongside
  agreement statistics.
- **Class imbalance**: Some classes (particularly `CHAIN_UNDERMINE`, which
  represents ~0.8% of the test set) may have very few representatives in the
  250-position sample. Report actual class counts in the benchmark, and note
  that per-class agreement statistics on small counts have high sampling
  uncertainty.
- **Sampling is not exhaustive**: 250 positions from a test set of 10,461 is a
  2.4% sample. Results should be interpreted as an estimate, not a census.

---

## 11. Annotator qualifications

Both annotators disclose their chess background in the final report, as
required by the evaluation protocol.

**Shahzad Ahmad** (24I-3090):
- Chess platform/organization: _______________
- Rating type (e.g., Rapid, Blitz, Classical): _______________
- Approximate rating: _______________
- Date of disclosure: _______________
- Years of chess experience: _______________
- Tournament or formal coaching experience: _______________

**Muhammad Ali** (24I-3158):
- Chess platform/organization: _______________
- Rating type (e.g., Rapid, Blitz, Classical): _______________
- Approximate rating: _______________
- Date of disclosure: _______________
- Years of chess experience: _______________
- Tournament or formal coaching experience: _______________

Ratings from different platforms (Lichess, Chess.com, FIDE, national
federations) are not directly comparable. Report the platform and rating type
alongside the number.

---

## 12. Sign-off

By signing below, each annotator confirms they have read and understood:
- This document in full.
- The authoritative taxonomy in `docs/06_CHESS_STRATEGY_TAXONOMY_AND_RUBRIC.md`.
- The independence and history-only constraints in §2.
- The evaluated-player definition in §4.

Both annotators must sign before beginning annotation.

- [ ] **Shahzad Ahmad** (24I-3090) — date: __________
- [ ] **Muhammad Ali** (24I-3158) — date: __________
