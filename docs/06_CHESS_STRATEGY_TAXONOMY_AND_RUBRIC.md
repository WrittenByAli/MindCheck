# MindCheck: Chess Strategy Taxonomy & Annotation Rubric

This document defines the formal operational definitions, bitboard predicates, trigger criteria, and human annotation rubric for the **5 strategic plan classes** and the **null class** recognized by MindCheck.

---

## 1. Class 0: Kingside Pawn Storm (`K_STORM`)

### 1.1 Strategic Definition
A coordinated tactical or positional advance of flank pawns on the kingside (primarily g- and h-pawns, optionally accompanied by the f-pawn) directed against the opponent's castled king position, intended to open files, breach pawn shelters, or force decisive weaknesses.

### 1.2 Operational Bitboard Predicates
- **Target Context**: Opponent King is castled on the kingside (squares g8/h8/f8 for Black, g1/h1/f1 for White).
- **Advancement Trigger**: Friendly g-pawn $\ge$ 4th rank (for White) / $\le$ 5th rank (for Black), or friendly h-pawn $\ge$ 4th rank (for White) / $\le$ 5th rank (for Black).
- **Supporting Context**: Recent moves include at least 2 pawn advances on the f-, g-, or h-files within the 8-ply window, or a major piece behind the advancing flank pawns (e.g. Rook on g1/h1 or g8/h8).
- **Invalidation Condition**: The kingside pawns are traded off without opening files, or friendly king is also exposed on the kingside without adequate defense.

---

## 2. Class 1: Queenside Expansion (`Q_EXP`)

### 2.1 Strategic Definition
A systematic pawn advance on the queenside (a-, b-, and c-pawns) aimed at gaining space, rolling over opponent pieces, fixing queenside pawn weaknesses, or opening flank files (e.g., Minority Attack in the Carlsbad structure, Sicilian queenside pushes).

### 2.2 Operational Bitboard Predicates
- **Territory / Space Metric**: Friendly pawns controlling or occupying squares on files a, b, c past the 4th rank (White) or past the 5th rank (Black).
- **Move Sequence**: At least 2 queenside pawn pushes (e.g., `a3` followed by `b4`, or `b5` followed by `a5`) in the 8-ply history.
- **Piece Alignment**: Minor pieces or rooks supporting queenside pawn anchors (e.g. Knight on c3/c6, Rook on b1/b8).
- **Invalidation Condition**: Opponent executes a central breakthrough that completely diverts play, or queenside pawns become overextended and statically captured.

---

## 3. Class 2: Outpost Occupation (`OUTPOST`)

### 3.1 Strategic Definition
The identification, maneuver toward, and eventual occupation of an advanced "hole" or outpost square in the enemy camp (typically on the 4th, 5th, or 6th rank for White; 5th, 4th, or 3rd for Black) that cannot be attacked or driven away by enemy pawns, supported by a friendly pawn.

### 3.2 Operational Bitboard Predicates
- **Outpost Hole Identification**: A square $S$ on files c, d, e, or f where opponent pawns cannot attack $S$ (neither currently nor by advancing, i.e., opponent pawns on adjacent files are either traded or past $S$).
- **Friendly Pawn Support**: Square $S$ is defended by at least one friendly pawn.
- **Piece Maneuver**: A friendly knight or bishop moves onto $S$, or moves toward an intermediate square whose subsequent legal destination is $S$ within the 8-ply window.
- **Invalidation Condition**: The outpost piece is immediately exchanged for an equal or greater value opponent piece, leaving no permanent positional control.

---

## 4. Class 3: Rook-File Buildup (`ROOK_FILE`)

### 4.1 Strategic Definition
The strategic concentration of heavy artillery along an open file (no pawns of either color) or semi-open file (no friendly pawns, only enemy pawns), involving rook doubling, tripling (Alekhine's Gun with Queen and Rooks), or penetrating to the 7th/8th rank.

### 4.2 Operational Bitboard Predicates
- **File Status**: File $F \in \{a, b, c, d, e, f, g, h\}$ has zero friendly pawns.
- **Heavy Piece Alignment**: At least two heavy pieces (Rook + Rook, or Rook + Queen) occupy file $F$ simultaneously, or a rook moves to file $F$ followed by a second heavy piece entering the file within the 8-ply window.
- **Invasion Threat**: Rooks on $F$ control key entry squares on the 7th or 8th rank (2nd or 1st rank for Black).
- **Invalidation Condition**: Opponent completely blocks the file with a permanent pawn or heavily protected piece, neutralizing penetration.

---

## 5. Class 4: Pawn-Chain Undermining (`CHAIN_UNDERMINE`)

### 5.1 Strategic Definition
A thematic pawn strike (lever) directed against the base, apex, or supporting links of a locked or semi-locked opponent pawn chain to destabilize the opponent's spatial foundation (e.g., French Defense `...c5` or `...f6` attacking White's `e5-d4` chain).

### 5.2 Operational Bitboard Predicates
- **Chain Existence**: Opponent possesses a diagonal chain of $\ge 2$ pawns mutually defending each other (e.g., Black pawns on e6-d5-c4, or White pawns on c3-d4-e5).
- **Pawn Lever Move**: Friendly pawn moves to a square adjacent to the base or apex pawn of the chain, creating an immediate pawn-tension / capture threat within the 8-ply history.
- **Context**: The pawn push is made not merely to gain tempo, but to force pawn trades that dismantle the enemy structure.
- **Invalidation Condition**: The tension is resolved without altering the structure (e.g., bypassed with a closed push) and no further pressure is applied.

---

## 6. Class 5: No Dominant Plan (`NO_PLAN`)

### 6.1 Definition & Inclusion Criteria
Assigned when the position exhibits:
- Early-game uncommitted opening moves (development of knights and bishops without a crystallized strategic direction).
- Short-term tactical skirmishes (forced checks, piece trades, hanging piece recaptures) that do not belong to a deliberate long-term plan.
- Balanced, quiet maneuvering where neither side has committed to a specific strategic theme.
- Conflicting or mutually canceling evidence between two different plans where neither achieves dominance.

---

## 7. Human Annotation Protocol (Double-Blind Benchmark)

To build the 200–300 position independent ground truth test set:
1. **Double-Blind Scoring**: Both **Muhammad Ali** and **Shahzad Ahmad** will inspect each sampled 8-ply move window independently without communicating.
2. **History-Only Constraint**: Annotators inspect **only** the board history up to ply $t$. They MUST NOT look at future plies (to avoid outcome bias) and MUST NOT look at model or rule predictions.
3. **Recorded Fields**:
   - `window_id`: Unique identifier
   - `game_id`: PGN game source
   - `ply_index`: Current ply
   - `primary_label`: Dominant plan (one of the 6 classes)
   - `confidence`: High (1.0), Medium (0.7), Low (0.4)
   - `secondary_label`: Secondary plausible plan (if applicable)
   - `supporting_moves`: Key moves in the 8-ply window that justify the choice
4. **Adjudication**: Following individual completion, inter-rater reliability ($\text{Cohen's } \kappa$) is calculated. Any disputed positions are discussed and adjudicated into a consensus label.
