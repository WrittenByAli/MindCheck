# MindCheck: History-Aware Chess Plan Recognition Using a Neural Network
## Project Overview & Mission Statement

**Course**: AI2002: Artificial Intelligence | Fall 2026  
**Institution**: National University of Computer and Emerging Sciences (NUCES FAST), School of Computing, Islamabad Campus  
**Instructor**: Dr. Muhammad Bilal  
**Group Number**: Group 29  
**Selected Path**: Path A - Software-Based (Strictly Software, Zero Hardware)  
**Team Members**:
- **Muhammad Ali** (Roll No: `24I-3158`)
- **Shahzad Ahmad** (Roll No: `24I-3090`)

---

## 1. Problem Statement & Real-World Justification
Beginner and intermediate chess players frequently struggle to synthesize individual opponent moves into a coherent, high-level strategic plan. Traditional chess engines (e.g., Stockfish) output numeric evaluation scores (e.g., `+1.8`) and concrete tactical principal variations (PVs), but completely fail to answer pedagogical strategic questions such as:
- *"What positional plan is my opponent orchestrating over the last several moves?"*
- *"Are they building pressure along the c-file, launching a kingside storm, or targeting a weakened outpost?"*

MindCheck solves this problem by framing chess strategy recognition as a **supervised temporal pattern recognition problem**. Given a sequence of recent moves (a sliding history window of 8 half-moves / plies), MindCheck predicts the underlying strategic plan, tracks plan progression across time, detects the onset and shift of plans, and renders grounded symbolic evidence inside an intuitive, interactive Streamlit application.

### Non-Goals & Integrity Boundaries
1. MindCheck **does not** claim to read a player's mind or determine verified psychological intent; it produces an empirical strategic hypothesis based on move evidence.
2. MindCheck is designed strictly for **post-game review, analysis, and pedagogical training**, never for live competitive play assistance.
3. MindCheck **does not** replace a trained neural network with static rules or engine search; the neural model is the primary inference engine.

---

## 2. Research Questions
1. **Primary Research Question**: Does incorporating recent move history (temporal board states) measurably improve the recognition of strategic plans compared to evaluating only the static current board?
2. **Secondary Research Question**: Does combining neural softmax predictions with symbolic rule-based predicates (hybrid fusion) and temporal state hysteresis improve prediction stability and reduce erratic plan flickering across consecutive plies?

---

## 3. Five-Plan Strategy Taxonomy (+1 Null Class)
MindCheck defines six mutually exclusive target classes:
1. **Kingside Pawn Storm (`K_STORM`)**: Coordinated advancement of g- and/or h-pawns (and f-pawn) directed toward the opponent's castled king position.
2. **Queenside Expansion (`Q_EXP`)**: Coordinated advancement of a-, b-, and c-pawns to secure spatial dominance, open flank files, or create queenside passers.
3. **Outpost Occupation (`OUTPOST`)**: Maneuvering a minor piece (knight or bishop) onto a protected, hole-in-the-camp square that cannot be dislodged by opponent pawns.
4. **Rook-File Buildup (`ROOK_FILE`)**: Doubling or tripling major pieces (rooks and queen) along an open or semi-open file to achieve deep penetration.
5. **Pawn-Chain Undermining (`CHAIN_UNDERMINE`)**: Tactical or strategic pawn levers attacking the base or apex of an established pawn chain (e.g., `...c5` attacking `d4`).
6. **No Dominant Plan (`NO_PLAN`)**: Mixed, ambiguous, transitional, or purely tactical positions lacking a distinct strategic focus among the five defined categories.

---

## 4. Key Architectural Pillars
```
[ PGN Move Stream / Lichess Elite Dataset ]
                   │
                   ▼
  [ Memory-Efficient Streaming Parser & Validator ]
                   │
                   ▼
  [ 8-Ply Sliding Window & Canonical Tensor Encoding ] (12x8x8 planes * 8 + metadata)
                   │
                   ▼
  [ Deep Residual Convolutional Neural Network (PyTorch) ]
                   │ (Softmax Probabilities)
                   ▼
  [ Temporal Plan State Tracker (Hysteresis & Onset) ]
                   │
                   ▼
  [ Symbolic Fact Extractor & Evidence Fusion ]
                   │
                   ▼
  [ Interactive Streamlit Web App (Board SVG, Timeline, Explanations) ]
```

---

## 5. Course Learning Outcomes (CLO) Alignment
- **CLO 1 (Problem Formulation & Justification)**: Formulate the strategic plan recognition problem from raw chess move sequences into high-dimensional spatial-temporal tensors and justify the selection of Deep Residual CNNs over conventional methods.
- **CLO 2 (Algorithm Implementation & Pipeline Engineering)**: Group implementation from scratch in PyTorch (Residual CNN, tensor stacking, bitboard fact extractors, temporal hysteresis tracker, Streamlit UI) without copying existing repositories.
- **CLO 3 (Evaluation, Testing & Scientific Communication)**: Rigorous evaluation on held-out test sets, independent double-blind student annotation (Cohen's Kappa), confusion matrices, history-length ablations, comprehensive project report, slides, and live Streamlit demonstration.
