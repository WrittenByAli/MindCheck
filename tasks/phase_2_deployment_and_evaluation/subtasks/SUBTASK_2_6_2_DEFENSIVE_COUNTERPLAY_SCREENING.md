# Subtask 2.6.2: Defender Counter-Move Screening & Plan Disruption

**Subtask ID**: `2.6.2.DEFENSIVE.COUNTERPLAY.SCREENING`  
**Parent Task**: `Task 2.6`  
**Phase**: Phase 2  
**Milestone**: Milestone 13  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Screen defender candidate moves to identify counter-moves that disrupt the opponent's detected plan.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.Board` and detected opponent plan.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: List of `EngineCandidateMove` with strategic disruption notes.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Method `screen_counterplay(board, detected_plan)`:
   - Identify candidate defender moves.
   - Check if move attacks opponent's outpost piece, blocks open file, or counter-attacks center.
   - Return top 3 moves with centipawn evaluation and strategic annotation.

---

## 5. Claude Code Action Plan
1. Implement defensive screening in `src/utils/stockfish_client.py`.
2. Test on tactical positions with obvious defensive replies.
3. Display suggestions in Streamlit Tab 3.

---

## 6. Unit Testing & Acceptance Criteria
- Produces sensible defensive recommendations.
- Explicitly disclaims engine output as study suggestions.
