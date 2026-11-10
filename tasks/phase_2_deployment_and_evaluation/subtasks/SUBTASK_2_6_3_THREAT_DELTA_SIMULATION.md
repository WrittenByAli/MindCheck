# Subtask 2.6.3: Threat Delta Simulation (Proactive vs Passive)

**Subtask ID**: `2.6.3.THREAT.DELTA.SIMULATION`  
**Parent Task**: `Task 2.6`  
**Phase**: Phase 2  
**Milestone**: Milestone 13  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Compute evaluation delta between proactive defensive moves and ignoring the opponent's plan.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: Current position and candidate moves.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: Threat delta numerical value $\Delta_{\text{threat}}$ and interpretation note.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Formula:
   $$\Delta_{\text{threat}} = \text{Eval}(\text{Proactive Defense}) - \text{Eval}(\text{Passive Move})$$
2. Illustrates positional consequence of allowing opponent to execute their plan unimpeded.

---

## 5. Claude Code Action Plan
1. Implement threat delta calculation in `src/utils/stockfish_client.py`.
2. Integrate delta display in Streamlit engine tab.
3. Verify proper centipawn formatting (+1.5, -0.8).

---

## 6. Unit Testing & Acceptance Criteria
- Accurately reports engine evaluation difference.
- Helps learners appreciate the severity of positional threats.
