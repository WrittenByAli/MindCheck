# Subtask 2.3.3: Strict Structured Claim Schema & Validation Filter

**Subtask ID**: `2.3.3.STRUCTURED.CLAIM.VERIFIER`  
**Parent Task**: `Task 2.3`  
**Phase**: Phase 2  
**Milestone**: Milestone 11  
**Primary Owner**: **Muhammad Ali (24I-3158)**  
**Collaborator**: **Shahzad Ahmad (24I-3090)**  

---

## 1. Context & Objective
Verify that every square and piece claim made in an explanation actually exists on the board.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `chess.Board` and generated `ExplanationCard`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `verified_claim_status: bool` and safe fallback if verification fails.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Claim Verifier:
   - Parse all mentioned squares (e.g. 'd5', 'e4') from explanation text.
   - Assert that claimed piece is currently on that square or visited it in the 8-ply window.
   - If any claim fails verification: discard card and fall back to safe generic summary:
     'Model predicts [Plan] with [X]% confidence based on recent move history.'

---

## 5. Claude Code Action Plan
1. Implement verification logic in `src/tracking/explanations.py`.
2. Write unit tests injecting intentional corrupted square claims; verify fallback triggers.
3. Test with genuine board states; verify 100% pass rate.

---

## 6. Unit Testing & Acceptance Criteria
- Zero hallucination: no non-existent pieces or squares are presented as facts.
- Automatic fallback on unexpected board states.
