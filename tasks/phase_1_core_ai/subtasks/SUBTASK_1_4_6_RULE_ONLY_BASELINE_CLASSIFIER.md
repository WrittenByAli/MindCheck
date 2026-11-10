# Subtask 1.4.6: Rule-Only Baseline Classifier & Ambiguity Resolver

**Subtask ID**: `1.4.6.RULE.ONLY.BASELINE.CLASSIFIER`  
**Parent Task**: `Task 1.4`  
**Phase**: Phase 1  
**Milestone**: Milestone 4  
**Primary Owner**: **Shahzad Ahmad (24I-3090)**  
**Collaborator**: **Muhammad Ali (24I-3158)**  

---

## 1. Context & Objective
Combine all 5 rule templates into a unified Rule-Only Classifier (Baseline M3) with ambiguity resolution.

---

## 2. Input Requirements
- **Prerequisites / Input Files**: `PositionWindow`.

---

## 3. Output Artifacts & Deliverables
- **Target Files & Formats**: `RuleEvaluationResult` containing predicted class, confidence, all scores, and evidence list.

---

## 4. Detailed Technical Specification
### Technical Implementation:
1. Class `SymbolicRuleEngine`:
   - Compute scores: $[S_0, S_1, S_2, S_3, S_4]$.
   - Let $S_{\max} = \max_k S_k$ and $S_{\text{second}} = \text{second\_highest}(S)$.
   - Decision Logic:
     - If $S_{\max} < 0.50$: assign `NO_PLAN` (Class 5).
     - If $S_{\max} - S_{\text{second}} < 0.15$: assign `NO_PLAN` (Ambiguous conflict).
     - Else: assign class $k = \arg\max_k S_k$.
2. Formats supporting evidence list for pedagogical display.

---

## 5. Claude Code Action Plan
1. Implement `SymbolicRuleEngine.evaluate_window` in `src/rules/rule_engine.py`.
2. Write `tests/test_rule_engine.py` verifying multi-class resolution and ambiguity filtering.
3. Verify deterministic output across repeat evaluations.

---

## 6. Unit Testing & Acceptance Criteria
- Predicts dominant class when evidence is strong.
- Gracefully falls back to `NO_PLAN` when scores are low or conflicting.
