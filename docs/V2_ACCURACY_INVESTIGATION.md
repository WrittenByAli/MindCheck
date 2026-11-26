# v2 Accuracy Investigation

Is v2 recognizing chess strategies, or mainly predicting the dominant class and benefiting from
bias adjustment?

**All scores are against rule-engine weak labels, not human judgement.** Nothing was trained,
relabelled or tuned on test data for this investigation. Models: v2 = `G_allpos_uniform` (seed 42
unless stated), v1 = `D_random_balance` (seed 999). Machine-readable results:
`experiments/v2/accuracy_investigation.json`. Reproduce with
`.venv\Scripts\python.exe scripts/investigate_v2.py --work-dir <dir>`.

The second holdout ("unseen test", 6,000 games) was already used for the v2 decision. Reading it
again here is diagnostic. **It can no longer serve as an untouched test for any new model.**

## 1. Executive summary

- **v2 does learn to tell strategies apart.** On positions that have a strategy, choosing among the
  five plans, it is right 81.8% of the time against 40.9% for always picking the commonest plan.
- **Raw v2 does not mainly predict "no plan"; it does the opposite.** The truth is 58.8% no-plan,
  raw v2 predicts it 25.2% of the time. That is why raw accuracy (55.5%) is *below* the
  always-no-plan baseline (58.8%).
- **The jump to 72.1% is almost all no-plan.** The bias adds 21.8 accuracy points on no-plan
  positions and removes 5.1 points on strategy positions. Balanced accuracy *falls* from 69.6% to
  60.1%. Macro-F1 rises from 0.488 to 0.583.
- **72.1% is not contaminated.** The bias is fitted on validation only and was reproduced exactly.
- **The most important finding:** a rule engine that sees only past moves scores 69.7% accuracy,
  0.556 macro-F1 and 60.3% balanced accuracy. Calibrated v2 beats it by only about 2 accuracy
  points and 2 to 3 macro-F1 points, with no reliable difference in balanced accuracy.
- **Quality: PROMISING**, not GOOD (section 11).

## 2. Dataset distribution

| Population | Positions | NO_PLAN | OUTPOST | Q_EXP | K_STORM | ROOK_FILE | CHAIN |
|---|---|---|---|---|---|---|---|
| Train, natural (unfiltered) | 1,645,332 | 58.6% | 16.0% | 12.2% | 6.6% | 5.7% | 0.87% |
| Train, rows v2 trained on (balanced) | 973,040 | 30.0% | 27.1% | 20.6% | 11.2% | 9.6% | 1.46% |
| Validation (unfiltered) | 294,774 | 58.6% | 16.1% | 12.0% | 6.7% | 5.8% | 0.94% |
| Unseen test, every position | 463,408 | 58.8% | 16.9% | 11.6% | 6.6% | 5.4% | 0.69% |
| Unseen test, filtered | 147,164 | 44.2% | 27.8% | 13.4% | 6.2% | 8.0% | 0.58% |

Correction: earlier documents gave v2's balanced training size as 968,901. That is v1's number.
v2 trained on **973,040** rows.

**The "59% no-plan baseline"** is simply the share of NO_PLAN among **unfiltered** positions
(validation 58.6%, unseen test 58.8%; the 59.3% quoted earlier was the first holdout). It does not
apply to filtered positions (44.2%) or to the balanced training rows (30.0%).

## 3. Baselines

Trivial baselines are built from training labels only. For every population, the majority class of
natural training data, "always NO_PLAN" and the most frequent class of the balanced training rows
are all NO_PLAN, so they are one row. Stratified random is the exact expected value of guessing
with natural training frequencies.

Accuracy / balanced accuracy / macro-F1:

| Predictor | Validation | Unseen, every position | Unseen, filtered |
|---|---|---|---|
| Always NO_PLAN (majority) | 58.6% / 16.7% / 0.123 | 58.8% / 16.7% / 0.123 | 44.2% / 16.7% / 0.102 |
| Stratified random | 39.1% / 16.7% / 0.167 | 39.3% / 16.7% / 0.167 | 29.9% / 16.7% / 0.167 |
| **Rule engine, past moves only** | 69.2% / 60.3% / 0.560 | **69.7% / 60.3% / 0.556** | **79.2% / 69.2% / 0.668** |
| v2 raw | 54.0% / 69.0% / 0.486 | 55.5% / 69.6% / 0.488 | 71.3% / 75.6% / 0.611 |
| v2 bias-adjusted | 71.5% / 59.5% / 0.581 | **72.1% / 60.1% / 0.583** | **83.5% / 69.6% / 0.705** |

On the balanced training rows, always-NO_PLAN scores 30.0% and v2 raw 68.8%. On natural training
data they score 58.6% and 55.9%.

The rule engine is a strong baseline with a home advantage: the labels come from the same engine
applied to later moves. It also sees 8 positions of history, where v2 sees 4. This baseline was
planned in the project documents and had never been run.

### v2 against the rule engine (95% game-bootstrap interval of the difference, points)

| Unseen test | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| Every position, v2 raw (seed 42) | −14.6 to −13.8 | −7.1 to −6.5 | **+8.8 to +9.8** |
| Every position, v2 adjusted (seed 42) | +2.2 to +2.6 | +2.4 to +3.1 | −0.7 to +0.2 |
| Every position, adjusted, all 3 seeds | +1.7 to +2.7 | +2.0 to +3.1 | −1.4 to +1.0 |
| Filtered, adjusted, all 3 seeds | +4.0 to +4.8 | +2.7 to +4.3 | −1.5 to +2.0 |

## 4. Classification metrics (v2, seed 42)

| Population | Output | Acc | Balanced acc | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|---|
| Train (trained rows) | raw | 68.8% | 74.7% | 0.618 | 0.747 | 0.634 | 0.682 |
| Train (trained rows) | adjusted | 71.0% | 64.0% | 0.719 | 0.640 | 0.669 | 0.714 |
| Train (natural) | raw | 55.9% | 74.6% | 0.481 | 0.746 | 0.509 | 0.567 |
| Train (natural) | adjusted | 72.9% | 63.9% | 0.602 | 0.639 | 0.617 | 0.731 |
| Validation | raw | 54.0% | 69.0% | 0.463 | 0.690 | 0.486 | 0.550 |
| Validation | adjusted | 71.5% | 59.5% | 0.572 | 0.595 | 0.581 | 0.717 |
| Unseen, every position | raw | 55.5% | 69.6% | 0.463 | 0.696 | 0.488 | 0.567 |
| Unseen, every position | adjusted | 72.1% | 60.1% | 0.572 | 0.601 | 0.583 | 0.723 |
| Unseen, filtered | raw | 71.3% | 75.6% | 0.596 | 0.756 | 0.611 | 0.734 |
| Unseen, filtered | adjusted | 83.5% | 69.6% | 0.728 | 0.696 | 0.705 | 0.835 |

(The 56.5% "training accuracy" reported earlier was the trained rows **resampled to validation's
class mix**. On their own balanced mix it is 68.8%.)

Across seeds 42, 123 and 999, adjusted macro-F1 on every unseen position is 0.583, 0.580 and
0.580. Per-class F1 varies by at most 0.014.

### Per class, unseen test (precision / recall / F1)

| Class | Every position: rule engine | Every position: v2 adjusted | Filtered: v2 adjusted | Support (every) |
|---|---|---|---|---|
| NO_PLAN | .79 / .73 / .76 | .80 / .76 / **.78** | .82 / .90 / .86 | 272,375 |
| OUTPOST | .71 / .78 / .74 | .69 / .80 / **.74** | .90 / .87 / .89 | 78,179 |
| Q_EXP | .58 / .58 / .58 | .61 / .61 / .61 | .83 / .73 / .78 | 53,940 |
| ROOK_FILE | .65 / .54 / **.59** | .60 / .56 / .58 | .87 / .67 / .76 | 25,036 |
| K_STORM | .48 / .55 / .51 | .54 / .56 / .55 | .78 / .69 / .73 | 30,676 |
| CHAIN_UNDERMINE | .10 / .44 / .16 | .20 / .31 / .24 | .18 / .31 / .23 | 3,202 |

On every position, **v2 does not beat the rule engine on OUTPOST** (equal) or **ROOK_FILE** (the
rule engine is slightly better). Its gains are on K_STORM, Q_EXP, NO_PLAN and CHAIN_UNDERMINE.

## 5. Confusion matrices

See the appendix at the end of this file. All other matrices (every population, every seed, raw
and adjusted) are in the JSON results.

## 6. NO_PLAN analysis

| Population, output | True share | Predicted share | Precision | Recall | Strategy called no-plan | No-plan given a strategy |
|---|---|---|---|---|---|---|
| Validation, raw | 58.6% | 24.0% | .90 | .37 | 6.1% | 63.3% |
| Validation, adjusted | 58.6% | 55.6% | .79 | .75 | 27.7% | 24.6% |
| Unseen every, raw | 58.8% | 25.2% | .90 | .39 | 5.9% | 61.2% |
| Unseen every, adjusted | 58.8% | 55.7% | .80 | .76 | 27.0% | 24.2% |
| Unseen filtered, raw | 44.2% | 24.8% | .95 | .53 | 2.3% | 46.7% |
| Unseen filtered, adjusted | 44.2% | 48.8% | .82 | .90 | 16.0% | 9.8% |

- **Raw v2 strongly under-predicts no-plan** and invents a plan for 61% of no-plan positions. This
  follows from training: balanced sampling *plus* inverse-frequency loss weights make the model
  act as if all six classes were equally common.
- **Adjusted v2 predicts no-plan about as often as it occurs** (on every position), at the price of
  calling 27% of real strategy positions "no plan".
- It is not relying on the dominant class to tell strategies apart (section 8).

## 7. Bias adjustment audit

| Question | Finding |
|---|---|
| What is applied? | One additive constant per class on the logits: K_STORM −0.54, Q_EXP −0.04, OUTPOST +0.46, ROOK_FILE −0.29, CHAIN −1.04, NO_PLAN +1.46 |
| How is it calculated? | Coordinate search (grid −3 to 3, step 0.25, 3 passes) maximizing **macro-F1**, then mean-centred |
| Learned from training data only? | **No.** From validation logits and validation labels |
| Uses test labels, distribution or games? | **No.** Code opens only validation; refitting from validation reproduces the stored vector exactly (difference 0.0); no validation game is in the unseen test; the bias file predates the test build |
| Contaminated? | **No.** |
| Caveats | The same validation set chose the variant, seed and epoch. It assumes deployment has validation's class mix. The later *recommendation* to prefer v2 did use test results |

### Controlled comparison (unseen test, every position)

| Adjustment (source) | Acc | Balanced acc | Macro-F1 | CHAIN recall | NO_PLAN recall |
|---|---|---|---|---|---|
| None | 55.5% | 69.6% | 0.488 | 63.1% | 38.8% |
| **Validation fit (shipped)** | 72.1% | 60.1% | **0.583** | 31.1% | 75.8% |
| Training labels only (class-prior correction) | **73.2%** | 52.6% | 0.539 | **0.2%** | 80.6% |
| Fit on filtered validation | 71.0% | 61.8% | 0.581 | 30.9% | 71.1% |

The training-only correction gives the **highest accuracy while essentially deleting
CHAIN_UNDERMINE** (recall 0.2%). This is the clearest demonstration that accuracy alone is
misleading here.

Where the +16.6 accuracy points of the shipped bias come from (change by true class): NO_PLAN
**+21.8**, Q_EXP −1.7, K_STORM −1.6, ROOK_FILE −1.1, OUTPOST −0.5, CHAIN −0.2.

It improves macro-F1 (+9.5 points) and **reduces balanced accuracy** (−9.5 points). On filtered
positions the pattern is the same: accuracy 71.3% → 83.5%, macro-F1 0.611 → 0.705, balanced
accuracy 75.6% → 69.6%.

## 8. Strategy recognition by subset (unseen test)

Subsets use only definitions that already existed: the dataset filter (confidence ≥ 0.55,
material limit, top 25 per game) and the labels. For B and C, averages run over the five
strategies. "Forced choice" picks the best of the five plan logits, ignoring NO_PLAN.

Accuracy / balanced accuracy / macro-F1:

| Predictor | A: all (463,408) | B: has a strategy (191,033) | C: clear strategy (82,148) |
|---|---|---|---|
| v2 raw | 55.5 / 69.6 / .488 | 79.4 / 75.8 / .728 | 85.6 / 80.1 / .749 |
| v2 adjusted | 72.1 / 60.1 / .583 | 67.0 / 57.0 / .675 | 78.3 / 65.5 / .731 |
| v1, same unfiltered-validation calibration | 70.0 / 58.7 / .556 | 67.1 / 56.1 / .648 | 79.7 / 67.2 / .727 |
| Rule engine, past only | 69.7 / 60.3 / .556 | 64.7 / 57.8 / .650 | 73.4 / 65.7 / .703 |
| Majority strategy (OUTPOST) | n/a | 40.9 / 20.0 / .116 | 49.8 / 20.0 / .133 |
| **v2 forced choice** | n/a | **81.8 / 79.0 / .719** | **86.5 / 81.7 / .744** |
| v1 forced choice | n/a | 76.8 / 76.6 / .673 | 85.0 / 82.1 / .722 |

| Predictor | D1: labelled NO_PLAN (272,375), recall | D2: confidence < 0.55 (82,033): acc / balanced / F1 |
|---|---|---|
| v2 raw | 38.8% | 20.5 / 54.0 / .219 |
| v2 adjusted | 75.8% | 42.3 / 37.6 / .274 |
| v1, unfiltered calibration | 72.1% | 32.9 / 32.0 / .212 |
| Rule engine, past only | 73.2% | **46.3 / 41.6 / .280** |

Per-class recall on subset B, v2 adjusted: K_STORM .56, Q_EXP .61, OUTPOST .80, ROOK_FILE .56,
CHAIN .31 (raw: .81, .76, .83, .76, .63). Per-class recall for every row is in the JSON.

- **When a strategy is present, v2 identifies which one well** (forced choice 81.8%).
- **Deciding whether any strategy is present is the weak part.**
- On low-confidence positions, which sit near the label's own threshold, **every predictor is
  poor, and the rule engine is best.**

## 9. CHAIN_UNDERMINE

**Examples:** 14,234 trained rows (1.46% of balanced training), 2,775 validation (0.94%), 3,202
unseen test (0.69%).

**Unseen test:**

| Output | Predicted | Correct | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Raw | 28,926 | 2,019 | .07 | .63 | .126 |
| Adjusted | 5,070 | 997 | .20 | .31 | .241 |

- **False positives (adjusted):** 4,073, of which 3,234 (79%) are NO_PLAN and 551 are Q_EXP.
- **False negatives (adjusted):** 2,205, of which 1,731 (79%) go to NO_PLAN, then Q_EXP 173,
  K_STORM 161. Raw misses go mostly to K_STORM (384) and Q_EXP (299).
- **Game phase (adjusted F1, validation / test):** opening .29 / .30, middlegame .15 / .16, late
  .15 / .13.
- **Not driven by a few games:** CHAIN positions come from 784 of 6,000 test games; the top 10 hold
  3.6%. Removing the 50 richest games changes F1 from .241 to .223 (validation .242 to .230).
- **Train vs validation recall gap:** 15 to 20 points across seeds (target was 10).
- **16 example errors** (FEN, ply, scores) are in the JSON under
  `chain_undermine.val_examples`.

**Likely causes, with evidence (validation; test agrees):**

1. **Much of the label is not visible in the past.** For **44%** of CHAIN-labelled validation
   positions (47% on test), the rule engine's past-only CHAIN score is exactly zero: the label
   comes entirely from later moves. v2 adjusted recall is **42% when there is past evidence and
   12% when there is none** (test: 48% and 13%). "Visible" is judged on the rule engine's 8-board
   window and v2 sees 4, so v2's real view is narrower still.
2. **It overlaps other pawn-push plans.** The runner-up plan in the label is Q_EXP (52%) or
   K_STORM (31%).
3. **It is very rare** (under 1%), so even a small false-alarm rate swamps the true positives.
4. **More examples did not help.** Round 1 variant F had about twice the CHAIN training rows of G
   (30,464 against 14,234) and the same CHAIN F1 (.245 against .242), with a larger overfitting gap
   (30 against 20 points).

Whether humans would find these labels ambiguous is **unknown** until annotation.

**This is not unique to CHAIN.** The past-only engine matches the look-ahead label for only 54% of
K_STORM, 58% of Q_EXP, 54% of ROOK_FILE, 77% of OUTPOST and 73% of NO_PLAN positions (46% for
CHAIN). v2 raw recovers more than that (76% to 83% for the first four), so disagreement does not
prove a position is unpredictable.

## 10. v1 against v2 on identical positions

### Every unseen position

| Model, calibration | Acc | Balanced | Macro-F1 | NO_PLAN P / R | CHAIN F1 |
|---|---|---|---|---|---|
| v1 raw | 59.5% | 67.6% | 0.504 | .87 / .49 | .118 |
| v2 raw | 55.5% | 69.6% | 0.488 | .90 / .39 | .126 |
| v1, unfiltered-validation bias | 70.0% | 58.7% | 0.556 | .81 / .72 | .179 |
| **v2, unfiltered-validation bias** | **72.1%** | **60.1%** | **0.583** | .80 / .76 | .241 |
| v1, filtered-validation bias | 68.8% | 61.0% | 0.554 | .82 / .68 | .188 |
| **v2, filtered-validation bias** | **71.0%** | **61.8%** | **0.581** | .82 / .71 | .242 |

v2 minus v1 (95% interval, points): with unfiltered bias, accuracy +2.0 to +2.3, macro-F1 +2.4 to
+3.0, balanced +1.0 to +1.8. With filtered bias, +2.0 to +2.4, +2.4 to +3.0 and +0.5 to +1.2.
**v2's advantage holds under both calibrations, so it is not only calibration.** Raw outputs sit
at different operating points: v2 raw is lower in accuracy and macro-F1, higher in balanced
accuracy.

### Filtered unseen positions

| Model, calibration | Acc | Balanced | Macro-F1 |
|---|---|---|---|
| v1 raw | 78.5% | 77.6% | 0.663 |
| v2 raw | 71.3% | 75.6% | 0.611 |
| v1, filtered-validation bias | **85.3%** | **73.0%** | **0.722** |
| v2, filtered-validation bias | 84.1% | 71.2% | 0.711 |

**v1 is better here under every calibration** (same filtered bias: v2 minus v1 accuracy −1.4 to
−1.0, macro-F1 −1.6 to −0.7).

### Strategy positions

At the same calibration, v2 and v1 get the **same strategy accuracy** on every strategy position
(67.0% against 67.1%), with v2 ahead on strategy macro-F1 (.675 against .648). v2's extra overall
accuracy comes mainly from **fewer false plans on no-plan positions** (recall 75.8% against 72.1%).
In forced choice v2 separates plans better (81.8% against 76.8%). On clear strategy positions v1
is slightly ahead in calibrated accuracy.

v1's high filtered training accuracy is not evidence that v1 is better; the comparisons above use
identical positions.

## 11. Model quality: PROMISING

| Criterion | Evidence |
|---|---|
| Beats trivial baselines | Yes, widely, on balanced accuracy and macro-F1; adjusted accuracy by 13 points |
| Learns individual strategies | Yes: forced choice 81.8% against 40.9% |
| Stable across seeds and unseen games | Yes: macro-F1 0.580 to 0.583; test matches validation |
| Clearly beats a non-neural baseline | **Only slightly**: about +2 accuracy and +2 to +3 macro-F1 over the past-only rule engine, none in balanced accuracy; not ahead on OUTPOST or ROOK_FILE |
| Reliable across important classes | **No**: on every position only NO_PLAN and OUTPOST reach F1 0.74 or more; CHAIN fails |

Not WEAK, because it clearly separates strategies. Not GOOD or STRONG, because of the small margin
over the rule engine and the class weaknesses.

## 12. Recommendations (the smallest justified set)

1. **Human annotation first.** Score v2, v1 and the past-only rule engine on the 227 clean
   windows. It is the only test of whether the labels, and therefore these margins, mean anything.
2. **Always report the rule engine beside the neural model**, with macro-F1 and balanced accuracy.
   Do not quote accuracy alone.
3. **Choose the app's operating point deliberately** (on validation): the shipped bias favours
   accuracy and macro-F1; raw favours catching strategies, at the cost of many false plans.
4. **One capacity check, on validation only:** a larger network, or 8-position input, with the
   same data. Capacity has not been tested upward, so architecture cannot yet be ruled out.
5. **Later, in Phase 2 fusion (task 2.2), tune any neural-plus-rule combination on validation.**
   The rule engine is competitive, so fusion is worth measuring; its benefit is untested.

**Not justified by the evidence:** more CHAIN oversampling (tried, no gain), stronger
regularization (tried, no gain), focal loss (the imbalance is already handled by sampling and
weights; the CHAIN problem is mainly invisible evidence and overlap), or simply more data (going
from 22k to 80k games gained 1.7 macro-F1 points in the v1 line). Probability calibration was
**not measured** (no reliability analysis), so no claim is made about it.

Any further model needs a **new unused holdout**; the archive still has many unused games.

## Answers to the final questions

1. **Is 55.5% raw accuracy better than meaningful baselines?** Not on accuracy: it is below
   always-NO_PLAN (58.8%) and the rule engine (69.7%), and above stratified random (39.3%). On
   balanced accuracy it is far above trivial baselines (69.6% against 16.7%) and above the rule
   engine (60.3%). On macro-F1 (0.488) it beats trivial baselines but not the rule engine (0.556).
2. **Learning strategies or predicting no-plan?** Learning strategies. Raw v2 under-predicts
   no-plan. The calibrated version predicts it about as often as it occurs.
3. **Is 72.1% methodologically valid?** Yes, as calibrated accuracy on unseen games with a
   validation-only bias. It is not the raw network's accuracy, and it should never be quoted
   without macro-F1 0.583, balanced accuracy 60.1% and the rule engine's 69.7%.
4. **Does v2 recognize strategy positions effectively?** Moderately. It names the right plan well
   when forced (81.8%), but calibrated it recovers 67.0% of strategy positions, and raw it finds
   79.4% while inventing plans for 61% of no-plan positions.
5. **Which classes are reliable?** NO_PLAN and OUTPOST are the best (F1 .78 and .74 on every
   position; precision .80 and .69). Q_EXP, ROOK_FILE and K_STORM are moderate (.55 to .61), and
   better on filtered positions (.73 to .78). CHAIN_UNDERMINE is not usable (precision .20).
6. **Why does CHAIN_UNDERMINE underperform?** Mostly because nearly half its labels have no
   evidence in past moves, it overlaps Q_EXP and K_STORM, and it is under 1% of positions. Twice
   as many examples did not help.
7. **Is v2 better than v1 for the real application?** For judging arbitrary positions, yes, by a
   modest, statistically reliable margin (+2.9 macro-F1 points over v1 as shipped; at equal
   calibration +2.4 to +3.0 macro-F1 and +2.0 to +2.4 accuracy points). v1 is better on confident, filtered positions. Which matters
   depends on what the app shows.
8. **Biggest bottleneck?** Most likely the **prediction target**: weak labels built from future
   moves. This is supported, not proven. Data quantity, class imbalance and regularization showed
   little further effect; **model capacity is untested.**
9. **Keep, improve or reconsider?** Keep v2 as the app candidate and do not retrain yet. Annotate
   first. If humans agree poorly with the weak labels, reconsider the labelling approach; if they
   agree well, run the capacity check.

v2 still **did not pass the pre-registered gate B**, so under [V2_PROTOCOL.md](V2_PROTOCOL.md) the
promoted model remains v1. Using v2 is a team decision and should be reported as an exception.

## Appendix: v2 (seed 42) per-class metrics and confusion matrices

Generated from `experiments/v2/accuracy_investigation.json`. Matrix rows are true labels, columns are predictions.

### Train (rows trained on), raw

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.566 | 0.856 | 0.681 | 109,180 | 165,100 |
| Q_EXP | 0.714 | 0.796 | 0.753 | 200,421 | 223,590 |
| OUTPOST | 0.836 | 0.829 | 0.832 | 263,457 | 261,275 |
| ROOK_FILE | 0.637 | 0.797 | 0.708 | 93,836 | 117,315 |
| CHAIN_UNDERMINE | 0.201 | 0.822 | 0.323 | 14,234 | 58,228 |
| NO_PLAN | 0.757 | 0.382 | 0.508 | 291,912 | 147,532 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 93,459 | 3,066 | 2,750 | 1,711 | 2,537 | 5,657 |
| **Q_EXP** | 7,844 | 159,574 | 6,285 | 3,816 | 10,509 | 12,393 |
| **OUTP** | 10,996 | 11,011 | 218,368 | 6,863 | 5,250 | 10,969 |
| **ROOK** | 4,294 | 3,461 | 3,956 | 74,746 | 1,073 | 6,306 |
| **CHAIN** | 870 | 611 | 289 | 194 | 11,704 | 566 |
| **NONE** | 47,637 | 45,867 | 29,627 | 29,985 | 27,155 | 111,641 |

### Train (rows trained on), adjusted

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.776 | 0.599 | 0.676 | 109,180 | 84,320 |
| Q_EXP | 0.825 | 0.651 | 0.728 | 200,421 | 158,185 |
| OUTPOST | 0.851 | 0.802 | 0.826 | 263,457 | 248,214 |
| ROOK_FILE | 0.816 | 0.579 | 0.677 | 93,836 | 66,592 |
| CHAIN_UNDERMINE | 0.492 | 0.443 | 0.466 | 14,234 | 12,830 |
| NO_PLAN | 0.554 | 0.765 | 0.643 | 291,912 | 402,899 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 65,437 | 1,188 | 2,824 | 605 | 256 | 38,870 |
| **Q_EXP** | 1,608 | 130,505 | 5,814 | 1,260 | 1,779 | 59,455 |
| **OUTP** | 2,565 | 4,921 | 211,233 | 2,176 | 556 | 42,006 |
| **ROOK** | 1,148 | 1,732 | 3,970 | 54,338 | 112 | 32,536 |
| **CHAIN** | 435 | 413 | 307 | 57 | 6,312 | 6,710 |
| **NONE** | 13,127 | 19,426 | 24,066 | 8,156 | 3,815 | 223,322 |

### Validation, raw

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.318 | 0.800 | 0.455 | 19,786 | 49,771 |
| Q_EXP | 0.460 | 0.768 | 0.575 | 35,289 | 58,854 |
| OUTPOST | 0.649 | 0.811 | 0.721 | 47,348 | 59,198 |
| ROOK_FILE | 0.375 | 0.751 | 0.500 | 16,969 | 33,959 |
| CHAIN_UNDERMINE | 0.080 | 0.643 | 0.143 | 2,775 | 22,246 |
| NO_PLAN | 0.895 | 0.367 | 0.521 | 172,607 | 70,746 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 15,825 | 786 | 756 | 511 | 680 | 1,228 |
| **Q_EXP** | 1,599 | 27,087 | 1,278 | 882 | 1,944 | 2,499 |
| **OUTP** | 2,237 | 2,156 | 38,422 | 1,373 | 1,074 | 2,086 |
| **ROOK** | 1,006 | 751 | 879 | 12,737 | 252 | 1,344 |
| **CHAIN** | 337 | 224 | 121 | 64 | 1,785 | 244 |
| **NONE** | 28,767 | 27,850 | 17,742 | 18,392 | 16,511 | 63,345 |

### Validation, adjusted

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.532 | 0.559 | 0.546 | 19,786 | 20,794 |
| Q_EXP | 0.615 | 0.629 | 0.622 | 35,289 | 36,109 |
| OUTPOST | 0.684 | 0.785 | 0.731 | 47,348 | 54,317 |
| ROOK_FILE | 0.599 | 0.553 | 0.575 | 16,969 | 15,672 |
| CHAIN_UNDERMINE | 0.208 | 0.289 | 0.242 | 2,775 | 3,857 |
| NO_PLAN | 0.793 | 0.754 | 0.773 | 172,607 | 164,025 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 11,069 | 361 | 696 | 179 | 92 | 7,389 |
| **Q_EXP** | 374 | 22,200 | 1,179 | 320 | 355 | 10,861 |
| **OUTP** | 540 | 960 | 37,153 | 456 | 126 | 8,113 |
| **ROOK** | 324 | 362 | 869 | 9,390 | 29 | 5,995 |
| **CHAIN** | 159 | 140 | 106 | 28 | 803 | 1,539 |
| **NONE** | 8,328 | 12,086 | 14,314 | 5,299 | 2,452 | 130,128 |

### Unseen test, every position, raw

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.316 | 0.809 | 0.454 | 30,676 | 78,601 |
| Q_EXP | 0.459 | 0.763 | 0.573 | 53,940 | 89,586 |
| OUTPOST | 0.659 | 0.827 | 0.733 | 78,179 | 98,162 |
| ROOK_FILE | 0.372 | 0.761 | 0.499 | 25,036 | 51,261 |
| CHAIN_UNDERMINE | 0.070 | 0.631 | 0.126 | 3,202 | 28,926 |
| NO_PLAN | 0.903 | 0.388 | 0.542 | 272,375 | 116,872 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 24,816 | 1,153 | 1,090 | 788 | 842 | 1,987 |
| **Q_EXP** | 2,521 | 41,155 | 1,991 | 1,393 | 2,958 | 3,922 |
| **OUTP** | 3,464 | 3,216 | 64,648 | 2,246 | 1,293 | 3,312 |
| **ROOK** | 1,463 | 1,148 | 1,250 | 19,046 | 299 | 1,830 |
| **CHAIN** | 384 | 299 | 131 | 105 | 2,019 | 264 |
| **NONE** | 45,953 | 42,615 | 29,052 | 27,683 | 21,515 | 105,557 |

### Unseen test, every position, adjusted

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.537 | 0.563 | 0.550 | 30,676 | 32,156 |
| Q_EXP | 0.612 | 0.614 | 0.613 | 53,940 | 54,148 |
| OUTPOST | 0.691 | 0.800 | 0.741 | 78,179 | 90,548 |
| ROOK_FILE | 0.596 | 0.559 | 0.577 | 25,036 | 23,468 |
| CHAIN_UNDERMINE | 0.197 | 0.311 | 0.241 | 3,202 | 5,070 |
| NO_PLAN | 0.800 | 0.758 | 0.778 | 272,375 | 258,018 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 17,278 | 502 | 1,056 | 242 | 108 | 11,490 |
| **Q_EXP** | 611 | 33,118 | 1,788 | 515 | 551 | 17,357 |
| **OUTP** | 865 | 1,443 | 62,528 | 800 | 135 | 12,408 |
| **ROOK** | 453 | 635 | 1,260 | 13,996 | 45 | 8,647 |
| **CHAIN** | 161 | 173 | 113 | 27 | 997 | 1,731 |
| **NONE** | 12,788 | 18,277 | 23,803 | 7,888 | 3,234 | 206,385 |

### Unseen test, filtered, raw

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.439 | 0.865 | 0.582 | 9,066 | 17,864 |
| Q_EXP | 0.634 | 0.834 | 0.720 | 19,659 | 25,876 |
| OUTPOST | 0.869 | 0.878 | 0.873 | 40,874 | 41,304 |
| ROOK_FILE | 0.636 | 0.826 | 0.719 | 11,700 | 15,197 |
| CHAIN_UNDERMINE | 0.049 | 0.603 | 0.091 | 849 | 10,422 |
| NO_PLAN | 0.949 | 0.533 | 0.682 | 65,016 | 36,501 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 7,842 | 343 | 266 | 186 | 216 | 213 |
| **Q_EXP** | 804 | 16,400 | 578 | 378 | 873 | 626 |
| **OUTP** | 1,471 | 1,448 | 35,880 | 1,041 | 428 | 606 |
| **ROOK** | 538 | 522 | 510 | 9,664 | 87 | 379 |
| **CHAIN** | 123 | 92 | 40 | 29 | 512 | 53 |
| **NONE** | 7,086 | 7,071 | 4,030 | 3,899 | 8,306 | 34,624 |

### Unseen test, filtered, adjusted

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| K_STORM | 0.777 | 0.687 | 0.729 | 9,066 | 8,019 |
| Q_EXP | 0.831 | 0.733 | 0.779 | 19,659 | 17,339 |
| OUTPOST | 0.900 | 0.870 | 0.885 | 40,874 | 39,479 |
| ROOK_FILE | 0.868 | 0.671 | 0.757 | 11,700 | 9,047 |
| CHAIN_UNDERMINE | 0.175 | 0.314 | 0.225 | 849 | 1,525 |
| NO_PLAN | 0.817 | 0.902 | 0.858 | 65,016 | 71,755 |

| true \ predicted | K_ST | Q_EXP | OUTP | ROOK | CHAIN | NONE |
|---|---|---|---|---|---|---|
| **K_ST** | 6,228 | 225 | 290 | 64 | 35 | 2,224 |
| **Q_EXP** | 250 | 14,405 | 543 | 154 | 186 | 4,121 |
| **OUTP** | 422 | 780 | 35,546 | 397 | 45 | 3,684 |
| **ROOK** | 190 | 347 | 608 | 7,851 | 12 | 2,692 |
| **CHAIN** | 80 | 68 | 33 | 12 | 267 | 389 |
| **NONE** | 849 | 1,514 | 2,459 | 569 | 980 | 58,645 |
