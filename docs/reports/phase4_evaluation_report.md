# Phase 4 — Evaluation and Bias Testing

Headline metrics, then the robustness tests that decide whether the tool is safe to put in front of a reviewer: does it read substance or surface, does it work equally well across engagement types, does it find the subtle cases, and does its confidence number mean anything.

**This is decision support.** Nothing here issues or overrides a Status Determination Statement, and the operating point is chosen on the assumption that a human reads every flag.

Evaluated on the Phase 1 synthetic corpus: 9,918 (submission, question) instances across 350 submissions, 70 planted contradictions (0.71%). Scores are out-of-fold.

> The Phase 3 transformer arms are absent: this sandbox's egress policy blocks `huggingface.co`, so no model weights could be fetched. See `docs/phase3_status.md`. Everything below therefore evaluates the methods that can actually run, and the same harness will accept the NLI and few-shot detectors unchanged once weights are available.

## 1. Headline metrics

| Method | PR-AUC | ROC-AUC | Macro-F1 | Threshold | Recall | 95% CI | Precision | Flags/form | Missed |
|---|---|---|---|---|---|---|---|---|---|
| `rules_as_authored` | 0.571 | 0.797 | 0.690 | 0.354 | **0.60** | 0.48–0.72 | 0.286 | 0.42 | 28 |
| `tfidf_logreg` | 0.699 | 0.949 | 0.580 | 0.073 | **0.80** | 0.71–0.89 | 0.105 | 1.52 | 14 |

- `rules_as_authored`: Recall target 80% is unreachable at any threshold; the best achievable is 60%. Reported at that point.

Macro-F1 is reported because the brief asks for it. It should be read with care at this class balance: it averages the F1 of the majority class — which is near 1.0 for everything — with the F1 of the minority class, so it compresses real differences. PR-AUC and flags-per-form are the metrics that describe what a reviewer experiences.

## 2. The operating point, and why recall

Thresholds are chosen as **the highest threshold that still reaches 80% recall**, not the threshold maximising F1.

The asymmetry is the whole argument. A missed contradiction reaches HMRC as a determination made without the reasonable care the off-payroll rules require — exposure measured in tax liability, interest and penalties, plus the dispute itself. A false flag costs a reviewer perhaps thirty seconds to read and dismiss. Optimising F1 would trade recall for precision at roughly one-to-one, which is the wrong exchange rate by two orders of magnitude.

**The limit on that argument is human, not statistical.** If reviewers see so many flags that they start dismissing without reading, recall on paper rises while recall in practice collapses — and the tool becomes worse than nothing, because it produces an audit trail suggesting the contradictions were considered. The precision floor is therefore set by reviewer fatigue, and we do not yet know where it is. That is why:

- **flags per submission** is reported at every operating point, not precision alone;
- **Phase 5 captures every accept/dismiss decision**, because those are the only data that will locate the real floor;
- the threshold is config, not code, so it can be moved once there is evidence.

At the chosen point, `rules_as_authored` shows a reviewer **0.42 flags per submission** and misses **28** of 70 contradictions.
At the chosen point, `tfidf_logreg` shows a reviewer **1.52 flags per submission** and misses **14** of 70 contradictions.

## 3. Style invariance

Every engagement re-rendered verbose, terse and hedged with the structured answers, the propositions and the planted contradictions held **literally identical** — only the wording differs. This is possible because the Phase 1 generator separates what a record says from how it is written, and it is what makes the test meaningful: any difference in the flags is attributable to surface form alone.

| Method | Flag-set agreement (Jaccard) | Records agreeing exactly | Agreement on true flags | Agreement on false flags | Mean score drift |
|---|---|---|---|---|---|
| `rules_as_authored` | 0.587 | 123/350 (35%) | 0.969 | 0.333 | 0.0142 |
| `tfidf_logreg` | 0.242 | 11/350 (3%) | 0.919 | 0.124 | 0.0227 |

### Recall by register, same contradictions

| Method | verbose | terse | hedged | spread |
|---|---|---|---|---|
| `rules_as_authored` | 0.60 | 0.60 | 0.60 | 0.00 |
| `tfidf_logreg` | 0.77 | 0.80 | 0.80 | 0.03 |

**The headline is the gap between the last two columns.** Recall barely moves across registers — the table below shows a spread of a few points at most — so an aggregate metric would report the detector as register-robust. But the flag *sets* agree far less well, and splitting the agreement by whether a flag was correct shows why: the true contradictions are found consistently, while a substantially different set of **false** flags is raised depending on how the manager wrote. Stable recall, unstable reviewer experience.

That distinction matters operationally. The same submission, rewritten in a different voice by a different manager, shows the reviewer a different set of fields to check. Aggregate recall cannot see this, and it is exactly the kind of inconsistency that erodes trust in a tool — a reviewer who notices that two similar forms produced different flags has no way to tell whether the difference is meaningful.

**Why this is a fairness test, not just a robustness test.** Register is not randomly distributed across engagement types. In this corpus visiting-lecturer submissions skew verbose and hedged; technical-trades submissions skew terse. A detector that performs worse on hedged prose therefore performs worse for a particular category of worker — and the per-archetype breakdown alone cannot tell you whether that is because the engagement type is harder or because those managers write differently. Holding the engagement fixed and varying only the writing separates the two.

`rules_as_authored`: score drift is largest on technical trades (0.019) and smallest on visiting lecturer (0.008).
`tfidf_logreg`: score drift is largest on technical trades (0.026) and smallest on visiting lecturer (0.018).

## 4. Recall by contradiction type

| Method | Type | n | Caught | Recall | 95% CI | CI width |
|---|---|---|---|---|---|---|
| `rules_as_authored` | category mismatch | 9 | 8 | 0.89 | 0.73–1.00 | 0.27 |
| `rules_as_authored` | direct negation | 10 | 8 | 0.80 | 0.50–1.00 | 0.50 |
| `rules_as_authored` | hedged non support | 10 | 2 | 0.20 | 0.00–0.50 | 0.50 |
| `rules_as_authored` | named person dependency | 10 | 4 | 0.40 | 0.10–0.70 | 0.60 |
| `rules_as_authored` | reimbursement flip | 10 | 6 | 0.60 | 0.30–0.90 | 0.60 |
| `rules_as_authored` | scope qualification | 11 | 8 | 0.73 | 0.40–1.00 | 0.60 |
| `rules_as_authored` | temporal inconsistency | 10 | 6 | 0.60 | 0.30–0.90 | 0.60 |
| `tfidf_logreg` | category mismatch | 9 | 7 | 0.78 | 0.55–1.00 | 0.45 |
| `tfidf_logreg` | direct negation | 10 | 8 | 0.80 | 0.50–1.00 | 0.50 |
| `tfidf_logreg` | hedged non support | 10 | 7 | 0.70 | 0.40–1.00 | 0.60 |
| `tfidf_logreg` | named person dependency | 10 | 10 | 1.00 | 1.00–1.00 | 0.00 |
| `tfidf_logreg` | reimbursement flip | 10 | 7 | 0.70 | 0.40–1.00 | 0.60 |
| `tfidf_logreg` | scope qualification | 11 | 8 | 0.73 | 0.50–1.00 | 0.50 |
| `tfidf_logreg` | temporal inconsistency | 10 | 9 | 0.90 | 0.70–1.00 | 0.30 |

Intervals are cluster bootstrap over **records**, not instances (2,000 replicates). Instances within one submission share a manager, a vocabulary, an engagement and a register, so resampling them independently would produce intervals narrower than the data supports.

Read the interval widths before the point estimates. At nine to eleven positives per type, a cell is consistent with a wide range of true values, and a difference of one or two caught cases is not a difference in capability.

## 5. Subtlety curve

| Method | Level | n | Caught | Recall | 95% CI | CI width |
|---|---|---|---|---|---|---|
| `rules_as_authored` | 1 | 23 | 15 | 0.65 | 0.43–0.84 | 0.41 |
| `rules_as_authored` | 2 | 34 | 22 | 0.65 | 0.50–0.79 | 0.29 |
| `rules_as_authored` | 3 | 13 | 5 | 0.38 | 0.15–0.62 | 0.46 |
| `tfidf_logreg` | 1 | 23 | 19 | 0.83 | 0.65–0.96 | 0.31 |
| `tfidf_logreg` | 2 | 34 | 26 | 0.76 | 0.62–0.89 | 0.27 |
| `tfidf_logreg` | 3 | 13 | 11 | 0.85 | 0.62–1.00 | 0.38 |

Level 1 is blatant — the text plainly says the opposite. Level 2 needs reading: a condition, a veto or a reimbursement flips the meaning. Level 3 needs domain knowledge, usually the form's own cost exclusions or the case law. **Level 3 is where the tool is most needed and least reliable**, because those are also the cases a busy reviewer misses.

## 6. Role-type disparity

| Method | Archetype | n | Caught | Recall | 95% CI | CI width |
|---|---|---|---|---|---|---|
| `rules_as_authored` | it contractor | 23 | 17 | 0.74 | 0.52–0.92 | 0.40 |
| `rules_as_authored` | research consultant | 18 | 10 | 0.56 | 0.32–0.80 | 0.48 |
| `rules_as_authored` | technical trades | 13 | 7 | 0.54 | 0.33–0.83 | 0.50 |
| `rules_as_authored` | visiting lecturer | 16 | 8 | 0.50 | 0.27–0.73 | 0.47 |
| `tfidf_logreg` | it contractor | 23 | 19 | 0.83 | 0.67–0.95 | 0.29 |
| `tfidf_logreg` | research consultant | 18 | 14 | 0.78 | 0.60–1.00 | 0.40 |
| `tfidf_logreg` | technical trades | 13 | 9 | 0.69 | 0.43–0.91 | 0.48 |
| `tfidf_logreg` | visiting lecturer | 16 | 14 | 0.88 | 0.69–1.00 | 0.31 |

For `rules_as_authored`, the gap runs from 50% on visiting lecturer engagements to 74% on it contractor ones. The intervals **overlap**, so this is a direction to investigate rather than a demonstrated effect.

Either way it belongs in the Information Governance conversation before real data. A tool that protects one category of worker better than another is a governance question first and a metric second — and the direction here is the one predicted in Phase 1: teaching engagements are structurally the hardest case, because genuine autonomy of method sits alongside a fixed timetable, on-site delivery and year-on-year repetition.

## 7. Calibration

| Method | ECE | MCE | Brier | Reliability | Resolution |
|---|---|---|---|---|---|
| `rules_as_authored` | 0.0066 | 0.3496 | 0.00485 | 0.001565 | 0.003727 |
| `tfidf_logreg` | 0.0164 | 0.1147 | 0.00375 | 0.000570 | 0.004032 |

Lower reliability is better calibrated; higher resolution means the scores separate positives from negatives. Both are components of the Brier score, which is why a method can have a good Brier score by being uniformly unconfident — the decomposition is what stops that being mistaken for skill.

**This matters because Phase 5 puts a confidence number in front of a reviewer.** A number that is not calibrated is worse than no number: a reviewer who learns that 0.9 usually means 'wrong' has been trained to distrust the tool, and one who takes 0.9 at face value has been misled into skipping a check. The rule baseline's score is a squashed weighted count of matched cues — monotone and useful for ranking, never intended as a probability — so its output should be shown as a band (low / medium / high) or a rank, not a percentage. That is a Phase 5 design constraint arising from this measurement.

## 8. Limitations

1. **The transformer arms are missing.** The comparison the assignment is built around is incomplete until the Phase 3 models can be run.
2. **Small cells.** Nine to eleven positives per contradiction type; the intervals are wide and are reported for that reason.
3. **Synthetic phrasing is discrete.** Real justifications vary continuously; template propositions do not. The style-invariance test varies register but not the underlying wording of a proposition, so it measures robustness to *style*, not to paraphrase.
4. **The corpus contradiction-type mix is stratified by design** (Phase 1), so per-type recall is estimable but the aggregate is not a field estimate.
5. **Reviewer fatigue is unmeasured.** The precision floor is asserted, not evidenced, until Phase 5 has collected accept/dismiss decisions.

