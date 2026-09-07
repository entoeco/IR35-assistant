# Phase 3 — Transformer and Few-Shot Comparison

> **VERIFICATION RUN, NOT A RESULT.** This run used the deterministic mock backend. It proves the pipeline end to end — premise construction, batching, score mapping, thresholding, reporting — but the mock returns canned distributions keyed on trigger phrases, so **no number below is evidence about model performance.** See `docs/phase3_status.md` for why the real run has not happened yet and how to produce it.

Backend: `mock` — n/a (local).

**This is decision support.** Every method produces a human-readable reason with every flag, and no method is asked for, or capable of producing, an employment status determination.

## Results

### `record` split

| Method | PR-AUC | ROC-AUC | Macro-F1 | Brier | Threshold | Recall | Precision | Flags/form | Missed |
|---|---|---|---|---|---|---|---|---|---|
| `fewshot_llm` | 0.008 | 0.508 | 0.461 | 0.0352 | 0.262 | **0.17** | 0.007 | 4.64 | 58 |
| `nli_cross_encoder` | 0.007 | 0.477 | 0.141 | 0.0887 | 0.115 | **0.80** | 0.007 | 23.93 | 14 |
| `rules_as_authored` | 0.571 | 0.797 | 0.690 | 0.0048 | 0.354 | **0.60** | 0.286 | 0.42 | 28 |
| `tfidf_logreg` | 0.732 | 0.942 | 0.592 | 0.0039 | 0.090 | **0.80** | 0.118 | 1.35 | 14 |

- `fewshot_llm`: Recall target 80% is unreachable at any threshold; the best achievable is 17%. Reported at that point.
- `rules_as_authored`: Recall target 80% is unreachable at any threshold; the best achievable is 60%. Reported at that point.

### `unit` split

| Method | PR-AUC | ROC-AUC | Macro-F1 | Brier | Threshold | Recall | Precision | Flags/form | Missed |
|---|---|---|---|---|---|---|---|---|---|
| `fewshot_llm` | 0.008 | 0.508 | 0.461 | 0.0352 | 0.262 | **0.17** | 0.007 | 4.64 | 58 |
| `nli_cross_encoder` | 0.007 | 0.477 | 0.141 | 0.0887 | 0.115 | **0.80** | 0.007 | 23.93 | 14 |
| `rules_as_authored` | 0.571 | 0.797 | 0.690 | 0.0048 | 0.354 | **0.60** | 0.286 | 0.42 | 28 |
| `tfidf_logreg` | 0.010 | 0.620 | 0.346 | 0.1379 | 0.050 | **0.80** | 0.011 | 14.11 | 14 |

- `fewshot_llm`: Recall target 80% is unreachable at any threshold; the best achievable is 17%. Reported at that point.
- `rules_as_authored`: Recall target 80% is unreachable at any threshold; the best achievable is 60%. Reported at that point.

### Phase 2 baselines, for comparison

| Method | Split | PR-AUC | Recall | Flags/form |
|---|---|---|---|---|
| `rules_as_authored` | `record` | 0.571 | 0.60 | 0.42 |
| `rules_phrase_deleaked` | `record` | 0.434 | 0.53 | 0.41 |
| `rules_vocab_stripped` | `record` | 0.035 | 0.03 | 0.01 |
| `tfidf_logreg` | `record` | 0.699 | 0.80 | 1.52 |
| `rules_as_authored` | `unit` | 0.571 | 0.60 | 0.42 |
| `rules_phrase_deleaked` | `unit` | 0.434 | 0.53 | 0.41 |
| `rules_vocab_stripped` | `unit` | 0.035 | 0.03 | 0.01 |
| `tfidf_logreg` | `unit` | 0.009 | 0.80 | 16.90 |

The comparison that matters is the `unit` split: TF-IDF collapsed there because it had memorised the templates, while the rule baseline was unaffected because it does not learn. Both methods added in this phase are also split-independent — zero-shot NLI and few-shot prompting fit nothing — so any gap between them and the rule baseline on the `unit` split is a real difference in ability to generalise, not a difference in what was memorised.

## Recall by contradiction type (`record` split)

| Method | category mismatch | direct negation | hedged non support | named person dependency | reimbursement flip | scope qualification | temporal inconsistency |
|---|---|---|---|---|---|---|---|
| `fewshot_llm` | 0.00 (0/9) | 0.20 (2/10) | 0.20 (2/10) | 0.40 (4/10) | 0.20 (2/10) | 0.09 (1/11) | 0.10 (1/10) |
| `nli_cross_encoder` | 0.67 (6/9) | 0.90 (9/10) | 0.70 (7/10) | 0.90 (9/10) | 0.90 (9/10) | 0.73 (8/11) | 0.80 (8/10) |
| `rules_as_authored` | 0.89 (8/9) | 0.80 (8/10) | 0.20 (2/10) | 0.40 (4/10) | 0.60 (6/10) | 0.73 (8/11) | 0.60 (6/10) |
| `tfidf_logreg` | 0.78 (7/9) | 0.80 (8/10) | 0.90 (9/10) | 0.80 (8/10) | 0.90 (9/10) | 0.55 (6/11) | 0.90 (9/10) |

The two types to watch are `hedged_non_support` and `named_person_dependency`, where the Phase 2 rule baseline scored 0.20 and 0.40. Those are the specific claims made for the transformer approach: that a model reasoning over the *relation* between two statements can detect a justification that fails to support a definite answer, where a keyword list has no phrase to match.

## Recall by subtlety (`record` split)

| Method | level 1 | level 2 | level 3 |
|---|---|---|---|
| `fewshot_llm` | 0.30 (7/23) | 0.12 (4/34) | 0.08 (1/13) |
| `nli_cross_encoder` | 0.96 (22/23) | 0.74 (25/34) | 0.69 (9/13) |
| `rules_as_authored` | 0.65 (15/23) | 0.65 (22/34) | 0.38 (5/13) |
| `tfidf_logreg` | 0.74 (17/23) | 0.82 (28/34) | 0.85 (11/13) |

## Runtime

| Method | Total inference time |
|---|---|
| `nli_cross_encoder` | 0.0s |
| `fewshot_llm` | 0.0s |
| `rules_as_authored` | 0.0s |
| `tfidf_logreg` | 12.6s |

Recorded because the deployment assumption is local-only. A method that needs a GPU, or that costs a hosted API call per field per submission, is a different proposition for the University than one that runs on a laptop — and that is a procurement and Information Governance question as much as an accuracy one.

## Dataset

- 9,918 instances across 350 submissions
- 70 planted contradictions (0.71%)
- Recall target 80%

