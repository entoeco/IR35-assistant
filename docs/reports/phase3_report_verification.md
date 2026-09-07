# Phase 3 — Transformer and Few-Shot Comparison

> **VERIFICATION RUN, NOT A RESULT.** This run used the deterministic mock backend. It proves the pipeline end to end — premise construction, batching, score mapping, thresholding, reporting — but the mock returns canned distributions keyed on trigger phrases, so **no number below is evidence about model performance.** See `docs/phase3_status.md` for why the real run has not happened yet and how to produce it.

Backend: `mock` — n/a (local).

**This is decision support.** Every method produces a human-readable reason with every flag, and no method is asked for, or capable of producing, an employment status determination.

## Results

### `record` split

| Method | PR-AUC | ROC-AUC | Macro-F1 | Brier | Threshold | Recall | Precision | Flags/form | Missed |
|---|---|---|---|---|---|---|---|---|---|
| `fewshot_llm` | 0.009 | 0.536 | 0.463 | 0.0350 | 0.262 | **0.23** | 0.010 | 4.66 | 54 |
| `nli_cross_encoder` | 0.008 | 0.514 | 0.174 | 0.0887 | 0.117 | **0.80** | 0.007 | 22.65 | 14 |
| `rules_as_authored` | 0.568 | 0.797 | 0.686 | 0.0050 | 0.354 | **0.60** | 0.276 | 0.43 | 28 |
| `tfidf_logreg` | 0.733 | 0.951 | 0.612 | 0.0035 | 0.087 | **0.80** | 0.142 | 1.12 | 14 |

- `fewshot_llm`: Recall target 80% is unreachable at any threshold; the best achievable is 23%. Reported at that point.
- `rules_as_authored`: Recall target 80% is unreachable at any threshold; the best achievable is 60%. Reported at that point.

### `unit` split

| Method | PR-AUC | ROC-AUC | Macro-F1 | Brier | Threshold | Recall | Precision | Flags/form | Missed |
|---|---|---|---|---|---|---|---|---|---|
| `fewshot_llm` | 0.009 | 0.536 | 0.463 | 0.0350 | 0.262 | **0.23** | 0.010 | 4.66 | 54 |
| `nli_cross_encoder` | 0.008 | 0.514 | 0.174 | 0.0887 | 0.117 | **0.80** | 0.007 | 22.65 | 14 |
| `rules_as_authored` | 0.568 | 0.797 | 0.686 | 0.0050 | 0.354 | **0.60** | 0.276 | 0.43 | 28 |
| `tfidf_logreg` | 0.023 | 0.636 | 0.357 | 0.1440 | 0.063 | **0.80** | 0.012 | 13.46 | 14 |

- `fewshot_llm`: Recall target 80% is unreachable at any threshold; the best achievable is 23%. Reported at that point.
- `rules_as_authored`: Recall target 80% is unreachable at any threshold; the best achievable is 60%. Reported at that point.

### Phase 2 baselines, for comparison

| Method | Split | PR-AUC | Recall | Flags/form |
|---|---|---|---|---|
| `rules_as_authored` | `record` | 0.568 | 0.60 | 0.43 |
| `rules_phrase_deleaked` | `record` | 0.425 | 0.53 | 0.42 |
| `rules_vocab_stripped` | `record` | 0.035 | 0.03 | 0.01 |
| `tfidf_logreg` | `record` | 0.718 | 0.80 | 0.71 |
| `rules_as_authored` | `unit` | 0.568 | 0.60 | 0.43 |
| `rules_phrase_deleaked` | `unit` | 0.425 | 0.53 | 0.42 |
| `rules_vocab_stripped` | `unit` | 0.035 | 0.03 | 0.01 |
| `tfidf_logreg` | `unit` | 0.009 | 0.80 | 16.18 |

The comparison that matters is the `unit` split: TF-IDF collapsed there because it had memorised the templates, while the rule baseline was unaffected because it does not learn. Both methods added in this phase are also split-independent — zero-shot NLI and few-shot prompting fit nothing — so any gap between them and the rule baseline on the `unit` split is a real difference in ability to generalise, not a difference in what was memorised.

## Recall by contradiction type (`record` split)

| Method | category mismatch | direct negation | hedged non support | named person dependency | reimbursement flip | scope qualification | temporal inconsistency |
|---|---|---|---|---|---|---|---|
| `fewshot_llm` | 0.00 (0/9) | 0.10 (1/10) | 0.30 (3/10) | 0.60 (6/10) | 0.30 (3/10) | 0.09 (1/11) | 0.20 (2/10) |
| `nli_cross_encoder` | 0.78 (7/9) | 0.50 (5/10) | 0.60 (6/10) | 0.90 (9/10) | 0.90 (9/10) | 1.00 (11/11) | 0.90 (9/10) |
| `rules_as_authored` | 0.89 (8/9) | 0.80 (8/10) | 0.20 (2/10) | 0.40 (4/10) | 0.60 (6/10) | 0.73 (8/11) | 0.60 (6/10) |
| `tfidf_logreg` | 0.78 (7/9) | 0.80 (8/10) | 0.90 (9/10) | 0.80 (8/10) | 0.90 (9/10) | 0.55 (6/11) | 0.90 (9/10) |

The two types to watch are `hedged_non_support` and `named_person_dependency`, where the Phase 2 rule baseline scored 0.20 and 0.40. Those are the specific claims made for the transformer approach: that a model reasoning over the *relation* between two statements can detect a justification that fails to support a definite answer, where a keyword list has no phrase to match.

## Recall by subtlety (`record` split)

| Method | level 1 | level 2 | level 3 |
|---|---|---|---|
| `fewshot_llm` | 0.26 (6/23) | 0.26 (9/34) | 0.08 (1/13) |
| `nli_cross_encoder` | 0.74 (17/23) | 0.85 (29/34) | 0.77 (10/13) |
| `rules_as_authored` | 0.65 (15/23) | 0.65 (22/34) | 0.38 (5/13) |
| `tfidf_logreg` | 0.74 (17/23) | 0.85 (29/34) | 0.77 (10/13) |

## Runtime

| Method | Total inference time |
|---|---|
| `nli_cross_encoder` | 0.0s |
| `fewshot_llm` | 0.0s |
| `rules_as_authored` | 0.0s |
| `tfidf_logreg` | 9.0s |

Recorded because the deployment assumption is local-only. A method that needs a GPU, or that costs a hosted API call per field per submission, is a different proposition for the University than one that runs on a laptop — and that is a procurement and Information Governance question as much as an accuracy one.

## Dataset

- 9,918 instances across 350 submissions
- 70 planted contradictions (0.71%)
- Recall target 80%

