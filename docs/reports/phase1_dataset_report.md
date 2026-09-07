# Phase 1 — Synthetic Dataset Report

- **Dataset version:** `1.0.0`
- **Seed:** `20260907`
- **Records:** 350
- **Provenance:** generated from `config/generation.yaml`, `config/text_bank.yaml` and `config/ir35_weights.yaml`. Regenerating from the same configs reproduces this corpus byte-for-byte, in any process.

  That last clause was not true until Phase 4. `render` seeded its random generator with `hash(register)`, and Python salts string hashing per process, so a corpus regenerated in a new interpreter did not match. Every determinism test passed throughout, because they all ran inside one interpreter where the salt is fixed. It is now seeded with `zlib.crc32`, and `tests/test_generator.py` asserts the property across two subprocesses with different `PYTHONHASHSEED` values — the only shape of test that could have caught it.

All content is synthetic (constraint 1). Every name, address, company, rate and email is drawn from the invented pools in the generation config.

## Acceptance against targets

| Measure | Target | Actual | |
|---|---|---|---|
| Inside rate | 25%–30% | 28.9% | PASS |
| Undetermined rate | 5%–12% | 12.0% | PASS |
| Records with a contradiction | 13%–17% | 14.9% | PASS |

## Status labels

| Label | n | share |
|---|---|---|
| inside | 101 | 28.9% |
| outside | 207 | 59.1% |
| undetermined | 42 | 12.0% |

`undetermined` is the rule engine abstaining, not a failure: it is the 'unable to determine' outcome CEST itself returns. These records are excluded from status-classifier training and kept for reviewer-facing evaluation.

## Engagement archetypes

| Archetype | n | inside | outside | undetermined |
|---|---|---|---|---|
| it_contractor | 111 | 18.9% | 69.4% | 11.7% |
| visiting_lecturer | 86 | 66.3% | 22.1% | 11.6% |
| technical_trades | 72 | 6.9% | 80.6% | 12.5% |
| research_consultant | 81 | 22.2% | 65.4% | 12.3% |

The spread is the point. Visiting lecturers skew inside — timetabled, on-site, repeated year on year — and are expected to be where the detector performs worst. That is a Phase 4 finding to report, not a defect to design away.

## Writing registers

| Register | n | share |
|---|---|---|
| hedged | 111 | 31.7% |
| terse | 127 | 36.3% |
| verbose | 112 | 32.0% |

Free-text length: median 122 characters, 95th percentile 244, max 329.

## Planted contradictions

70 contradictions across 52 records (14.9% of the corpus).

### By type

| Contradiction type | n |
|---|---|
| scope_qualification | 11 |
| reimbursement_flip | 10 |
| named_person_dependency | 10 |
| hedged_non_support | 10 |
| direct_negation | 10 |
| temporal_inconsistency | 10 |
| category_mismatch | 9 |

**The type mix is stratified by design, not sampled.** Independent per-record sampling produced `named_person_dependency` once in 350 records, which makes per-type recall unmeasurable — and per-type precision and recall is Phase 4's headline deliverable. The corpus therefore deals a balanced pool of target types across the records selected to carry contradictions.

The trade-off must be stated in the evaluation: per-type recall is estimable, but the type *prior* is a property of the experiment design and is not evidence about how contradictions are distributed in real submissions. No aggregate detection rate from this corpus should be reported as a field estimate.

### By subtlety

| Level | Meaning | n |
|---|---|---|
| 1 | blatant — the text plainly says the opposite | 23 |
| 2 | requires reading — a condition, veto or reimbursement flips it | 34 |
| 3 | requires domain knowledge — the form's own exclusions, or case law | 13 |

### By IR35 test

| Test | n |
|---|---|
| business_on_own_account | 21 |
| substitution | 15 |
| financial_risk | 13 |
| mutuality_of_obligation | 7 |
| part_and_parcel | 6 |
| control | 4 |
| contract_basis | 2 |
| office_holder | 2 |

59 of 70 contradictions undermine an outside-leaning answer. Those are the audit risk under HMRC's reasonable-care duty — a justification that quietly guts the answer supporting an *outside* determination — and Phase 4 reports recall on them separately.

Pair coverage: 23 of 31 pairs carry at least one contradiction. Pairs with fewer than three instances are not separately evaluable; Phase 4 reports per *type* and per *test*, which have workable counts, rather than per pair.

## Anomalies (not contradictions)

| Kind | n |
|---|---|
| blank | 6 |
| non_responsive | 10 |
| routing_violation | 5 |

Kept separately labelled. A blank rationale is a completeness failure the form itself cares about — 'Your ESQ will be rejected if not all fields have been completed' — and a reviewer needs to see it, but scoring it as a contradiction would flatter the detector.

## Pipeline verification

- Schema validation: **350/350 records valid** (no value outside its declared domain, no outcome leakage).
- Warnings raised (expected — these are the injected anomalies):
  - `missing_justification`: 6
  - `justification_too_short`: 4
- De-identification: **517 entities removed** from free text across 286 records, plus field-level treatment on every record. The stage is exercised on the whole corpus rather than assumed.

## Files

| File | Contents |
|---|---|
| `esq_synthetic_v1.jsonl` | De-identified records. The canonical modelling input. |
| `ground_truth_v1.jsonl` | Status labels with full score breakdown, planted contradictions, anomalies. |
| `generator_state_v1.jsonl` | Per-record generator state. Re-render in any register for the Phase 4 style-invariance test. |

