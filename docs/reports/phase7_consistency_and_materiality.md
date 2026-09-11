# Phase 7 — Tick-Box Consistency Checks and Materiality

Two additions to the Phase 5 reviewer app, both requested directly by the
project owner after reviewing Phase 6: a way to flag when two tick-box
answers on the same form disagree with each other (not just a tick-box
against its own written explanation), and "an indication if this is one to
be worried about falling into IR35."

The second half of that request is the one worth being careful about, and
this report spends most of its length on it, because the obvious way to
build it is also the way this project has spent six phases arguing against.

## What was asked for, and the design question it raised

The project owner's request, in full: *"consistency check with an
indication if this is one to be worried about falling into IR35. it would
be good for the tool to flag to reviewers where the inconsistency is, both
in the tick box vs text, as well as where this could impact the IR35
determination."*

"An indication of whether to be worried about falling into IR35" can mean
two quite different things:

1. **A status or lean for this submission** — "this one looks like it's
   heading inside IR35." Computed per-record, from this record's specific
   answers. This is exactly the RAG-status idea raised and explicitly
   rejected earlier in this project (see the `AskUserQuestion` decision
   recorded in the project history: *"Don't build this — explore a safer
   alternative instead"*). Building it here, under a different name, would
   undo that decision rather than honour the direction that replaced it.
2. **A description of how much a *category* of mismatch typically matters**
   — "a mismatch on personal service or control is worth taking seriously,
   because case law treats those as foundational; a mismatch on which
   equipment the worker used is a smaller factor." Computed once, from the
   rulebook, the same for every submission that touches that test. This
   does not predict anything about the record in front of the reviewer — it
   describes the test itself.

This report calls the second one **materiality**, to keep it verbally
distinct from a **status**. The rest of this section is the argument for
why materiality is what was actually asked for, not a hedge that waters the
request down.

Read literally, "this could impact the IR35 determination" is about
*impact* — how much a given fact matters to the outcome in general — not
about *direction* — which way this outcome is trending. A reviewer who
already knows a flag exists (section 3 already tells them that) gains
something real from being told "this is one of the two tests case law
treats as foundational" that they do not gain from being told "flags like
this are usually right 1 time in 9" (the existing confidence band, which
is about evidence strength, not stakes). Materiality answers a question the
band cannot: *if this flag turns out to be genuine, how much does it
matter?* That is a fair reading of "worried about falling into IR35" that
adds real information without crossing into a determination.

A status/lean was rejected for the same three reasons raised earlier in the
project and unchanged by this new request: automation bias (HMRC's own CEST
tool is the documented example of reviewers anchoring on a machine-produced
status), it would only reflect the half of the evidence this tool actually
scores (tick-boxes and their justifications) while ignoring "all the
circumstances" case law also requires, and no rule engine in this codebase
has been calibrated against real HMRC outcomes — `CestRuleEngine` produces
the synthetic corpus's training label, not a validated prediction. None of
that changed between the earlier decision and this request.

This interpretation was stated to the project owner before building, so it
could be corrected if it had misjudged the request. It was not corrected,
and the build below followed it.

## Part 1 — Tick-box vs tick-box consistency checks

Phase 0's `config/contradiction_pairs.yaml` already described six
cross-field checks in plain English, under `cross_field_checks:`, before
any detection code for them existed — a documented gap, not new scope.
Three had a shape simple enough to state as a small, honest boolean
condition:

- **`x_route_section2`** — the form says the engagement is *not* through an
  intermediary (`q2_01_via_third_party = "No"`), but a company name or
  registration number was filled in anyway.
- **`x_no_financial_risk_but_fixed_price`** — paid a fixed price for the
  project (the kind of payment basis that carries real financial risk), but
  every individual risk indicator (equipment, vehicle costs, unreimbursed
  materials, other costs, put-right-defects-unpaid) was answered "no."
- **`x_started_but_not_applicable`** — the work has already started, but the
  substitution question was answered "not applicable, work has not
  started."

The other three (`x_route_substitution`, `x_duties_vs_control`,
`x_missing_justification`) are left without an executable condition on
purpose: two of them duplicate checks `src/ingest/validate.py` already
performs under different names, and the third needs a real text comparison
(`method: document_level_nli`), not a boolean rule — building a fake
approximation of that would check something other than what its
description claims. This is recorded directly in the config file's
comments, next to each check, so the gap stays visible rather than being
silently duplicated or silently faked.

**The condition language is deliberately small** — three clause types
(`eq`, `in`, `any_not_empty`) combined with `when_all` — so that adding a
seventh check is a config edit a non-programmer can read and verify, per
this project's constraint 5 (no field names hardcoded in Python outside
config). `src/models/cross_field.py` is the evaluator; it raises on an
unrecognised operator rather than silently doing nothing, so a config typo
surfaces immediately in testing instead of quietly never firing.

These findings carry a severity (`low` / `medium` / `high`, authored per
check in config) rather than a confidence band — a boolean condition either
holds or it does not, so there is no calibration question to hedge, unlike
the statistical scores section 3 shows.

## Part 2 — Materiality

`src/review/materiality.py` attaches a short, static "why this matters"
line to every flag in section 3 and every finding in the new section 4.
The tiers are a direct restatement of the case-law hierarchy already
written into `config/ir35_weights.yaml`'s own comments since Phase 2 —
personal service (substitution) and control as the "irreducible minimum"
(*Ready Mixed Concrete*), financial risk next, the rest as "all the
circumstances" — not a new judgement invented for this phase:

| Tier | Applies to | Wording |
|---|---|---|
| Foundational | substitution, control (weight ≥ 0.25) | "One of the two foundational tests" |
| Major factor | financial_risk (weight ≥ 0.15) | "A major factor" |
| Contributing | part_and_parcel, business_on_own_account, mutuality_of_obligation | "One of several contributing factors" |
| Gate | any field named in a determinative gate (`ir35_weights.yaml: gates`) | "Could be decisive on its own" — takes priority over a test's weight tier whenever it applies |

`contract_basis` and any field with no configured weight get no tier at
all, rather than a guessed one.

### Why this is safe to add without reopening the status question

The function's signature is the safety argument, made structural rather
than just documented:

```python
def materiality_for(
    review_config, ir35_weights_config, *, ir35_test=None, field_id=None
) -> MaterialityTier | None:
```

There is no `record` parameter and no `score` parameter — not "we chose
not to use them," but **there is nowhere to pass them**.
`tests/test_materiality.py::test_function_signature_has_no_record_or_score_parameter`
checks this with `inspect.signature`, the same structural-guarantee pattern
Phase 5 used for `test_assess_module_never_imports_the_labelling_engine`:
a fact about what the code *can* do, not a promise about what it currently
does. Calling `materiality_for` twice with the same test id always returns
an equal result (`test_same_test_id_always_returns_the_same_tier`) — there
is nothing in the function that could vary it. And
`test_tier_labels_never_mention_inside_or_outside` scans every configured
tier's label and explanation text and fails if the words "inside IR35" or
"outside IR35" ever appear — the same drift-detector pattern
`test_scope_banner_never_mentions_a_status` already applies to the app's
banner text, extended to this new wording.

The one deliberate judgement call: a gate field (e.g.
`q4_03_substitute_sent`) gets the gate tier **regardless of its current
value** — because it names a fact case law treats as potentially
decisive, not because of which way this particular answer leans. This is
structural, not value-dependent, and is the correct reading of "could be
decisive" — a field does not stop being capable of settling the question
just because, on this form, it happens to point the ordinary way.

### What a reviewer actually sees

A flag card in section 3 now reads, underneath its existing confidence
band:

> **Why this matters:** A major factor. Financial risk is weighted heavily
> in most determinations, though it is not treated as part of the
> essential minimum alongside personal service and control.

Deliberately grayscale — `#2c2c2c` on `#f7f7f5`, no colour — so it cannot
be visually confused with the confidence band above it, which keeps its
existing blue/amber/red. The two lines answer different questions ("how
strong is the evidence" vs. "how much would it matter if true") and
conflating their colours would blur that distinction the way the rejected
RAG status would have. A sidebar expander,
*"Why this matters" — what materiality does and does not tell you*, spells
this out for a reviewer in plain language, including the structural
guarantee above.

## Automated tests

`tests/test_cross_field.py` (11 tests) — every configured check fires and
does not fire on the right inputs, condition-less checks are silently
skipped rather than erroring, a regression guard fails loudly if a new
check is added to config without anyone deciding which bucket it belongs
in, and findings never assert a determination (same forbidden-phrase guard
used throughout this project).

`tests/test_materiality.py` (16 tests) — the structural signature
guarantee, correct tier for every weighted test and every gate field, gate
priority over test weight, determinism, and the "never mentions inside or
outside IR35" scan across every configured test.

`tests/test_review.py` gained 6 tests covering `assess_submission`'s new
`ir35_weights_config` parameter: materiality is `None` when the config is
omitted (so a caller that only wants Phase 5 behaviour is unaffected),
materiality is attached across the sample corpus when it is supplied, a
minimal hand-built record produces the expected cross-field finding with
gate-tier materiality, the same import-guard pattern extended to the two
new modules, and cross-field findings never assert a determination.

`tests/test_ui_app.py` gained 3 tests: the default sample's known
materiality line reaches the rendered page, the default sample's (known,
checked-in-advance) absence of a cross-field finding renders the "nothing
found" state without an exception, and a hand-built pasted record that
trips `x_started_but_not_applicable` produces a section-4 card whose
accept control writes a decision with `method: "cross_field"` to the same
decision log section 3 uses.

**267 tests pass across the whole project as of this phase**, up from 232
at the end of Phase 5.

## Rejected alternatives

**A combined per-submission status ("likely inside" / "likely outside").**
The request this phase started from, read one way. Rejected for the three
reasons in "What was asked for" above — automation bias, partial evidence,
no real-world calibration — none of which changed since the earlier
rejection of the RAG-status idea.

**Colouring the materiality line the same way as the confidence band**
(e.g. red for "foundational," matching "High priority"). Rejected because
it would make two independent signals look like one combined score,
recreating by colour what was rejected by wording — a reviewer skimming
colours alone should not be able to reconstruct anything resembling a
status.

**A single combined "materiality × confidence" score.** Considered and
rejected immediately: multiplying an evidence-strength number by an
importance-of-category number produces exactly the single combined figure
this whole design avoids, and it would carry false precision on top —
neither factor is itself a calibrated probability.

**Reusing a new, parallel decision-log table for cross-field findings.**
Considered, rejected in favour of reusing `DecisionLog` / `ReviewDecision`
as-is (with the check's id standing in for `pair_id` and its touched
fields, joined, standing in for `field_id`) — a second audit trail for what
is, for governance purposes, the same event (a reviewer looked at a flagged
inconsistency and made a call) would fragment the record Phase 5 built this
log to keep in one place.
