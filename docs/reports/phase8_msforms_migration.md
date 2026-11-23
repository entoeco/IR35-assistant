# Phase 8 — Microsoft Forms Migration

HR moved manager data collection from the June-2026 Excel template to a
Microsoft Forms form. The request: *"how can the process we have built here
be adapted to integrate into the MS Form data entry?"* This report covers
what actually changed on the live form, what was deliberately left alone
and why, and the one real behavioural consequence worth a reviewer's
attention. The project owner confirmed the dropped/changed fields were
intentional on HR's part, so this phase treats the live form as correct and
adapts around it rather than questioning it.

## What changed

Working from a real test submission (a Microsoft Forms "View results" PDF
and the matching spreadsheet export), three kinds of change showed up.

**The export shape itself is simpler and more reliable.** One header row,
one row per response, one column per question. No more row-position
heuristics, merged cells, or label-text searches — the fragility the
June-2026 template's converter (`scripts/convert_esq_xlsx_to_json.py`) was
built to work around is gone for submissions coming through the new form.

**Three genuinely new questions exist only on the live form:** is the
worker providing their services through an Umbrella Company (with a
free-text rationale box), whether the individual is registered for Self
Assessment with HMRC, and the worker's address collected as one multi-line
box rather than four separate line/postcode cells. All three were added to
`config/schema.yaml` as new, additive fields —
`q1_09_umbrella_company`/`q1_09_umbrella_company_details`,
`q3_02_self_assessment_registered`, and `q1_04_address_full`. Nothing
existing was renamed or removed to make room for them.

**Four questions from the June-2026 template are not asked on the live
form at all:** 2.4 (relationship between the contracting entity and the
worker), 2.5 (years in business), 3.1 (engaged directly, in the worker's
own name), and 5.2 (name of the engaging officer). These stay in
`config/schema.yaml` unchanged, for a specific reason: `config/
generation.yaml`'s `identity_field_map`/`engagement_fields` and
`src/generate/generator.py` reference them by id to build the synthetic
corpus, and the already-committed `data/synthetic/esq_synthetic_v1.jsonl`
(350 records) and every Phase 2–4 evaluation result were built against
them. Deleting them would break the generator and invalidate that work for
no benefit — a real submission converted by the new converter simply omits
them, and nothing in `src/ingest/validate.py` treats that as an error:
`required_when` in schema.yaml is documentation, never enforced in code,
and the routing/justification/domain checks only ever look at fields that
are actually present.

## Wording drift, and the one change that is not just wording

A few value_domain options came back from the live form with cosmetic
differences from the exact strings in `config/schema.yaml` — stray or
doubled whitespace, a trailing period, a capitalisation/apostrophe
difference. These are handled in the new converter
(`scripts/convert_esq_msforms_xlsx_to_json.py`'s `normalize_domain_value`)
by matching case- and punctuation-insensitively and substituting the
canonical schema string, rather than by editing schema.yaml. That is
deliberate: those exact strings are load-bearing keys in `config/
text_bank.yaml` and already appear verbatim throughout the committed
synthetic corpus, so changing them in schema.yaml would silently break
corpus regeneration without fixing anything for real submissions. If the
converter cannot confidently match a value to a declared option, it leaves
the raw value alone — `RecordValidator` then reports a
`value_not_in_domain` error, which is the right failure mode: a human looks
at it, rather than the script guessing.

One change is not cosmetic. The live form merged two of question 4.3's
("would the worker's business send someone else to do the work") four
options — "No - they have not sent a substitute" and "Not applicable - work
has not started" — into a single new option, "No - it has not happened".
Both original options routed to the same next question (4.5), so the merge
changes no routing, and it was added to `q4_03_substitute_sent`'s
`value_domain` in schema.yaml (additively — the original two options stay,
for the synthetic generator's sake) rather than normalized away.

**Consequence:** the cross-field check `x_started_but_not_applicable`
(`config/contradiction_pairs.yaml`) — one of the project's three
tick-box-vs-tick-box checks, catching "4.1 says work has started, but 4.3
says it hasn't" — keys on the literal string "Not applicable - work has not
started". The live form no longer offers that exact option, so this check
can no longer fire on any real submission, only on synthetic data. This is
flagged here as an accepted, known gap rather than left to be discovered
silently: if this check matters enough to keep working on real data, the
live form's new "No - it has not happened" option would need its own
equivalent check (comparing it against 4.1, the same way the original did).

## What was deliberately not done

`q1_09_umbrella_company`/`q1_09_umbrella_company_details` is structurally
the same shape as the existing 1.8/1.8a contract pair — a tick-box with a
free-text rationale, a natural candidate for the same kind of
tick-box-vs-text contradiction detection the rest of the tool does. It was
**not** added to `config/contradiction_pairs.yaml` or `config/
text_bank.yaml`'s contradiction templates in this phase. Every other pair
in this project went through a deliberate design pass — polarity, what an
"outside-leaning" answer looks like, which contradiction types are
plausible for it, generator support for injecting them — and this field
deserves the same treatment rather than being bolted on as a side effect of
a form migration. **Recommended as a follow-on.**

## The new converter

`scripts/convert_esq_msforms_xlsx_to_json.py` converts a Microsoft Forms
export into the same schema-field-id-keyed JSON shape the app already
expects. It is a separate script from the June-2026 template's converter —
the two source shapes have little in common — but follows the same
contract: never populates the four outcome-only fields, never guesses a
value for a blank cell, and runs the result through `RecordValidator`
before writing it out. It additionally:

- Maps the export's branching rationale columns (one column per possible
  triggering option, at most one populated per response) back onto the
  single rationale field id the schema expects.
- Casts fields Microsoft Forms exports as numbers but which are really
  identifiers — telephone numbers, company numbers — back to strings.
- Derives `record_id` from the export's own numeric `Id` column rather than
  the submitting manager's name or email, both of which the export also
  carries (as the `Name`/`Email` columns) but which identify the manager,
  not the worker, and are not part of the data contract.
- Converts every response row in one export, since a single Microsoft
  Forms export can hold many.

Verified end to end against the real test submission used for this
analysis: zero schema errors, one wording normalization applied (4.17's
stray double-space-plus-period), and the expected set of
justification-too-short warnings reflecting the test data's deliberately
short placeholder answers.

Tests: `tests/test_convert_esq_msforms_xlsx.py` (header matching and
normalization, nbsp handling, branching-column merge, value-domain
normalization including the no-op and no-match cases, numeric-to-string
coercion, record-id derivation, multi-response handling, and an explicit
check that no column mapping can populate the four retired fields).

## What this did not touch

`src/ui/app.py`, `src/generate/generator.py`, `config/generation.yaml`,
and the committed synthetic corpus are all unchanged. A real submission
converted by the new script is a drop-in input to the existing reviewer
app, pipeline, and validator — nothing downstream of the JSON needed to
know which form it came from.
