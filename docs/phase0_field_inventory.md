# Phase 0 — Field Inventory and Data Contract

**Source:** `Copy of Employment Status Questionnaire ESQ.xlsx` — single sheet,
`Employment Status Questionnaire`, footer marked *"ESQ form updated June 2026"*.
Extracted 2026-09-03 with `openpyxl`, reading cell values, merged ranges and
**data validation lists** (the dropdowns are where the authoritative value
domains live — they are not visible in the printed form).

**Scope constraint:** this system is decision support. It flags inconsistency
and offers an indicative signal; it does not issue or override a Status
Determination Statement. See the header of `config/schema.yaml`.

---

## 1. Shape of the form

| Section | Content | Fields |
|---|---|---|
| 1. Worker Details | Name, address, phone, web presence, engagement dates, contract existence | 8 questions (12 cells) |
| 2. Company Status | Third-party/PSC route, company name, number, relationship, trading history | 5 |
| 3. Worker Status | Direct engagement flag (only if §2 = No) | 1 |
| 4. Employment Status Questions | Role title, duties narrative, **Q4.1–Q4.31** | 2 + 31 |
| 5. Declaration | School/division, engaging officer, contact details | 5 |
| IR35 Team use only | Assessment date, assessor, HR authorisation, outcome | 4 |

**Totals: 64 addressable fields.** 36 structured, 32 free-text, 26 metadata
(some fields count in more than one row of the summary above only where a
question has both a dropdown and a rationale box).

The critical structural fact for this project: **30 of the 31 Section 4
questions have a paired "Please explain the reason/rationale for your answer in
4.x" free-text box directly beneath the dropdown.** That pairing is the entire
substrate for contradiction detection — the form is, by design, 30 matched
(claim, justification) pairs. Q4.1 is the sole exception and has no rationale
box.

---

## 2. Field inventory by IR35 test

Value domains below are **verbatim from the Excel data-validation lists**,
normalised only for trailing whitespace. Where an option's wording is odd
(inconsistent capitalisation, `a percentage of the organisations profit`), it is
preserved — the real form is the contract.

### Office holder — determinative on its own

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.2 | Will the worker be an office holder? | structured | Yes / No |
| 4.2r | rationale | free-text | — |

### Personal service / right of substitution

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.3 | Has the worker's business sent a substitute during this engagement? | structured | Yes - you accepted them / Yes - you did not accept them / No - they have not sent a substitute / Not applicable - work has not started |
| 4.4 | Will the worker pay their substitute? | structured | Yes / No |
| 4.5 | Do you have the right to reject a substitute? | structured | Yes / No |
| 4.6 | Has the worker paid another person to do a significant portion? | structured | Yes / No |
| 4.7 | If a substitute is provided, would the worker have to pay them? | structured | Yes / No |
| 4.3r–4.7r | rationales | free-text | — |

Q4.3 drives a branch: *accepted them* → 4.4; *did not accept* → 4.6; *no
substitute* / *not applicable* → 4.5. Q4.5 then branches Yes → 4.6, No → 4.7.
Blank fields off the active route are legitimate, not missing data — the schema
records routing so validation doesn't false-positive on them.

### Control

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.8 | Can the worker be moved from the agreed task? | structured | Yes / No - they would have to agree / No - that would require a new contract or formal working arrangement |
| 4.9 | Can you decide **how** the work is done? | structured | Yes / No - the worker solely decides / No - the University and the worker agree together / Not Relevant - it is highly skilled work |
| 4.10 | Can you decide the **schedule** of working hours? | structured | Yes / No - the worker solely decides / No - the engager and the worker agree / No - the work is based on agreed deadlines |
| 4.11 | Can you choose **where** the work is done? | structured | Yes / No - the worker decides / No - the task sets the location / No - some work has to be done in an agreed location and some can be done at the workers choice |
| 4.8r–4.11r | rationales | free-text | — |

Note Q4.9's fourth option, *"Not Relevant - it is highly skilled work"*. This is
a CEST-style escape hatch and it is where I expect the highest rate of
undermining justifications — a manager selects it in good faith and then
describes daily supervision.

### Financial risk

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.12 | Must the worker buy equipment before being paid? | structured | Yes / No |
| 4.13 | Must the worker fund vehicle costs? | structured | Yes / No |
| 4.14 | Does the worker pay for materials **without being reimbursed**? | structured | Yes / No |
| 4.15 | Any other costs funded before payment? | structured | Yes / No |
| 4.16 | How will the worker be paid? | structured | An hourly/daily or weekly rate / A fixed price for the project / A fixed amount for each piece of work completed / A percentage of the sales the worker generates / a percentage of the organisations profit or savings |
| 4.17 | Must the worker put unsatisfactory work right? | structured | Yes - unpaid & they would have extra costs the University would not pay for / Yes - unpaid but their only cost would be loss of opportunity / Yes - in their usual hours at their usual rate or fee / No - the work is time-specific or for a single event / No |
| 4.12r–4.17r | rationales | free-text | — |

4.12, 4.13 and 4.14 carry help text that **excludes** specific costs (laptops,
phones, commuting fuel, minor consumables). That exclusion is a rich source of
subtle contradiction: a "Yes" justified entirely by an excluded cost is wrong
without being a plain negation. Those are my subtlety-level 2–3 cases.

### Part and parcel / integration

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.18 | Paid-for benefits from the University? | structured | Yes / No |
| 4.19 | Responsible for hiring/dismissing/appraising University workers? | structured | Yes / No |
| 4.20 | How does the worker identify themselves to students/partners? | structured | They work for the organisation (the University) / They are an independent worker acting on behalf of the organisation / They work for their own business / This would not happen |
| 4.18r–4.20r | rationales | free-text | — |

### In business on own account / exclusivity / IP

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.21 | Does the contract stop similar work for others? | structured | Yes / No |
| 4.22 | Must the worker ask permission to work elsewhere? | structured | Yes / No |
| 4.23 | Any ownership rights (IP, copyright, trademarks, patents, image rights)? | structured | Yes / No |
| 4.24 | Do the rights belong to the University? | structured | Yes / No |
| 4.25 | Option for the University to buy rights for a separate fee? | structured | Yes / No |
| 4.30 | Will this take up the majority of the worker's available working time? | structured | Yes / No |
| 4.31 | Similar self-employed work for other clients in the last 12 months? | structured | Yes / No |
| rationales | | free-text | — |

4.23 branches: Yes → 4.24, No → 4.25. 4.24 branches: Yes → 4.26, No → 4.25.

### Mutuality of obligation / continuity

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 4.26 | Previous contract with the University? | structured | Yes / No |
| 4.27 | Is this the first in a series of contracts? | structured | Yes / No |
| 4.28 | Will this start immediately after the previous one ended? | structured | Yes / No |
| 4.29 | Does the contract allow extension? | structured | Yes / No |
| rationales | | free-text | — |

### Contract and engagement metadata

| Ref | Field | Type | Value domain |
|---|---|---|---|
| 1.7 | Length of Engagement (From / To) | metadata | dates |
| 1.8 | Is there, or will there be, a contract in place? | structured | Yes - contract already in place and provided with ESQ / Yes - contract not yet in place and will be provided later / No |
| 1.8a | What form does the contract take? | free-text | — |
| 2.1 | Engagement through a third party / company / agency? | structured | Yes / No |
| 2.4 | Relationship between contracting entity and worker | free-text | — |
| 2.5 | If company in worker's own name, how long in business? | semi-structured | no validation — free text |
| 3.1 | Engaged directly, not through a company? | structured | Yes / No |
| 4.0a | Contract or role title | metadata | — |
| 4.0b | **Description of Duties of Engagement** | free-text | — (merged F64:J73 — the largest box on the form) |

### PII fields (§1, §2.2–2.3, §5, and the IR35-team block)

Title, first name, surname, four address lines, telephone, internet presence,
company name, registered company number, engaging officer name, contact name,
phone, email, assessor name, HR signature. All classified in `schema.yaml` as
`direct_identifier`, `quasi_identifier` or `sensitive_commercial`. These are the
de-identifier's field-level targets.

School/Division is a quasi-identifier that I want to **keep in hashed form**,
because it is the cohort variable for the Phase 4 disparity analysis — a
detector that works well for IT contracts and badly for visiting lecturers is a
real fairness problem, and I can't measure it if I throw the division away.

---

## 3. Contradiction-checkable pairs

31 pairs are defined in `config/contradiction_pairs.yaml` — the 30 Section 4
question/rationale pairs plus 1.8/1.8a. Each carries a plain-English description
of what a contradiction looks like, the contradiction types the Phase 1
generator is permitted to plant into it, and lexical cues for the Phase 2 rule
baseline.

Seven are marked `priority: high` — these are the pairs where a contradiction
most directly undermines a determination, and where Phase 4 will report recall
separately:

| Pair | Why it matters |
|---|---|
| 4.5 right to reject a substitute | The canonical failure. "No right to reject" + "we specifically need this individual, nobody else has the expertise". An unfettered right of substitution is close to dispositive for *outside*; undermining it flips the determination. |
| 4.9 who decides how | *"Not Relevant - it is highly skilled work"* + a narrative of supervision and sign-off gates. |
| 4.10 who decides the schedule | *"based on agreed deadlines"* + "must be on site Monday to Friday, 9 to 5". |
| 4.14 unreimbursed materials | "Yes" + "costs are recharged to the department on production of receipts". A reimbursed cost is not financial risk. |
| 4.16 payment basis | "A fixed price for the project" + "invoiced monthly at a day rate". Category mismatch, and a direct CEST input. |
| 4.20 how they identify themselves | "They work for their own business" + a sussex.ac.uk address and a staff badge. |
| 4.30 / 4.31 other clients and time share | "No, not the majority of their time" + "full-time, five days a week for nine months"; or "yes, other clients" + "we are their only client". |

**Contradiction taxonomy** (used for ground-truth labels in Phase 1, and for
per-type precision/recall in Phase 4):

`direct_negation`, `scope_qualification`, `named_person_dependency`,
`reimbursement_flip`, `category_mismatch`, `temporal_inconsistency`,
`hedged_non_support`, `non_responsive`.

Splitting the taxonomy this way is deliberate. `direct_negation` is what a
keyword baseline can catch; `scope_qualification` and `hedged_non_support` are
what should separate a transformer from a keyword list. If the transformer
doesn't beat the baseline on those two types specifically, the method comparison
in the assignment has a real finding to report rather than a foregone one.

**Cross-field checks** (structured-vs-structured, and duties-narrative-vs-many)
are defined separately in the same file. They are rule-based by nature and are
not part of the NLI evaluation, but they belong in the same reviewer view.

---

## 4. Design choices, and what I rejected

**Stable field IDs (`q4_05_right_to_reject`) rather than cell references or
question numbers as keys.** Cell references break the moment anyone inserts a
row. Question numbers are unstable across form versions and, as it turns out,
aren't even reliable inside this version (see §5). The ID is the contract; `cell`
and `form_ref` are provenance metadata that can be edited freely.

*Rejected:* using the question text as the key. Reads better, but the text is
long, contains punctuation that fights YAML, and changes between form revisions
for cosmetic reasons.

**Value domains lifted from Excel data validations rather than retyped.** The
dropdowns are the only place the exact option strings exist. Retyping them
introduces silent mismatches that would break structured-answer parsing on real
submissions in ways that are hard to debug.

**Routing captured in config.** Without it, every skipped branch looks like a
missing answer and the completeness check becomes useless noise.

**Contradiction pairs in a separate file from the schema.** The schema describes
what the form *is*; the pairs file describes what the detector *looks for*. They
change for different reasons and on different timescales — the schema changes
when HR revises the form, the pairs file changes when we learn something about
detection.

**Two detector classes, not one.** Paired NLI handles (structured, adjacent
free-text). A rule engine handles routing violations, completeness, and the
duties narrative against many answers at once. Forcing the second class through
an NLI model would be a worse fit and harder to explain to a reviewer.

**Outcome fields declared in the schema and explicitly stripped, not merely
omitted.** The IR35-team block sits in the same sheet as the inputs. Declaring it
with `role: outcome` means ingest removes it by an assertion I can unit-test,
rather than by hoping nobody widens a cell range later. This is the leakage
guard.

---

## 5. Things I need you to clarify

**a. Broken/orphan data validations.** Six validation rules in the workbook
point at `#REF!` — they reference ranges that were deleted. They sit at rows 37,
123, 138–143 and 181–187. Separately, column M holds two orphan option lists
that no live dropdown uses:

- `M10:M13` — *Regular payment based on an hourly, daily, weekly, or monthly
  rate / Piece rate or work measure / Basic salary / Fixed price*
- `M16, M18, M20` — three working-time patterns (regular days-hours-shifts / freedom
  to choose / no set hours but agreed deadlines)

These look like remnants of an earlier version, **or** like newer CEST wording
that has been pasted in but not yet wired up. Which is it? It matters because
`M10:M13` is a different four-option payment taxonomy from the five-option list
actually attached to Q4.16, and I need to know which one the real form will use
when this is pointed at live submissions.

**b. Question numbering.** Cells `B132`, `B204` and `B274` read `4.1`, `4.2` and
`4.3`. These are Excel numeric cells that dropped a trailing zero — they are
Q4.10, Q4.20 and Q4.30. I've modelled them as such. Please confirm, because it
means Q4.1 and Q4.10 are visually indistinguishable on the printed form and a
manager could plausibly cite the wrong one.

**c. Q4.5 polarity.** *"Do you have the right to reject a substitute?"* — I've
modelled **"No" as the outside-leaning answer** (no veto = unfettered right of
substitution). The form's routing is consistent with that reading. Confirm the
IR35 team reads it the same way; if not, the polarity flips for the most
important pair in the set.

**d. Q4.15 input cell.** Its data validation spans `I169:J170` where every other
question's spans a single merged row. Cosmetic in the template, but the ingest
adapter needs to know whether the answer lands in `I169` or `I170`.

**e. How do submissions actually arrive?** The form says email the .xlsx to
`IR35info@sussex.ac.uk`. Is that still the route, or is there a workflow system
(Ivanti, a Forms front-end, InfoBase) behind it? This decides whether `ingest`
parses .xlsx or consumes JSON, and whether free-text arrives with formatting.

**f. Realistic free-text length.** No character limits are set on the rationale
cells. Typical and worst-case lengths matter: a cross-encoder NLI model truncates
at 512 tokens, so if managers routinely write 800 words in the duties box I need
a chunking strategy, and I'd rather design for it now than discover it in Phase 3.

**g. The `inside`/`outside` label for synthetic data.** The ESQ itself doesn't
produce a determination — HR runs CEST afterwards. So for Phase 1 I need a
labelling function, and there are two honest routes:

1. **Rule engine first:** implement a documented, CEST-approximating rule set,
   generate answers freely, derive the label. Labels are causally consistent with
   the answers, and the rule engine doubles as an interpretable Phase 2 baseline.
   Risk: my rule engine becomes the ground truth, so Phase 2's rule baseline is
   evaluated against its own logic and scores unfairly well. I'd mitigate by
   using a deliberately *different* and simpler rule set for the baseline, and
   saying so plainly in the write-up.
2. **Label first:** sample `inside`/`outside`, then generate answers conditioned
   on it. Cleaner separation from the baseline, but risks generating internally
   implausible combinations unless the conditioning is careful.

I lean towards (1) with the mitigation, because a dataset where the answers and
the label are causally linked is more defensible than one where they're
correlated by construction. Your call — it affects how much of Phase 2's result
we can claim.

**h. Class balance.** You asked for ~50 of 350 inside (14%). Two things worth
weighing: HMRC/CEST outcomes at universities often skew more inside than that,
particularly for visiting lecturers; and 14% is a thin positive class for the
status classifier, though it's fine for the contradiction detector (which is a
separate, better-balanced target at 15%). Happy to build to 14% — just confirm
it's a deliberate reflection of Sussex's mix rather than an estimate.

**i. Does Q4.1 really have no rationale box?** It's the only Section 4 question
without one. If that's an oversight in the template rather than a decision, the
live form may gain one, and I'd rather the schema anticipate it.

---

## 6. Information Governance / DPO flags

None of these block the synthetic build. All of them need clearing **before this
touches a real submission.**

1. **Article 22 posture.** An employment status determination has legal effect
   and affects tax treatment, so a solely automated decision is prohibited
   without an Article 22(2) basis. The design keeps a human decision-maker
   throughout and the tool never issues a determination. IG should confirm the
   posture and record it. My view: this is a *decision-support* tool and the
   determination stays with the IR35 team, but that conclusion needs to be
   theirs, not mine.
2. **DPIA.** Processing employment/tax-relevant personal data with an automated
   system that influences a decision about a person is very likely to require a
   DPIA under Art.35. Should be started early, not retrofitted.
3. **Lawful basis and purpose limitation.** The ESQ is collected to make a status
   determination. Using the same data to train or evaluate a model is arguably a
   *different* purpose. Compatible-purpose assessment needed. This is precisely
   why Phase 1 is synthetic.
4. **Third-country transfer.** Constraint 2 (swappable backend) exists partly for
   this. If a hosted API is used on real data, submissions containing named
   individuals and day rates leave University control. IG will want the
   local-model path to be the production default, and I've designed for
   local-only production accordingly.
5. **Retention of reviewer decisions.** Phase 5 captures accept/dismiss actions.
   That log is itself personal data about *the reviewer* as well as the worker,
   and needs a retention period and a stated purpose (model improvement? audit
   evidence of reasonable care? they have different retention answers).
6. **Special category and criminal-offence data.** Rationale boxes are free text
   and managers do occasionally write things like "they're on long-term sick" or
   reference a DBS outcome. The de-identifier must handle Art.9 and Art.10
   content, not just names. Worth telling IG that this risk exists in the
   *existing* process too — the tool surfaces it rather than creating it.
7. **Transparency to the worker.** The worker is the data subject of the
   determination but does not complete the form. Whether the privacy notice
   covers automated assistance in the process is a question for the DPO.
8. **Records of reasonable care.** HMRC requires reasonable care in each
   determination. If the tool flags a contradiction and a reviewer dismisses it,
   that dismissal is now part of the audit trail — and arguably *strengthens*
   the reasonable-care position. Worth raising with the IR35 team as a benefit,
   but also as a discoverability risk they should enter into knowingly.

---

## 7. Suggested first commit

```
Phase 0: extract ESQ schema and define contradiction pairs

Parse the June 2026 ESQ template and derive a config-driven data contract:
64 fields with types, Excel-authoritative value domains, routing and PII
classification, mapped to eight IR35 tests. Define 31 structured/free-text
contradiction pairs plus six cross-field checks.

No data generated. No model code. Schema only.
```
