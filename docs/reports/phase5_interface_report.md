# Phase 5 — Reviewer Interface

A Streamlit app that takes one ESQ submission and shows a reviewer where the
tick-box answers and the written explanations next to them seem to disagree,
so they know which parts of a long form to check first instead of reading
every line of every submission with the same amount of attention.

Run it with:

```
streamlit run src/ui/app.py
```

## What it does, in plain terms

1. You load a submission — a sample from the test data, a pasted block of
   text, or an uploaded file.
2. The app checks it for personal data first and removes it, then checks the
   form is filled in properly.
3. It compares every tick-box answer against the explanation written next to
   it, and shows you the ones that look inconsistent — worst first.
4. Each flag comes with a plain-language reason: which question, what was
   ticked, what was written, and why the two do not sit well together.
5. You accept the flag (it needs a closer look) or dismiss it (it does not),
   optionally with a note. That decision is saved, with your name and the
   time, so there is a record of what was looked at and by whom.

Nothing in this tool decides employment status. That is on purpose, not an
oversight — see "The determination is never shown" below.

## Why a confidence band, not a percentage

Phase 4 checked whether the numbers these methods report can be trusted at
face value, and found they cannot. The rule-based method, for example,
reported "46% confident" in a range where the true rate — checked against
the test data — was closer to 11%. That is not a rounding error, it is
roughly a four-times overstatement, and it comes from how these methods were
built: none of them were trained to produce a correct probability, only to
separate likely cases from unlikely ones.

Recalibrating the models was considered and rejected. Fixing a probability
curve needs a reasonably large, representative sample of outcomes, and the
one available here — 70 known contradictions out of 9,918 checked pairs — is
both too small and drawn from made-up text, not real submissions. Fitting a
calibration curve to that and then presenting the result as trustworthy would
trade one misleading number for a differently-shaped misleading number.

Instead, every flag is shown with a **band** — *Worth a look*, *Priority*, or
*High priority* — worded to make only the claim the evidence actually
supports: flags in a higher band were, in testing, more often right than
flags in a lower one. It does not claim to know the odds for this particular
case, and the app says so next to every explanation of the bands. The cut
points and the sentence justifying each band live in `config/review.yaml`,
sourced directly from the Phase 4 results, so re-running the evaluation and
updating the bands is a config edit, not a rewrite.

**Rejected alternative:** hiding the number entirely and showing only a
tick or a colour. Rejected because a reviewer working through many
submissions needs to know *why* something is worth their time in more than
one dimension — a full sentence, not just a triage colour — and because a
band with no supporting sentence is exactly the kind of unexplained score
Phase 4 flagged as a problem with raw percentages in the first place, just
smaller.

## The determination is never shown

The rule engine that labels the synthetic training data (`CestRuleEngine`,
`src/models/cest_rules.py`) computes something that looks like a status —
inside, outside, or undetermined. Phase 5's brief is explicit that a real
status determination must never be an output of this tool, so that engine is
never called from anywhere in the reviewer path. `src/review/assess.py`,
which runs a submission through the whole pipeline, does not import it, and
a test (`tests/test_review.py::test_assess_module_never_imports_the_labelling_engine`)
checks the actual import statements and bound names in that file — not just
the page's wording — so this cannot regress silently if someone adds a
"quick" status hint later without reading this document first.

The banner at the top of the app repeats this in plain terms — *"Indicative
only — for reviewer context, not a determination"* — and stays visible
throughout, rather than being a one-time notice that scrolls out of view.

## Reviewer decisions

Accepting or dismissing a flag is a judgement about *that one inconsistency*,
never about the submission's outcome — there is no field anywhere in the
decision record for a status. Every decision carries who made it, when, which
flag it was on, and what band the flag was in at the time, plus an optional
free-text reason.

The reason is treated the same way `config/pipeline.yaml`'s
`logging.log_content` flag treats a manager's justification text: it is
content, and content is redacted to a digest by default
(`config/review.yaml`: `decision_log.log_reason_content`). A reviewer's note
can describe a real person's circumstances just as easily as the original
form can, so the same discipline applies to it.

**Rejected alternative:** storing decisions only in the browser session
(`st.session_state`), with no file on disk. Rejected because a decision log
that vanishes when the tab closes cannot support an audit trail, and the
build brief calls for exactly that trail as part of the human-in-the-loop
posture. The current JSONL file is explicitly a placeholder for a real
deployment's database — see `docs/phase_plan.md` Phase 6 — not a proposal
for production; it is the right amount of engineering for a prototype that
needs to be inspectable, not a running service.

## Detection methods on offer

Three, selectable from the sidebar, all built and evaluated in earlier
phases: the rule baseline as authored, the same rules with cue phrases
copied verbatim from the training text removed (a more conservative check),
and the TF-IDF logistic regression baseline. The Phase 3 transformer and
few-shot methods are not offered here — Phase 3's report
(`docs/reports/phase3_status.md`) explains why they could not be run for
real in this environment (the model host was blocked by network policy, and
no API key was available), so there is no genuine result to plug in. Wiring
them in once real weights are available is a few lines in
`src/review/assess.py:build_detector`, not a redesign.

The TF-IDF baseline is fitted on the synthetic corpus each time the app
starts, cached for the rest of the session. **This is a demonstration
convenience, not the production design** — see "What changes for real data"
below.

## What changes for real data

This is written for a marker and for the IR35 team to read before this ever
touches a real submission, not as a formality.

- **The TF-IDF model must stop being fitted at start-up.** A production
  service does not retrain itself on every launch; it loads one specific,
  versioned model artefact that has been evaluated and signed off, the same
  way the rule engine's thresholds are a reviewed config file rather than
  numbers picked at run time. Fitting in-process is fine for a synthetic
  demo where the corpus is 350 records and nothing is at stake; it is the
  wrong design the moment a real submission is involved.
- **The de-identifier needs a validation run against real text.** Phase 1's
  documentation already says this plainly: regex-and-gazetteer scrubbing
  misses names it has never seen and cannot parse. On the committed
  synthetic corpus it is largely a no-op because the generator already wrote
  placeholders — the app shows "no personal data was found" almost every
  time, and says so, rather than pretending that proves anything about real
  text.
- **The decision log needs a retention policy and access control**, not a
  JSONL file anyone with shell access can read. This is a Phase 6 item and a
  DPIA question, not a code change here.
- **The production backend must be local**, per constraint 2 — this app
  never calls a hosted API, and the detector choices offered here are the
  ones that run entirely offline already.
- **A schema-driven input form**, rather than paste/upload JSON, is worth
  building once there is a real intake to connect to. Nothing in the schema
  layer stops this — `src/ingest/schema_loader.py`'s whole point is that no
  field name is hardcoded outside config, so a generated form would read the
  same `schema.yaml` this app already does. It was not built for this phase
  because the brief specifically asks for "paste or upload a submission,"
  and because a 64-field dynamic form with the ESQ's routing logic is a
  substantial second interface in its own right, better scoped as a
  follow-on than squeezed in alongside the review workflow this phase is
  actually about.

## WCAG 2.2 AA — what was checked and how

Each item below was checked against the running app, not assumed from the
CSS. The commands and a screenshot trail are in the session's working notes;
the concrete numbers are reproduced here so the claim can be checked without
re-running anything.

**Colour is never the only signal.** Every flag band, validation finding and
recorded decision pairs an icon or symbol with a full text label — "Worth a
look", "Priority", "Error:", "Accepted by …" — and the text is the real
content; the icon is marked `aria-hidden` so it cannot become the only thing
a screen reader announces. Turning off colour entirely (a Chromium
grayscale-filter check) leaves every distinction still legible from the text
alone.

**Text and UI-component contrast (1.4.3 / 1.4.11).** Every colour used for
text or for a meaningful border was checked against its actual background
with the WCAG relative-luminance formula, not eyeballed:

| Use | Foreground | Background | Ratio | Needs |
|---|---|---|---|---|
| Scope banner text | `#2c2c2c` | `#f7f7f5` | 13.0:1 | 4.5:1 |
| "Worth a look" badge | `#0b3d7a` | `#ffffff` | 10.7:1 | 4.5:1 |
| "Priority" badge | `#7a4a00` | `#ffffff` | 7.5:1 | 4.5:1 |
| "High priority" badge | `#8c1d18` | `#ffffff` | 9.1:1 | 4.5:1 |
| "Accepted" note | `#146c40` | `#ffffff` | 6.5:1 | 4.5:1 |
| Focus outline | `#0b3d7a` | `#ffffff` | 10.7:1 | 3:1 |

All comfortably clear their thresholds; none was picked for looks and
checked afterwards; they were checked first and the CSS written to match.

**Visible keyboard focus (2.4.11).** A global `:focus-visible` rule applies
a 3px high-contrast outline to every focusable element. Checked by tabbing
through the rendered page in a real Chromium instance and reading each
focused element's computed `outline-width` and `outline-color` back from the
DOM: every stop — sidebar inputs, radio options, expanders, tabs, the flag
buttons — returned `3px` / `rgb(11, 61, 122)`, none returned `0px` except the
inert `<body>` element at the end of the tab sequence.

**Full keyboard operation (2.1.1).** Every action available with a mouse —
choosing a sample, switching detector, opening an expander, accepting or
dismissing a flag, downloading the decision log — was exercised with
`Tab`/`Shift+Tab`/`Enter`/`Space` only, via both a scripted Chromium session
and Streamlit's own `AppTest` harness (`tests/test_ui_app.py`), which drives
the real widget tree without a mouse at all. Nothing requires a drag (2.5.7
is satisfied by having nothing to satisfy it against — the file uploader's
only required action is its native Browse button).

**Target size (2.5.8).** Buttons were measured, not assumed: the rendered
bounding box came back 132.9 × 44 CSS pixels, well clear of the 24×24
minimum. The CSS sets a 44px minimum height deliberately, matching the
touch-target guidance many platforms use above the AA floor.

**Programmatic labels (4.1.2) and status messages (4.1.3).** Every input has
a visible `label` (Streamlit sets the accessible name from it), never a
placeholder-only field. A visually-hidden `role="status" aria-live="polite"`
region reports the flag count and method whenever the page reruns, so a
screen reader user is told a new count without having to go looking for it.

**A skip link (2.4.1), with an honestly-documented limit.** A "Skip to main
content" link is the first thing this app's own markup renders. In practice
it lands eighth in the tab sequence, after the sidebar's controls — because
Streamlit's DOM places the sidebar before the main content region
structurally, which cannot be reordered from application code, and which
also matches the page's actual left-to-right visual reading order, so it is
a defensible structure rather than a bug. `.streamlit/config.toml` sets
`client.toolbarMode = "minimal"`, found during this check to remove two pure
platform-chrome stops (Streamlit's own "Deploy" and menu buttons) that
otherwise sat ahead of the skip link for no reason connected to this app.
The skip link's practical benefit today is modest, because the sidebar is
short; it is kept because the sidebar is exactly the part of the page most
likely to grow (more detector options, more filters), and a skip link earns
its keep as that happens, not necessarily on day one.

**Not fully covered by this pass:** a screen-reader-software walkthrough
(NVDA/VoiceOver) rather than DOM-level ARIA inspection, and testing with a
reviewer who has a disability rather than automated tooling. Both are named
directly as Phase 6 handover gaps rather than implied to be done.

## Automated tests

`tests/test_review.py` (19 tests) covers the confidence-band logic, the
decision log's redaction behaviour, detector construction for all three
methods, and the end-to-end assessment pipeline — including a check that the
pipeline finds at least one of the corpus's known planted contradictions, and
the import-level guarantee that the status-labelling engine is unreachable
from the reviewer path.

`tests/test_ui_app.py` (8 tests) drives the actual Streamlit script via
`streamlit.testing.v1.AppTest` — the same code path `streamlit run` uses —
to check the app loads without error, a sample submission produces flags,
decision buttons are disabled until a reviewer name is entered, accepting or
dismissing a flag writes a correctly-redacted decision to the log and is
reflected back in the page, switching detector method does not crash, and
pasting invalid JSON shows an error rather than a stack trace.

232 tests pass across the whole project as of this phase.

## Rejected alternatives, project-level

**A custom Flask/React app instead of Streamlit.** Would allow finer control
over accessibility semantics (real ARIA live regions with guaranteed
announcement timing, full control over DOM order for the skip link). Rejected
for this phase on time and scope grounds: Streamlit's per-rerun re-render
model is a genuine limitation for some fine-grained interaction patterns, but
it let the reviewer workflow, decision capture and WCAG checklist above all
be built, tested and evidenced within the phase, which a from-scratch
frontend would not have left room for. If this tool moves towards a real
deployment, a custom frontend is exactly the kind of thing that belongs in
the Phase 6 "what changes for real data" conversation.

**Showing the rule engine's abstention cases specially** (surfacing
"undetermined" engagements differently from clean ones). Rejected because it
is a half-step towards showing a status by another name — even labelling a
submission as "this one is hard to call" is a comment on the outcome, which
is precisely what this tool must not do. The app treats every submission
identically: check for personal data, check completeness, score the pairs,
show what clears a band.
