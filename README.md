# IR35 ESQ contradiction detector and anomaly highlighter

A decision-support tool for the University's Employment Status Questionnaire
(ESQ) process. It compares a hiring manager's tick-box answers against the
free-text justification written next to each one, flags the pairs that seem
to disagree, and shows a reviewer a plain-language reason for each flag —
strongest evidence first. It never issues an employment-status determination;
that stays a human decision throughout, by design (see "Scope and posture"
below).

Built in phases, each gated on the last: schema first, then a synthetic
dataset, then two baseline detection methods, then a transformer-based
approach, then bias and calibration testing, then a reviewer interface. The
full reasoning for each phase — including rejected alternatives and honestly
reported limitations — is in `docs/reports/`. This README is the practical
"how do I run this" companion to those; start there for *why* things are
built the way they are.

## Quick start

Requires Python 3.11+. No GPU, no network access, and no account with any
external service is needed for anything below.

```bash
git clone <this repository>
cd ir35-assistant
python3 -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest                             # 267 tests, all offline, ~1 minute
streamlit run src/ui/app.py        # the reviewer app, at http://localhost:8501
```

That's the whole install. `requirements.txt` deliberately contains nothing
that needs a GPU or a package index other than PyPI — see "Optional: the
Phase 3 transformer backend" below for the one piece of the project that
does, and why it is kept separate.

The synthetic test corpus (`data/synthetic/`) is already generated and
committed, so nothing needs to run before the app or the tests do. Loading a
sample submission in the reviewer app or running `pytest` both work straight
after `pip install`.

## What's in this repository

```
config/            Every field name, threshold, weight and band lives here —
                    not in the code. See "Config-driven, on purpose" below.
data/synthetic/     The generated test corpus: 350 records, labelled,
                    de-identified, committed to the repo.
docs/               Phase reports (the "why"), the field inventory, the
                    original phase plan, and (this phase) the stakeholder
                    runbook and deployment notes.
docs/reports/       One report per phase: what was built, what was measured,
                    what was rejected and why, and any bug the phase's own
                    rigour caught.
scripts/            One entry point per pipeline stage — generate the
                    corpus, run the baselines, run Phase 3, run Phase 4.
src/deidentify/     Personal-data scrubbing. Runs first, on every record.
src/generate/       The synthetic-data generator (Phase 1).
src/ingest/         The schema loader and the record/completeness validator.
src/models/         Every detection method behind one shared interface.
src/evaluate/       Metrics, splits, leakage checks, calibration, plots.
src/review/         Phase 5: turning a raw score into a reviewer-facing band,
                    running the pipeline on one live submission, and
                    capturing accept/dismiss decisions. Phase 7 added
                    src/review/materiality.py — see below.
src/models/cross_field.py  Phase 7: tick-box vs tick-box consistency checks
                    (as opposed to tick-box vs its own free text).
src/ui/app.py       The Streamlit reviewer app.
tests/              267 tests. Run all of them with `pytest`.
```

## Running things

**The test suite.**

```bash
pytest                 # everything
pytest tests/test_review.py tests/test_ui_app.py   # just the Phase 5 pieces
pytest -k calibration  # anything with "calibration" in the test name
```

**The reviewer app.**

```bash
streamlit run src/ui/app.py
```

Opens in a browser at `localhost:8501`. Load a sample submission from the
dropdown, or paste/upload a submission as JSON (there's a "download an
example" button to get a valid starting point). See
`docs/reports/phase5_interface_report.md` for what it does and why it shows
a confidence *band* rather than a percentage.

**Regenerating the synthetic corpus.** Not required — the committed corpus
already reflects the current config — but useful after editing
`config/generation.yaml`:

```bash
python scripts/generate_dataset.py
python scripts/generate_dataset.py --n 500 --out data/synthetic_v2
```

Deterministic: the same config and seed reproduce the corpus byte-for-byte,
including across a fresh Python process (see
`docs/reports/phase4_evaluation_report.md` for the bug this used to fail on
and the test that now catches it).

**The Phase 2 baselines** (rule-based and TF-IDF, evaluated on a record split
and a content-unit split):

```bash
python scripts/run_baselines.py
```

Writes `docs/reports/phase2_baseline_report.md`,
`docs/reports/phase2_pr_curves.png` and `docs/reports/phase2_results.json`.

**Phase 4 evaluation** (calibration, style-invariance, bootstrap intervals):

```bash
python scripts/run_phase4.py
```

Writes `docs/reports/phase4_evaluation_report.md` and its supporting plots
and JSON.

**Phase 3** (the transformer and few-shot methods) needs real model weights
and, for the few-shot arm, an API key — neither of which this build sandbox
had access to. See "Optional: the Phase 3 transformer backend" below and
`docs/phase3_status.md` for the full story, including what was built and
tested without them (40 tests, all passing) and exactly what running it for
real involves.

## Config-driven, on purpose

No field name, threshold, or wording lives in `.py` code outside `config/`.
Pointing this system at a revised ESQ, retuning a threshold after a new
evaluation run, or rewording a reviewer-facing explanation is a config edit,
checked mechanically by tests that scan the source for hardcoded field names.
The files, in the order you'd usually touch them:

| File | Governs |
|---|---|
| `config/schema.yaml` | Every field on the form: id, type, PII class, IR35 test, routing. |
| `config/contradiction_pairs.yaml` | Which (tick-box, free-text) pairs are checked, and the taxonomy of contradiction types. Also (Phase 7) the `cross_field_checks:` conditions — tick-box vs tick-box, as opposed to tick-box vs free text. |
| `config/generation.yaml` | The synthetic corpus: archetypes, registers, contradiction rate, reproducibility seed. |
| `config/ir35_weights.yaml` | The rule engine that labels the synthetic training data (never used at inference time on a real submission — see below). |
| `config/rule_baseline.yaml` | The Phase 2 keyword/rule detector's cue lists and scoring. |
| `config/model.yaml` | Which inference backend runs (mock / local transformer / local LLM / hosted API) and its prompts. |
| `config/pipeline.yaml` | De-identification policy and what gets logged. |
| `config/review.yaml` | The Phase 5 app: confidence-band cut-points and wording, decision-log policy, sample submissions offered. Phase 7 added the materiality tiers and cross-field wording. |

## Scope and posture (read this before touching real data)

This tool is decision support. It highlights where a manager's own answers
disagree with each other; it does not decide, has never decided, and must
never be made to decide, employment status. That is not a UI choice — it is
enforced at the code level: `src/review/assess.py`, which runs a live
submission through the pipeline, never imports the rule engine
(`src/models/cest_rules.py`) that computes a status for the synthetic
training data, and a test inspects the actual import statements in that file
to make sure this cannot change silently. Under UK GDPR Article 22, an
employment-status determination has legal effect, so a solely automated
decision is out of scope by design, not by omission — a human in the IR35
team remains the decision-maker for every record, every time.

Everything downstream of `src/deidentify/` only ever sees a scrubbed record.
De-identification runs first in the pipeline, before validation, before
scoring, before anything reaches a model — on every record, including
synthetic ones, so that it is a live, tested component rather than a comment
that would only be noticed missing the day it mattered.

## Optional: the Phase 3 transformer backend

`requirements-transformers.txt` installs `torch` and `transformers`, needed
only to run the Phase 3 NLI and few-shot methods against real model weights:

```bash
pip install -r requirements-transformers.txt
python scripts/fetch_models.py
python scripts/run_phase3.py
```

Nothing else needs this file. It is kept separate from `requirements.txt`
because torch's build is machine-specific (CPU vs a particular CUDA version)
— see the comments in that file for the detail, and
`docs/reports/phase6_handover_report.md` for why this split exists (short
version: the original single pinned `torch==...+cu130` could not be
installed on an ordinary laptop at all).

## Reproducing the exact build environment

`requirements.txt` pins this project's direct dependencies.
`requirements-lock.txt` pins everything underneath them too — Streamlit's own
dependency tree, pandas, click, and so on — at the exact versions this was
tested against:

```bash
pip install -r requirements-lock.txt
```

Regenerate it (after changing `requirements.txt`) from a bare virtual
environment — see the header of `requirements-lock.txt` for the exact
commands — so it never silently accumulates packages from an unrelated
project on the same machine.

## For the IR35 team

`docs/stakeholder_runbook.md` is written for you, not for a developer — what
the tool does, what a flag means, what to do with one, and who to contact if
something looks wrong. `docs/deployment_notes.md` covers what has to happen
— a DPIA, a de-identification validation run against real text, a retention
decision for the reviewer decision log — before this tool goes anywhere near
a real submission; it is aimed at whoever signs that off, not at a reviewer
using the finished tool day to day.

## Project reports

Each phase's full write-up — including honestly reported limitations and
what was rejected and why — lives in `docs/`:

- `docs/phase0_field_inventory.md` — the ESQ schema extraction and the gate
  decisions taken before any code was written.
- `docs/reports/phase1_dataset_report.md` — the synthetic data generator.
- `docs/reports/phase2_baseline_report.md` — the rule and TF-IDF baselines.
- `docs/phase3_status.md` — the transformer/few-shot methods, and the
  environment blocker that stopped them running for real.
- `docs/reports/phase4_evaluation_report.md` — calibration, style
  invariance, and the two real bugs this phase's rigour caught.
- `docs/reports/phase5_interface_report.md` — the reviewer app and its WCAG
  2.2 AA walkthrough.
- `docs/reports/phase6_handover_report.md` — the install-gate fix, the
  lockfile, and the handover materials referenced below.
- `docs/reports/phase7_consistency_and_materiality.md` — tick-box vs
  tick-box consistency checks, and "materiality" (how much a *kind* of
  mismatch typically matters) — including the design argument for why this
  is not the RAG-style status this project deliberately does not build.
