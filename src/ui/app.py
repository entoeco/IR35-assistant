"""Phase 5 — the reviewer interface.

Run with:

    streamlit run src/ui/app.py

WHAT THIS APP IS AND IS NOT
It takes one ESQ submission, de-identifies it, checks it against the data
contract, scores every (structured answer, justification) pair for a possible
contradiction, and shows a reviewer the ones worth a look — with a plain-
language reason for each, never a bare number. The reviewer accepts or
dismisses each flag; that decision is captured with who made it, when, and
why.

It never shows a status. There is no "this looks INSIDE / OUTSIDE IR35"
anywhere in this file, and ``src/review/assess.py`` never imports the engine
that could compute one (a fact a test enforces, not just a promise this file
keeps). See ``docs/reports/phase5_interface_report.md`` for the reasoning and
the WCAG 2.2 AA walkthrough.

Three things added after Phase 5 (``docs/reports/phase7_consistency_and_materiality.md``):
tick-box-vs-tick-box consistency findings (section 5 below), a "why this
matters" line attached to every flag and finding, describing how much that
KIND of mismatch typically matters to a determination — never which way
this submission leans (``src/review/materiality.py`` has the argument in
full) — and a five-point "review priority" summary (section 3) aggregating
every flag and finding into "how much attention does this need". Review
priority is emphatically NOT a Likert scale for "how likely is this inside
or outside IR35" — that was considered and rejected for the same reasons the
original RAG-status idea was; see ``src/review/review_priority.py``.

ACCESSIBILITY NOTES ARE INLINE, next to the code they apply to, because a
checklist kept only in a separate document is the first thing that goes stale.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run src/ui/app.py` sets sys.path[0] to src/ui/, not the repo
# root, so the `src.*` package imports below fail unless the repo root is on
# the path — independent of the caller's working directory or PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deidentify.deidentifier import Deidentifier
from src.features.dataset import build_instances, load_corpus
from src.ingest.schema_loader import CONFIG_DIR, Schema, load_schema, load_yaml
from src.ingest.validate import RecordValidator, Severity
from src.models.base import ContradictionDetector
from src.review.assess import AssessmentResult, assess_submission, build_detector
from src.review.bands import load_band_config
from src.review.decision_log import DecisionLog, ReviewDecision

DATA_DIR = REPO_ROOT / "data" / "synthetic"

st.set_page_config(
    page_title="IR35 ESQ reviewer assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# WCAG 2.2 AA: global styling.
#
# * Visible focus indicator (2.4.11/2.4.13): a thick, high-contrast outline on
#   every focusable element, not just the browser default some UAs suppress.
# * Target size (2.5.8): buttons get a comfortable minimum footprint, well
#   above the 24x24px CSS-pixel floor the criterion sets.
# * A skip link (2.4.1) so a keyboard user does not have to tab through the
#   whole sidebar to reach the submission each visit.
# * Colours below are checked for contrast against white in
#   docs/reports/phase5_interface_report.md (all >= 4.5:1 for text). Bands
#   are rendered as a bordered outline with the label as text, never as a
#   colour swatch alone, so removing colour entirely still leaves the
#   information intact — the actual test of "not colour alone".
# =============================================================================
st.markdown(
    """
    <style>
    a.skip-link {
        position: absolute; left: -9999px; top: 0; z-index: 1000;
        background: #ffffff; color: #0b3d7a; padding: 0.75rem 1rem;
        border: 2px solid #0b3d7a; font-weight: 600;
    }
    a.skip-link:focus { left: 0.5rem; top: 0.5rem; }

    *:focus-visible {
        outline: 3px solid #0b3d7a !important;
        outline-offset: 2px !important;
    }

    div.stButton > button, div.stDownloadButton > button {
        min-height: 44px;
        min-width: 44px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .visually-hidden {
        position: absolute !important;
        width: 1px; height: 1px;
        overflow: hidden;
        clip: rect(0 0 0 0);
        white-space: nowrap;
    }

    .scope-banner {
        border: 2px solid #2c2c2c;
        border-left-width: 8px;
        background: #f7f7f5;
        color: #2c2c2c;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }

    .band-badge {
        display: inline-block;
        font-weight: 700;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        border: 2px solid currentColor;
        background: #ffffff;
        margin-bottom: 0.35rem;
    }
    .band-worth-a-look { color: #0b3d7a; }
    .band-priority { color: #7a4a00; }
    .band-high-priority { color: #8c1d18; }

    .flag-card {
        border: 1px solid #b8b8b8;
        border-left: 6px solid #7a4a00;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        background: #ffffff;
    }

    .decided-note {
        border: 2px solid #146c40;
        color: #146c40;
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .decided-note.dismissed { border-color: #2c2c2c; color: #2c2c2c; }

    /* Materiality ("why this matters") — deliberately grayscale, never the
       band colours above. Confidence bands say how strong the EVIDENCE is
       for a mismatch; materiality says how much that KIND of mismatch
       typically matters. Keeping them visually distinct (and structurally
       separate — see src/review/materiality.py) stops a reviewer reading
       "high stakes if true" as "highly likely true". */
    .materiality-line {
        color: #2c2c2c;
        border-left: 3px solid #2c2c2c;
        padding-left: 0.6rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        background: #f7f7f5;
    }
    .materiality-gate {
        font-weight: 700;
        border-left-color: #000000;
        border-left-width: 4px;
    }

    .cross-field-card {
        border: 1px solid #b8b8b8;
        border-left: 6px solid #2c2c2c;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        background: #ffffff;
    }
    .severity-badge {
        display: inline-block;
        font-weight: 700;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        border: 2px solid currentColor;
        background: #ffffff;
        margin-bottom: 0.35rem;
        color: #2c2c2c;
    }

    /* Review priority — deliberately a black/white "filled segments" meter,
       never the band colours or a red/green scale. This is an AGGREGATE
       of what sections 4-5 already show (how many flags, how strong the
       evidence, how much it typically matters) — not a new, different kind
       of claim — so it shares materiality's grayscale palette rather than
       inventing a third colour language that could read as a traffic light
       for "inside vs outside IR35". The filled-segment count is always
       paired with the level's text label, never colour alone (WCAG 1.4.1). */
    .priority-banner {
        border: 2px solid #2c2c2c;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        background: #ffffff;
    }
    .priority-meter {
        display: inline-flex;
        gap: 5px;
        margin-right: 0.75rem;
        vertical-align: middle;
    }
    .priority-segment {
        width: 16px;
        height: 16px;
        border: 2px solid #2c2c2c;
        border-radius: 3px;
        display: inline-block;
        background: #ffffff;
    }
    .priority-segment.filled { background: #2c2c2c; }
    .priority-label {
        font-weight: 700;
        font-size: 1.1rem;
        vertical-align: middle;
    }
    </style>
    <a class="skip-link" href="#main-heading">Skip to main content</a>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Cached resources. Loaded once per process, not once per rerun — Streamlit
# reruns this whole script on every widget interaction, so anything that
# reads a config file or fits a model must be cached or the app would re-read
# and re-fit on every button click.
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_schema() -> Schema:
    return load_schema()


@st.cache_resource(show_spinner=False)
def get_configs() -> dict[str, Any]:
    return {
        "review": load_yaml(CONFIG_DIR / "review.yaml"),
        "rule_baseline": load_yaml(CONFIG_DIR / "rule_baseline.yaml"),
        "text_bank": load_yaml(CONFIG_DIR / "text_bank.yaml"),
        "pipeline": load_yaml(CONFIG_DIR / "pipeline.yaml"),
        "ir35_weights": load_yaml(CONFIG_DIR / "ir35_weights.yaml"),
    }


@st.cache_resource(show_spinner=False)
def get_deidentifier(_schema: Schema, configs: dict[str, Any]) -> Deidentifier:
    return Deidentifier(_schema, configs["pipeline"].get("deidentification"))


@st.cache_resource(show_spinner=False)
def get_validator(_schema: Schema) -> RecordValidator:
    return RecordValidator(_schema)


@st.cache_resource(show_spinner=False)
def get_decision_log(configs: dict[str, Any]) -> DecisionLog:
    cfg = configs["review"]["decision_log"]
    return DecisionLog(
        REPO_ROOT / cfg["path"],
        log_reason_content=bool(cfg.get("log_reason_content", False)),
        reason_max_chars=int(cfg.get("reason_max_chars", 500)),
    )


@st.cache_data(show_spinner=False)
def load_sample_pool() -> list[dict[str, Any]]:
    """A curated spread of sample submissions for the demo picker.

    There is no live feed of real ESQ submissions to paste from, so the app
    offers records from the Phase 1 synthetic corpus instead — deliberately
    mixed between records with a planted contradiction and clean ones, so a
    reviewer trying the tool sees both a hit and a quiet form, not a rigged
    demo that only ever shows a flag.
    """
    records, truth, _states = load_corpus(str(DATA_DIR))
    with_contradiction = [
        r for r, t in zip(records, truth) if t.get("contradictions")
    ]
    clean = [r for r, t in zip(records, truth) if not t.get("contradictions")]
    n = get_configs()["review"]["sample_records"].get("n_offered", 12)
    half = max(1, n // 2)
    pool = with_contradiction[:half] + clean[: n - half]
    return pool


@st.cache_resource(show_spinner=False)
def get_training_instances_cache_key() -> str:
    """A cheap fingerprint so the fitted TF-IDF model is invalidated if the
    corpus on disk changes, without re-reading the whole corpus every rerun."""
    records_path = DATA_DIR / "esq_synthetic_v1.jsonl"
    stat = records_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_resource(show_spinner="Fitting the TF-IDF baseline on the synthetic corpus…")
def get_detector(method: str, _schema: Schema, configs: dict[str, Any], corpus_key: str) -> ContradictionDetector:
    """Build and, where needed, fit a detector. Cached per (method, corpus).

    See ``src.review.assess.build_detector`` for why ``tfidf_logreg`` being
    fitted here, at app start-up, is a demonstration convenience and not the
    production design.
    """
    if method == "tfidf_logreg":
        records, truth, states = load_corpus(str(DATA_DIR))
        training_instances = build_instances(_schema, records, truth, states)
        return build_detector(method, _schema, training_instances=training_instances)
    return build_detector(
        method,
        _schema,
        rule_config=configs["rule_baseline"],
        text_bank=configs["text_bank"],
    )


# =============================================================================
# Session state
# =============================================================================
st.session_state.setdefault("reviewer_id", "")
st.session_state.setdefault("current_record", None)
st.session_state.setdefault("record_source_label", "")
st.session_state.setdefault("reopened_flags", set())


schema = get_schema()
configs = get_configs()
deidentifier = get_deidentifier(schema, configs)
validator = get_validator(schema)
decision_log = get_decision_log(configs)

DETECTOR_DESCRIPTIONS = {
    "rules_as_authored": "Keyword and phrasing rules — fast, fully explainable, the weakest at subtle wording.",
    "rules_phrase_deleaked": "The same rules with cue phrases copied verbatim from the training text removed — a more conservative check.",
    "tfidf_logreg": "A statistical model trained on the synthetic corpus — catches more subtle wording, harder for a person to predict in advance.",
}

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Reviewer session")
    st.text_input(
        "Your name or reviewer ID",
        key="reviewer_id",
        help="Recorded against every decision you make, so the decision log shows who actioned each flag.",
    )
    if not st.session_state["reviewer_id"].strip():
        st.caption("Enter your name to enable the accept/dismiss controls below.")

    st.divider()
    st.header("Detection method")
    available_methods = configs["review"]["detectors"]["available"]
    default_method = configs["review"]["detectors"]["default"]
    method = st.radio(
        "Which method should score this submission?",
        options=available_methods,
        index=available_methods.index(default_method),
        format_func=lambda m: m,
        key="detector_method",
        help="Changing this re-scores the current submission with a different method.",
    )
    st.caption(DETECTOR_DESCRIPTIONS.get(method, ""))

    st.divider()
    with st.expander("How to read the flags"):
        st.markdown(
            "Each flag shows a **band** — *Worth a look*, *Priority*, or "
            "*High priority* — instead of a percentage.\n\n"
            "We tried showing a raw confidence score first. It turned out to "
            "be misleading: in testing, a score the model reported as "
            "\"46% confident\" was actually right about the true rate only "
            "roughly 1 time in 9. The band tells you which flags to look at "
            "first, based on what happened with similar scores in testing — "
            "it is not a probability for this specific answer, and it should "
            "not be read as one."
        )
        st.caption(configs["review"]["detectors"]["band_disclaimer"])

    with st.expander("About this tool"):
        st.markdown(
            "This tool never decides employment status. It compares a tick-box "
            "answer against the free-text explanation next to it and flags "
            "cases where they seem to pull in different directions, so a human "
            "reviewer can look at exactly those cases rather than the whole "
            "form. It also checks tick-box answers against *each other* — "
            "section 5 below — for places where two answers on the form don't "
            "sit well together. Accepting or dismissing a flag or finding "
            "records your judgement about *that one inconsistency* — it does "
            "not set the questionnaire's outcome, which stays entirely a "
            "matter for the IR35 team's usual process.\n\n"
            "All personal data is checked and, where necessary, replaced with "
            "placeholders before anything else happens to a submission. "
            "This build only ever runs on synthetic, made-up submissions."
        )

    with st.expander('"Why this matters" — what materiality does and does not tell you'):
        st.markdown(
            "Most flags and findings carry a short **\"Why this matters\"** "
            "line. It describes how much *that kind* of test or fact "
            "typically matters to an IR35 determination in general — for "
            "example, that personal service and control are treated in case "
            "law as the foundation of the question, or that a particular "
            "answer is one of a handful of facts that can settle the "
            "question on its own.\n\n"
            "**It is not a prediction about this submission.** It is looked "
            "up from two static configuration files — the test weights and "
            "determinative-gate rules used since Phase 2 — using only *which* "
            "test or field a flag concerns. It never reads this submission's "
            "answers, its score, or which way anything here leans, and it "
            "reads exactly the same whether the underlying flag turns out to "
            "be a genuine inconsistency or a false alarm. A test enforces "
            "that this function cannot even be passed a record — see "
            "``src/review/materiality.py`` — so this is a structural "
            "guarantee, not just a promise this screen keeps."
        )

# =============================================================================
# Main content
# =============================================================================
st.markdown('<div id="main-heading"></div>', unsafe_allow_html=True)
st.title("IR35 ESQ reviewer assistant")

st.markdown(
    """
    <div class="scope-banner" role="note">
    <strong>Indicative only — for reviewer context, not a determination.</strong>
    This tool flags places where a manager's tick-box answer and their written
    explanation seem to disagree. It does not say whether the engagement is
    inside or outside IR35, and nothing on this page should be read as if it
    did. The employment-status decision is made by the IR35 team through
    their normal process, using this tool's flags as one input among others.
    </div>
    """,
    unsafe_allow_html=True,
)

st.header("1. Load a submission")

tab_sample, tab_paste, tab_upload = st.tabs(
    ["Use a sample submission", "Paste a submission", "Upload a file"]
)

with tab_sample:
    st.caption(
        "There is no live intake to pull a real submission from here, so this "
        "list is drawn from the synthetic test corpus built in Phase 1 — some "
        "of these carry a made-up inconsistency, most do not, same as a real "
        "batch of forms would."
    )
    pool = load_sample_pool()
    labels = [f"{r['record_id']}  ·  {r.get('archetype', 'unknown').replace('_', ' ')}" for r in pool]
    choice = st.selectbox("Sample submission", options=range(len(pool)), format_func=lambda i: labels[i])
    if st.button("Load this sample", key="load_sample"):
        st.session_state["current_record"] = pool[choice]
        st.session_state["record_source_label"] = f"sample {pool[choice]['record_id']}"

with tab_paste:
    st.caption(
        "Paste a submission as JSON — field id to value. Use the download "
        "below to get a valid example to start from."
    )
    example_bytes = json.dumps(load_sample_pool()[0], indent=2).encode("utf-8")
    st.download_button(
        "Download an example submission (JSON)",
        data=example_bytes,
        file_name="example_esq_submission.json",
        mime="application/json",
        key="download_example",
    )
    pasted = st.text_area("Submission JSON", height=180, key="pasted_json")
    if st.button("Load pasted submission", key="load_pasted"):
        try:
            record = json.loads(pasted)
            if not isinstance(record, dict):
                raise ValueError("top level of the JSON must be an object")
            record.setdefault("record_id", f"PASTED-{uuid.uuid4().hex[:8]}")
            st.session_state["current_record"] = record
            st.session_state["record_source_label"] = "pasted submission"
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Could not read that as a submission: {exc}")

with tab_upload:
    uploaded = st.file_uploader("Submission file (.json)", type=["json"], key="uploaded_file")
    if uploaded is not None and st.button("Load uploaded submission", key="load_uploaded"):
        try:
            record = json.loads(uploaded.getvalue().decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("top level of the JSON must be an object")
            record.setdefault("record_id", f"UPLOAD-{uuid.uuid4().hex[:8]}")
            st.session_state["current_record"] = record
            st.session_state["record_source_label"] = f"uploaded file {uploaded.name}"
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Could not read that file as a submission: {exc}")

record = st.session_state["current_record"]
if record is None:
    st.info("Load a submission above to see its flags.")
    st.stop()

st.success(f"Loaded {st.session_state['record_source_label']} — record {record.get('record_id', 'unknown')}.")

# =============================================================================
# 2. Assess
# =============================================================================
detector = get_detector(method, schema, configs, get_training_instances_cache_key())
result: AssessmentResult = assess_submission(
    record,
    schema=schema,
    deidentifier=deidentifier,
    validator=validator,
    detector=detector,
    review_config=configs["review"],
    ir35_weights_config=configs["ir35_weights"],
)

st.header("2. Before the flags: privacy and completeness checks")

col_deid, col_valid = st.columns(2)

with col_deid:
    st.subheader("Personal data check")
    report = result.deid_report
    if report.is_clean:
        st.markdown(
            "No personal data was found in the free-text answers. "
            "(This corpus is already de-identified upstream, so that is "
            "expected here — the check still runs on every submission, "
            "including this one.)"
        )
    else:
        st.markdown(f"**{report.total_entities} item(s) removed** before anything else ran:")
        for kind, count in sorted(report.entity_counts.items()):
            st.markdown(f"- {count} × {kind.replace('_', ' ')}")
        st.caption(
            "Only counts are shown here — never the values that were removed."
        )

with col_valid:
    st.subheader("Form completeness")
    findings = result.validation.findings
    if not findings:
        st.markdown("No completeness or contract issues found.")
    else:
        errors = [f for f in findings if f.severity is Severity.ERROR]
        warnings = [f for f in findings if f.severity is Severity.WARNING]
        for f in errors:
            st.markdown(f"**Error:** {f.message}")
        for f in warnings[:6]:
            st.markdown(f"**Note:** {f.message}")
        if len(warnings) > 6:
            st.caption(f"…and {len(warnings) - 6} more completeness notes.")

# =============================================================================
# 3. Review priority — an AGGREGATE of sections 4-5 below (how many flags and
# findings, how strong the evidence for each, how much that kind of mismatch
# typically matters), collapsed into one five-point scale so a reviewer
# triaging many submissions can tell "needs a close look" from "nothing much
# here" at a glance.
#
# THIS IS NOT A STATUS. Its own config comments (config/review.yaml) and
# module docstring (src/review/review_priority.py) both say so, and this is
# the third place: none of count, evidence strength, or importance-of-
# category — the only three things that go into this number — encode which
# way any answer leans. It is worded around "how much attention", never
# "how likely inside/outside", and the wording is checked against a
# forbidden-phrase list the same way the scope banner and materiality text
# are (tests/test_review_priority.py::test_configured_levels_never_mention_a_lean_or_a_status).
# =============================================================================
st.header("3. How much attention does this submission need?")

priority = result.review_priority
PRIORITY_LEVEL_ORDER = ["none", "light", "moderate", "close", "urgent"]
filled = PRIORITY_LEVEL_ORDER.index(priority.key) + 1 if priority is not None and priority.key in PRIORITY_LEVEL_ORDER else 0
segments_html = "".join(
    f'<span class="priority-segment{" filled" if i < filled else ""}" aria-hidden="true"></span>'
    for i in range(5)
)
st.markdown(
    f'<div class="priority-banner" role="note">'
    f'<span class="priority-meter">{segments_html}</span>'
    f'<span class="priority-label">{priority.label if priority else "Not available"}</span>'
    f"<p>{priority.description if priority else ''}</p>"
    f"<p><em>This is a summary of the flags below, not a prediction of the "
    f"outcome — it does not say whether the engagement is inside or outside "
    f"IR35, only how much of what this tool checks disagreed with itself.</em></p>"
    f"</div>",
    unsafe_allow_html=True,
)

with st.expander("What goes into this, and what doesn't"):
    st.markdown(
        "This combines three things also shown below: **how many** flags "
        "and tick-box consistency findings there are, **how strong the "
        "evidence** is for each (its confidence band or severity), and "
        "**how much that kind of question typically matters** (its "
        "materiality tier, when available). A single finding touching a "
        "fact case law treats as potentially decisive on its own always "
        "puts a submission at *Urgent review*, however small everything "
        "else looks.\n\n"
        "It never reads which way a tick-box or a piece of free text "
        "actually leans — only that a flag exists, how confident the tool "
        "is that it's a genuine inconsistency, and how important that kind "
        "of question generally is. Two submissions with opposite-leaning "
        "answers but the same number, strength and importance of "
        "inconsistencies score identically — that's checked directly in "
        "the test suite, not just claimed here."
    )

# =============================================================================
# 4. Flags
# =============================================================================
st.header("4. Flags for review")

# WCAG 2.2: 4.1.3 status messages. This text changes whenever the flag count
# or method changes, and screen readers announce an aria-live region's
# updated text without the user needing to navigate to it.
st.markdown(
    f'<div class="visually-hidden" role="status" aria-live="polite">'
    f"{len(result.flagged)} flag(s) found using the {result.method.replace('_', ' ')} method, "
    f"out of {result.n_instances_scored} answered question pairs checked, "
    f"and {len(result.cross_field_findings)} tick-box consistency finding(s). "
    f"Review priority: {priority.label if priority else 'not available'}."
    f"</div>",
    unsafe_allow_html=True,
)

if not result.flagged:
    # Not a `st.stop()` here: even with no tick-box-vs-text flags, section 5
    # below may still have a tick-box-vs-tick-box consistency finding to show,
    # and the decision log section always renders.
    st.markdown(
        "No flags at or above this method's reporting threshold. "
        f"({result.n_instances_scored} answered question pairs were checked.)"
    )
else:
    st.caption(
        f"{len(result.flagged)} of {result.n_instances_scored} answered question pairs flagged. "
        "Shown strongest evidence first within each test."
    )

section_order = list(configs["review"]["test_section_order"])
by_test: dict[str, list] = {}
for fi in result.flagged:
    by_test.setdefault(fi.instance.ir35_test, []).append(fi)
ordered_tests = [t for t in section_order if t in by_test] + [
    t for t in by_test if t not in section_order
]

BAND_CSS_CLASS = {
    "Worth a look": "band-worth-a-look",
    "Priority": "band-priority",
    "High priority": "band-high-priority",
}
BAND_ICON = {"Worth a look": "●", "Priority": "▲", "High priority": "■"}


def _render_materiality(tier: Any) -> None:
    """Render a "why this matters" line for a materiality tier, if present.

    Deliberately grayscale (see the .materiality-line / .materiality-gate
    CSS above) so this is never mistaken for a confidence band. Says nothing
    about this record — only how much this KIND of test or fact typically
    matters, per config/review.yaml and config/ir35_weights.yaml.
    """
    if tier is None:
        return
    css_class = "materiality-line materiality-gate" if tier.is_gate else "materiality-line"
    st.markdown(
        f'<div class="{css_class}">'
        f"<strong>Why this matters:</strong> {tier.label}. {tier.explanation}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _existing_decision(record_id: str, pair_id: str, field_id: str) -> dict[str, Any] | None:
    """The most recent decision already logged for this exact flag, if any."""
    matches = [
        row
        for row in decision_log.for_record(record_id)
        if row.get("pair_id") == pair_id and row.get("field_id") == field_id
    ]
    return matches[-1] if matches else None


for test in ordered_tests:
    test_meta = schema.ir35_tests.get(test, {})
    st.subheader(test_meta.get("label", test.replace("_", " ").title()))
    if test_meta.get("note"):
        st.caption(test_meta["note"])

    for fi in by_test[test]:
        flag = fi.flag
        instance = fi.instance
        pair = schema.pairs[flag.pair_id]
        band = fi.band
        css_class = BAND_CSS_CLASS.get(band.label, "band-worth-a-look") if band else ""
        icon = BAND_ICON.get(band.label, "●") if band else ""

        flag_key = f"{record.get('record_id')}::{flag.pair_id}::{flag.field_id}::{result.method}"
        existing = _existing_decision(str(record.get("record_id")), flag.pair_id, flag.field_id)

        with st.container():
            st.markdown('<div class="flag-card">', unsafe_allow_html=True)
            if band:
                st.markdown(
                    f'<span class="band-badge {css_class}">'
                    f'<span aria-hidden="true">{icon}</span> {band.label}'
                    f"</span>",
                    unsafe_allow_html=True,
                )
                st.caption(band.rationale)

            _render_materiality(fi.materiality)

            st.markdown(f"**Question {flag.form_ref}** — {schema[instance.structured_field].label}")
            st.markdown(f"Answer given: **{instance.structured_value}**")
            st.markdown(f"> {instance.free_text}")
            st.markdown(flag.explanation)

            with st.expander("What would count as a contradiction here?"):
                st.markdown(pair.contradiction or "No description recorded for this pair.")

            if existing and flag_key not in st.session_state["reopened_flags"]:
                decided_class = "dismissed" if existing["decision"] == "dismiss" else ""
                verb = "Accepted" if existing["decision"] == "accept" else "Dismissed"
                st.markdown(
                    f'<div class="decided-note {decided_class}">'
                    f"{verb} by {existing['reviewer_id']} at {existing['timestamp_utc']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Change this decision", key=f"reopen_{flag_key}"):
                    st.session_state["reopened_flags"].add(flag_key)
                    st.rerun()
            else:
                reviewer_id = st.session_state["reviewer_id"].strip()
                reason = st.text_input(
                    "Optional note (why you accepted or dismissed this)",
                    key=f"reason_{flag_key}",
                    label_visibility="visible",
                )
                b_accept, b_dismiss = st.columns(2)
                disabled = not reviewer_id
                if disabled:
                    st.caption("Enter your reviewer name in the sidebar to record a decision.")
                with b_accept:
                    if st.button("Accept — needs follow-up", key=f"accept_{flag_key}", disabled=disabled):
                        decision_log.record(
                            ReviewDecision(
                                decision_id=uuid.uuid4().hex,
                                record_id=str(record.get("record_id")),
                                pair_id=flag.pair_id,
                                field_id=flag.field_id,
                                method=result.method,
                                score=flag.score,
                                band_label=band.label if band else None,
                                decision="accept",
                                reviewer_id=reviewer_id,
                                reason=reason or None,
                            )
                        )
                        st.session_state["reopened_flags"].discard(flag_key)
                        st.rerun()
                with b_dismiss:
                    if st.button("Dismiss — not a concern", key=f"dismiss_{flag_key}", disabled=disabled):
                        decision_log.record(
                            ReviewDecision(
                                decision_id=uuid.uuid4().hex,
                                record_id=str(record.get("record_id")),
                                pair_id=flag.pair_id,
                                field_id=flag.field_id,
                                method=result.method,
                                score=flag.score,
                                band_label=band.label if band else None,
                                decision="dismiss",
                                reviewer_id=reviewer_id,
                                reason=reason or None,
                            )
                        )
                        st.session_state["reopened_flags"].discard(flag_key)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 5. Tick-box vs tick-box consistency findings
#
# Same idea as section 4, but comparing two structured answers against each
# other instead of an answer against its own free-text justification (see
# src/models/cross_field.py). These have no confidence band — a boolean
# condition either fired or it did not, so severity (authored per-check in
# config, not computed) is shown instead.
#
# Decisions on these findings are recorded through the same DecisionLog /
# ReviewDecision machinery as section 4, reusing its `pair_id` field for the
# check's id and `field_id` for the touched fields (joined), rather than
# adding a parallel log for what is, for audit purposes, the same kind of
# event: a reviewer looked at a flagged inconsistency and made a call on it.
# =============================================================================
st.header("5. " + configs["review"]["cross_field"]["section_heading"])
st.caption(configs["review"]["cross_field"]["section_intro"])

if not result.cross_field_findings:
    st.markdown("No tick-box consistency findings for this submission.")
else:
    severity_label = configs["review"]["cross_field"]["severity_label"]
    for cf in result.cross_field_findings:
        finding = cf.finding
        cf_pair_id = finding.check_id
        cf_field_id = ",".join(finding.fields)
        cf_key = f"{record.get('record_id')}::{cf_pair_id}::{cf_field_id}::cross_field"
        existing_cf = _existing_decision(str(record.get("record_id")), cf_pair_id, cf_field_id)

        with st.container():
            st.markdown('<div class="cross-field-card">', unsafe_allow_html=True)
            st.markdown(
                f'<span class="severity-badge">'
                f"{severity_label.get(finding.severity, finding.severity)}"
                f"</span>",
                unsafe_allow_html=True,
            )

            _render_materiality(cf.materiality)

            st.markdown(finding.description)
            st.markdown(
                "Fields to look at: "
                + ", ".join(
                    f"**{schema[fid].label}**" if fid in schema and schema[fid].label else f"`{fid}`"
                    for fid in finding.fields
                )
            )

            if existing_cf and cf_key not in st.session_state["reopened_flags"]:
                decided_class = "dismissed" if existing_cf["decision"] == "dismiss" else ""
                verb = "Accepted" if existing_cf["decision"] == "accept" else "Dismissed"
                st.markdown(
                    f'<div class="decided-note {decided_class}">'
                    f"{verb} by {existing_cf['reviewer_id']} at {existing_cf['timestamp_utc']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Change this decision", key=f"reopen_{cf_key}"):
                    st.session_state["reopened_flags"].add(cf_key)
                    st.rerun()
            else:
                reviewer_id = st.session_state["reviewer_id"].strip()
                reason = st.text_input(
                    "Optional note (why you accepted or dismissed this)",
                    key=f"reason_{cf_key}",
                    label_visibility="visible",
                )
                b_accept, b_dismiss = st.columns(2)
                disabled = not reviewer_id
                if disabled:
                    st.caption("Enter your reviewer name in the sidebar to record a decision.")
                with b_accept:
                    if st.button("Accept — needs follow-up", key=f"accept_{cf_key}", disabled=disabled):
                        decision_log.record(
                            ReviewDecision(
                                decision_id=uuid.uuid4().hex,
                                record_id=str(record.get("record_id")),
                                pair_id=cf_pair_id,
                                field_id=cf_field_id,
                                method="cross_field",
                                score=1.0,
                                band_label=severity_label.get(finding.severity, finding.severity),
                                decision="accept",
                                reviewer_id=reviewer_id,
                                reason=reason or None,
                            )
                        )
                        st.session_state["reopened_flags"].discard(cf_key)
                        st.rerun()
                with b_dismiss:
                    if st.button("Dismiss — not a concern", key=f"dismiss_{cf_key}", disabled=disabled):
                        decision_log.record(
                            ReviewDecision(
                                decision_id=uuid.uuid4().hex,
                                record_id=str(record.get("record_id")),
                                pair_id=cf_pair_id,
                                field_id=cf_field_id,
                                method="cross_field",
                                score=1.0,
                                band_label=severity_label.get(finding.severity, finding.severity),
                                decision="dismiss",
                                reviewer_id=reviewer_id,
                                reason=reason or None,
                            )
                        )
                        st.session_state["reopened_flags"].discard(cf_key)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 6. Decision log for this record
# =============================================================================
st.header("6. Decisions recorded for this submission")
record_decisions = decision_log.for_record(str(record.get("record_id")))
if not record_decisions:
    st.markdown("No decisions recorded yet for this submission.")
else:
    st.dataframe(
        [
            {
                "when": d["timestamp_utc"],
                "reviewer": d["reviewer_id"],
                "pair / check": d["pair_id"],
                "decision": d["decision"],
                "band / severity at the time": d.get("band_label"),
            }
            for d in record_decisions
        ],
        width="stretch",
        hide_index=True,
    )

all_decisions = decision_log.load()
if all_decisions:
    st.download_button(
        "Download the full decision log (JSON)",
        data=json.dumps(all_decisions, indent=2).encode("utf-8"),
        file_name="reviewer_decisions.json",
        mime="application/json",
        key="download_decisions",
    )
