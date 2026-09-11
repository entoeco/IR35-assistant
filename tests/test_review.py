"""Phase 5 — reviewer app plumbing: bands, decision capture, the assessment
pipeline.

The tests that matter most here are the ones that would catch the interface
quietly drifting into the thing it must never do: showing a raw percentage as
if it were a calibrated probability, or surfacing a status determination. See
``test_flags_never_carry_a_status_determination`` and
``test_bands_never_expose_a_raw_percentage_as_the_label``.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import pytest

from src.deidentify.deidentifier import Deidentifier
from src.ingest.schema_loader import CONFIG_DIR, load_schema, load_yaml
from src.ingest.validate import RecordValidator
from src.review import assess as assess_module
from src.review.assess import assess_submission, build_detector
from src.review.bands import band_for_score, load_band_config
from src.review.decision_log import DecisionLog, ReviewDecision

DATA = "data/synthetic"


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def review_config():
    return load_yaml(CONFIG_DIR / "review.yaml")


@pytest.fixture(scope="module")
def rule_config():
    return load_yaml(CONFIG_DIR / "rule_baseline.yaml")


@pytest.fixture(scope="module")
def text_bank():
    return load_yaml(CONFIG_DIR / "text_bank.yaml")


@pytest.fixture(scope="module")
def pipeline_config():
    return load_yaml(CONFIG_DIR / "pipeline.yaml")


@pytest.fixture(scope="module")
def ir35_weights_config():
    return load_yaml(CONFIG_DIR / "ir35_weights.yaml")


@pytest.fixture(scope="module")
def sample_records():
    import json

    path = Path(DATA) / "esq_synthetic_v1.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()][:40]


@pytest.fixture(scope="module")
def training_instances(schema):
    from src.features.dataset import build_instances, load_corpus

    records, truth, states = load_corpus(DATA)
    return build_instances(schema, records, truth, states)


# =============================================================================
# Confidence bands
# =============================================================================


def test_band_config_loads_ascending(review_config):
    bands = load_band_config(review_config, "rules_as_authored")
    assert len(bands) >= 2
    assert [b.min_score for b in bands] == sorted(b.min_score for b in bands)


def test_score_below_every_band_is_unflagged(review_config):
    bands = load_band_config(review_config, "rules_as_authored")
    assert band_for_score(0.0, bands) is None
    assert band_for_score(bands[0].min_score - 0.001, bands) is None


def test_score_lands_in_highest_cleared_band(review_config):
    bands = load_band_config(review_config, "rules_as_authored")
    top = bands[-1]
    assert band_for_score(top.min_score, bands).label == top.label
    assert band_for_score(1.0, bands).label == top.label


def test_unknown_method_has_no_bands(review_config):
    """A detector with no configured bands flags nothing, rather than the
    code inventing a default cut-point that lives nowhere in config."""
    assert load_band_config(review_config, "some_future_method") == ()


def test_bands_never_expose_a_raw_percentage_as_the_label(review_config):
    """The whole point of banding is to stop showing a number that Phase 4
    proved was miscalibrated. A label like "46%" would smuggle it back in."""
    for method_bands in review_config["detectors"]["confidence_bands"].values():
        for entry in method_bands:
            label = str(entry["label"])
            assert "%" not in label
            assert not any(ch.isdigit() for ch in label)


# =============================================================================
# Decision log
# =============================================================================


def test_decision_log_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        log = DecisionLog(Path(tmp) / "decisions.jsonl")
        decision = ReviewDecision(
            decision_id="d1",
            record_id="SYN-0001",
            pair_id="p_x",
            field_id="q_x_rationale",
            method="rules_as_authored",
            score=0.42,
            band_label="Priority",
            decision="accept",
            reviewer_id="j.reviewer",
            reason="Matches what they told us on the phone.",
        )
        stored = log.record(decision)
        assert stored["decision"] == "accept"
        assert log.load() == [stored]
        assert log.for_record("SYN-0001") == [stored]
        assert log.for_record("SYN-9999") == []


def test_decision_log_redacts_reason_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        log = DecisionLog(Path(tmp) / "decisions.jsonl")
        decision = ReviewDecision(
            decision_id="d1",
            record_id="SYN-0001",
            pair_id="p_x",
            field_id="q_x_rationale",
            method="rules_as_authored",
            score=0.42,
            band_label="Priority",
            decision="dismiss",
            reviewer_id="j.reviewer",
            reason="Dr Ashwood confirmed this by email on the 3rd.",
        )
        stored = log.record(decision)
        assert "reason" not in stored
        assert stored["reason_logged"] is False
        assert "Ashwood" not in str(stored)
        assert stored["reason_digest"]["chars"] == len(decision.reason)


def test_decision_log_can_be_configured_to_keep_the_reason():
    with tempfile.TemporaryDirectory() as tmp:
        log = DecisionLog(Path(tmp) / "decisions.jsonl", log_reason_content=True)
        decision = ReviewDecision(
            decision_id="d1",
            record_id="SYN-0001",
            pair_id="p_x",
            field_id="q_x_rationale",
            method="rules_as_authored",
            score=0.42,
            band_label="Priority",
            decision="accept",
            reviewer_id="j.reviewer",
            reason="Kept verbatim in this configuration.",
        )
        stored = log.record(decision)
        assert stored["reason"] == decision.reason
        assert stored["reason_logged"] is True


def test_decision_never_carries_a_status_field():
    """The reviewer decides on a flag. Nothing in the schema lets them, or the
    log, record a status determination."""
    fields = {f.name for f in ReviewDecision.__dataclass_fields__.values()}
    assert not any("status" in f.lower() for f in fields)
    assert not any(f.lower() in ("outcome", "verdict") for f in fields)


# =============================================================================
# Detector construction
# =============================================================================


def test_build_detector_rules_as_authored(schema, rule_config):
    detector = build_detector("rules_as_authored", schema, rule_config=rule_config)
    assert detector.name == "rules_as_authored"


def test_build_detector_rules_phrase_deleaked(schema, rule_config, text_bank):
    detector = build_detector(
        "rules_phrase_deleaked", schema, rule_config=rule_config, text_bank=text_bank
    )
    assert detector.name == "rules_phrase_deleaked"


def test_build_detector_tfidf_requires_training_instances(schema):
    with pytest.raises(ValueError):
        build_detector("tfidf_logreg", schema)


def test_build_detector_tfidf(schema, training_instances):
    # The full corpus, not a slice: at a 0.7% positive rate an arbitrary
    # slice can easily land on a single class, which is TfidfBaseline's own
    # guard rail (see src/models/baseline_tfidf.py) doing its job, not a bug
    # in this test.
    detector = build_detector("tfidf_logreg", schema, training_instances=training_instances)
    assert detector.name == "tfidf_logreg"


def test_build_detector_rejects_unknown_method(schema, rule_config):
    with pytest.raises(ValueError):
        build_detector("made_up_method", schema, rule_config=rule_config)


# =============================================================================
# End-to-end assessment
# =============================================================================


@pytest.fixture(scope="module")
def deidentifier(schema, pipeline_config):
    return Deidentifier(schema, pipeline_config.get("deidentification"))


@pytest.fixture(scope="module")
def validator(schema):
    return RecordValidator(schema)


@pytest.fixture(scope="module")
def rules_detector(schema, rule_config):
    return build_detector("rules_as_authored", schema, rule_config=rule_config)


def test_assess_submission_runs_deidentify_before_scoring(
    schema, deidentifier, validator, rules_detector, review_config, sample_records
):
    record = sample_records[0]
    result = assess_submission(
        record,
        schema=schema,
        deidentifier=deidentifier,
        validator=validator,
        detector=rules_detector,
        review_config=review_config,
    )
    assert result.record_id == str(record["record_id"])
    assert result.n_instances_scored > 0
    # Every flagged instance cleared the detector's lowest band.
    bands = load_band_config(review_config, rules_detector.name)
    for fi in result.flagged:
        assert fi.flag.score >= bands[0].min_score
        assert fi.band is not None


def test_assess_submission_finds_at_least_one_planted_contradiction(
    schema, deidentifier, validator, rules_detector, review_config, sample_records
):
    """Sanity check the whole chain against records we know carry a planted
    contradiction, using the ground truth file purely as a test oracle."""
    import json

    truth_by_id = {}
    with (Path(DATA) / "ground_truth_v1.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            truth_by_id[str(row["record_id"])] = row

    found_any = False
    for record in sample_records:
        truth = truth_by_id[str(record["record_id"])]
        planted_pairs = {c["pair_id"] for c in truth["contradictions"]}
        if not planted_pairs:
            continue
        result = assess_submission(
            record,
            schema=schema,
            deidentifier=deidentifier,
            validator=validator,
            detector=rules_detector,
            review_config=review_config,
        )
        flagged_pairs = {fi.instance.pair_id for fi in result.flagged}
        if flagged_pairs & planted_pairs:
            found_any = True
            break
    assert found_any, "the rule detector caught none of the planted contradictions in the sample"


def test_flags_never_carry_a_status_determination(
    schema, deidentifier, validator, rules_detector, review_config, sample_records
):
    """The Art.22 posture as a fact about the data, not just the UI copy:
    nothing produced by the assessment pipeline can be rendered as a status."""
    for record in sample_records[:10]:
        result = assess_submission(
            record,
            schema=schema,
            deidentifier=deidentifier,
            validator=validator,
            detector=rules_detector,
            review_config=review_config,
        )
        for fi in result.flagged:
            explanation = fi.flag.explanation.lower()
            for forbidden in ("is inside ir35", "is outside ir35", "determination is", "we determine"):
                assert forbidden not in explanation


def test_assess_module_never_imports_the_labelling_engine():
    """A stronger guarantee than checking output text: the module that scores
    a live submission cannot compute a status at all, because it never
    imports the engine that knows how.

    Checks actual imports and bound names (via ``ast`` and the module's own
    namespace), not the docstring — the docstring explains *why* the engine
    is deliberately absent, which mentions its name on purpose.
    """
    import ast

    source = inspect.getsource(assess_module)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)

    assert not any("cest_rules" in m for m in imported_modules)
    assert "CestRuleEngine" not in imported_names
    assert "Status" not in imported_names
    assert not hasattr(assess_module, "CestRuleEngine")
    assert not hasattr(assess_module, "Status")


def test_assessment_result_has_no_status_field():
    fields = {f.name for f in assess_module.AssessmentResult.__dataclass_fields__.values()}
    assert not any("status" in f.lower() for f in fields)


# =============================================================================
# Cross-field findings and materiality, wired into assess_submission
# =============================================================================


def test_assess_submission_without_weights_config_leaves_materiality_none(
    schema, deidentifier, validator, rules_detector, review_config, sample_records
):
    """The ir35_weights_config argument is optional -- a caller that only
    wants original Phase 5 behaviour should not be forced to load a config
    it never asked for."""
    record = sample_records[0]
    result = assess_submission(
        record,
        schema=schema,
        deidentifier=deidentifier,
        validator=validator,
        detector=rules_detector,
        review_config=review_config,
    )
    for fi in result.flagged:
        assert fi.materiality is None
    for cf in result.cross_field_findings:
        assert cf.materiality is None


def test_assess_submission_attaches_materiality_when_weights_config_given(
    schema, deidentifier, validator, rules_detector, review_config, ir35_weights_config, sample_records
):
    found_any = False
    for record in sample_records:
        result = assess_submission(
            record,
            schema=schema,
            deidentifier=deidentifier,
            validator=validator,
            detector=rules_detector,
            review_config=review_config,
            ir35_weights_config=ir35_weights_config,
        )
        for fi in result.flagged:
            found_any = True
            # Every flagged pair touches a test that is either weighted or a
            # gate field, per the schema -- so materiality should resolve.
            assert fi.materiality is not None
    assert found_any, "no flags found across the sample to check materiality on"


def test_cross_field_findings_appear_for_a_record_built_to_trigger_one(
    schema, deidentifier, validator, rules_detector, review_config, ir35_weights_config
):
    record = {
        "record_id": "TEST-CROSS-0001",
        "q4_01_already_started": "Yes",
        "q4_03_substitute_sent": "Not applicable - work has not started",
    }
    result = assess_submission(
        record,
        schema=schema,
        deidentifier=deidentifier,
        validator=validator,
        detector=rules_detector,
        review_config=review_config,
        ir35_weights_config=ir35_weights_config,
    )
    ids = {cf.finding.check_id for cf in result.cross_field_findings}
    assert "x_started_but_not_applicable" in ids
    hit = next(cf for cf in result.cross_field_findings if cf.finding.check_id == "x_started_but_not_applicable")
    # q4_03_substitute_sent is named in the substitute_actually_sent_and_paid
    # gate, so this finding should carry the gate tier, not a test-weight one.
    assert hit.materiality is not None
    assert hit.materiality.is_gate


def test_materiality_modules_never_import_the_labelling_engine():
    """The same import-level guarantee as assess.py, extended to the new
    modules this addition and the follow-up (review priority) introduce."""
    import ast

    from src.models import cross_field as cross_field_module
    from src.review import materiality as materiality_module
    from src.review import review_priority as review_priority_module

    for module in (cross_field_module, materiality_module, review_priority_module):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        assert not any("cest_rules" in m for m in imported_modules)
        assert not hasattr(module, "CestRuleEngine")


def test_cross_field_findings_never_assert_a_determination(
    schema, deidentifier, validator, rules_detector, review_config, ir35_weights_config
):
    record = {
        "record_id": "TEST-CROSS-0002",
        "q2_01_via_third_party": "No",
        "q2_02_company_name": "Acme Consulting Ltd",
        "q4_16_payment_basis": "A fixed price for the project",
        "q4_12_buys_equipment": "No",
        "q4_13_vehicle_costs": "No",
        "q4_14_materials_unreimbursed": "No",
        "q4_15_other_costs": "No",
        "q4_17_put_right": "No",
    }
    result = assess_submission(
        record,
        schema=schema,
        deidentifier=deidentifier,
        validator=validator,
        detector=rules_detector,
        review_config=review_config,
        ir35_weights_config=ir35_weights_config,
    )
    assert result.cross_field_findings
    forbidden = ("is inside ir35", "is outside ir35", "determination is", "we determine")
    for cf in result.cross_field_findings:
        lowered = cf.finding.description.lower()
        assert not any(phrase in lowered for phrase in forbidden)
        if cf.materiality:
            tier_text = (cf.materiality.label + " " + cf.materiality.explanation).lower()
            assert not any(phrase in tier_text for phrase in forbidden)


# =============================================================================
# Review priority, wired into assess_submission
# =============================================================================


def test_assess_submission_always_attaches_a_review_priority(
    schema, deidentifier, validator, rules_detector, review_config, sample_records
):
    """Unlike materiality, review priority needs no ir35_weights_config --
    it is computable from bands/severities alone, with a 1.0 materiality
    multiplier when no tier is available (see review_priority.py)."""
    for record in sample_records[:10]:
        result = assess_submission(
            record,
            schema=schema,
            deidentifier=deidentifier,
            validator=validator,
            detector=rules_detector,
            review_config=review_config,
        )
        assert result.review_priority is not None
        assert result.review_priority.key in {"none", "light", "moderate", "close", "urgent"}


def test_review_priority_is_none_when_nothing_is_flagged(
    schema, deidentifier, validator, rules_detector, review_config
):
    record = {"record_id": "TEST-PRIORITY-CLEAN"}
    result = assess_submission(
        record,
        schema=schema,
        deidentifier=deidentifier,
        validator=validator,
        detector=rules_detector,
        review_config=review_config,
    )
    assert not result.flagged
    assert not result.cross_field_findings
    assert result.review_priority.key == "none"


def test_review_priority_is_urgent_for_a_gate_touching_record(
    schema, deidentifier, validator, rules_detector, review_config, ir35_weights_config
):
    """The same record used to prove the cross-field gate-tier materiality
    test above should escalate the whole submission straight to 'urgent',
    the aggregate-level equivalent of that per-item override."""
    record = {
        "record_id": "TEST-PRIORITY-URGENT",
        "q4_01_already_started": "Yes",
        "q4_03_substitute_sent": "Not applicable - work has not started",
    }
    result = assess_submission(
        record,
        schema=schema,
        deidentifier=deidentifier,
        validator=validator,
        detector=rules_detector,
        review_config=review_config,
        ir35_weights_config=ir35_weights_config,
    )
    assert result.review_priority.key == "urgent"


def test_review_priority_never_mentions_a_lean(
    schema, deidentifier, validator, rules_detector, review_config, ir35_weights_config, sample_records
):
    """Same forbidden-phrase guard used throughout this project, applied to
    the aggregate scale's own wording -- including every configured level,
    not just the ones a sample happens to reach."""
    forbidden = (
        "is inside ir35",
        "is outside ir35",
        "likely inside",
        "likely outside",
        "probably inside",
        "probably outside",
        "determination is",
        "we determine",
    )
    for level in review_config["review_priority"]["levels"]:
        lowered = (level["label"] + " " + level["description"]).lower()
        assert not any(phrase in lowered for phrase in forbidden)

    for record in sample_records[:15]:
        result = assess_submission(
            record,
            schema=schema,
            deidentifier=deidentifier,
            validator=validator,
            detector=rules_detector,
            review_config=review_config,
            ir35_weights_config=ir35_weights_config,
        )
        lowered = (result.review_priority.label + " " + result.review_priority.description).lower()
        assert not any(phrase in lowered for phrase in forbidden)


def test_review_priority_does_not_encode_direction(review_config):
    """The property review_priority.py's module docstring claims: two flags
    whose *content* points opposite ways (one instance leaning towards
    employment, the other away from it, different structured values and
    different free text) but with the same evidence strength and the same
    materiality tier score identically. review_priority_for reads only
    ``fi.band`` and ``fi.materiality`` off each item -- this test builds
    real, differently-leaning ``PairInstance``/``Flag`` objects to prove
    that, rather than passing placeholders that would make the claim true
    by construction."""
    from src.features.dataset import PairInstance
    from src.models.base import Flag
    from src.review.assess import FlaggedInstance
    from src.review.bands import ConfidenceBand
    from src.review.materiality import MaterialityTier
    from src.review.review_priority import review_priority_for

    band = ConfidenceBand(label="Priority", min_score=0.4, rationale="testing")
    tier = MaterialityTier(label="A major factor", explanation="testing", is_gate=False)

    def make_flagged(structured_value: str, free_text: str, is_outside_leaning: bool) -> FlaggedInstance:
        instance = PairInstance(
            record_id="TEST-DIRECTION",
            pair_id="p_test",
            structured_field="q4_16_payment_basis",
            free_text_field="q4_18_rationale",
            ir35_test="financial_risk",
            structured_value=structured_value,
            free_text=free_text,
            is_outside_leaning=is_outside_leaning,
            label=0,
            contradiction_type=None,
            subtlety=None,
            unit_key="unit-1",
            anomaly=None,
            archetype="test_archetype",
            register="formal",
            ir35_label="unknown",
        )
        flag = Flag(
            record_id="TEST-DIRECTION",
            pair_id="p_test",
            field_id="q4_18_rationale",
            form_ref="4.18",
            ir35_test="financial_risk",
            score=0.6,
            explanation="A tick-box and its explanation seem to disagree.",
        )
        return FlaggedInstance(flag=flag, instance=instance, band=band, materiality=tier)

    leaning_employed = [
        make_flagged("A fixed price for the project", "We pay a fixed monthly fee regardless.", False)
    ]
    leaning_self_employed = [
        make_flagged("Paid only for hours actually worked, no minimum", "They invoice us for hours logged.", True)
    ]

    priority_a = review_priority_for(leaning_employed, [], review_config)
    priority_b = review_priority_for(leaning_self_employed, [], review_config)
    assert priority_a == priority_b
