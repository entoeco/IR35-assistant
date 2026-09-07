"""The labelling function must behave the way the documentation claims.

This engine produces the dataset's ground truth, so a silent error here
propagates into every downstream result. These tests pin the behaviour that the
Phase 2 and Phase 4 write-ups will rely on.
"""

from __future__ import annotations

import pytest

from src.ingest.schema_loader import load_schema
from src.models.cest_rules import CestRuleEngine, Status


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def engine(schema) -> CestRuleEngine:
    return CestRuleEngine(schema)


STRONGLY_OUTSIDE = {
    "q4_02_office_holder": "No",
    "q4_05_right_to_reject": "No",
    "q4_07_would_pay_substitute": "Yes",
    "q4_08_moved_from_task": "No - that would require a new contract or formal working arrangement",
    "q4_09_decide_how": "No - the worker solely decides",
    "q4_10_decide_schedule": "No - the work is based on agreed deadlines",
    "q4_11_decide_where": "No - the worker decides",
    "q4_12_buys_equipment": "Yes",
    "q4_14_materials_unreimbursed": "Yes",
    "q4_16_payment_basis": "A fixed price for the project",
    "q4_17_put_right": "Yes - unpaid & they would have extra costs that the University would not pay for",
    "q4_18_benefits": "No",
    "q4_20_identifies_as": "They work for their own business",
    "q4_31_other_clients": "Yes",
}

STRONGLY_INSIDE = {
    "q4_02_office_holder": "No",
    "q4_05_right_to_reject": "Yes",
    "q4_07_would_pay_substitute": "No",
    "q4_08_moved_from_task": "Yes",
    "q4_09_decide_how": "Yes",
    "q4_10_decide_schedule": "Yes",
    "q4_11_decide_where": "Yes",
    "q4_12_buys_equipment": "No",
    "q4_14_materials_unreimbursed": "No",
    "q4_16_payment_basis": "An hourly/daily or weekly rate",
    "q4_17_put_right": "No",
    "q4_18_benefits": "Yes",
    "q4_20_identifies_as": "They work for the organisation (the University)",
    "q4_31_other_clients": "No",
    "q4_28_starts_immediately_after": "Yes",
}


# --- determinative gates -----------------------------------------------------

def test_office_holder_is_determinative_inside(engine: CestRuleEngine) -> None:
    """An office holder is inside regardless of everything else. The gate must
    beat an otherwise overwhelmingly outside-leaning form."""
    answers = dict(STRONGLY_OUTSIDE, **{"q4_02_office_holder": "Yes"})
    result = engine.evaluate(answers)
    assert result.status is Status.INSIDE
    assert result.gate == "office_holder"
    assert result.score > 0  # the weighted score still points outside...
    # ...and the gate overrides it. That asymmetry is the point of a gate.


def test_substitute_sent_and_paid_is_determinative_outside(engine: CestRuleEngine) -> None:
    answers = dict(
        STRONGLY_INSIDE,
        **{
            "q4_03_substitute_sent": "Yes - you accepted them",
            "q4_04_worker_pays_substitute": "Yes",
        },
    )
    result = engine.evaluate(answers)
    assert result.status is Status.OUTSIDE
    assert result.gate == "substitute_actually_sent_and_paid"


def test_paid_another_person_is_determinative_outside(engine: CestRuleEngine) -> None:
    answers = dict(STRONGLY_INSIDE, **{"q4_06_paid_another_person": "Yes"})
    assert engine.evaluate(answers).gate == "paid_another_for_significant_portion"


def test_gate_order_puts_office_holder_first(engine: CestRuleEngine) -> None:
    """Both an office holder and a paid substitute: inside wins, because being
    an office holder brings the engagement in regardless of substitution."""
    answers = {
        "q4_02_office_holder": "Yes",
        "q4_06_paid_another_person": "Yes",
    }
    result = engine.evaluate(answers)
    assert result.status is Status.INSIDE
    assert result.gate == "office_holder"


def test_gate_reason_is_recorded(engine: CestRuleEngine) -> None:
    result = engine.evaluate({"q4_02_office_holder": "Yes"})
    assert "office holder" in result.reason.lower()
    assert "\n" not in result.reason


# --- scoring -----------------------------------------------------------------

def test_strongly_outside_answers_label_outside(engine: CestRuleEngine) -> None:
    result = engine.evaluate(STRONGLY_OUTSIDE)
    assert result.status is Status.OUTSIDE
    assert result.gate is None
    assert result.score > 0.12


def test_strongly_inside_answers_label_inside(engine: CestRuleEngine) -> None:
    result = engine.evaluate(STRONGLY_INSIDE)
    assert result.status is Status.INSIDE
    assert result.score < -0.12


def test_q4_05_polarity_moves_the_score_outward(engine: CestRuleEngine) -> None:
    """The Phase 0 gate decision, expressed as behaviour rather than config."""
    with_veto = engine.evaluate({"q4_05_right_to_reject": "Yes"})
    without_veto = engine.evaluate({"q4_05_right_to_reject": "No"})
    assert without_veto.score > with_veto.score
    assert without_veto.status is Status.OUTSIDE
    assert with_veto.status is Status.INSIDE


def test_opposing_tests_produce_undetermined(engine: CestRuleEngine) -> None:
    """Strong substitution rights alongside heavy day-to-day control. This is
    the 'unable to determine' case CEST itself returns, and the corpus needs it.
    """
    answers = {
        "q4_05_right_to_reject": "No",
        "q4_07_would_pay_substitute": "Yes",
        "q4_09_decide_how": "Yes",
        "q4_10_decide_schedule": "Yes",
        "q4_11_decide_where": "Yes",
    }
    result = engine.evaluate(answers)
    assert result.status is Status.UNDETERMINED
    assert "opposite directions" in result.reason


def test_empty_answers_are_undetermined_not_neutral(engine: CestRuleEngine) -> None:
    result = engine.evaluate({})
    assert result.status is Status.UNDETERMINED
    assert result.n_answered == 0
    assert result.test_scores == ()


def test_weights_are_renormalised_over_answered_tests(engine: CestRuleEngine) -> None:
    """A form answering only control questions is scored on control, not
    dragged towards neutral by the tests it is silent on."""
    result = engine.evaluate({"q4_09_decide_how": "Yes", "q4_10_decide_schedule": "Yes"})
    assert len(result.test_scores) == 1
    assert result.test_scores[0].test == "control"
    assert result.test_scores[0].weight == pytest.approx(1.0)
    assert result.status is Status.INSIDE


def test_unknown_values_and_fields_are_ignored(engine: CestRuleEngine) -> None:
    """Robustness: a revised form with a new option must not crash the engine."""
    result = engine.evaluate(
        {
            "q4_09_decide_how": "Some option that does not exist",
            "not_a_field": "Yes",
            "q4_10_decide_schedule": "Yes",
        }
    )
    assert result.n_answered == 1


def test_free_text_is_ignored(engine: CestRuleEngine) -> None:
    """The engine reads structured answers only. If it read the justifications
    the ground truth would move whenever a contradiction was planted, and there
    would be nothing stable to evaluate against."""
    clean = engine.evaluate(STRONGLY_OUTSIDE)
    with_text = engine.evaluate(
        dict(STRONGLY_OUTSIDE, **{"q4_05_rationale": "we specifically need this individual"})
    )
    assert clean.score == with_text.score
    assert clean.status is with_text.status


# --- inspectability ----------------------------------------------------------

def test_result_serialises_with_full_breakdown(engine: CestRuleEngine) -> None:
    """A labelling function nobody can inspect is not evidence of anything."""
    payload = engine.evaluate(STRONGLY_OUTSIDE).as_dict()
    assert payload["status"] == "outside"
    assert "control" in payload["test_scores"]
    control = payload["test_scores"]["control"]
    assert {"raw", "normalised", "weight", "contribution", "n_answered"} <= control.keys()


def test_test_scores_are_ordered_by_influence(engine: CestRuleEngine) -> None:
    scores = engine.evaluate(STRONGLY_OUTSIDE).test_scores
    contributions = [abs(s.contribution) for s in scores]
    assert contributions == sorted(contributions, reverse=True)


def test_reason_names_the_driving_tests(engine: CestRuleEngine) -> None:
    reason = engine.evaluate(STRONGLY_OUTSIDE).reason
    assert "points outside" in reason
    assert any(t in reason for t in ("substitution", "control", "financial_risk"))


def test_engine_reports_its_config_version(engine: CestRuleEngine) -> None:
    """Ground truth is only reproducible if the weighting is versioned."""
    assert engine.version != "unknown"
