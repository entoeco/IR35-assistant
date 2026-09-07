"""Record validation behaviour.

The interesting case is the routing-aware completeness check. A validator that
reports every skipped question as missing produces a wall of findings on every
record, and a check that cries wolf on every record gets switched off.
"""

from __future__ import annotations

import pytest

from src.ingest.schema_loader import load_generation_config, load_schema
from src.ingest.validate import RecordValidator, Severity


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def validator(schema) -> RecordValidator:
    return RecordValidator(schema, load_generation_config().get("validation"))


def codes(result) -> set[str]:
    return {f.code for f in result.findings}


def test_value_outside_the_declared_domain_is_an_error(validator) -> None:
    """Config and data have diverged; every downstream assumption is void."""
    result = validator.validate(
        {"record_id": "T1", "q4_05_right_to_reject": "Maybe, we would discuss it"}
    )
    assert not result.is_valid
    assert "value_not_in_domain" in codes(result)


def test_valid_option_passes(validator) -> None:
    result = validator.validate(
        {
            "record_id": "T2",
            "q4_05_right_to_reject": "No",
            "q4_05_rationale": "The supplier may substitute at their own discretion.",
        }
    )
    assert result.is_valid
    assert "value_not_in_domain" not in codes(result)


def test_blank_justification_is_a_warning_not_an_error(validator) -> None:
    """The form says an ESQ is rejected if fields are incomplete, so a reviewer
    must see this — but the record is still usable, so it does not fail the
    contract."""
    result = validator.validate(
        {"record_id": "T3", "q4_05_right_to_reject": "No", "q4_05_rationale": ""}
    )
    assert result.is_valid
    assert "missing_justification" in codes(result)
    finding = next(f for f in result.findings if f.code == "missing_justification")
    assert finding.severity is Severity.WARNING


def test_justification_too_short_is_flagged(validator) -> None:
    """A rationale too short to say anything is a blank one wearing a hat."""
    result = validator.validate(
        {"record_id": "T4", "q4_05_right_to_reject": "No", "q4_05_rationale": "n/a"}
    )
    assert "justification_too_short" in codes(result)


def test_skipped_questions_do_not_report_as_missing(validator) -> None:
    """Routing awareness. 4.3 = 'not applicable' skips 4.4, so 4.4 being blank
    is correct rather than a finding."""
    result = validator.validate(
        {
            "record_id": "T5",
            "q4_03_substitute_sent": "Not applicable - work has not started",
            "q4_03_rationale": "The engagement has not begun, so this does not arise.",
        }
    )
    missing = [f.field_id for f in result.findings if f.code == "missing_on_route"]
    assert "q4_04_worker_pays_substitute" not in missing


def test_answering_a_skipped_question_is_flagged(validator) -> None:
    result = validator.validate(
        {
            "record_id": "T6",
            "q4_03_substitute_sent": "Not applicable - work has not started",
            "q4_04_worker_pays_substitute": "Yes",
        }
    )
    off_route = [f.field_id for f in result.findings if f.code == "answered_off_route"]
    assert "q4_04_worker_pays_substitute" in off_route


def test_outcome_leakage_is_an_error(validator) -> None:
    """The determination travelling with a record is silent damage: the model
    just gets better and nobody asks why."""
    result = validator.validate({"record_id": "T7", "ir35_outcome": "inside"})
    assert not result.is_valid
    assert "outcome_leakage" in codes(result)


def test_result_summary_carries_no_content(validator) -> None:
    """Validation results are logged under constraint 4."""
    result = validator.validate(
        {
            "record_id": "T8",
            "q4_05_right_to_reject": "No",
            "q4_05_rationale": "Dr Ashwood cannot be replaced by anyone else.",
        }
    )
    assert "Ashwood" not in str(result.as_log_meta())


def test_routing_checks_can_be_disabled(schema) -> None:
    """Some consumers only care about contract conformance."""
    validator = RecordValidator(schema, {"enforce_routing": False,
                                         "require_justification_for_routed_fields": False})
    result = validator.validate(
        {"record_id": "T9", "q4_03_substitute_sent": "Not applicable - work has not started"}
    )
    assert not any(f.code in ("missing_on_route", "answered_off_route") for f in result.findings)


def test_empty_record_is_contract_valid(validator) -> None:
    """Nothing to contradict the contract. Completeness findings still appear
    where a question was answered — of which there are none here."""
    result = validator.validate({"record_id": "T10"})
    assert result.is_valid
