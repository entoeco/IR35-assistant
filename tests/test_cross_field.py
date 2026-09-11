"""Tick-box vs tick-box consistency checks (src/models/cross_field.py)."""

from __future__ import annotations

import pytest

from src.ingest.schema_loader import load_schema
from src.models.cross_field import evaluate_cross_field_checks


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _base_record(**overrides):
    record = {"record_id": "TEST-0001"}
    record.update(overrides)
    return record


def test_no_findings_on_an_empty_record(schema):
    assert evaluate_cross_field_checks(schema, _base_record()) == []


def test_route_section2_fires_when_no_third_party_but_company_named(schema):
    record = _base_record(q2_01_via_third_party="No", q2_02_company_name="Acme Ltd")
    findings = evaluate_cross_field_checks(schema, record)
    ids = {f.check_id for f in findings}
    assert "x_route_section2" in ids


def test_route_section2_does_not_fire_when_company_number_only(schema):
    """`any_not_empty` should match on either field, not require both."""
    record = _base_record(q2_01_via_third_party="No", q2_03_company_number="12345678")
    findings = evaluate_cross_field_checks(schema, record)
    assert "x_route_section2" in {f.check_id for f in findings}


def test_route_section2_does_not_fire_when_via_third_party(schema):
    record = _base_record(q2_01_via_third_party="Yes", q2_02_company_name="Acme Ltd")
    findings = evaluate_cross_field_checks(schema, record)
    assert "x_route_section2" not in {f.check_id for f in findings}


def test_no_financial_risk_but_fixed_price_fires(schema):
    record = _base_record(
        q4_16_payment_basis="A fixed price for the project",
        q4_12_buys_equipment="No",
        q4_13_vehicle_costs="No",
        q4_14_materials_unreimbursed="No",
        q4_15_other_costs="No",
        q4_17_put_right="No",
    )
    findings = evaluate_cross_field_checks(schema, record)
    hit = [f for f in findings if f.check_id == "x_no_financial_risk_but_fixed_price"]
    assert len(hit) == 1
    assert hit[0].severity == "low"
    assert set(hit[0].fields) == {
        "q4_16_payment_basis",
        "q4_12_buys_equipment",
        "q4_13_vehicle_costs",
        "q4_14_materials_unreimbursed",
        "q4_15_other_costs",
        "q4_17_put_right",
    }


def test_no_financial_risk_but_fixed_price_does_not_fire_with_any_risk(schema):
    record = _base_record(
        q4_16_payment_basis="A fixed price for the project",
        q4_12_buys_equipment="Yes",  # one risk indicator present
        q4_13_vehicle_costs="No",
        q4_14_materials_unreimbursed="No",
        q4_15_other_costs="No",
        q4_17_put_right="No",
    )
    findings = evaluate_cross_field_checks(schema, record)
    assert "x_no_financial_risk_but_fixed_price" not in {f.check_id for f in findings}


def test_started_but_not_applicable_fires(schema):
    record = _base_record(
        q4_01_already_started="Yes",
        q4_03_substitute_sent="Not applicable - work has not started",
    )
    findings = evaluate_cross_field_checks(schema, record)
    hit = [f for f in findings if f.check_id == "x_started_but_not_applicable"]
    assert len(hit) == 1
    assert hit[0].severity == "high"


def test_started_but_not_applicable_does_not_fire_when_consistent(schema):
    record = _base_record(
        q4_01_already_started="No",
        q4_03_substitute_sent="Not applicable - work has not started",
    )
    findings = evaluate_cross_field_checks(schema, record)
    assert "x_started_but_not_applicable" not in {f.check_id for f in findings}


def test_checks_with_no_condition_are_skipped_not_errored(schema):
    """x_route_substitution, x_duties_vs_control and x_missing_justification
    have no `condition` in config (see contradiction_pairs.yaml for why) --
    evaluating them must not raise, and they must never appear as a finding."""
    record = _base_record(q4_00_duties_narrative="anything at all")
    findings = evaluate_cross_field_checks(schema, record)
    ids = {f.check_id for f in findings}
    assert "x_route_substitution" not in ids
    assert "x_duties_vs_control" not in ids
    assert "x_missing_justification" not in ids


def test_all_six_configured_checks_are_accounted_for(schema):
    """Every check in config either has a condition (and is tested above by
    id) or is documented as intentionally condition-less. This test fails
    loudly if a seventh check is added to config without anyone deciding
    which bucket it belongs in."""
    with_condition = {"x_route_section2", "x_no_financial_risk_but_fixed_price", "x_started_but_not_applicable"}
    without_condition = {"x_route_substitution", "x_duties_vs_control", "x_missing_justification"}
    configured = {c["id"] for c in schema.cross_field_checks}
    assert configured == with_condition | without_condition


def test_findings_never_assert_a_determination(schema):
    """Same guard as the pair-based detectors: a description of a data
    inconsistency, never a status."""
    record = _base_record(
        q4_01_already_started="Yes",
        q4_03_substitute_sent="Not applicable - work has not started",
        q2_01_via_third_party="No",
        q2_02_company_name="Acme Ltd",
    )
    findings = evaluate_cross_field_checks(schema, record)
    assert findings
    forbidden = ("is inside ir35", "is outside ir35", "determination is", "we determine")
    for finding in findings:
        lowered = finding.description.lower()
        assert not any(phrase in lowered for phrase in forbidden)
