"""The typed view of the data contract must agree with the YAML it loads."""

from __future__ import annotations

import pytest

from src.ingest.schema_loader import (
    FieldType,
    PIIClass,
    Schema,
    load_generation_config,
    load_pipeline_config,
    load_schema,
)


@pytest.fixture(scope="module")
def schema() -> Schema:
    return load_schema()


def test_loads_and_indexes_every_field(schema: Schema) -> None:
    assert len(schema.fields) > 50
    assert "q4_05_right_to_reject" in schema
    assert schema["q4_05_right_to_reject"].type is FieldType.STRUCTURED


def test_pairs_expose_their_halves_as_fields(schema: Schema) -> None:
    pair = schema.pairs["p_4_05_right_to_reject"]
    assert schema[pair.structured].type is FieldType.STRUCTURED
    assert schema[pair.free_text].type is FieldType.FREE_TEXT
    assert pair.is_high_priority


def test_q4_05_polarity_matches_the_phase0_gate_decision(schema: Schema) -> None:
    """Confirmed with the user: no right to reject == unfettered substitution.

    This is the polarity of the highest-priority pair in the set. If it is ever
    flipped it must be a deliberate config edit with this test updated, not a
    silent change.
    """
    assert schema.pairs["p_4_05_right_to_reject"].outside_leaning == "No"


def test_input_fields_exclude_the_outcome_block(schema: Schema) -> None:
    """The leakage guard, expressed as a derived view rather than a filter that
    each caller has to remember to apply."""
    input_ids = {f.id for f in schema.input_fields}
    assert "ir35_outcome" not in input_ids
    assert schema.outcome_fields
    for field in schema.outcome_fields:
        assert field.id not in input_ids


def test_justification_map_covers_thirty_section_four_questions(schema: Schema) -> None:
    section_four = [
        f for f in schema.structured_answer_fields
        if str(f.form_ref or "").startswith("4.")
    ]
    mapped = [f for f in section_four if f.id in schema.justification_map]
    assert len(section_four) == 31
    assert len(mapped) == 30


def test_pii_fields_are_the_ones_the_deidentifier_will_see(schema: Schema) -> None:
    pii_ids = {f.id for f in schema.pii_fields}
    assert "q1_02_first_name" in pii_ids
    assert "q1_04_postcode" in pii_ids
    assert "q5_05_email" in pii_ids
    # A structured status answer carries no personal data and must not be
    # scrubbed — scrubbing it would destroy the signal.
    assert "q4_05_right_to_reject" not in pii_ids
    assert schema["q1_02_first_name"].pii_class is PIIClass.DIRECT_IDENTIFIER


def test_premise_rendering_uses_the_config_template(schema: Schema) -> None:
    pair = schema.pairs["p_4_05_right_to_reject"]
    premise = schema.premise_for(pair, "No")
    assert "right to reject a substitute" in premise
    assert premise.endswith("No")


def test_routing_marks_off_branch_fields_as_unreachable(schema: Schema) -> None:
    """4.3 = 'not applicable' routes to 4.5, so 4.4 is legitimately blank."""
    answers = {"q4_03_substitute_sent": "Not applicable - work has not started"}
    reachable = schema.routed_fields(answers)
    assert "q4_05_right_to_reject" in reachable


def test_for_test_groups_fields_by_ir35_test(schema: Schema) -> None:
    control = {f.id for f in schema.for_test("control")}
    assert {"q4_09_decide_how", "q4_10_decide_schedule", "q4_11_decide_where"} <= control
    assert "q4_31_other_clients" not in control


def test_pairs_for_test(schema: Schema) -> None:
    subs = schema.pairs_for_test("substitution")
    assert {p.id for p in subs} >= {"p_4_05_right_to_reject", "p_4_03_substitute_sent"}


def test_contradiction_taxonomy_is_available_to_consumers(schema: Schema) -> None:
    assert "named_person_dependency" in schema.contradiction_types
    assert schema.subtlety_levels == (1, 2, 3)


def test_pipeline_config_defaults_to_content_logging_off() -> None:
    cfg = load_pipeline_config()
    assert cfg["logging"]["log_content"] is False
    assert cfg["deidentification"]["enabled"] is True


def test_generation_config_is_the_reproducibility_contract() -> None:
    cfg = load_generation_config()
    assert isinstance(cfg["seed"], int)
    assert cfg["n_records"] == 350
    assert abs(sum(cfg["archetype_mix"].values()) - 1.0) < 1e-9
    assert set(cfg["archetypes"]) == set(cfg["archetype_mix"])


def test_missing_config_raises_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schema(tmp_path)
