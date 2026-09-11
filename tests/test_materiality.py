"""Materiality tiers (src/review/materiality.py).

The property that matters most here is the safety one: a tier is a function
of the RULEBOOK (a test id, a field id), never of a RECORD. These tests check
that behaviourally (same input, same output, always) and structurally (the
function cannot even accept a record) rather than trusting the module
docstring's claim.
"""

from __future__ import annotations

import inspect

import pytest

from src.ingest.schema_loader import CONFIG_DIR, load_yaml
from src.review.materiality import materiality_for


@pytest.fixture(scope="module")
def review_config():
    return load_yaml(CONFIG_DIR / "review.yaml")


@pytest.fixture(scope="module")
def ir35_weights_config():
    return load_yaml(CONFIG_DIR / "ir35_weights.yaml")


def test_function_signature_has_no_record_or_score_parameter():
    """Structural guarantee, not just a behavioural one: this function could
    not be fed a record or a score even by mistake, because the parameter
    does not exist."""
    params = set(inspect.signature(materiality_for).parameters)
    assert params == {"review_config", "ir35_weights_config", "ir35_test", "field_id"}


@pytest.mark.parametrize("test_id", ["substitution", "control"])
def test_foundational_tests_get_the_foundational_tier(review_config, ir35_weights_config, test_id):
    tier = materiality_for(review_config, ir35_weights_config, ir35_test=test_id)
    assert tier is not None
    assert tier.label == "One of the two foundational tests"
    assert not tier.is_gate


def test_financial_risk_gets_the_major_factor_tier(review_config, ir35_weights_config):
    tier = materiality_for(review_config, ir35_weights_config, ir35_test="financial_risk")
    assert tier is not None
    assert tier.label == "A major factor"


@pytest.mark.parametrize(
    "test_id", ["part_and_parcel", "business_on_own_account", "mutuality_of_obligation"]
)
def test_all_the_circumstances_tests_get_the_contributing_tier(
    review_config, ir35_weights_config, test_id
):
    tier = materiality_for(review_config, ir35_weights_config, ir35_test=test_id)
    assert tier is not None
    assert tier.label == "One of several contributing factors"


def test_unweighted_test_gets_no_tier(review_config, ir35_weights_config):
    """contract_basis is explicit in schema.yaml as "not a test in itself" and
    carries no weight -- there is nothing to describe."""
    tier = materiality_for(review_config, ir35_weights_config, ir35_test="contract_basis")
    assert tier is None


def test_unknown_test_id_gets_no_tier(review_config, ir35_weights_config):
    tier = materiality_for(review_config, ir35_weights_config, ir35_test="not_a_real_test")
    assert tier is None


@pytest.mark.parametrize(
    "field_id",
    [
        "q4_02_office_holder",
        "q4_03_substitute_sent",
        "q4_04_worker_pays_substitute",
        "q4_06_paid_another_person",
    ],
)
def test_gate_fields_get_the_gate_tier_regardless_of_test_weight(
    review_config, ir35_weights_config, field_id
):
    """These fields are named in a determinative gate's `when` clause in
    ir35_weights.yaml. The gate tier must win even though some of them
    (q4_03, q4_04, q4_06) also belong to a weighted test."""
    tier = materiality_for(
        review_config, ir35_weights_config, ir35_test="substitution", field_id=field_id
    )
    assert tier is not None
    assert tier.is_gate
    assert tier.label == "Could be decisive on its own"


def test_non_gate_field_falls_back_to_the_test_tier(review_config, ir35_weights_config):
    tier = materiality_for(
        review_config,
        ir35_weights_config,
        ir35_test="financial_risk",
        field_id="q4_12_buys_equipment",
    )
    assert tier is not None
    assert not tier.is_gate
    assert tier.label == "A major factor"


def test_same_test_id_always_returns_the_same_tier(review_config, ir35_weights_config):
    """The determinism the safety argument depends on, checked directly."""
    first = materiality_for(review_config, ir35_weights_config, ir35_test="control")
    second = materiality_for(review_config, ir35_weights_config, ir35_test="control")
    assert first == second


def test_tier_labels_never_mention_inside_or_outside(review_config, ir35_weights_config):
    """A tier describes IMPORTANCE, never DIRECTION. If the word "inside" or
    "outside" ever appears in a tier's wording, it has drifted into being a
    lean rather than a materiality description."""
    weights = ir35_weights_config["test_weights"]
    checked = 0
    for test_id in weights:
        tier = materiality_for(review_config, ir35_weights_config, ir35_test=test_id)
        if tier is None:
            continue
        checked += 1
        lowered = (tier.label + " " + tier.explanation).lower()
        assert "inside ir35" not in lowered
        assert "outside ir35" not in lowered
    assert checked == len(weights)
