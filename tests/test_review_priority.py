"""Review priority (src/review/review_priority.py).

Unit-level tests against the scoring function in isolation, using minimal
stand-in ``FlaggedInstance``/``CrossFieldFindingWithMateriality`` objects
built directly rather than run through the full pipeline -- ``test_review.py``
covers the pipeline wiring (``assess_submission`` attaching a priority) and
the "does not encode direction" property against real, differently-leaning
flag content. This file is about the scoring rules themselves: thresholds,
the gate override, and the never-a-lean wording guard applied to every
configured level.
"""

from __future__ import annotations

import inspect

import pytest

from src.ingest.schema_loader import CONFIG_DIR, load_yaml
from src.models.cross_field import CrossFieldFinding
from src.review.assess import CrossFieldFindingWithMateriality, FlaggedInstance
from src.review.bands import ConfidenceBand
from src.review.materiality import MaterialityTier
from src.review.review_priority import review_priority_for


@pytest.fixture(scope="module")
def review_config():
    return load_yaml(CONFIG_DIR / "review.yaml")


def _flag(band_label: str, tier: MaterialityTier | None) -> FlaggedInstance:
    band = ConfidenceBand(label=band_label, min_score=0.3, rationale="testing")
    return FlaggedInstance(flag=None, instance=None, band=band, materiality=tier)


def _finding(severity: str, tier: MaterialityTier | None) -> CrossFieldFindingWithMateriality:
    finding = CrossFieldFinding(
        check_id="x_test", description="testing", severity=severity, fields=("f1",)
    )
    return CrossFieldFindingWithMateriality(finding=finding, materiality=tier)


CONTRIBUTING = MaterialityTier(label="One of several contributing factors", explanation="x", is_gate=False)
MAJOR = MaterialityTier(label="A major factor", explanation="x", is_gate=False)
FOUNDATIONAL = MaterialityTier(label="One of the two foundational tests", explanation="x", is_gate=False)
GATE = MaterialityTier(label="Could be decisive on its own", explanation="x", is_gate=True)


def test_nothing_flagged_gives_the_none_level(review_config):
    priority = review_priority_for([], [], review_config)
    assert priority.key == "none"


def test_a_single_low_stakes_flag_gives_light(review_config):
    priority = review_priority_for([_flag("Worth a look", CONTRIBUTING)], [], review_config)
    assert priority.key == "light"


def test_a_strong_foundational_flag_reaches_close_on_its_own(review_config):
    """High priority (weight 3) x foundational (weight 2) = 6, clearing the
    'close' threshold from a single flag -- the scenario the scoring
    comments in config/review.yaml describe explicitly."""
    priority = review_priority_for([_flag("High priority", FOUNDATIONAL)], [], review_config)
    assert priority.key == "close"


def test_several_low_stakes_flags_add_up_to_moderate(review_config):
    flags = [_flag("Worth a look", CONTRIBUTING) for _ in range(3)]
    priority = review_priority_for(flags, [], review_config)
    assert priority.key == "moderate"


def test_gate_materiality_on_a_flag_forces_urgent_regardless_of_score(review_config):
    """A gate-tier flag alone -- even the weakest band -- must still reach
    'urgent', because the override is checked before any score is summed,
    not as a side effect of a big number."""
    priority = review_priority_for([_flag("Worth a look", GATE)], [], review_config)
    assert priority.key == "urgent"


def test_gate_materiality_on_a_cross_field_finding_forces_urgent(review_config):
    priority = review_priority_for([], [_finding("low", GATE)], review_config)
    assert priority.key == "urgent"


def test_a_gate_flag_among_many_low_stakes_ones_still_wins(review_config):
    """The override must not be diluted by averaging -- one determinative
    fact plus a pile of low-stakes ones is still 'urgent', not 'moderate'."""
    flags = [_flag("Worth a look", CONTRIBUTING) for _ in range(2)] + [_flag("Worth a look", GATE)]
    priority = review_priority_for(flags, [], review_config)
    assert priority.key == "urgent"


def test_cross_field_findings_alone_can_drive_the_score(review_config):
    priority = review_priority_for([], [_finding("high", MAJOR)], review_config)
    assert priority.key in {"moderate", "close"}


def test_items_with_no_materiality_use_the_default_weight(review_config):
    """When ir35_weights_config was not supplied upstream, materiality is
    None throughout -- this must not crash, and must still produce a
    sensible (non-gate) level."""
    priority = review_priority_for([_flag("Priority", None)], [], review_config)
    assert priority.key in {"light", "moderate"}


def test_same_inputs_always_score_the_same(review_config):
    flags = [_flag("Priority", MAJOR)]
    first = review_priority_for(flags, [], review_config)
    second = review_priority_for(flags, [], review_config)
    assert first == second


def test_function_signature_takes_no_raw_record():
    """Structural guarantee, same pattern as materiality_for: there is no
    parameter this function's caller could even mistakenly pass a raw
    record or a computed status through."""
    params = set(inspect.signature(review_priority_for).parameters)
    assert params == {"flagged", "cross_field_findings", "review_config"}


def test_configured_levels_never_mention_a_lean_or_a_status(review_config):
    """Same drift-detector pattern as materiality's tier-wording test and
    the app's scope-banner test, applied to every configured review-priority
    level -- not just the ones a particular scenario happens to reach."""
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
    levels = review_config["review_priority"]["levels"]
    assert len(levels) == 5
    for level in levels:
        lowered = (level["label"] + " " + level["description"]).lower()
        assert not any(phrase in lowered for phrase in forbidden)


def test_unknown_band_or_severity_label_does_not_crash(review_config):
    """A band/severity label absent from the scoring config's weight tables
    should fall back gracefully (get(..., default)), not raise -- config and
    code are edited in different files, so this guards against a renamed
    band label silently breaking the aggregate instead of just under-scoring."""
    priority = review_priority_for([_flag("Some New Band", CONTRIBUTING)], [], review_config)
    assert priority.key in {"none", "light", "moderate", "close", "urgent"}
