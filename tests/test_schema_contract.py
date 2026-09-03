"""Phase 0 verification: the data contract must be internally consistent.

These tests run before any data exists. They check that the two config files
agree with each other and with the structural facts extracted from the ESQ
template, so that a later edit to config cannot silently break ingest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    """The ESQ data contract."""
    return yaml.safe_load((CONFIG / "schema.yaml").read_text())


@pytest.fixture(scope="module")
def pairs() -> dict[str, Any]:
    """The contradiction-pair definitions."""
    return yaml.safe_load((CONFIG / "contradiction_pairs.yaml").read_text())


@pytest.fixture(scope="module")
def by_id(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in schema["fields"]}


def test_field_ids_unique(schema: dict[str, Any]) -> None:
    ids = [f["id"] for f in schema["fields"]]
    assert len(ids) == len(set(ids)), "duplicate field ids in schema.yaml"


def test_every_structured_field_has_a_value_domain(schema: dict[str, Any]) -> None:
    """Structured means 'drawn from a closed list' — except the two the template
    genuinely leaves open (2.5 trading history, and the outcome block)."""
    exempt = {"q2_05_years_in_business"}
    for f in schema["fields"]:
        if f["type"] == "structured" and f["id"] not in exempt:
            assert f.get("value_domain"), f"{f['id']} has no value_domain"


def test_every_field_carries_a_pii_class(schema: dict[str, Any]) -> None:
    """The de-identifier dispatches on pii_class, so an unclassified field is a
    field that would pass through unscrubbed."""
    valid = set(schema["pii_classes"])
    for f in schema["fields"]:
        assert f.get("pii_class") in valid, f"{f['id']} has no valid pii_class"


def test_free_text_justifications_point_at_a_real_field(by_id: dict[str, Any]) -> None:
    for fid, f in by_id.items():
        target = f.get("justification_for")
        if target is not None:
            assert target in by_id, f"{fid} justifies unknown field {target}"


def test_section_four_pairing(by_id: dict[str, Any]) -> None:
    """30 of the 31 Section 4 questions have a rationale box. Q4.1 does not.

    This is the structural fact the whole detector rests on, so it is asserted
    rather than assumed.
    """
    q4 = [f for f in by_id.values()
          if str(f.get("form_ref", "")).startswith("4.")
          and f["type"] == "structured"]
    assert len(q4) == 31, f"expected 31 Section 4 questions, found {len(q4)}"

    justified = {f["justification_for"] for f in by_id.values()
                 if f.get("justification_for")}
    unpaired = [f["id"] for f in q4 if f["id"] not in justified]
    assert unpaired == ["q4_01_already_started"], f"unexpected unpaired: {unpaired}"


def test_pairs_resolve_against_schema(pairs: dict[str, Any], by_id: dict[str, Any]) -> None:
    for p in pairs["pairs"]:
        assert p["structured"] in by_id, f"{p['id']}: unknown structured field"
        assert p["free_text"] in by_id, f"{p['id']}: unknown free_text field"
        assert by_id[p["structured"]]["type"] == "structured"
        assert by_id[p["free_text"]]["type"] == "free_text"


def test_pairs_are_within_one_ir35_test(pairs: dict[str, Any], by_id: dict[str, Any]) -> None:
    """A pair only means something if both halves address the same test."""
    for p in pairs["pairs"]:
        assert (by_id[p["structured"]]["ir35_test"]
                == by_id[p["free_text"]]["ir35_test"]
                == p["ir35_test"]), f"{p['id']}: ir35_test mismatch"


def test_outside_leaning_value_is_in_the_domain(pairs: dict[str, Any], by_id: dict[str, Any]) -> None:
    for p in pairs["pairs"]:
        domain = by_id[p["structured"]]["value_domain"]
        assert p["outside_leaning"] in domain, (
            f"{p['id']}: outside_leaning {p['outside_leaning']!r} not in domain"
        )


def test_plantable_types_are_declared(pairs: dict[str, Any], schema: dict[str, Any]) -> None:
    known = set(schema["contradiction_types"])
    for p in pairs["pairs"]:
        for t in p.get("plantable_types", []):
            assert t in known, f"{p['id']}: unknown contradiction type {t}"


def test_outcome_fields_are_flagged_for_stripping(schema: dict[str, Any]) -> None:
    """Leakage guard. The IR35 team's block sits in the same sheet as the
    inputs; ingest strips it by assertion, not by omission."""
    outcomes = [f["id"] for f in schema["fields"] if f.get("role") == "outcome"]
    assert "ir35_outcome" in outcomes
    assert len(outcomes) >= 4


def test_no_pair_uses_an_outcome_field(pairs: dict[str, Any], by_id: dict[str, Any]) -> None:
    for p in pairs["pairs"]:
        for side in ("structured", "free_text"):
            assert by_id[p[side]].get("role") != "outcome", (
                f"{p['id']} would leak the determination"
            )
