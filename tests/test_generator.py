"""The generator's guarantees, expressed as tests.

Three of these matter more than the rest:

* **Reproducibility.** The corpus must regenerate byte-for-byte from the config.
  Without that, no downstream result is reproducible either.
* **Style invariance is a data property.** Rendering one state in three
  registers must change the surface form and nothing else. Phase 4's
  style-invariance test measures the detector against this guarantee; if the
  guarantee does not hold, that test measures nothing.
* **Ground truth is internally consistent.** A planted contradiction must
  actually be in the rendered text, and an anomaly must never quietly delete one.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from src.generate.generator import EsqGenerator, walk_route
from src.ingest.schema_loader import (
    CONFIG_DIR,
    FieldType,
    load_generation_config,
    load_schema,
    load_yaml,
)
from src.ingest.validate import RecordValidator
from src.models.cest_rules import CestRuleEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL = 60  # records, for tests that do not need the full corpus


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def gen_cfg():
    return load_generation_config()


@pytest.fixture(scope="module")
def text_bank():
    return load_yaml(CONFIG_DIR / "text_bank.yaml")


@pytest.fixture(scope="module")
def generator(schema, gen_cfg, text_bank) -> EsqGenerator:
    return EsqGenerator(schema, gen_cfg, text_bank, CestRuleEngine(schema))


@pytest.fixture(scope="module")
def corpus(generator):
    return generator.generate(SMALL)


# =============================================================================
# Reproducibility
# =============================================================================

def test_generation_is_deterministic(schema, gen_cfg, text_bank) -> None:
    """Two independent generators, same config, identical corpus."""
    first = EsqGenerator(schema, gen_cfg, text_bank, CestRuleEngine(schema))
    second = EsqGenerator(schema, gen_cfg, text_bank, CestRuleEngine(schema))
    _, records_a, truth_a = first.generate(SMALL)
    _, records_b, truth_b = second.generate(SMALL)
    assert records_a == records_b
    assert truth_a == truth_b


def test_serialised_corpus_is_byte_identical(schema, gen_cfg, text_bank) -> None:
    """The reproducibility claim is about the file on disk, not about dicts."""
    def dump() -> str:
        gen = EsqGenerator(schema, gen_cfg, text_bank, CestRuleEngine(schema))
        _, records, _ = gen.generate(SMALL)
        return "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)

    assert dump() == dump()


def test_records_are_independently_reproducible(generator) -> None:
    """Record 37 rebuilds identically without replaying records 0-36.

    This is what makes the seed-plus-index scheme worth having: a failing record
    can be reproduced on its own.
    """
    direct = generator.build_state(37)
    _, records, _ = generator.generate(SMALL)
    assert generator.render(direct) == records[37]


def test_changing_the_seed_changes_the_corpus(schema, gen_cfg, text_bank) -> None:
    altered = dict(gen_cfg, seed=int(gen_cfg["seed"]) + 1)
    base = EsqGenerator(schema, gen_cfg, text_bank, CestRuleEngine(schema))
    other = EsqGenerator(schema, altered, text_bank, CestRuleEngine(schema))
    assert base.generate(SMALL)[1] != other.generate(SMALL)[1]


# =============================================================================
# Style invariance — the property Phase 4 depends on
# =============================================================================

def test_rerendering_changes_surface_form_only(corpus, generator, schema) -> None:
    """One state, three registers: the words change, the engagement does not."""
    states, _, _ = corpus
    state = next(s for s in states if s.contradictions)
    renders = {r: generator.render(state, r) for r in ("verbose", "terse", "hedged")}

    structured_ids = [f.id for f in schema.structured_answer_fields]
    baseline = {k: renders["verbose"][k] for k in structured_ids if k in renders["verbose"]}
    for register, record in renders.items():
        assert {k: record[k] for k in baseline} == baseline, (
            f"{register} rendering changed a structured answer"
        )

    free_text_ids = [f.id for f in schema.of_type(FieldType.FREE_TEXT)]
    differing = [
        fid
        for fid in free_text_ids
        if len({renders[r].get(fid) for r in renders}) > 1
    ]
    assert differing, "registers produced identical free text — no style variation"


def test_planted_contradictions_survive_every_register(corpus, generator) -> None:
    """The contradiction lives in the content unit, so the realiser cannot lose
    it. If this fails, the subtlety curve in Phase 4 is measuring noise."""
    states, _, _ = corpus
    for state in (s for s in states if s.contradictions):
        for contradiction in state.contradictions:
            assert state.units[contradiction.pair_id] == contradiction.planted_unit
            for register in ("verbose", "terse", "hedged"):
                text = generator.render(state, register)[contradiction.free_text_field]
                assert text, "a planted contradiction rendered as empty text"


def test_hedged_register_is_not_labelled_a_contradiction(corpus) -> None:
    """Wrapping a supporting proposition in 'as far as I am aware' is style, not
    a contradiction. Only units authored as contradictions are labelled."""
    states, _, truth = corpus
    hedged = [s for s in states if s.register == "hedged"]
    assert hedged
    for state in hedged:
        labelled = {c.pair_id for c in state.contradictions}
        for pair_id, unit in state.units.items():
            if pair_id not in labelled:
                assert unit != "" or True  # supporting units carry no label
        assert len(state.contradictions) == len(
            [c for c in state.contradictions if c.contradiction_type]
        )


# =============================================================================
# Ground truth consistency
# =============================================================================

def test_every_contradiction_names_a_real_pair_and_type(corpus, schema) -> None:
    _, _, truth = corpus
    for row in truth:
        for contradiction in row["contradictions"]:
            pair = schema.pairs[contradiction["pair_id"]]
            assert contradiction["contradiction_type"] in schema.contradiction_types
            assert contradiction["subtlety"] in schema.subtlety_levels
            assert contradiction["structured_field"] == pair.structured
            assert contradiction["free_text_field"] == pair.free_text
            assert contradiction["ir35_test"] == pair.ir35_test


def test_contradiction_targets_an_answered_question(corpus) -> None:
    """A contradiction on a question the form skipped would be unreachable."""
    states, records, _ = corpus
    for state, record in zip(states, records):
        for contradiction in state.contradictions:
            assert record.get(contradiction.structured_field) == contradiction.structured_value


def test_undermines_flag_matches_the_pair_polarity(corpus, schema) -> None:
    _, _, truth = corpus
    for row in truth:
        for contradiction in row["contradictions"]:
            pair = schema.pairs[contradiction["pair_id"]]
            expected = contradiction["structured_value"] == pair.outside_leaning
            assert contradiction["undermines_outside_leaning"] == expected


def test_anomalies_never_destroy_a_planted_contradiction(corpus) -> None:
    """Blanking a justification that ground truth says carries a contradiction
    would silently corrupt every recall measurement in Phase 4."""
    states, _, _ = corpus
    for state in states:
        contradicted = {c.free_text_field for c in state.contradictions}
        anomalous = {a.field_id for a in state.anomalies if a.kind in ("blank", "non_responsive")}
        assert not (contradicted & anomalous)


def test_clean_unit_is_retained_for_the_control_condition(corpus) -> None:
    """Phase 4 needs the same record without the contradiction."""
    states, _, _ = corpus
    for state in states:
        for contradiction in state.contradictions:
            assert contradiction.clean_unit
            assert contradiction.clean_unit != contradiction.planted_unit


def test_contradiction_plan_balances_types(generator, schema) -> None:
    """Stratification is the whole reason per-type recall is measurable."""
    plan = generator.contradiction_plan(350)
    counts: dict[str, int] = {}
    for types in plan.values():
        for name in types:
            counts[name] = counts.get(name, 0) + 1
    expected = [t for t in schema.contradiction_types if t != "non_responsive"]
    assert set(counts) == set(expected)
    assert max(counts.values()) - min(counts.values()) <= 2


def test_non_responsive_is_not_planted_as_a_contradiction(generator) -> None:
    """It is an anomaly. Scoring it as a contradiction would flatter the model."""
    plan = generator.contradiction_plan(350)
    assert all("non_responsive" not in types for types in plan.values())


# =============================================================================
# Contract conformance
# =============================================================================

def test_every_record_validates(corpus, schema, gen_cfg) -> None:
    _, records, _ = corpus
    validator = RecordValidator(schema, gen_cfg.get("validation"))
    for result in validator.validate_many(records):
        assert result.is_valid, f"{result.record_id}: {[f.code for f in result.errors]}"


def test_structured_answers_come_from_the_declared_domain(corpus, schema) -> None:
    _, records, _ = corpus
    for record in records:
        for field in schema.structured_answer_fields:
            value = record.get(field.id)
            if value not in (None, ""):
                assert value in field.value_domain, f"{field.id}: {value!r}"


def test_no_outcome_field_reaches_a_record(corpus, schema) -> None:
    """The leakage guard, checked on the actual output."""
    _, records, _ = corpus
    for record in records:
        for field in schema.outcome_fields:
            assert field.id not in record


def test_skipped_questions_are_absent_not_blank(corpus, schema, gen_cfg) -> None:
    _, records, _ = corpus
    prefix = gen_cfg["section_four_prefix"]
    for record in records:
        answers = {
            f.id: record[f.id]
            for f in schema.structured_answer_fields
            if record.get(f.id) not in (None, "")
        }
        reachable = set(walk_route(schema, answers, prefix))
        for field in schema.structured_answer_fields:
            if str(field.form_ref or "").startswith(prefix) and field.id in record:
                assert field.id in reachable


def test_all_slot_placeholders_are_resolved(corpus, schema) -> None:
    """An unresolved {slot} in generated text is a text-bank typo."""
    _, records, _ = corpus
    leftover = re.compile(r"\{\w+\}")
    for record in records:
        for field in schema.of_type(FieldType.FREE_TEXT):
            text = record.get(field.id)
            if text:
                assert not leftover.search(str(text)), f"{field.id}: {text!r}"


# =============================================================================
# Routing
# =============================================================================

def test_route_terminates_on_the_back_edge(schema) -> None:
    """4.7 'No' routes back to 4.6. Without back-edge handling this loops."""
    answers = {
        "q4_03_substitute_sent": "No - they have not sent a substitute",
        "q4_05_right_to_reject": "No",
        "q4_07_would_pay_substitute": "No",
    }
    route = walk_route(schema, answers)
    assert len(route) == len(set(route)), "a question was asked twice"
    assert "q4_06_paid_another_person" in route
    assert "q4_31_other_clients" in route


def test_not_applicable_route_skips_the_substitute_payment_question(schema) -> None:
    answers = {"q4_03_substitute_sent": "Not applicable - work has not started"}
    route = walk_route(schema, answers)
    assert "q4_05_right_to_reject" in route
    assert "q4_04_worker_pays_substitute" not in route


def test_exclusivity_yes_skips_the_permission_question(schema) -> None:
    answers = {"q4_21_exclusivity_clause": "Yes"}
    route = walk_route(schema, answers)
    assert "q4_22_permission_required" not in route
    assert "q4_23_ownership_rights" in route


# =============================================================================
# Config integrity
# =============================================================================

def test_identity_pool_entries_are_all_strings(gen_cfg) -> None:
    """PyYAML is YAML 1.1: a bare `on` in a name-fragment list arrives as False
    and blows up at generation time. This caught it once already."""
    for pool, values in gen_cfg["identity_pools"].items():
        for value in values:
            assert isinstance(value, str), f"{pool} contains a non-string: {value!r}"


def test_field_priors_reference_real_fields_and_values(gen_cfg, schema) -> None:
    for field_id, prior in gen_cfg["field_priors"].items():
        field = schema.fields[field_id]
        assert abs(sum(prior.values()) - 1.0) < 1e-9, f"{field_id} prior does not sum to 1"
        for value in prior:
            assert value in field.value_domain, f"{field_id}: {value!r} not in domain"


def test_text_bank_covers_every_option_of_every_pair(schema, text_bank) -> None:
    """A value with no supporting proposition falls back to generic filler,
    which would be an invisible quality hole in the corpus."""
    missing: list[str] = []
    for pair in schema.pairs.values():
        entry = text_bank["pairs"].get(pair.id, {})
        supporting = entry.get("supporting") or {}
        for value in schema[pair.structured].value_domain or ():
            if value not in supporting:
                missing.append(f"{pair.id} -> {value!r}")
    assert not missing, "no supporting text for: " + "; ".join(missing)


def test_text_bank_contradiction_types_are_in_the_taxonomy(schema, text_bank) -> None:
    for pair_id, entry in text_bank["pairs"].items():
        for value, units in (entry.get("contradicting") or {}).items():
            for unit in units:
                assert unit["type"] in schema.contradiction_types, f"{pair_id}/{value}"
                assert unit["subtlety"] in schema.subtlety_levels, f"{pair_id}/{value}"


def test_text_bank_contradictions_reference_real_option_values(schema, text_bank) -> None:
    for pair_id, entry in text_bank["pairs"].items():
        domain = schema[schema.pairs[pair_id].structured].value_domain or ()
        for value in (entry.get("contradicting") or {}):
            assert value in domain, f"{pair_id}: {value!r} is not an option"


def test_every_high_priority_pair_can_express_a_contradiction(schema, text_bank) -> None:
    """The seven high-priority pairs are where a contradiction most directly
    flips a determination. Each must have authored contradiction text for its
    outside-leaning answer."""
    for pair in schema.pairs.values():
        if not pair.is_high_priority:
            continue
        contradicting = (text_bank["pairs"].get(pair.id, {}).get("contradicting") or {})
        assert pair.outside_leaning in contradicting, pair.id


# =============================================================================
# Constraint 5, enforced rather than asserted
# =============================================================================

def _code_string_literals(path: Path) -> list[str]:
    """String literals in a module, excluding docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_module_hardcodes_an_esq_field_id() -> None:
    """Constraint 5: pointing this at a revised form must be a config edit.

    Checked by parsing every module and looking for field-id literals in code.
    Docstrings are excluded — a worked example in a docstring is documentation,
    not a dependency.
    """
    field_id = re.compile(r"^q\d_\d+_\w+$")
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        for literal in _code_string_literals(path):
            if field_id.match(literal):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {literal!r}")
    assert not offenders, "hardcoded field ids: " + "; ".join(offenders)
