"""Baselines, leakage control and metrics.

The tests that matter most here are the ones guarding against flattering
results: that the rule baseline handles negation rather than firing on
"we do NOT require them on site", that a degenerate threshold cannot be
reported as perfect recall, and that no detector's explanation asserts an
employment-status determination.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate.harness import cross_val_scores
from src.evaluate.leakage import analyse_cue_leakage, cue_list, deleaked_cues
from src.evaluate.metrics import evaluate, threshold_for_recall
from src.features.dataset import PairInstance, build_instances, load_corpus
from src.ingest.schema_loader import CONFIG_DIR, load_schema, load_yaml
from src.models.baseline_rules import RuleBaseline
from src.models.baseline_tfidf import TfidfBaseline

DATA = "data/synthetic"


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def rule_config():
    return load_yaml(CONFIG_DIR / "rule_baseline.yaml")


@pytest.fixture(scope="module")
def text_bank():
    return load_yaml(CONFIG_DIR / "text_bank.yaml")


@pytest.fixture(scope="module")
def instances(schema):
    records, truth, states = load_corpus(DATA)
    return build_instances(schema, records, truth, states)


@pytest.fixture(scope="module")
def rules(schema, rule_config) -> RuleBaseline:
    return RuleBaseline(schema, rule_config)


def make_instance(**overrides) -> PairInstance:
    """A minimal instance for unit-level tests."""
    defaults = dict(
        record_id="T",
        pair_id="p_4_11_decide_where",
        structured_field="q4_11_decide_where",
        free_text_field="q4_11_rationale",
        ir35_test="control",
        structured_value="No - the worker decides",
        free_text="",
        is_outside_leaning=True,
        label=0,
        contradiction_type=None,
        subtlety=None,
        unit_key="u",
        anomaly=None,
        archetype="it_contractor",
        register="terse",
        ir35_label="outside",
    )
    defaults.update(overrides)
    return PairInstance(**defaults)


# --- cue leakage -------------------------------------------------------------

def test_leakage_is_detected_and_quantified(schema, text_bank) -> None:
    """The finding this whole control exists for: the Phase 0 cue list and the
    Phase 1 text bank overlap substantially."""
    result = analyse_cue_leakage(schema, text_bank)
    assert result.n_units > 50
    assert 0.2 < result.unit_leakage_rate < 0.6
    assert result.worst_examples


def test_phrase_deleak_removes_copied_phrasing(schema, text_bank) -> None:
    surviving = deleaked_cues(schema, text_bank, min_leak_words=3)
    all_surviving = {c for v in surviving.values() for c in v}
    assert "we specifically need this individual" not in all_surviving


def test_phrase_deleak_keeps_short_domain_vocabulary(schema, text_bank) -> None:
    """One- and two-word domain terms are what an independent analyst would
    write anyway; stripping them measures nothing but the domain."""
    surviving = {c for v in deleaked_cues(schema, text_bank, 3).values() for c in v}
    assert any(len(c.split()) <= 2 for c in surviving)
    assert "trustee" in surviving


def test_vocab_strip_is_more_aggressive_than_phrase_deleak(schema, text_bank) -> None:
    phrase = sum(len(v) for v in deleaked_cues(schema, text_bank, 3).values())
    vocab = sum(len(v) for v in deleaked_cues(schema, text_bank, 1).values())
    assert vocab < phrase


def test_cue_list_ignores_regex_entries(schema) -> None:
    pair = schema.pairs["p_4_16_payment_basis"]
    assert all(isinstance(c, str) for c in cue_list(pair.cues))


# --- rule baseline -----------------------------------------------------------

def test_negation_suppresses_a_cue(rules) -> None:
    """Without this the matcher fires on evidence FOR the answer. It is the
    usual way a keyword baseline is turned into a strawman."""
    positive = make_instance(free_text="They are required on site every working day.")
    negative = make_instance(free_text="They are not required on site at any point.")
    scores = rules.score([positive, negative])
    assert scores[0] > 0
    assert scores[1] == 0.0


def test_cues_only_fire_against_an_outside_leaning_answer(rules) -> None:
    """The same phrase is evidence of contradiction only when the answer leans
    the way the phrase argues against."""
    text = "They are required on site every working day."
    outside = make_instance(free_text=text, is_outside_leaning=True)
    inside = make_instance(free_text=text, is_outside_leaning=False, structured_value="Yes")
    assert rules.score([outside])[0] > rules.score([inside])[0]


def test_blank_text_scores_zero(rules) -> None:
    assert rules.score([make_instance(free_text="")])[0] == 0.0


def test_scores_are_probabilities(rules, instances) -> None:
    scores = rules.score(instances[:500])
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_rule_baseline_is_deterministic(rules, instances) -> None:
    assert np.array_equal(rules.score(instances[:300]), rules.score(instances[:300]))


def test_rule_baseline_finds_the_canonical_case(rules, instances) -> None:
    """The example the whole project started from: an unfettered right of
    substitution asserted, then undermined in the next clause."""
    canonical = [
        i
        for i in instances
        if i.pair_id == "p_4_05_right_to_reject"
        and i.label == 1
        and "specifically need this individual" in i.free_text
    ]
    assert canonical, "canonical contradiction not present in the corpus"
    assert rules.score(canonical).min() > 0.5


def test_de_leaked_variant_scores_lower_overall(schema, rule_config, text_bank, instances) -> None:
    full = RuleBaseline(schema, rule_config)
    reduced = RuleBaseline.de_leaked(schema, rule_config, text_bank, 3)
    y = np.array([i.label for i in instances])
    assert (reduced.score(instances) > 0)[y == 1].sum() <= (full.score(instances) > 0)[y == 1].sum()


# --- TF-IDF baseline ---------------------------------------------------------

def test_scoring_before_fit_raises(schema) -> None:
    with pytest.raises(RuntimeError):
        TfidfBaseline(schema).score([make_instance()])


def test_single_class_training_fails_loudly(schema) -> None:
    """At a 0.7% base rate a badly formed fold really can be single-class, and
    silently producing constant scores would corrupt the comparison."""
    with pytest.raises(ValueError):
        TfidfBaseline(schema).fit([make_instance(), make_instance()])


def test_fit_and_score_shape(schema, instances) -> None:
    model = TfidfBaseline(schema).fit(instances[:4000])
    scores = model.score(instances[:200])
    assert scores.shape == (200,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_interaction_features_encode_the_answer_polarity(schema) -> None:
    """The feature that lets a linear model say 'this phrase contradicts THIS
    answer' rather than 'this phrase is suspicious'."""
    outside = TfidfBaseline._interaction_tokens(make_instance(free_text="on campus"))
    inside = TfidfBaseline._interaction_tokens(
        make_instance(free_text="on campus", is_outside_leaning=False)
    )
    assert outside.startswith("OUT~")
    assert inside.startswith("IN~")
    assert outside != inside


# --- explanations ------------------------------------------------------------

FORBIDDEN = ("is inside ir35", "is outside ir35", "determination is", "we determine")


def test_rule_explanations_never_assert_a_determination(rules, instances) -> None:
    """UK GDPR Art.22 posture, expressed in the output the reviewer reads."""
    flagged = [i for i in instances if rules.score([i])[0] > 0.4][:25]
    assert flagged
    for instance in flagged:
        text = rules.explain(instance).lower()
        assert not any(phrase in text for phrase in FORBIDDEN)


def test_rule_explanations_name_the_question_and_the_evidence(rules, instances) -> None:
    """A confidence score is not a reason. A reviewer must be able to check the
    claim against the form in front of them."""
    flagged = next(i for i in instances if rules.score([i])[0] > 0.5)
    explanation = rules.explain(flagged)
    assert "Question" in explanation
    assert flagged.structured_value in explanation
    assert "“" in explanation


def test_no_evidence_produces_an_honest_explanation(rules) -> None:
    assert "No lexical evidence" in rules.explain(make_instance(free_text="They set the method."))


# --- metrics -----------------------------------------------------------------

def test_degenerate_zero_threshold_is_never_chosen() -> None:
    """A detector emitting 0 for 'no evidence' must not be handed threshold 0,
    flag every field on every form and be recorded as achieving 100% recall.
    That is switching the detector off and calling it perfect."""
    y = np.array([0] * 90 + [1] * 10)
    scores = np.zeros(100)
    scores[90:93] = 0.9  # catches 3 of 10
    point = threshold_for_recall(y, scores, target_recall=0.8, n_records=10)
    assert point.threshold > 0
    assert point.recall == pytest.approx(0.3)


def test_unreachable_recall_target_is_reported_not_hidden() -> None:
    y = np.array([0] * 90 + [1] * 10)
    scores = np.zeros(100)
    scores[90:93] = 0.9
    instances = [
        make_instance(label=int(v), contradiction_type="direct_negation" if v else None,
                      subtlety=1 if v else None, record_id=f"R{i}")
        for i, v in enumerate(y)
    ]
    result = evaluate("test", "record", instances, scores, target_recall=0.8)
    assert result.chosen.recall < 0.8
    assert any("unreachable" in note for note in result.notes)


def test_flags_per_record_is_computed(instances, rules) -> None:
    """The operational metric. Precision means nothing to the IR35 team;
    'one flag every other form' means everything."""
    scores = rules.score(instances)
    result = evaluate("rules", "record", instances, scores, target_recall=0.5)
    assert 0 < result.chosen.flags_per_record < 30
    assert result.n_records == 350


def test_breakdowns_are_populated(instances, rules) -> None:
    result = evaluate("rules", "record", instances, rules.score(instances), 0.5)
    assert result.by_type and result.by_subtlety
    assert result.by_archetype and result.by_register
    for entry in result.by_type.values():
        assert 0.0 <= entry["recall"] <= 1.0


# --- harness -----------------------------------------------------------------

def test_cross_validation_scores_every_instance(schema, rule_config, instances) -> None:
    scored = cross_val_scores(lambda: RuleBaseline(schema, rule_config), instances, "record", 3)
    assert scored.scores.shape == (len(instances),)
    assert all(count > 0 for count in scored.fold_positives)


def test_rule_baseline_scores_identically_in_and_out_of_the_loop(
    schema, rule_config, instances
) -> None:
    """It has nothing to fit, so cross-validation must not change its scores.
    If it did, something in the harness would be leaking."""
    direct = RuleBaseline(schema, rule_config).score(instances)
    scored = cross_val_scores(lambda: RuleBaseline(schema, rule_config), instances, "record", 3)
    assert np.allclose(direct, scored.scores)


def test_tfidf_collapses_on_unseen_phrasing(schema, instances) -> None:
    """The headline Phase 2 result, pinned as a test.

    On a record-disjoint split the model looks strong; on a unit-disjoint split
    it falls to the no-skill floor, because most of what it learned was the
    template rather than the contradiction. If a future change makes this test
    fail, the claim in the Phase 2 report needs rewriting.
    """
    by_record = cross_val_scores(lambda: TfidfBaseline(schema), instances, "record", 3)
    by_unit = cross_val_scores(lambda: TfidfBaseline(schema), instances, "unit", 3)
    record_result = evaluate("tfidf", "record", instances, by_record.scores, 0.8)
    unit_result = evaluate("tfidf", "unit", instances, by_unit.scores, 0.8)
    assert record_result.pr_auc > 0.4
    assert unit_result.pr_auc < 0.1
    assert record_result.pr_auc > unit_result.pr_auc * 4
