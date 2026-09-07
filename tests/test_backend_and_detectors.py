"""Phase 3 plumbing, proved without model weights.

The mock backend exists so that everything between the config file and the
score can be tested properly: premise construction, the three-class to
one-score mapping, bidirectional aggregation, chunking, JSON parse robustness,
retry behaviour, and the constraint-2 guarantee that swapping backends is a
config edit.

That is worth doing on its own terms, not only because this sandbox cannot
reach huggingface.co. The bugs these tests catch — a reversed label order, a
score that ignores the reverse direction, a parse failure imputed as zero
without a record — are exactly the ones that otherwise surface only after a
long inference run, or not at all.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pytest

from src.features.dataset import PairInstance
from src.ingest.schema_loader import CONFIG_DIR, load_schema, load_yaml
from src.models.backend import (
    BackendUnavailable,
    GenerationResult,
    HostedApiBackend,
    InferenceBackend,
    LocalTransformersBackend,
    MockBackend,
    NliResult,
    UnsupportedOperation,
    get_backend,
)
from src.models.fewshot import FewShotDetector
from src.models.nli import NliDetector


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def model_config():
    return load_yaml(CONFIG_DIR / "model.yaml")


@pytest.fixture
def mock_backend(model_config) -> MockBackend:
    return get_backend(model_config, override="mock")


def make_instance(**overrides) -> PairInstance:
    """A minimal instance for unit-level tests."""
    defaults = dict(
        record_id="T",
        pair_id="p_4_05_right_to_reject",
        structured_field="q4_05_right_to_reject",
        free_text_field="q4_05_rationale",
        ir35_test="substitution",
        structured_value="No",
        free_text="The supplier may substitute at their own discretion.",
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


CONTRADICTORY = "We specifically need this individual; nobody else has the expertise."
HEDGED = "I believe that would be fine, though the arrangement has not been tested."


# =============================================================================
# Constraint 2: the backend is swappable by config alone
# =============================================================================

def test_every_configured_backend_builds_from_config(model_config) -> None:
    """The whole of constraint 2: name a backend in config, get that backend.

    Construction must not require weights or credentials — those are checked by
    health_check, so a misconfiguration is diagnosable without a model present.
    """
    for name in model_config["backends"]:
        backend = get_backend(model_config, override=name)
        assert isinstance(backend, InferenceBackend)
        assert backend.name == name


def test_switching_backend_is_a_config_edit(model_config) -> None:
    edited = dict(model_config, backend="mock")
    assert get_backend(edited).name == "mock"
    edited = dict(model_config, backend="local_transformers")
    assert isinstance(get_backend(edited), LocalTransformersBackend)


def test_unknown_backend_name_raises(model_config) -> None:
    with pytest.raises(KeyError):
        get_backend(model_config, override="does_not_exist")


def test_require_local_blocks_the_hosted_backend(model_config) -> None:
    """ESQ submissions carry personal and tax-sensitive data. The guard fails at
    construction, not at first inference part-way through a batch."""
    locked = dict(model_config, require_local=True, backend="hosted_api")
    with pytest.raises(BackendUnavailable) as excinfo:
        get_backend(locked)
    assert "require_local" in str(excinfo.value)


def test_require_local_permits_local_backends(model_config) -> None:
    locked = dict(model_config, require_local=True, backend="mock")
    assert get_backend(locked).is_local


def test_backends_declare_whether_they_leave_the_machine(model_config) -> None:
    assert get_backend(model_config, override="mock").is_local
    assert get_backend(model_config, override="local_transformers").is_local
    assert not get_backend(model_config, override="hosted_api").is_local


def test_unsupported_operations_raise_rather_than_degrade(model_config) -> None:
    """A cross-encoder cannot generate; pretending otherwise would produce
    plausible nonsense that nothing downstream could detect."""
    nli_only = get_backend(model_config, override="local_transformers")
    with pytest.raises(UnsupportedOperation):
        nli_only.generate(["hello"])
    generative = get_backend(model_config, override="hosted_api")
    with pytest.raises(UnsupportedOperation):
        generative.classify_nli([("a", "b")])


def test_health_check_reports_a_usable_message(model_config) -> None:
    ok, message = get_backend(model_config, override="mock").health_check()
    assert ok and message
    ok, message = get_backend(model_config, override="local_transformers").health_check()
    if not ok:
        # The expected state in an environment without cached weights. The
        # message has to say what to do about it.
        assert "fetch_models" in message or "transformers" in message


def test_hosted_backend_refuses_identifying_text(model_config) -> None:
    """A backstop on the one code path that sends data outside the University."""
    backend = HostedApiBackend(model_config["backends"]["hosted_api"])
    with pytest.raises(ValueError):
        backend._check_deidentified("Contact them at bera.kirkwell@example.com about this.")
    backend._check_deidentified("Contact them at [EMAIL_1] about this.")


def test_mock_backend_is_deterministic(mock_backend) -> None:
    pairs = [("premise one", "hypothesis one"), ("premise two", "hypothesis two")]
    assert mock_backend.classify_nli(pairs) == mock_backend.classify_nli(pairs)


# =============================================================================
# NLI detector
# =============================================================================

def test_premise_carries_the_question_and_the_option(schema, mock_backend, model_config) -> None:
    """"No" is not a proposition anything can contradict. The premise has to
    state what the manager actually asserted."""
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    premise = detector.premise(make_instance())
    assert "right to reject a substitute" in premise
    assert premise.endswith("No")


def test_contradictory_text_scores_above_supporting_text(
    schema, mock_backend, model_config
) -> None:
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    scores = detector.score(
        [make_instance(free_text=CONTRADICTORY), make_instance()]
    )
    assert scores[0] > 0.7
    assert scores[0] > scores[1]


def test_neutral_class_lifts_a_hedged_justification(schema, mock_backend, model_config) -> None:
    """`hedged_non_support` is the type the rule baseline scored 0.20 on. NLI's
    neutral class is the same shape, so it is mixed in — and the weight is a
    config parameter precisely so this effect can be measured."""
    with_signal = NliDetector(schema, mock_backend, dict(model_config["nli"]))
    without = NliDetector(
        schema, mock_backend, dict(model_config["nli"], non_support_weight=0.0)
    )
    hedged = make_instance(free_text=HEDGED)
    assert with_signal.score([hedged])[0] > without.score([hedged])[0]


def test_non_support_cannot_outrank_a_real_contradiction(
    schema, mock_backend, model_config
) -> None:
    """It is supporting evidence for a reviewer, not a claim of conflict."""
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    scores = detector.score(
        [make_instance(free_text=CONTRADICTORY), make_instance(free_text=HEDGED)]
    )
    assert scores[0] > scores[1]


def test_blank_justification_scores_zero_not_flagged(
    schema, mock_backend, model_config
) -> None:
    """A blank box is a completeness finding, not a contradiction. Conflating
    the two would let the detector take credit for finding empty fields."""
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    assert detector.score([make_instance(free_text="")])[0] == 0.0


def test_bidirectional_scoring_issues_both_directions(schema, model_config) -> None:
    """NLI is asymmetric, so both directions are scored and the maximum taken."""

    class RecordingBackend(MockBackend):
        def __init__(self) -> None:
            super().__init__({"seed": 1})
            self.seen: list[tuple[str, str]] = []

        def classify_nli(self, pairs: Sequence[tuple[str, str]]) -> list[NliResult]:
            self.seen.extend(pairs)
            return super().classify_nli(pairs)

    backend = RecordingBackend()
    detector = NliDetector(schema, backend, dict(model_config["nli"], bidirectional=True))
    instance = make_instance()
    detector.score([instance])
    forward = detector.premise(instance)
    assert any(p == forward for p, _ in backend.seen)
    assert any(h == forward for _, h in backend.seen)


def test_forward_only_mode_issues_one_direction(schema, model_config) -> None:
    class CountingBackend(MockBackend):
        def __init__(self) -> None:
            super().__init__({"seed": 1})
            self.calls = 0

        def classify_nli(self, pairs: Sequence[tuple[str, str]]) -> list[NliResult]:
            self.calls += len(pairs)
            return super().classify_nli(pairs)

    backend = CountingBackend()
    detector = NliDetector(schema, backend, dict(model_config["nli"], bidirectional=False))
    detector.score([make_instance()])
    assert backend.calls == 1


def test_long_text_is_chunked_not_truncated(schema, mock_backend, model_config) -> None:
    """Truncation drops the END of an argument, which for a justification is
    where the qualification usually lives."""
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    long_text = ("This is a long sentence about the engagement. " * 40).strip()
    assert len(detector.chunks(long_text)) > 1
    assert all(len(c) <= detector.chunk_target_chars + 200 for c in detector.chunks(long_text))


def test_a_contradiction_late_in_long_text_is_still_found(
    schema, mock_backend, model_config
) -> None:
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    padding = "The engagement proceeds as agreed with the department. " * 30
    scores = detector.score([make_instance(free_text=padding + CONTRADICTORY)])
    assert scores[0] > 0.7


def test_score_mode_normalised_discounts_neutral(schema, mock_backend, model_config) -> None:
    """Most justifications are neither strict entailments nor strict
    contradictions of a dropdown value, so the raw contradiction probability is
    compressed. Dividing out neutral keeps the score usable."""
    raw = NliDetector(
        schema, mock_backend, dict(model_config["nli"], score_mode="contradiction_only",
                                   non_support_weight=0.0)
    )
    normalised = NliDetector(
        schema, mock_backend, dict(model_config["nli"], score_mode="normalised",
                                   non_support_weight=0.0)
    )
    instance = make_instance(free_text=CONTRADICTORY)
    assert normalised.score([instance])[0] > raw.score([instance])[0]


def test_nli_is_split_independent(schema, mock_backend, model_config) -> None:
    """Zero-shot, so fitting changes nothing. This is the property that made the
    rule baseline stable across splits and TF-IDF collapse."""
    detector = NliDetector(schema, mock_backend, model_config["nli"])
    instances = [make_instance(free_text=CONTRADICTORY), make_instance()]
    before = detector.score(instances)
    detector.fit(instances)
    assert np.allclose(before, detector.score(instances))


# =============================================================================
# Few-shot detector
# =============================================================================

def test_prompt_contains_the_question_option_and_justification(
    schema, mock_backend, model_config
) -> None:
    detector = FewShotDetector(schema, mock_backend, model_config["fewshot"])
    prompt = detector.build_prompt(make_instance(free_text=CONTRADICTORY))
    assert "right to reject a substitute" in prompt
    assert "SELECTED OPTION: No" in prompt
    assert CONTRADICTORY in prompt


def test_prompt_never_asks_for_a_status_determination(
    schema, mock_backend, model_config
) -> None:
    """Art.22 posture built into the prompt: the model is not asked for a
    status, and is not given the other answers a status would need."""
    detector = FewShotDetector(schema, mock_backend, model_config["fewshot"])
    prompt = detector.build_prompt(make_instance()).lower()
    assert "inside ir35" not in prompt
    assert "outside ir35" not in prompt
    assert "not determining employment status" in detector.system_prompt.lower()


def test_shots_are_constant_across_pairs(schema, mock_backend, model_config) -> None:
    """Per-pair prompt tuning would confound the comparison: differences between
    pairs would reflect prompt effort rather than the method."""
    detector = FewShotDetector(schema, mock_backend, model_config["fewshot"])
    first = detector.build_prompt(make_instance())
    second = detector.build_prompt(
        make_instance(
            pair_id="p_4_14_materials_unreimbursed",
            structured_field="q4_14_materials_unreimbursed",
            free_text_field="q4_14_rationale",
            ir35_test="financial_risk",
            structured_value="Yes",
        )
    )
    shot = "The supplier may send any suitably qualified engineer"
    assert shot in first and shot in second


@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict": "contradicts", "confidence": 0.9, "reason": "x"}',
        '```json\n{"verdict": "contradicts", "confidence": 0.9, "reason": "x"}\n```',
        'Here is my answer:\n{"verdict": "contradicts", "confidence": 0.9, "reason": "x"}',
    ],
)
def test_parser_tolerates_common_wrappers(schema, mock_backend, model_config, raw) -> None:
    """Models wrap JSON in prose or a fence. Retrying that wastes a call."""
    detector = FewShotDetector(schema, mock_backend, model_config["fewshot"])
    verdict = detector.parse(raw)
    assert verdict is not None
    assert verdict.verdict == "contradicts"


@pytest.mark.parametrize(
    "raw",
    ["", "not json at all", '{"verdict": "maybe"}', '{"confidence": 0.9}'],
)
def test_parser_refuses_to_guess(schema, mock_backend, model_config, raw) -> None:
    """Guessing would put a fabricated judgement in front of a reviewer."""
    detector = FewShotDetector(schema, mock_backend, model_config["fewshot"])
    assert detector.parse(raw) is None


def test_parse_failure_is_recorded_not_imputed(schema, model_config) -> None:
    """An imputed score is invisible in the results and quietly biases them."""

    class BrokenBackend(MockBackend):
        def generate(self, prompts, system=None):  # type: ignore[override]
            return [GenerationResult(text="I cannot answer that.") for _ in prompts]

    detector = FewShotDetector(schema, BrokenBackend({"seed": 1}), model_config["fewshot"])
    scores = detector.score([make_instance()])
    assert scores[0] == 0.0
    assert detector.parse_failures == 1
    assert detector.usage()["parse_failures"] == 1
    assert "could not be read" in detector.explain(make_instance())


def test_parse_failure_is_retried_before_being_recorded(schema, model_config) -> None:
    class FlakyBackend(MockBackend):
        def __init__(self) -> None:
            super().__init__({"seed": 1})
            self.calls = 0

        def generate(self, prompts, system=None):  # type: ignore[override]
            self.calls += 1
            if self.calls == 1:
                return [GenerationResult(text="sorry") for _ in prompts]
            return super().generate(prompts, system)

    backend = FlakyBackend()
    detector = FewShotDetector(schema, backend, model_config["fewshot"])
    detector.score([make_instance(free_text=CONTRADICTORY)])
    assert backend.calls >= 2
    assert detector.parse_failures == 0


def test_verdicts_map_to_the_configured_scores(schema, mock_backend, model_config) -> None:
    detector = FewShotDetector(schema, mock_backend, model_config["fewshot"])
    scores = detector.score(
        [
            make_instance(free_text=CONTRADICTORY),
            make_instance(free_text=HEDGED),
            make_instance(),
        ]
    )
    assert scores[0] > scores[1] > scores[2]
    assert scores[2] == 0.0


def test_confidence_never_lifts_a_consistent_verdict(schema, model_config) -> None:
    """A confidently-consistent answer is still zero evidence of contradiction."""

    class ConfidentBackend(MockBackend):
        def generate(self, prompts, system=None):  # type: ignore[override]
            return [
                GenerationResult(
                    text=json.dumps(
                        {"verdict": "consistent", "confidence": 1.0, "reason": "clear"}
                    )
                )
                for _ in prompts
            ]

    detector = FewShotDetector(schema, ConfidentBackend({"seed": 1}), model_config["fewshot"])
    assert detector.score([make_instance()])[0] == 0.0


# =============================================================================
# Explanations
# =============================================================================

FORBIDDEN = ("is inside ir35", "is outside ir35", "determination is", "we determine")


@pytest.mark.parametrize("text", [CONTRADICTORY, HEDGED, "The worker sets the method."])
def test_no_detector_explanation_asserts_a_determination(
    schema, mock_backend, model_config, text
) -> None:
    """The Art.22 posture, expressed in the output the reviewer reads."""
    instance = make_instance(free_text=text)
    for detector in (
        NliDetector(schema, mock_backend, model_config["nli"]),
        FewShotDetector(schema, mock_backend, model_config["fewshot"]),
    ):
        detector.score([instance])
        explanation = detector.explain(instance).lower()
        assert not any(phrase in explanation for phrase in FORBIDDEN)


def test_explanations_name_the_question_and_the_answer(
    schema, mock_backend, model_config
) -> None:
    """A confidence score is not a reason. A reviewer must be able to check the
    claim against the form in front of them."""
    instance = make_instance(free_text=CONTRADICTORY)
    for detector in (
        NliDetector(schema, mock_backend, model_config["nli"]),
        FewShotDetector(schema, mock_backend, model_config["fewshot"]),
    ):
        detector.score([instance])
        explanation = detector.explain(instance)
        assert "4.5" in explanation
        assert instance.structured_value in explanation


def test_detectors_satisfy_the_shared_interface(schema, mock_backend, model_config) -> None:
    """One interface for every method, so the comparison is not contaminated by
    differences in how each is scored or thresholded."""
    from src.models.base import ContradictionDetector

    for detector in (
        NliDetector(schema, mock_backend, model_config["nli"]),
        FewShotDetector(schema, mock_backend, model_config["fewshot"]),
    ):
        assert isinstance(detector, ContradictionDetector)
        assert detector.fit([make_instance()]) is detector
        assert detector.score([make_instance()]).shape == (1,)
        assert isinstance(detector.explain(make_instance()), str)
