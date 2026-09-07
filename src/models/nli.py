"""Natural language inference framing of contradiction detection.

THE FRAMING
The structured answer is the premise — what the manager asserted by ticking the
box. The justification they wrote beside it is the hypothesis. A cross-encoder
NLI model then classifies the relation: entailment, neutral, or contradiction.

This is a good fit for the problem in a way the Phase 2 baselines are not. A
contradiction here is a *relation between two texts*, and both baselines had to
approximate that: the rule baseline by gating cue lists on the answer's
polarity, the TF-IDF model by crossing tokens with polarity to fake an
interaction. An NLI model represents the relation directly, and — the reason it
should generalise where TF-IDF did not — it was trained on the relation rather
than on this corpus's vocabulary.

THREE DESIGN CHOICES WORTH DEFENDING

**Bidirectional scoring.** NLI is not symmetric: P(contradiction | premise,
hypothesis) is not P(contradiction | hypothesis, premise). Scoring both ways and
taking the maximum is a recall-oriented choice consistent with the operating
point rationale — a conflict visible from either direction is worth showing a
reviewer. Configurable, so the asymmetry can be measured rather than assumed.

**The neutral class is a signal, not a leftover.** ``hedged_non_support`` is the
contradiction type the rule baseline scored 0.20 on, because there is no phrase
to match: the manager selected a definite option and then wrote something that
fails to support it. That is precisely what NLI neutral means. So neutral mass
above a floor is mixed in at a low weight, and the weight is a config parameter
so its contribution can be measured by setting it to zero.

**No polarity gating.** The rule baseline could only fire against
outside-leaning answers, because its cue lists were written in one direction.
NLI needs no such restriction — a justification arguing *for* self-employment
beside an inside-leaning answer is just as much a contradiction, and this
detector will find it. That is a capability difference, not a tuning difference.

WHAT THIS COSTS
Zero-shot transfer, no fine-tuning. Seventy positives is nowhere near enough to
fine-tune, and doing so would re-import the template-memorisation failure Phase 2
exposed. The model has never seen an ESQ, and the premise wording is doing real
work in bridging that gap — which is why the templates live in config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.dataset import PairInstance
from src.ingest.schema_loader import Schema
from src.models.backend import InferenceBackend, NliResult
from src.models.base import ContradictionDetector

__all__ = ["NliScore", "NliDetector"]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class NliScore:
    """The full working for one instance, kept for explanation and analysis.

    Attributes:
        forward: Distribution with the answer as premise.
        reverse: Distribution with the justification as premise, when
            bidirectional scoring is on.
        contradiction: Aggregated contradiction evidence in [0, 1].
        non_support: Aggregated neutral-class evidence in [0, 1].
        score: The final blended score the harness thresholds.
        n_chunks: How many chunks the justification was split into.
    """

    forward: NliResult
    reverse: NliResult | None
    contradiction: float
    non_support: float
    score: float
    n_chunks: int = 1


class NliDetector(ContradictionDetector):
    """Zero-shot cross-encoder NLI contradiction detector."""

    name = "nli_cross_encoder"
    description = (
        "Zero-shot cross-encoder NLI: structured answer as premise, justification "
        "as hypothesis, bidirectional, with a non-support signal from the neutral class"
    )

    def __init__(
        self,
        schema: Schema,
        backend: InferenceBackend,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create the detector.

        Args:
            schema: The loaded data contract, for question labels.
            backend: Any backend supporting ``classify_nli``.
            config: The ``nli`` block of ``config/model.yaml``.
        """
        cfg = dict(config or {})
        self.schema = schema
        self.backend = backend
        self.premise_template: str = cfg.get(
            "premise_template",
            "In answer to '{label}', the engaging manager stated: {value}",
        )
        self.hypothesis_template: str = cfg.get("hypothesis_template", "{text}")
        self.bidirectional: bool = bool(cfg.get("bidirectional", True))
        self.aggregate: str = str(cfg.get("aggregate", "max"))
        self.score_mode: str = str(cfg.get("score_mode", "normalised"))
        self.non_support_weight: float = float(cfg.get("non_support_weight", 0.25))
        self.non_support_floor: float = float(cfg.get("non_support_floor", 0.55))
        self.chunk_long_text: bool = bool(cfg.get("chunk_long_text", True))
        self.chunk_target_chars: int = int(cfg.get("chunk_target_chars", 900))
        self.name = f"nli_{backend.name}"
        self._cache: dict[str, NliScore] = {}

    # -- text construction -------------------------------------------------
    def premise(self, instance: PairInstance) -> str:
        """Render the structured answer as a natural-language statement.

        The model has never seen an ESQ. The premise has to carry both the
        question and the selected option, or "No" is not a proposition anything
        can contradict.

        Args:
            instance: The pair instance.

        Returns:
            The premise sentence.
        """
        label = self.schema[instance.structured_field].label or instance.structured_field
        return self.premise_template.format(
            label=" ".join(str(label).split()), value=instance.structured_value
        )

    def chunks(self, text: str) -> list[str]:
        """Split long justifications on sentence boundaries.

        A cross-encoder truncates at its maximum length, and truncation drops
        the *end* of an argument — which for a justification is usually where
        the qualification lives ("... though we would need to approve them").
        Splitting and taking the strongest chunk keeps that.

        Args:
            text: The justification.

        Returns:
            One or more chunks. Never empty for non-empty input.
        """
        stripped = text.strip()
        if not stripped:
            return []
        if not self.chunk_long_text or len(stripped) <= self.chunk_target_chars:
            return [stripped]

        parts: list[str] = []
        current = ""
        for sentence in _SENTENCE.split(stripped):
            if current and len(current) + len(sentence) + 1 > self.chunk_target_chars:
                parts.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            parts.append(current.strip())
        return parts or [stripped]

    # -- scoring -----------------------------------------------------------
    def _blend(self, forward: NliResult, reverse: NliResult | None) -> tuple[float, float]:
        """Combine one or both directions into contradiction and non-support."""
        directions = [forward] + ([reverse] if reverse is not None else [])

        def contradiction_of(result: NliResult) -> float:
            if self.score_mode == "contradiction_only":
                return result.contradiction
            denominator = result.contradiction + result.entailment
            # A justification that is neither a strict entailment nor a strict
            # contradiction of a dropdown value is the common case, so dividing
            # out the neutral mass keeps the score usable. When both classes are
            # near zero the ratio is meaningless, so fall back to the raw value.
            if denominator < 1e-6:
                return result.contradiction
            return result.contradiction / denominator

        contradictions = [contradiction_of(d) for d in directions]
        neutrals = [d.neutral for d in directions]
        if self.aggregate == "mean":
            return float(np.mean(contradictions)), float(np.mean(neutrals))
        return float(np.max(contradictions)), float(np.max(neutrals))

    def _combine(self, contradiction: float, non_support: float) -> float:
        """Blend the contradiction and non-support evidence into one score.

        Non-support only counts above a floor, and is capped so it can never
        push an instance above a genuine contradiction on its own. It is
        supporting evidence for a reviewer to look at, not a claim of conflict.
        """
        score = contradiction
        if self.non_support_weight > 0 and non_support >= self.non_support_floor:
            excess = (non_support - self.non_support_floor) / max(
                1e-6, 1.0 - self.non_support_floor
            )
            score += self.non_support_weight * excess * (1.0 - contradiction)
        return float(min(max(score, 0.0), 1.0))

    def score_detailed(self, instances: Sequence[PairInstance]) -> list[NliScore]:
        """Score instances and keep the full working.

        Args:
            instances: Instances to score.

        Returns:
            One ``NliScore`` per instance, in order.
        """
        # Build every (premise, hypothesis) pair first, then make one batched
        # call. Per-instance calls would multiply inference cost by the batch
        # size for no benefit, and the local-only deployment assumption makes
        # throughput a real constraint rather than a nicety.
        jobs: list[tuple[int, str, str, bool]] = []
        chunk_counts: list[int] = []
        for index, instance in enumerate(instances):
            premise = self.premise(instance)
            parts = self.chunks(instance.free_text)
            chunk_counts.append(len(parts))
            for part in parts:
                hypothesis = self.hypothesis_template.format(text=part)
                jobs.append((index, premise, hypothesis, False))
                if self.bidirectional:
                    jobs.append((index, hypothesis, premise, True))

        pairs = [(premise, hypothesis) for _, premise, hypothesis, _ in jobs]
        outputs = self.backend.classify_nli(pairs) if pairs else []

        forward_by_instance: dict[int, list[NliResult]] = {}
        reverse_by_instance: dict[int, list[NliResult]] = {}
        for (index, _, _, is_reverse), result in zip(jobs, outputs):
            target = reverse_by_instance if is_reverse else forward_by_instance
            target.setdefault(index, []).append(result)

        scores: list[NliScore] = []
        for index, instance in enumerate(instances):
            forwards = forward_by_instance.get(index, [])
            if not forwards:
                # No usable text. Blank justifications are a completeness
                # finding, not a contradiction, and are scored zero here so the
                # two are never conflated.
                empty = NliResult(entailment=0.0, neutral=1.0, contradiction=0.0)
                scores.append(NliScore(empty, None, 0.0, 0.0, 0.0, 0))
                continue

            # Strongest chunk wins: a contradiction in one sentence of a long
            # justification is still a contradiction.
            best = max(forwards, key=lambda r: r.contradiction)
            reverses = reverse_by_instance.get(index)
            best_reverse = (
                max(reverses, key=lambda r: r.contradiction) if reverses else None
            )
            contradiction, non_support = self._blend(best, best_reverse)
            scores.append(
                NliScore(
                    forward=best,
                    reverse=best_reverse,
                    contradiction=contradiction,
                    non_support=non_support,
                    score=self._combine(contradiction, non_support),
                    n_chunks=chunk_counts[index],
                )
            )
        return scores

    # -- detector interface ------------------------------------------------
    def fit(self, instances: Sequence[PairInstance]) -> "NliDetector":
        """No-op. Zero-shot by design.

        Kept so the evaluation harness treats every method identically. It also
        means the record and unit splits give this detector identical scores,
        which is the same property that made the rule baseline split-independent
        — and the direct point of comparison with TF-IDF, which collapsed.
        """
        return self

    def score(self, instances: Sequence[PairInstance]) -> np.ndarray:
        """Contradiction probability per instance.

        Args:
            instances: Instances to score.

        Returns:
            Scores in [0, 1].
        """
        detailed = self.score_detailed(instances)
        for instance, result in zip(instances, detailed):
            self._cache[self._key(instance)] = result
        return np.asarray([d.score for d in detailed], dtype=float)

    @staticmethod
    def _key(instance: PairInstance) -> str:
        """Cache key for explanations."""
        return f"{instance.record_id}|{instance.pair_id}"

    def explain(self, instance: PairInstance) -> str:
        """Say what the model found, in terms a reviewer can check.

        Args:
            instance: The instance.

        Returns:
            A reviewer-facing sentence. Reports a relation between two
            statements; never asserts a status.
        """
        cached = self._cache.get(self._key(instance))
        if cached is None:
            cached = self.score_detailed([instance])[0]
        form_ref = self.schema[instance.structured_field].form_ref
        test = instance.ir35_test.replace("_", " ")

        if cached.n_chunks == 0:
            return f"Question {form_ref} has no justification to check."
        if cached.contradiction >= 0.5:
            return (
                f"Question {form_ref} was answered “{instance.structured_value}”, but the "
                f"justification reads as a conflicting statement rather than a supporting "
                f"one (contradiction {cached.contradiction:.2f} against entailment "
                f"{cached.forward.entailment:.2f}). For the {test} test these point in "
                "opposite directions. Worth a reviewer's eye."
            )
        if cached.non_support >= self.non_support_floor:
            return (
                f"Question {form_ref} was answered “{instance.structured_value}”, a definite "
                f"option, but the justification neither supports nor conflicts with it "
                f"(neutral {cached.non_support:.2f}). It may not evidence the answer given, "
                f"which matters for the {test} test. Worth a reviewer's eye."
            )
        return (
            f"The justification for question {form_ref} is consistent with the answer "
            f"“{instance.structured_value}”."
        )
