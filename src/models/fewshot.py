"""Few-shot LLM contradiction detector, through the same backend interface.

The third arm of the method comparison. It goes through ``InferenceBackend``
exactly as the NLI detector does, so the choice between a local instruction-tuned
model and a hosted API is a config edit — which is the point of constraint 2 and
also the point of this comparison. The interesting question for the University is
not only "is it more accurate" but "is it viable at all when the data cannot
leave the building".

WHAT THE MODEL IS ASKED, AND WHAT IT IS NOT
It sees one question, the selected option, and the justification for that same
question. It is asked whether those two agree. It is never shown the other
answers, never asked for an employment status, and the system prompt says so
explicitly. That is the Art.22 posture built into the prompt rather than bolted
on afterwards: the model is not withheld from making a determination by
instruction alone, it is not given the information a determination would need.

WHY A STRICT OUTPUT CONTRACT
A free-text answer needs parsing that fails quietly and produces plausible
rubbish. A JSON contract fails loudly, gets retried, and on repeated failure is
recorded as a failure rather than imputed — a silently guessed score is worse
than a recorded miss, because it is invisible in the results.

THE THREE-WAY VERDICT
``consistent`` / ``contradicts`` / ``unclear``. The third is not a hedge: it is
the same signal as the NLI neutral class, aimed at the ``hedged_non_support``
type the rule baseline could not touch. It maps to a low but non-zero score, so
it can raise a flag when combined with a low threshold but will not outrank a
plain contradiction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.dataset import PairInstance
from src.ingest.schema_loader import Schema
from src.models.backend import InferenceBackend
from src.models.base import ContradictionDetector

__all__ = ["FewShotVerdict", "FewShotDetector"]

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass
class FewShotVerdict:
    """One parsed model response.

    Attributes:
        verdict: ``consistent``, ``contradicts`` or ``unclear``.
        confidence: The model's stated confidence in [0, 1].
        reason: One sentence, shown to the reviewer.
        score: Verdict mapped to a contradiction probability.
        raw: The raw completion, kept for debugging a parse failure.
        parse_failed: True when no valid JSON could be recovered.
        input_tokens: Prompt tokens, for the runtime and cost comparison.
        output_tokens: Completion tokens.
        latency_seconds: Wall-clock time.
    """

    verdict: str = "unclear"
    confidence: float = 0.0
    reason: str = ""
    score: float = 0.0
    raw: str = ""
    parse_failed: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0


class FewShotDetector(ContradictionDetector):
    """Few-shot prompted contradiction detector."""

    name = "fewshot_llm"
    description = (
        "Few-shot prompted LLM through the shared backend interface, with a strict "
        "JSON verdict contract and a three-way consistent/contradicts/unclear output"
    )

    def __init__(
        self,
        schema: Schema,
        backend: InferenceBackend,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create the detector.

        Args:
            schema: The loaded data contract, for question labels and the
                per-pair contradiction descriptions.
            backend: Any backend supporting ``generate``.
            config: The ``fewshot`` block of ``config/model.yaml``.
        """
        cfg = dict(config or {})
        self.schema = schema
        self.backend = backend
        self.system_prompt: str = " ".join(str(cfg.get("system_prompt", "")).split())
        self.response_format: str = str(cfg.get("response_format", "")).strip()
        self.include_pair_description: bool = bool(cfg.get("include_pair_description", True))
        self.n_shots: int = int(cfg.get("n_shots", 4))
        self.verdict_scores: Mapping[str, float] = cfg.get(
            "verdict_scores", {"contradicts": 1.0, "unclear": 0.35, "consistent": 0.0}
        )
        self.max_parse_retries: int = int(cfg.get("max_parse_retries", 2))
        self.on_parse_failure: str = str(cfg.get("on_parse_failure", "zero_and_record"))
        self.name = f"fewshot_{backend.name}"
        self._verdicts: dict[str, FewShotVerdict] = {}
        self.parse_failures: int = 0

    # -- prompt construction ----------------------------------------------
    @staticmethod
    def _shots() -> list[tuple[str, str, str, dict[str, Any]]]:
        """Worked examples, held constant across every pair.

        Deliberately not drawn from the corpus and deliberately not tuned per
        pair. Per-pair shots would confound the comparison — differences between
        pairs would then reflect prompt effort rather than the method — and
        corpus-drawn shots would reintroduce the leakage problem Phase 2 had to
        control for. These are written from the domain, in engagement types the
        corpus does not contain.

        Returns:
            ``(question, selected option, justification, expected verdict)``.
        """
        return [
            (
                "Do you have the right to reject a substitute?",
                "No",
                "The supplier may send any suitably qualified engineer; we have no "
                "contractual right to refuse.",
                {
                    "verdict": "consistent",
                    "confidence": 0.9,
                    "reason": "The justification restates the absence of a veto over substitutes.",
                },
            ),
            (
                "Do you have the right to reject a substitute?",
                "No",
                "In practice the contract names her personally and we could not accept "
                "anyone else on this project.",
                {
                    "verdict": "contradicts",
                    "confidence": 0.9,
                    "reason": "The answer says there is no right to refuse, but the text "
                    "describes a dependency on one named individual.",
                },
            ),
            (
                "Does the worker pay for materials without being reimbursed?",
                "Yes",
                "They purchase the fittings and then invoice us for them at cost "
                "alongside the fee.",
                {
                    "verdict": "contradicts",
                    "confidence": 0.85,
                    "reason": "The answer claims an unreimbursed cost, but the text "
                    "describes the cost being passed back to the engager.",
                },
            ),
            (
                "Does the current contract allow for it to be extended?",
                "No",
                "Nothing has been agreed, but these things do tend to carry on if the "
                "budget is there.",
                {
                    "verdict": "unclear",
                    "confidence": 0.6,
                    "reason": "The justification neither confirms nor denies an extension "
                    "provision and does not support the definite answer given.",
                },
            ),
        ]

    def build_prompt(self, instance: PairInstance) -> str:
        """Build the prompt for one instance.

        Args:
            instance: The pair instance.

        Returns:
            The user prompt.
        """
        label = " ".join(str(self.schema[instance.structured_field].label or "").split())
        pair = self.schema.pairs[instance.pair_id]

        parts: list[str] = []
        for question, option, justification, expected in self._shots()[: self.n_shots]:
            parts.append(
                f"QUESTION: {question}\n"
                f"SELECTED OPTION: {option}\n"
                f"JUSTIFICATION: {justification}\n"
                f"ANSWER: {json.dumps(expected)}"
            )

        if self.include_pair_description and pair.contradiction:
            parts.append(
                "For this particular question, a contradiction looks like: "
                + " ".join(pair.contradiction.split())
            )

        parts.append(
            f"Now assess this one.\n"
            f"QUESTION: {label}\n"
            f"SELECTED OPTION: {instance.structured_value}\n"
            f"JUSTIFICATION: {instance.free_text or '(left blank)'}\n\n"
            f"{self.response_format}"
        )
        return "\n\n".join(parts)

    # -- parsing -----------------------------------------------------------
    def parse(self, text: str) -> FewShotVerdict | None:
        """Recover a verdict from a completion.

        Tolerant of a model wrapping JSON in prose or a code fence, because that
        is the common failure and retrying it wastes a call. Not tolerant of a
        missing or unrecognised verdict, because guessing there would put a
        fabricated judgement in front of a reviewer.

        Args:
            text: The raw completion.

        Returns:
            A verdict, or ``None`` if nothing valid could be recovered.
        """
        if not text or not text.strip():
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-z]*\s*|\s*```$", "", candidate, flags=re.IGNORECASE)

        payload: dict[str, Any] | None = None
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(candidate)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
        if not isinstance(payload, dict):
            return None

        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in self.verdict_scores:
            return None
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return FewShotVerdict(
            verdict=verdict,
            confidence=min(max(confidence, 0.0), 1.0),
            reason=" ".join(str(payload.get("reason", "")).split()),
            score=float(self.verdict_scores[verdict]),
            raw=text,
        )

    # -- detector interface ------------------------------------------------
    def fit(self, instances: Sequence[PairInstance]) -> "FewShotDetector":
        """No-op. Few-shot, not fine-tuned.

        Like the NLI detector, this makes the method split-independent: it
        scores the record and unit splits identically, which is the direct
        comparison with the TF-IDF baseline that collapsed on unseen phrasing.
        """
        return self

    def score(self, instances: Sequence[PairInstance]) -> np.ndarray:
        """Score instances by prompting the backend.

        Args:
            instances: Instances to score.

        Returns:
            Scores in [0, 1].
        """
        prompts = [self.build_prompt(i) for i in instances]
        results = self.backend.generate(prompts, system=self.system_prompt or None)

        scores = np.zeros(len(instances), dtype=float)
        retry_indices: list[int] = []
        parsed: list[FewShotVerdict | None] = []

        for index, (instance, result) in enumerate(zip(instances, results)):
            verdict = self.parse(result.text) if not result.error else None
            if verdict is None:
                retry_indices.append(index)
            else:
                verdict.input_tokens = result.input_tokens
                verdict.output_tokens = result.output_tokens
                verdict.latency_seconds = result.latency_seconds
            parsed.append(verdict)

        for _ in range(self.max_parse_retries):
            if not retry_indices:
                break
            retry_prompts = [
                prompts[i] + "\n\nYour previous reply could not be parsed. "
                "Reply with the JSON object only." for i in retry_indices
            ]
            retried = self.backend.generate(retry_prompts, system=self.system_prompt or None)
            still_failing: list[int] = []
            for index, result in zip(retry_indices, retried):
                verdict = self.parse(result.text) if not result.error else None
                if verdict is None:
                    still_failing.append(index)
                else:
                    parsed[index] = verdict
            retry_indices = still_failing

        for index, (instance, verdict) in enumerate(zip(instances, parsed)):
            if verdict is None:
                # Record the failure rather than impute a score. An imputed
                # value is invisible in the results and quietly biases them.
                self.parse_failures += 1
                verdict = FewShotVerdict(parse_failed=True, raw=results[index].text)
                self.backend.log.warning(
                    "fewshot.parse_failed",
                    meta={"record_id": instance.record_id, "pair_id": instance.pair_id},
                )
            # Confidence scales a contradiction verdict but never a consistent
            # one: a confidently-consistent answer is still zero evidence of
            # contradiction, and letting confidence raise it would invert the
            # meaning of the field.
            scaled = verdict.score * (0.5 + 0.5 * verdict.confidence) if verdict.score > 0 else 0.0
            verdict.score = float(min(max(scaled, 0.0), 1.0))
            scores[index] = verdict.score
            self._verdicts[f"{instance.record_id}|{instance.pair_id}"] = verdict

        return scores

    def explain(self, instance: PairInstance) -> str:
        """Return the model's own stated reason, framed for a reviewer.

        Args:
            instance: The instance.

        Returns:
            A reviewer-facing sentence. Reports agreement between two
            statements; never asserts a status.
        """
        key = f"{instance.record_id}|{instance.pair_id}"
        verdict = self._verdicts.get(key)
        if verdict is None:
            self.score([instance])
            verdict = self._verdicts.get(key)
        form_ref = self.schema[instance.structured_field].form_ref

        if verdict is None or verdict.parse_failed:
            return (
                f"Question {form_ref} could not be assessed automatically — the model's "
                "response could not be read. Please review this field manually."
            )
        if verdict.verdict == "consistent":
            return (
                f"The justification for question {form_ref} is consistent with the answer "
                f"“{instance.structured_value}”."
            )
        prefix = (
            "reads as inconsistent with"
            if verdict.verdict == "contradicts"
            else "does not clearly support"
        )
        reason = verdict.reason or "no reason given"
        return (
            f"Question {form_ref} was answered “{instance.structured_value}”, and the "
            f"justification {prefix} it: {reason} Worth a reviewer's eye."
        )

    def usage(self) -> dict[str, Any]:
        """Token and latency totals, for the runtime comparison."""
        verdicts = list(self._verdicts.values())
        return {
            "n_scored": len(verdicts),
            "parse_failures": self.parse_failures,
            "input_tokens": sum(v.input_tokens for v in verdicts),
            "output_tokens": sum(v.output_tokens for v in verdicts),
            "total_latency_seconds": round(sum(v.latency_seconds for v in verdicts), 2),
        }
