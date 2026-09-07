"""CEST-approximating rule engine — the synthetic dataset's labelling function.

Chosen at the Phase 0 gate (option g1): the label is *derived* from the
structured answers rather than sampled independently, so that answers and label
are causally linked. That makes the dataset defensible — an engagement labelled
"inside" is inside *because of* the answers it contains, which is what a real
determination is.

WHAT THIS IS NOT
----------------
This is not CEST, not HMRC guidance, and not a determination. It is a
transparent approximation of the case-law weighting CEST encodes, built to
generate internally-consistent synthetic labels. Under UK GDPR Art.22 an
employment status determination has legal effect, so the system never issues
one; this module exists to label synthetic records, and its output is called a
``label``, never a ``determination``.

THE METHODOLOGICAL CAVEAT
-------------------------
Because this engine produces the ground truth, any rule-based classifier
evaluated against that ground truth inherits an advantage. The Phase 2 rule
baseline therefore operates on a different substrate entirely — lexical cues
over free text, not weighted structured answers — and Phase 4's write-up must
state the limitation outright. A rule baseline graded against rule-generated
labels is evidence about the generator, not about the world.

STRUCTURE
---------
1. **Determinative gates**, in order, first match wins. These encode the small
   number of answers that settle the question alone: office holder (inside), a
   substitute actually sent and paid (outside).
2. **Weighted scoring** otherwise. Value scores in [-2, +2] where positive
   points outside, averaged within each IR35 test, normalised to [-1, +1], then
   combined with test weights following the case-law hierarchy — personal
   service and control as the irreducible minimum (*Ready Mixed Concrete*),
   financial risk next, the rest as "all the circumstances".
3. **An abstention band** in the middle producing ``undetermined``. This is not
   a failure mode: it is the "unable to determine" outcome CEST itself returns,
   and the corpus needs those cases or it overstates how decidable real
   submissions are.

Every result carries the score breakdown and the reasoning, because a labelling
function nobody can inspect is not evidence of anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from src.ingest.schema_loader import CONFIG_DIR, load_yaml

__all__ = ["Status", "TestScore", "StatusResult", "CestRuleEngine"]


class Status(str, Enum):
    """The label assigned to a synthetic record."""

    INSIDE = "inside"
    OUTSIDE = "outside"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class TestScore:
    """Contribution of one IR35 test to the total.

    Attributes:
        test: IR35 test name.
        raw: Mean value score across the answered fields, in [-2, +2].
        normalised: ``raw / 2``, in [-1, +1].
        weight: Test weight from config.
        contribution: ``normalised * weight``.
        n_answered: How many of the test's fields were actually answered. A test
            with no answered fields contributes nothing and its weight is
            redistributed, so an incomplete form is not silently scored as
            neutral.
        detail: Field id -> value score, for inspection.
    """

    test: str
    raw: float
    normalised: float
    weight: float
    contribution: float
    n_answered: int
    detail: Mapping[str, float] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class StatusResult:
    """The label and everything needed to justify it.

    Attributes:
        status: inside / outside / undetermined.
        score: Total weighted score in [-1, +1]. Positive points outside.
        gate: Id of the determinative gate that fired, if any.
        reason: Human-readable explanation.
        test_scores: Per-test breakdown.
        n_answered: Total structured answers scored.
    """

    status: Status
    score: float
    gate: str | None
    reason: str
    test_scores: tuple[TestScore, ...]
    n_answered: int

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the ground-truth file."""
        return {
            "status": self.status.value,
            "score": round(self.score, 4),
            "gate": self.gate,
            "reason": self.reason,
            "n_answered": self.n_answered,
            "test_scores": {
                ts.test: {
                    "raw": round(ts.raw, 4),
                    "normalised": round(ts.normalised, 4),
                    "weight": ts.weight,
                    "contribution": round(ts.contribution, 4),
                    "n_answered": ts.n_answered,
                }
                for ts in self.test_scores
            },
        }


class CestRuleEngine:
    """Derives a status label from a record's structured answers.

    Example:
        >>> from src.ingest.schema_loader import load_schema
        >>> engine = CestRuleEngine(load_schema())
        >>> engine.evaluate({"q4_02_office_holder": "Yes"}).status
        <Status.INSIDE: 'inside'>
    """

    def __init__(self, schema, config_path: Path | str | None = None) -> None:
        """Create the engine.

        Args:
            schema: The loaded data contract, used to map fields to IR35 tests.
            config_path: Path to ``ir35_weights.yaml``. Defaults to the repo's
                ``config/``.
        """
        path = Path(config_path) if config_path else CONFIG_DIR / "ir35_weights.yaml"
        cfg = load_yaml(path)
        self.schema = schema
        self.version: str = cfg.get("weights_version", "unknown")
        self.test_weights: dict[str, float] = dict(cfg["test_weights"])
        self.value_scores: dict[str, dict[str, float]] = {
            k: {str(kk): float(vv) for kk, vv in v.items()}
            for k, v in cfg["value_scores"].items()
        }
        self.gates: list[Mapping[str, Any]] = list(cfg.get("gates", []))
        thresholds = cfg["thresholds"]
        self.outside_above: float = float(thresholds["outside_above"])
        self.inside_below: float = float(thresholds["inside_below"])

    # -- gates ------------------------------------------------------------
    def _check_gates(self, answers: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Return the first determinative gate whose conditions all hold."""
        for gate in self.gates:
            conditions: Mapping[str, Any] = gate["when"]
            if all(answers.get(field) == value for field, value in conditions.items()):
                return gate
        return None

    # -- scoring ----------------------------------------------------------
    def score_tests(self, answers: Mapping[str, Any]) -> list[TestScore]:
        """Score each IR35 test from the answers present.

        Weights are renormalised across the tests that actually have answers, so
        a partially completed form is scored on what it says rather than being
        dragged towards neutral by the tests it is silent on.

        Args:
            answers: Field id -> selected option value.

        Returns:
            One ``TestScore`` per test that had at least one scored answer.
        """
        by_test: dict[str, dict[str, float]] = {}
        for field_id, value in answers.items():
            table = self.value_scores.get(field_id)
            if table is None or value is None:
                continue
            if value not in table:
                continue
            field = self.schema.fields.get(field_id)
            test = field.ir35_test if field else None
            if test is None or test not in self.test_weights:
                continue
            by_test.setdefault(test, {})[field_id] = table[value]

        live_weight = sum(self.test_weights[t] for t in by_test) or 1.0
        scores: list[TestScore] = []
        for test, detail in by_test.items():
            raw = sum(detail.values()) / len(detail)
            normalised = raw / 2.0
            weight = self.test_weights[test] / live_weight
            scores.append(
                TestScore(
                    test=test,
                    raw=raw,
                    normalised=normalised,
                    weight=round(weight, 6),
                    contribution=normalised * weight,
                    n_answered=len(detail),
                    detail=detail,
                )
            )
        return sorted(scores, key=lambda s: -abs(s.contribution))

    def evaluate(self, answers: Mapping[str, Any]) -> StatusResult:
        """Assign a status label to one record's structured answers.

        Args:
            answers: Field id -> selected option value. Free text is ignored;
                this engine reads only the structured answers, by design — the
                whole premise of the project is that the free text sometimes
                disagrees with them, and a labelling function that read both
                would have no stable ground truth to offer.

        Returns:
            A ``StatusResult`` carrying the label, the score and the reasoning.
        """
        gate = self._check_gates(answers)
        test_scores = tuple(self.score_tests(answers))
        n_answered = sum(ts.n_answered for ts in test_scores)
        total = sum(ts.contribution for ts in test_scores)

        if gate is not None:
            return StatusResult(
                status=Status(gate["label"]),
                score=total,
                gate=gate["id"],
                reason=" ".join(str(gate["reason"]).split()),
                test_scores=test_scores,
                n_answered=n_answered,
            )

        if not test_scores:
            return StatusResult(
                status=Status.UNDETERMINED,
                score=0.0,
                gate=None,
                reason="No scoreable structured answers were present.",
                test_scores=(),
                n_answered=0,
            )

        if total > self.outside_above:
            status = Status.OUTSIDE
        elif total < self.inside_below:
            status = Status.INSIDE
        else:
            status = Status.UNDETERMINED

        return StatusResult(
            status=status,
            score=total,
            gate=None,
            reason=self._explain(status, total, test_scores),
            test_scores=test_scores,
            n_answered=n_answered,
        )

    def _explain(
        self, status: Status, total: float, test_scores: tuple[TestScore, ...]
    ) -> str:
        """Build a short justification naming the tests that drove the result."""
        if status is Status.UNDETERMINED:
            opposed = [ts for ts in test_scores if abs(ts.normalised) > 0.3]
            outside = [ts.test for ts in opposed if ts.normalised > 0]
            inside = [ts.test for ts in opposed if ts.normalised < 0]
            if outside and inside:
                return (
                    f"Weighted score {total:+.3f} falls in the abstention band. "
                    f"Tests pull in opposite directions: {', '.join(outside)} point "
                    f"outside while {', '.join(inside)} point inside. A reviewer "
                    "should decide this one."
                )
            return (
                f"Weighted score {total:+.3f} falls in the abstention band "
                f"({self.inside_below:+.2f} to {self.outside_above:+.2f}). "
                "No test is decisive."
            )
        direction = "outside" if status is Status.OUTSIDE else "inside"
        drivers = [ts for ts in test_scores[:3] if abs(ts.contribution) > 0.01]
        named = ", ".join(f"{ts.test} ({ts.normalised:+.2f})" for ts in drivers)
        return f"Weighted score {total:+.3f} points {direction}. Driven by {named}."
