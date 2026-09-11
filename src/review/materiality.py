""""Why does this matter?" — described from the rulebook, never from a record.

Phase 5 shows a reviewer *where* a tick-box answer and its justification (or,
as of this addition, two tick-box answers) disagree. This module adds *how
much that kind of disagreement typically matters* to an IR35 determination —
which is a different question from "does this submission lean inside or
outside," and answering only the first is what keeps this addition on the
right side of the line the whole project has held since Phase 0.

THE SAFETY PROPERTY, MADE MECHANICAL
Every function here takes an IR35 test name and, optionally, a field id —
identifiers describing *the form's structure* — and returns a description
drawn from ``config/ir35_weights.yaml``'s test weights and determinative
gates. Nothing here takes a record, a score, an answer value, or anything
else that could differ between two submissions. Two calls with the same test
id always return the same tier, on any submission, because the function
cannot see the submission at all. `tests/test_materiality.py` checks this
directly — not just by testing behaviour, but by inspecting the function
signatures — so a future change cannot smuggle per-record information in
without a test failing.

Concretely: this describes the same thing a footnote in the ESQ guidance
could say — "personal service and control are treated as the essential
minimum" — attached automatically to the right flag instead of requiring a
reviewer to already know it. It is public, static, and true regardless of
which way any particular form leans, which is exactly why it is safe to show
unconditionally, unlike a status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["MaterialityTier", "materiality_for"]


@dataclass(frozen=True)
class MaterialityTier:
    """A plain-language description of how much a test or field typically
    weighs in an IR35 determination — never how this submission scored.

    Attributes:
        label: Short reviewer-facing phrase, e.g. "A major factor".
        explanation: A sentence justifying the label, in plain language.
        is_gate: True if this came from a determinative gate rather than a
            weighted test — i.e. this single fact, taken alone, is capable of
            settling the question under the case law this project's rule
            engine approximates (see ``src/models/cest_rules.py``).
    """

    label: str
    explanation: str
    is_gate: bool = False


def materiality_for(
    review_config: Mapping[str, Any],
    ir35_weights_config: Mapping[str, Any],
    *,
    ir35_test: str | None,
    field_id: str | None = None,
) -> MaterialityTier | None:
    """Describe how much a test or field typically matters, structurally.

    Args:
        review_config: Parsed ``config/review.yaml`` (the ``materiality``
            block: tier cut-points and their wording).
        ir35_weights_config: Parsed ``config/ir35_weights.yaml`` (test
            weights and determinative gates).
        ir35_test: The IR35 test id a flag or finding relates to, e.g.
            ``"substitution"``. Structural — the same for every submission
            that touches this test.
        field_id: The specific structured field involved, if known. Checked
            against every determinative gate's condition fields; a match
            takes priority over the test-weight tier, because a gate field
            can settle the question alone regardless of the test's usual
            weight.

    Returns:
        A tier, or ``None`` if the test is not weighted at all (e.g.
        ``contract_basis``, which ``schema.yaml`` documents as context rather
        than a test) and matches no gate.
    """
    materiality_cfg = review_config.get("materiality", {})

    if field_id:
        for gate in ir35_weights_config.get("gates", []):
            if field_id in gate.get("when", {}):
                return MaterialityTier(
                    label=str(materiality_cfg.get("gate_label", "Could be decisive on its own")),
                    explanation=" ".join(
                        str(materiality_cfg.get("gate_explanation", "")).split()
                    ),
                    is_gate=True,
                )

    if ir35_test is None:
        return None
    weight = ir35_weights_config.get("test_weights", {}).get(ir35_test)
    if weight is None:
        return None

    tiers = sorted(
        materiality_cfg.get("tiers", []), key=lambda t: -float(t["min_weight"])
    )
    for tier in tiers:
        if float(weight) >= float(tier["min_weight"]):
            return MaterialityTier(
                label=str(tier["label"]),
                explanation=" ".join(str(tier["explanation"]).split()),
                is_gate=False,
            )
    return None
