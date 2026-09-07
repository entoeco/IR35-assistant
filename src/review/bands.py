"""Turn a raw detector score into a reviewer-facing confidence band.

WHY THIS MODULE EXISTS
Phase 4's calibration check found the rule baseline overstates itself: it
predicts roughly 0.46 in a score range where the true rate of a genuine
contradiction was roughly 0.11 (``docs/reports/phase4_evaluation_report.md``,
calibration section). Showing that raw number to a reviewer as "46%
confidence" would be showing them a number that is wrong by a factor of four.
Every method evaluated in Phase 2-4 has some degree of the same problem, and
none of them were built or tuned to produce calibrated probabilities in the
first place — that was never the target of the training objective.

The fix is not to recalibrate the models (a small, imbalanced synthetic corpus
is a poor basis for fitting a calibration curve that would then be trusted on
real submissions) but to stop presenting a percentage at all. A **band** —
"Worth a look" / "Priority" / "High priority" — makes only the ordinal claim
the evidence supports: higher-scoring flags were more often right than
lower-scoring ones in testing. It does not claim to know the odds for this
specific case.

Band cut-points and their supporting rationale live in ``config/review.yaml``,
not here, so re-tuning them after a future evaluation run is a config edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = ["ConfidenceBand", "band_for_score", "load_band_config"]


@dataclass(frozen=True)
class ConfidenceBand:
    """One named band a flag can fall into.

    Attributes:
        label: Short reviewer-facing name, e.g. ``"Priority"``.
        min_score: The band applies at or above this raw score.
        rationale: A sentence explaining the label in terms of what was
            observed in testing, never a raw probability for this case.
    """

    label: str
    min_score: float
    rationale: str


def load_band_config(
    review_config: Mapping[str, Any], method: str
) -> tuple[ConfidenceBand, ...]:
    """Read the ordered bands configured for one detector.

    Args:
        review_config: Parsed ``config/review.yaml``.
        method: Detector name, e.g. ``"rules_as_authored"``.

    Returns:
        Bands sorted ascending by ``min_score``. Empty if the method has no
        configured bands — callers should treat that as "show nothing",
        never invent a threshold.
    """
    raw = review_config.get("detectors", {}).get("confidence_bands", {}).get(method, [])
    bands = tuple(
        ConfidenceBand(
            label=str(entry["label"]),
            min_score=float(entry["min_score"]),
            rationale=" ".join(str(entry["rationale"]).split()),
        )
        for entry in raw
    )
    return tuple(sorted(bands, key=lambda b: b.min_score))


def band_for_score(score: float, bands: Sequence[ConfidenceBand]) -> ConfidenceBand | None:
    """Place a score into the highest band it clears.

    Args:
        score: Raw detector score in [0, 1].
        bands: Bands for this detector, ascending by ``min_score`` (as
            returned by :func:`load_band_config`).

    Returns:
        The highest-``min_score`` band the score clears, or ``None`` if the
        score is below every band — meaning this flag should not be shown to
        a reviewer at all.
    """
    matched: ConfidenceBand | None = None
    for band in bands:
        if score >= band.min_score:
            matched = band
        else:
            break
    return matched
