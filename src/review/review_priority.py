"""Review priority: how much a submission's flags, taken together, deserve
attention -- never a lean on the underlying employment-status question.

WHAT THIS IS AND IS NOT
This is the aggregate counterpart to ``src.review.materiality``: where a
materiality tier describes one flag or finding, review priority collapses
*every* flag and finding on one submission into a single five-point scale --
"Nothing flagged" through "Urgent review" -- so a reviewer triaging many
submissions can tell which ones need a close look without opening every
card first.

It is emphatically NOT a Likert scale for "how likely is this inside or
outside IR35". That framing was considered and rejected -- see
``docs/reports/phase7_consistency_and_materiality.md`` -- for the same
reasons the original RAG-status idea was rejected earlier in this project:
automation bias, partial evidence, and no calibration against real HMRC
outcomes. What is built here instead only ever combines three things that
are already shown to a reviewer elsewhere on the page and that none of, on
their own, encode a direction:

* **how many** flags/findings there are,
* **how strong the evidence** is for each (its confidence band or, for a
  cross-field finding, its authored severity),
* **how much that kind of mismatch typically matters** (its materiality
  tier -- read, never computed, same as ``materiality_for``).

Count, evidence strength and importance-of-category do not, even combined,
say which way an answer leans -- a submission can score "Urgent review" with
every flag pointing towards more evidence of self-employment just as easily
as towards more evidence of employment. That is the property this module
depends on, and ``tests/test_review_priority.py`` checks it directly by
constructing both and asserting the same priority-scoring machinery treats
them identically.

STRUCTURAL SAFETY, SAME PATTERN AS materiality.py
``review_priority_for`` takes only the already-computed flags and findings
(each carrying a confidence band / severity and a materiality tier, nothing
else) plus config -- never the record, never a raw score, never anything
the labelling engine would need to be imported to produce. A gate-tier item
(the same override ``materiality_for`` applies per-item) short-circuits
straight to the top level regardless of everything else's score, so one
determinative fact cannot be diluted by a pile of low-stakes ones the way a
naive weighted average might.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from src.review.assess import CrossFieldFindingWithMateriality, FlaggedInstance

__all__ = ["ReviewPriority", "RankedItem", "review_priority_for", "rank_by_impact"]


@dataclass(frozen=True)
class ReviewPriority:
    """One point on the five-point review-priority scale.

    Attributes:
        key: The level's config key (``"none"``, ``"light"``, ``"moderate"``,
            ``"close"``, ``"urgent"``) -- stable, for tests and logging.
        label: Reviewer-facing short label, e.g. ``"Close review"``.
        description: One or two sentences explaining what that level means,
            worded around attention needed, never around a lean.
    """

    key: str
    label: str
    description: str


def _level(levels: Mapping[str, Mapping[str, Any]], key: str) -> ReviewPriority:
    entry = levels[key]
    return ReviewPriority(
        key=str(entry["key"]), label=str(entry["label"]), description=str(entry["description"])
    )


def _materiality_weight(
    materiality: Any, materiality_weight: Mapping[str, Any], default_weight: float
) -> float:
    if materiality is None:
        return default_weight
    return float(materiality_weight.get(materiality.label, default_weight))


def _flag_impact(fi: "FlaggedInstance", scoring: Mapping[str, Any]) -> tuple[float, bool]:
    """One flag's (impact_score, is_gate) -- the exact per-item contribution
    ``review_priority_for`` sums over, factored out so a caller that wants to
    rank individual items (see :func:`rank_by_impact`) uses the same numbers
    the aggregate is built from rather than a second, divergent notion of
    "how much this matters"."""
    if fi.materiality is not None and fi.materiality.is_gate:
        return float("inf"), True
    band_weight = scoring["band_weight"]
    default_weight = float(scoring.get("default_materiality_weight", 1.0))
    b = float(band_weight.get(fi.band.label, 1)) if fi.band is not None else 1.0
    m = _materiality_weight(fi.materiality, scoring["materiality_weight"], default_weight)
    return b * m, False


def _cross_field_impact(
    cf: "CrossFieldFindingWithMateriality", scoring: Mapping[str, Any]
) -> tuple[float, bool]:
    """Same idea as :func:`_flag_impact`, for a cross-field finding."""
    if cf.materiality is not None and cf.materiality.is_gate:
        return float("inf"), True
    severity_weight = scoring["cross_field_severity_weight"]
    default_weight = float(scoring.get("default_materiality_weight", 1.0))
    s = float(severity_weight.get(cf.finding.severity, 1))
    m = _materiality_weight(cf.materiality, scoring["materiality_weight"], default_weight)
    return s * m, False


def review_priority_for(
    flagged: Sequence["FlaggedInstance"],
    cross_field_findings: Sequence["CrossFieldFindingWithMateriality"],
    review_config: Mapping[str, Any],
) -> ReviewPriority:
    """Aggregate one submission's flags and findings into a priority level.

    Args:
        flagged: The submission's tick-box-vs-text flags, as produced by
            :func:`~src.review.assess.assess_submission` -- each item's
            already-computed confidence band and materiality tier are read;
            nothing else about the flag (its score, its underlying text) is.
        cross_field_findings: The submission's tick-box-vs-tick-box findings,
            same source -- each item's authored severity and materiality
            tier are read.
        review_config: Parsed ``config/review.yaml``, for
            ``review_priority``'s levels and scoring weights.

    Returns:
        The :class:`ReviewPriority` this submission's flags add up to.
        ``"none"`` if there is nothing flagged at all. ``"urgent"`` if any
        flag or finding carries gate-tier materiality, regardless of every
        other score -- checked before, and independently of, the weighted
        sum below, the same override :func:`~src.review.materiality.materiality_for`
        applies per-item.
    """
    cfg = review_config["review_priority"]
    levels = {lvl["key"]: lvl for lvl in cfg["levels"]}
    scoring = cfg["scoring"]

    if not flagged and not cross_field_findings:
        return _level(levels, "none")

    score = 0.0
    for fi in flagged:
        item_score, is_gate = _flag_impact(fi, scoring)
        if is_gate:
            return _level(levels, "urgent")
        score += item_score

    for cf in cross_field_findings:
        item_score, is_gate = _cross_field_impact(cf, scoring)
        if is_gate:
            return _level(levels, "urgent")
        score += item_score

    for threshold in scoring["thresholds"]:
        if score >= float(threshold["min_score"]):
            return _level(levels, str(threshold["key"]))

    return _level(levels, "light")


@dataclass(frozen=True)
class RankedItem:
    """One flag or cross-field finding, carrying how much it drove the
    submission's review priority.

    Lets a reviewer see *which questions* are behind a "Close review" or
    "Urgent review" result, not just the aggregate level -- without inventing
    a second notion of importance: ``impact_score`` (and the gate override,
    via ``is_gate``) is read directly from the same per-item numbers
    :func:`review_priority_for` sums over.

    Attributes:
        kind: ``"flag"`` or ``"cross_field"`` -- which of ``flag``/
            ``cross_field`` is populated.
        flag: Set when ``kind == "flag"``.
        cross_field: Set when ``kind == "cross_field"``.
        impact_score: This item's contribution to the aggregate score.
            ``float("inf")`` for a gate item (see ``is_gate``) -- it is not a
            meaningful magnitude to display, only an ordering device.
        is_gate: True if this item alone carries gate-tier materiality, the
            same override that sends the whole submission straight to
            "Urgent review" regardless of everything else.
    """

    kind: str
    flag: "FlaggedInstance | None"
    cross_field: "CrossFieldFindingWithMateriality | None"
    impact_score: float
    is_gate: bool


def rank_by_impact(
    flagged: Sequence["FlaggedInstance"],
    cross_field_findings: Sequence["CrossFieldFindingWithMateriality"],
    review_config: Mapping[str, Any],
    limit: int | None = None,
) -> list[RankedItem]:
    """Rank every flag and cross-field finding by its contribution to the
    submission's review priority, strongest first.

    Gate items sort first (ties broken by original order), then everything
    else by ``impact_score`` descending. This is a *view* onto numbers
    ``review_priority_for`` already computes -- it never reads a record,
    a raw score, or which way an answer leans, for the same reason that
    function does not.

    Args:
        flagged: As passed to :func:`review_priority_for`.
        cross_field_findings: As passed to :func:`review_priority_for`.
        review_config: Parsed ``config/review.yaml``.
        limit: If given, only the top ``limit`` items are returned.

    Returns:
        Ranked items, strongest driver first.
    """
    scoring = review_config["review_priority"]["scoring"]
    ranked: list[RankedItem] = []
    for fi in flagged:
        item_score, is_gate = _flag_impact(fi, scoring)
        ranked.append(RankedItem(kind="flag", flag=fi, cross_field=None, impact_score=item_score, is_gate=is_gate))
    for cf in cross_field_findings:
        item_score, is_gate = _cross_field_impact(cf, scoring)
        ranked.append(RankedItem(kind="cross_field", flag=None, cross_field=cf, impact_score=item_score, is_gate=is_gate))

    ranked.sort(key=lambda r: (r.is_gate, r.impact_score), reverse=True)
    return ranked[:limit] if limit is not None else ranked
