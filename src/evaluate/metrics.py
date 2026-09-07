"""Evaluation metrics for contradiction detection.

WHY ACCURACY APPEARS NOWHERE
The positive rate is about 0.7%. A detector that never fires scores 99.3%
accuracy and finds nothing. Every metric here is one that a do-nothing detector
fails.

THE OPERATIONAL METRIC IS FLAGS PER SUBMISSION
Precision of 0.15 means nothing to the IR35 team. "You will see two flags on the
average form and dismiss most of them" means everything, and it is the number
that decides whether the tool gets used or switched off. Reviewer burden is
therefore reported alongside every threshold choice, and thresholds are selected
by asking for a recall target and reporting what it costs.

WHY RECALL IS THE OBJECTIVE
A missed contradiction reaches HMRC as a determination made without the
"reasonable care" the off-payroll rules require — a compliance and dispute
exposure measured in tax liability. A false flag costs a reviewer perhaps thirty
seconds to read and dismiss. The asymmetry is enormous, so the operating point
is chosen for recall.

That reasoning has a limit, and the limit is human rather than statistical: if
reviewers see so many flags that they start dismissing without reading, recall
on paper rises while recall in practice collapses. The precision floor is set by
reviewer fatigue, not by a metric, which is why ``flags_per_record`` is reported
at every operating point and why Phase 5 captures accept/dismiss decisions —
those are the only data that will tell us where the real floor is.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.features.dataset import PairInstance

__all__ = [
    "OperatingPoint",
    "EvaluationResult",
    "evaluate",
    "threshold_for_recall",
    "breakdown",
]


@dataclass(frozen=True)
class OperatingPoint:
    """Performance at one threshold.

    Attributes:
        threshold: Score at or above which a flag is raised.
        precision: Of the flags raised, the share that were real.
        recall: Of the real contradictions, the share flagged.
        f1: Harmonic mean of the two.
        n_flags: Total flags raised.
        flags_per_record: Reviewer burden — the number that decides adoption.
        n_true_positives: Contradictions correctly flagged.
        n_false_negatives: Contradictions missed. The costly errors.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    n_flags: int
    flags_per_record: float
    n_true_positives: int
    n_false_negatives: int

    def as_dict(self) -> dict[str, Any]:
        """Serialise for reporting."""
        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_flags": self.n_flags,
            "flags_per_record": round(self.flags_per_record, 2),
            "true_positives": self.n_true_positives,
            "missed": self.n_false_negatives,
        }


@dataclass
class EvaluationResult:
    """Threshold-free and threshold-based results for one method on one split.

    Attributes:
        method: Detector name.
        split: Split kind the scores came from.
        n_instances: Instances scored.
        n_positive: Ground-truth contradictions present.
        n_records: Submissions represented.
        pr_auc: Average precision. The headline threshold-free metric at this
            class balance — ROC-AUC is optimistic when negatives dominate.
        roc_auc: Reported because the brief asks for it, with that caveat.
        macro_f1: Macro-F1 across both classes at the chosen operating point.
        brier: Calibration. Low is good; a rule baseline emitting arbitrary
            scores will look poor here and should.
        operating_points: Performance across a threshold sweep.
        chosen: The recall-targeted operating point.
        by_type: Recall per contradiction type.
        by_subtlety: Recall per subtlety level.
        by_archetype: Recall per engagement archetype.
        by_register: Recall per writing register.
        notes: Free-text caveats carried into the report.
    """

    method: str
    split: str
    n_instances: int
    n_positive: int
    n_records: int
    pr_auc: float
    roc_auc: float
    macro_f1: float
    brier: float
    operating_points: list[OperatingPoint] = dc_field(default_factory=list)
    chosen: OperatingPoint | None = None
    by_type: dict[str, dict[str, Any]] = dc_field(default_factory=dict)
    by_subtlety: dict[str, dict[str, Any]] = dc_field(default_factory=dict)
    by_archetype: dict[str, dict[str, Any]] = dc_field(default_factory=dict)
    by_register: dict[str, dict[str, Any]] = dc_field(default_factory=dict)
    notes: list[str] = dc_field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for reporting."""
        return {
            "method": self.method,
            "split": self.split,
            "n_instances": self.n_instances,
            "n_positive": self.n_positive,
            "n_records": self.n_records,
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "macro_f1": round(self.macro_f1, 4),
            "brier": round(self.brier, 5),
            "chosen": self.chosen.as_dict() if self.chosen else None,
            "by_type": self.by_type,
            "by_subtlety": self.by_subtlety,
            "by_archetype": self.by_archetype,
            "by_register": self.by_register,
            "notes": self.notes,
        }


def _point(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, n_records: int
) -> OperatingPoint:
    """Compute performance at one threshold."""
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    true_positives = int(((predicted == 1) & (y_true == 1)).sum())
    return OperatingPoint(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        n_flags=int(predicted.sum()),
        flags_per_record=float(predicted.sum()) / max(n_records, 1),
        n_true_positives=true_positives,
        n_false_negatives=int(y_true.sum()) - true_positives,
    )


def threshold_for_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
    n_records: int,
) -> OperatingPoint:
    """Find the highest threshold that still reaches a recall target.

    Chosen this way round — highest threshold meeting the target, rather than
    the threshold maximising F1 — because the objective is a stated recall level
    at the least reviewer burden that achieves it. F1 would trade away recall to
    buy precision, which is the wrong trade when a miss is a compliance failure
    and a false flag is thirty seconds of a reviewer's time.

    Args:
        y_true: Ground-truth labels.
        scores: Predicted probabilities.
        target_recall: Recall to reach, e.g. 0.8.
        n_records: Submissions represented, for the burden figure.

    Returns:
        The operating point. If the target is unreachable at any threshold, the
        point with the highest achievable recall is returned instead — and the
        caller is expected to say so rather than quietly report a lower recall.
    """
    # Only thresholds above zero are candidates. A detector that emits exactly
    # zero for "no evidence" — the rule baseline does — would otherwise be
    # handed threshold 0.0, flag every field on every form, and be recorded as
    # achieving 100% recall. That is not an operating point; it is switching the
    # detector off and calling it perfect. Excluding it makes the real result
    # visible: some methods cannot reach the recall target at all.
    positive = scores[scores > 0]
    if positive.size == 0:
        return _point(y_true, scores, 1.0, n_records)

    candidates = np.sort(np.unique(positive))[::-1]
    best: OperatingPoint | None = None
    for threshold in candidates:
        point = _point(y_true, scores, threshold, n_records)
        if point.recall >= target_recall:
            return point
        if best is None or point.recall > best.recall:
            best = point
    return best if best is not None else _point(y_true, scores, float(candidates[-1]), n_records)


def _group_recall(
    instances: Sequence[PairInstance],
    y_true: np.ndarray,
    predicted: np.ndarray,
    key: str,
) -> dict[str, dict[str, Any]]:
    """Recall within each level of a grouping variable.

    Only positives contribute, so this answers "of the contradictions of this
    kind, how many did we catch" — which is the question Phase 4 asks of the
    subtlety curve and the archetype breakdown.
    """
    out: dict[str, dict[str, Any]] = {}
    for index, instance in enumerate(instances):
        if y_true[index] != 1:
            continue
        value = getattr(instance, key)
        if value is None:
            continue
        bucket = out.setdefault(str(value), {"n": 0, "caught": 0})
        bucket["n"] += 1
        bucket["caught"] += int(predicted[index] == 1)
    for bucket in out.values():
        bucket["recall"] = round(bucket["caught"] / bucket["n"], 4) if bucket["n"] else 0.0
    return dict(sorted(out.items()))


def breakdown(
    instances: Sequence[PairInstance],
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Recall broken down by every cohort variable Phase 4 needs.

    Args:
        instances: The scored instances.
        y_true: Ground-truth labels.
        scores: Predicted probabilities.
        threshold: Operating threshold.

    Returns:
        Nested dict keyed by variable then by level.
    """
    predicted = (scores >= threshold).astype(int)
    return {
        "by_type": _group_recall(instances, y_true, predicted, "contradiction_type"),
        "by_subtlety": _group_recall(instances, y_true, predicted, "subtlety"),
        "by_archetype": _group_recall(instances, y_true, predicted, "archetype"),
        "by_register": _group_recall(instances, y_true, predicted, "register"),
    }


def evaluate(
    method: str,
    split: str,
    instances: Sequence[PairInstance],
    scores: np.ndarray,
    target_recall: float = 0.80,
    thresholds: Sequence[float] | None = None,
) -> EvaluationResult:
    """Evaluate one method's scores on one split.

    Args:
        method: Detector name.
        split: Split kind, for the report.
        instances: The scored instances, carrying labels and cohort variables.
        scores: Predicted probabilities, aligned with instances.
        target_recall: Recall the operating point aims for.
        thresholds: Optional explicit sweep. Defaults to a fixed grid so that
            methods are compared at the same points.

    Returns:
        An ``EvaluationResult``.
    """
    y_true = np.asarray([i.label for i in instances])
    scores = np.asarray(scores, dtype=float)
    n_records = len({i.record_id for i in instances})
    grid = list(thresholds) if thresholds is not None else [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
    ]

    single_class = len(np.unique(y_true)) < 2
    result = EvaluationResult(
        method=method,
        split=split,
        n_instances=len(instances),
        n_positive=int(y_true.sum()),
        n_records=n_records,
        pr_auc=float(average_precision_score(y_true, scores)) if not single_class else float("nan"),
        roc_auc=float(roc_auc_score(y_true, scores)) if not single_class else float("nan"),
        macro_f1=0.0,
        brier=float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
    )
    result.operating_points = [_point(y_true, scores, t, n_records) for t in grid]
    result.chosen = threshold_for_recall(y_true, scores, target_recall, n_records)

    predicted = (scores >= result.chosen.threshold).astype(int)
    result.macro_f1 = float(f1_score(y_true, predicted, average="macro", zero_division=0))

    parts = breakdown(instances, y_true, scores, result.chosen.threshold)
    result.by_type = parts["by_type"]
    result.by_subtlety = parts["by_subtlety"]
    result.by_archetype = parts["by_archetype"]
    result.by_register = parts["by_register"]

    if result.chosen.recall < target_recall:
        result.notes.append(
            f"Recall target {target_recall:.0%} is unreachable at any threshold; "
            f"the best achievable is {result.chosen.recall:.0%}. Reported at that point."
        )
    if result.n_positive < 30:
        result.notes.append(
            f"Only {result.n_positive} positives on this split — per-type and "
            "per-subtlety cells are small and the intervals are wide."
        )
    return result
