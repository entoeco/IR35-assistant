"""Calibration, and honest uncertainty on small cells.

WHY CALIBRATION IS NOT OPTIONAL HERE
Phase 5 shows a reviewer a confidence number next to every flag. If that number
is not calibrated it is worse than showing nothing: a reviewer who learns that
"0.9" means "usually wrong" has been trained to distrust the tool, and one who
takes 0.9 at face value when it is really 0.3 has been misled into skipping a
check. A confidence score displayed to a human is a claim about the world, and
this module is where that claim is tested.

Expect the rule baseline to calibrate badly. Its score is a weighted count of
matched cues squashed through a logistic — monotone, and useful for ranking, but
never intended to be a probability. That is a finding to report and a reason to
present its output as a rank or a band rather than a percentage.

WHY A CLUSTER BOOTSTRAP
Per-type and per-archetype recall rests on nine to eleven positives per cell.
A bare point estimate at that size invites over-reading, and the ordinary
bootstrap over instances would understate the interval: instances within one
submission are not independent — same manager, same vocabulary, same engagement,
and the register is a record-level property. Resampling whole records keeps that
correlation structure, which is the difference between an interval that means
something and one that flatters the estimate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Any, Sequence

import numpy as np

from src.features.dataset import PairInstance

__all__ = [
    "CalibrationResult",
    "assess_calibration",
    "bootstrap_recall",
    "GroupRecall",
    "recall_with_intervals",
]


@dataclass
class CalibrationResult:
    """How well a detector's scores behave as probabilities.

    Attributes:
        method: Detector name.
        bin_centres: Mean predicted score within each occupied bin.
        bin_observed: Observed positive rate within each bin.
        bin_counts: Instances per bin.
        ece: Expected calibration error — count-weighted mean gap between
            predicted and observed. 0 is perfect.
        mce: Maximum calibration error over occupied bins.
        brier: Brier score.
        reliability: Brier decomposition term. Lower is better calibrated.
        resolution: Brier decomposition term. Higher means the scores separate
            positives from negatives.
        uncertainty: Base-rate variance. Fixed by the data, not the model.
        n_scored: Instances scored.
    """

    method: str
    bin_centres: list[float] = dc_field(default_factory=list)
    bin_observed: list[float] = dc_field(default_factory=list)
    bin_counts: list[int] = dc_field(default_factory=list)
    ece: float = 0.0
    mce: float = 0.0
    brier: float = 0.0
    reliability: float = 0.0
    resolution: float = 0.0
    uncertainty: float = 0.0
    n_scored: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the report."""
        return {
            "method": self.method,
            "ece": round(self.ece, 5),
            "mce": round(self.mce, 5),
            "brier": round(self.brier, 5),
            "reliability": round(self.reliability, 6),
            "resolution": round(self.resolution, 6),
            "uncertainty": round(self.uncertainty, 6),
            "n_scored": self.n_scored,
            "bins": [
                {"predicted": round(c, 4), "observed": round(o, 4), "n": n}
                for c, o, n in zip(self.bin_centres, self.bin_observed, self.bin_counts)
            ],
        }


def assess_calibration(
    method: str,
    y_true: Sequence[int],
    scores: Sequence[float],
    n_bins: int = 10,
    min_bin_count: int = 20,
) -> CalibrationResult:
    """Bin scores and compare predicted against observed frequency.

    Args:
        method: Detector name, for reporting.
        y_true: Ground-truth labels.
        scores: Predicted probabilities.
        n_bins: Number of equal-width bins over [0, 1].
        min_bin_count: Bins with fewer instances are excluded from ECE and from
            the reliability curve. At a 0.7% base rate a bin holding four
            instances produces an observed rate of 0.00 or 0.25 and nothing in
            between, which is noise plotted as a finding.

    Returns:
        A ``CalibrationResult``.
    """
    labels = np.asarray(y_true, dtype=float)
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    base_rate = labels.mean() if labels.size else 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    weighted_gap = 0.0
    total_weight = 0
    max_gap = 0.0
    reliability = 0.0
    resolution = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (values >= lower) & (values < upper if upper < 1.0 else values <= upper)
        n = int(in_bin.sum())
        if n == 0:
            continue
        predicted = float(values[in_bin].mean())
        actual = float(labels[in_bin].mean())
        reliability += n * (predicted - actual) ** 2
        resolution += n * (actual - base_rate) ** 2
        if n >= min_bin_count:
            centres.append(predicted)
            observed.append(actual)
            counts.append(n)
            weighted_gap += n * abs(predicted - actual)
            total_weight += n
            max_gap = max(max_gap, abs(predicted - actual))

    total = max(len(values), 1)
    return CalibrationResult(
        method=method,
        bin_centres=centres,
        bin_observed=observed,
        bin_counts=counts,
        ece=weighted_gap / total_weight if total_weight else 0.0,
        mce=max_gap,
        brier=float(np.mean((values - labels) ** 2)),
        reliability=reliability / total,
        resolution=resolution / total,
        uncertainty=float(base_rate * (1 - base_rate)),
        n_scored=int(values.size),
    )


@dataclass
class GroupRecall:
    """Recall within one level of a grouping variable, with an interval.

    Attributes:
        group: Level name.
        n: Positives in this group.
        caught: Positives flagged.
        recall: ``caught / n``.
        low: Lower bound of the bootstrap interval.
        high: Upper bound.
    """

    group: str
    n: int
    caught: int
    recall: float
    low: float
    high: float

    @property
    def interval_width(self) -> float:
        """How wide the interval is — the honest measure of how much this cell
        can support."""
        return self.high - self.low

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the report."""
        return {
            "group": self.group,
            "n": self.n,
            "caught": self.caught,
            "recall": round(self.recall, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
        }


def bootstrap_recall(
    instances: Sequence[PairInstance],
    y_true: np.ndarray,
    predicted: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260907,
) -> tuple[float, float]:
    """Cluster-bootstrap interval for overall recall.

    Records, not instances, are resampled. Instances within a submission share a
    manager, a vocabulary and a register, so treating them as independent draws
    would produce an interval narrower than the data supports.

    Args:
        instances: The scored instances.
        y_true: Ground-truth labels.
        predicted: Binary predictions at the operating threshold.
        n_boot: Bootstrap replicates.
        alpha: Two-sided significance level.
        seed: Reproducibility seed.

    Returns:
        ``(low, high)`` percentile interval. ``(0.0, 1.0)`` when there are no
        positives to resample.
    """
    by_record: dict[str, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        by_record[instance.record_id].append(index)
    record_ids = list(by_record)
    if not record_ids or y_true.sum() == 0:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(len(record_ids), size=len(record_ids), replace=True)
        indices = [i for pick in sampled for i in by_record[record_ids[pick]]]
        labels = y_true[indices]
        if labels.sum() == 0:
            continue
        estimates.append(float((predicted[indices] & (labels == 1)).sum() / labels.sum()))
    if not estimates:
        return 0.0, 1.0
    return (
        float(np.percentile(estimates, 100 * alpha / 2)),
        float(np.percentile(estimates, 100 * (1 - alpha / 2))),
    )


def recall_with_intervals(
    instances: Sequence[PairInstance],
    scores: np.ndarray,
    threshold: float,
    attribute: str,
    n_boot: int = 2000,
    seed: int = 20260907,
) -> list[GroupRecall]:
    """Per-group recall with cluster-bootstrap intervals.

    Args:
        instances: The scored instances.
        scores: Predicted probabilities.
        threshold: Operating threshold.
        attribute: Instance attribute to group by, e.g. ``"archetype"``.
        n_boot: Bootstrap replicates.
        seed: Reproducibility seed.

    Returns:
        One ``GroupRecall`` per level, ordered by name.
    """
    y_true = np.asarray([i.label for i in instances])
    predicted = (np.asarray(scores) >= threshold).astype(int)

    levels: dict[str, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        value = getattr(instance, attribute)
        if instance.label == 1 and value is not None:
            levels[str(value)].append(index)

    out: list[GroupRecall] = []
    for name, positive_indices in sorted(levels.items()):
        # Resample within the group's own records so the interval reflects this
        # cell's size rather than the corpus's.
        group_records = {instances[i].record_id for i in positive_indices}
        group_indices = [
            i for i, instance in enumerate(instances) if instance.record_id in group_records
        ]
        subset = [instances[i] for i in group_indices]
        subset_true = y_true[group_indices]
        subset_pred = predicted[group_indices]
        # Restrict to this level's positives so a record carrying two different
        # types does not contaminate the cell.
        mask = np.array(
            [
                1 if (idx in set(positive_indices)) else 0
                for idx in group_indices
            ]
        )
        subset_true = subset_true * mask

        low, high = bootstrap_recall(subset, subset_true, subset_pred, n_boot, seed=seed)
        caught = int((predicted[positive_indices] == 1).sum())
        n = len(positive_indices)
        out.append(
            GroupRecall(
                group=name,
                n=n,
                caught=caught,
                recall=caught / n if n else 0.0,
                low=low,
                high=high,
            )
        )
    return out
