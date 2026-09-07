"""Style invariance: does the detector read substance or surface?

THE TEST
Every record in the Phase 1 corpus was built in two stages — ``build_state``
decided what the submission says, ``render`` decided how it is written. So the
same engagement can be re-rendered verbose, terse and hedged with the structured
answers, the propositions and the planted contradictions held literally
identical. Only the wording moves.

A detector that reads meaning should produce the same flags on all three. Any
difference is attributable to surface form alone, because nothing else differs.
That is what makes this a real test rather than a comparison of three texts that
happen to say roughly the same thing.

WHY IT MATTERS BEYOND ROBUSTNESS
This is a fairness question wearing a technical hat. Register is not randomly
distributed across engagement types: in the Phase 1 corpus, visiting-lecturer
submissions skew verbose and hedged, technical-trades submissions skew terse. A
detector that performs worse on hedged prose therefore performs worse on a
particular category of worker — and it would do so invisibly, because the
per-archetype breakdown alone cannot separate "this engagement type is harder"
from "these managers write differently". Holding the engagement fixed and
varying only the writing separates the two.

WHAT IS MEASURED
1. **Flag-set agreement** — Jaccard overlap of the fields flagged, per record,
   across the three renderings. The reviewer-facing quantity: did the tool show
   the same fields?
2. **Score drift** — mean and maximum absolute movement of the score for the
   same field. Catches a detector that keeps its flag set but wobbles enough
   that a threshold change would break it.
3. **Per-register recall** at one fixed threshold, on the same planted
   contradictions.

TRAINING DISCIPLINE
For a fitted detector, the model is trained on the ORIGINAL rendering of the
training records and evaluated on all three renderings of the held-out records.
Training on the re-rendered text would leak the test conditions into the model
and turn the invariance test into a memorisation test.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.features.dataset import PairInstance, build_instances
from src.features.splits import make_folds
from src.generate.generator import EsqGenerator, RecordState
from src.ingest.schema_loader import Schema
from src.models.base import ContradictionDetector
from src.utils.logging import StructuredLogger, get_logger

__all__ = ["StyleInvarianceResult", "render_registers", "measure_style_invariance"]

REGISTERS = ("verbose", "terse", "hedged")


@dataclass
class StyleInvarianceResult:
    """Drift for one detector across three renderings of the same engagements.

    Attributes:
        method: Detector name.
        threshold: Operating threshold flags were taken at.
        n_records: Records re-rendered.
        n_instances: Instances per register.
        mean_jaccard: Mean per-record flag-set agreement across register pairs.
            1.0 means every rendering flagged exactly the same fields.
        records_fully_agreeing: Records where all three renderings produced an
            identical flag set.
        records_disagreeing: Records where at least one field was flagged in one
            rendering and not another.
        mean_abs_drift: Mean absolute score movement for the same field across
            renderings.
        max_abs_drift: Largest single movement observed.
        recall_by_register: Recall on planted contradictions, per rendering.
        flags_by_register: Flags raised per record, per rendering.
        pairwise_jaccard: Agreement for each register pair.
        drift_by_archetype: Mean absolute drift within each engagement type —
            the fairness cut.
        tp_jaccard: Agreement across renderings on which TRUE contradictions
            were flagged.
        fp_jaccard: Agreement across renderings on which FALSE flags were
            raised. Low here with high `tp_jaccard` means the detector reliably
            finds the real contradictions but raises a different set of spurious
            flags depending on how the manager writes — stable recall, unstable
            reviewer experience.
        worst_examples: A few (record, pair, register, score) rows where a flag
            appeared in one rendering and vanished in another.
    """

    method: str
    threshold: float
    n_records: int
    n_instances: int
    mean_jaccard: float
    records_fully_agreeing: int
    records_disagreeing: int
    mean_abs_drift: float
    max_abs_drift: float
    recall_by_register: dict[str, float] = dc_field(default_factory=dict)
    flags_by_register: dict[str, float] = dc_field(default_factory=dict)
    pairwise_jaccard: dict[str, float] = dc_field(default_factory=dict)
    drift_by_archetype: dict[str, float] = dc_field(default_factory=dict)
    tp_jaccard: float = 1.0
    fp_jaccard: float = 1.0
    worst_examples: list[dict[str, Any]] = dc_field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        """Share of records where all three renderings agreed exactly."""
        total = self.records_fully_agreeing + self.records_disagreeing
        return self.records_fully_agreeing / total if total else 1.0

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the report."""
        return {
            "method": self.method,
            "threshold": round(self.threshold, 4),
            "n_records": self.n_records,
            "n_instances": self.n_instances,
            "mean_jaccard": round(self.mean_jaccard, 4),
            "agreement_rate": round(self.agreement_rate, 4),
            "records_fully_agreeing": self.records_fully_agreeing,
            "records_disagreeing": self.records_disagreeing,
            "mean_abs_drift": round(self.mean_abs_drift, 4),
            "max_abs_drift": round(self.max_abs_drift, 4),
            "recall_by_register": {k: round(v, 4) for k, v in self.recall_by_register.items()},
            "flags_by_register": {k: round(v, 3) for k, v in self.flags_by_register.items()},
            "pairwise_jaccard": {k: round(v, 4) for k, v in self.pairwise_jaccard.items()},
            "tp_jaccard": round(self.tp_jaccard, 4),
            "fp_jaccard": round(self.fp_jaccard, 4),
            "drift_by_archetype": {k: round(v, 4) for k, v in self.drift_by_archetype.items()},
            "n_worst_examples": len(self.worst_examples),
        }


def render_registers(
    schema: Schema,
    generator: EsqGenerator,
    states: Sequence[RecordState],
    truth: Sequence[Mapping[str, Any]],
) -> dict[str, list[PairInstance]]:
    """Re-render every state in all three registers.

    Args:
        schema: The loaded data contract.
        generator: The Phase 1 generator, used only to render — no state is
            rebuilt, so the engagements are identical by construction.
        states: Record states, in corpus order.
        truth: Ground-truth rows, aligned with states.

    Returns:
        Register name -> instances, index-aligned across registers.
    """
    by_register: dict[str, list[PairInstance]] = {}
    state_dicts = [s.as_dict() for s in states]
    for register in REGISTERS:
        records = [generator.render(state, register) for state in states]
        by_register[register] = build_instances(schema, records, truth, state_dicts)
    return by_register


def _flag_sets(
    instances: Sequence[PairInstance], scores: np.ndarray, threshold: float
) -> dict[str, set[str]]:
    """Fields flagged, grouped by record."""
    flags: dict[str, set[str]] = {}
    for instance, score in zip(instances, scores):
        flags.setdefault(instance.record_id, set())
        if score >= threshold:
            flags[instance.record_id].add(instance.pair_id)
    return flags


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard overlap. Two empty sets agree completely."""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def measure_style_invariance(
    schema: Schema,
    generator: EsqGenerator,
    states: Sequence[RecordState],
    truth: Sequence[Mapping[str, Any]],
    factory: Callable[[], ContradictionDetector],
    threshold: float,
    original: Sequence[PairInstance],
    n_folds: int = 5,
    seed: int = 20260907,
    logger: StructuredLogger | None = None,
) -> StyleInvarianceResult:
    """Score three renderings of the same engagements and quantify the drift.

    Args:
        schema: The loaded data contract.
        generator: The Phase 1 generator, for rendering.
        states: Record states.
        truth: Ground-truth rows, aligned with states.
        factory: Callable returning a fresh detector. A factory rather than an
            instance so each fold trains from scratch.
        threshold: Operating threshold at which flags are taken.
        original: The corpus as generated, with each record in its own register.
            This is what a fitted detector trains on, because it is what the
            model would see in deployment — a mixed-register corpus. Training on
            one register and testing on three would measure a train/test
            distribution mismatch rather than sensitivity to register, which is
            a different and less interesting question.
        n_folds: Cross-validation folds.
        seed: Reproducibility seed.
        logger: Structured logger.

    Returns:
        A ``StyleInvarianceResult``.
    """
    log = logger or get_logger("evaluate.style")
    rendered = render_registers(schema, generator, states, truth)
    baseline = rendered[REGISTERS[0]]

    # Folds are computed once, on the record split, and reused for every
    # register. All four views — the original corpus and the three renderings —
    # are index-aligned, so the same fold indices select the same engagements in
    # each. That is the point: the model must never have trained on the record
    # it is being asked about, in any rendering.
    folds = make_folds(list(original), "record", n_folds, seed)
    scores: dict[str, np.ndarray] = {r: np.zeros(len(baseline)) for r in REGISTERS}

    for train_idx, test_idx in folds:
        detector = factory()
        # Trained on the corpus as generated, never on the re-rendered text.
        # Training on a rendering would leak the test conditions into the model
        # and turn an invariance test into a memorisation test.
        detector.fit([original[i] for i in train_idx])
        for register in REGISTERS:
            instances = rendered[register]
            scores[register][test_idx] = detector.score([instances[i] for i in test_idx])

    labels = np.asarray([i.label for i in baseline])
    flags = {r: _flag_sets(rendered[r], scores[r], threshold) for r in REGISTERS}

    pairwise: dict[str, float] = {}
    per_record: dict[str, list[float]] = {}
    for left, right in combinations(REGISTERS, 2):
        values = []
        for record_id in flags[left]:
            value = _jaccard(flags[left][record_id], flags[right].get(record_id, set()))
            values.append(value)
            per_record.setdefault(record_id, []).append(value)
        pairwise[f"{left} vs {right}"] = float(np.mean(values)) if values else 1.0

    fully_agreeing = sum(1 for values in per_record.values() if all(v == 1.0 for v in values))
    disagreeing = len(per_record) - fully_agreeing

    stacked = np.vstack([scores[r] for r in REGISTERS])
    drift = stacked.max(axis=0) - stacked.min(axis=0)

    by_archetype: dict[str, list[float]] = {}
    for instance, value in zip(baseline, drift):
        by_archetype.setdefault(instance.archetype, []).append(float(value))

    # Split the agreement measure by whether a flag was correct. Recall can look
    # perfectly stable across registers while the reviewer's actual experience —
    # which fields light up — changes completely, and only this separation shows
    # that.
    def flagged_set(register: str, want_positive: bool) -> set[tuple[str, str]]:
        return {
            (baseline[i].record_id, baseline[i].pair_id)
            for i in range(len(baseline))
            # bool(...) == want_positive, never `is`: labels is a numpy array,
            # so labels[i] == 1 yields numpy.bool_, and `numpy.True_ is True` is
            # False. The identity test silently produced two empty sets and a
            # Jaccard of 1.0 for both — a perfect score reported for a
            # measurement that never ran.
            if scores[register][i] >= threshold
            and bool(labels[i] == 1) == want_positive
        }

    tp_values = [
        _jaccard(flagged_set(left, True), flagged_set(right, True))
        for left, right in combinations(REGISTERS, 2)
    ]
    fp_values = [
        _jaccard(flagged_set(left, False), flagged_set(right, False))
        for left, right in combinations(REGISTERS, 2)
    ]

    worst: list[dict[str, Any]] = []
    for index, instance in enumerate(baseline):
        raised = {r for r in REGISTERS if scores[r][index] >= threshold}
        if raised and len(raised) < len(REGISTERS):
            worst.append(
                {
                    "record_id": instance.record_id,
                    "pair_id": instance.pair_id,
                    "label": instance.label,
                    "flagged_in": sorted(raised),
                    "scores": {r: round(float(scores[r][index]), 3) for r in REGISTERS},
                }
            )
    worst.sort(key=lambda row: -max(row["scores"].values()))

    result = StyleInvarianceResult(
        method=factory().name,
        threshold=threshold,
        n_records=len(per_record),
        n_instances=len(baseline),
        mean_jaccard=float(np.mean(list(pairwise.values()))) if pairwise else 1.0,
        records_fully_agreeing=fully_agreeing,
        records_disagreeing=disagreeing,
        mean_abs_drift=float(np.mean(drift)),
        max_abs_drift=float(np.max(drift)) if drift.size else 0.0,
        recall_by_register={
            r: float(((scores[r] >= threshold) & (labels == 1)).sum() / max(labels.sum(), 1))
            for r in REGISTERS
        },
        flags_by_register={
            r: float((scores[r] >= threshold).sum() / max(len(per_record), 1))
            for r in REGISTERS
        },
        pairwise_jaccard=pairwise,
        drift_by_archetype={
            k: float(np.mean(v)) for k, v in sorted(by_archetype.items())
        },
        tp_jaccard=float(np.mean(tp_values)) if tp_values else 1.0,
        fp_jaccard=float(np.mean(fp_values)) if fp_values else 1.0,
        worst_examples=worst[:10],
    )
    log.event("evaluate.style_invariance", meta=result.as_dict())
    return result
