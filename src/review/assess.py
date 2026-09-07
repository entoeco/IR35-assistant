"""Run one submission through the full pipeline and produce reviewer flags.

Pipeline position, same order as everywhere else in this project and as
constraint 3 requires: **de-identify, then validate, then score.** Nothing
downstream of this module ever sees a field the de-identifier was configured
to scrub, and a record that fails a contract check (wrong option, outcome
leakage) is reported to the reviewer rather than silently scored.

This module deliberately does NOT call ``CestRuleEngine``. That engine
produces the synthetic corpus's *ground-truth label* for Phases 1-4's
evaluation — it is a stand-in for a status determination, and the Phase 5
brief is explicit that a determination must never be an output of this tool.
Importing it here would make "don't show a status" a UI-layer promise rather
than a fact about what was computed, and a promise like that does not survive
a stakeholder asking for "just the answer". So it is simply never invoked
from this module, and there is nothing to hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.deidentify.deidentifier import DeidentificationReport, Deidentifier
from src.features.dataset import PairInstance, build_instances_single
from src.ingest.schema_loader import Schema
from src.ingest.validate import ValidationResult, RecordValidator
from src.models.base import ContradictionDetector, Flag
from src.models.baseline_rules import RuleBaseline
from src.models.baseline_tfidf import TfidfBaseline
from src.review.bands import ConfidenceBand, band_for_score, load_band_config

__all__ = ["FlaggedInstance", "AssessmentResult", "build_detector", "assess_submission"]


@dataclass(frozen=True)
class FlaggedInstance:
    """One flag, with the reviewer-facing band already resolved.

    Attributes:
        flag: The underlying :class:`~src.models.base.Flag`.
        instance: The scored instance, kept for building a decision (needs
            ``ir35_test`` etc.) and for a "why was this pair checked" view.
        band: The confidence band this score fell into, or ``None`` if the
            score was below every configured band — such instances are
            filtered out before they reach the reviewer (see
            :func:`assess_submission`), but the field exists so a caller can
            assert on it directly.
    """

    flag: Flag
    instance: PairInstance
    band: ConfidenceBand | None


@dataclass(frozen=True)
class AssessmentResult:
    """Everything the reviewer app shows for one submission.

    Attributes:
        record_id: The submission's identifier.
        method: Which detector produced these flags.
        deid_report: What the de-identifier removed. Metadata only.
        validation: Contract and completeness findings.
        flagged: Flags at or above their method's lowest confidence band,
            sorted by score descending — the reviewer sees the strongest
            evidence first.
        n_instances_scored: Total (structured answer, justification) pairs
            considered, flagged or not — context for "how much of the form
            did this actually check".
    """

    record_id: str
    method: str
    deid_report: DeidentificationReport
    validation: ValidationResult
    flagged: tuple[FlaggedInstance, ...]
    n_instances_scored: int


def build_detector(
    method: str,
    schema: Schema,
    *,
    rule_config: Mapping[str, Any] | None = None,
    text_bank: Mapping[str, Any] | None = None,
    training_instances: Sequence[PairInstance] | None = None,
) -> ContradictionDetector:
    """Construct one of the detectors offered by ``config/review.yaml``.

    Args:
        method: One of ``"rules_as_authored"``, ``"rules_phrase_deleaked"``,
            ``"tfidf_logreg"``.
        schema: The loaded data contract.
        rule_config: Parsed ``rule_baseline.yaml``. Required for the two rule
            variants.
        text_bank: Parsed ``text_bank.yaml``. Required for the de-leaked rule
            variant, which needs it to know what counts as "copied phrasing".
        training_instances: Labelled instances to fit on. Required for
            ``tfidf_logreg``; ignored by the rule variants, which need no
            training data.

    Returns:
        A fitted, ready-to-score detector.

    Raises:
        ValueError: Unknown method name, or a required argument is missing
            for the requested method.

    Note:
        Fitting ``tfidf_logreg`` here, at call time, on the synthetic corpus
        is a convenience for this demonstration app only. A real deployment
        would load a persisted model artefact that has been through its own
        validation and sign-off, for the same reason a production service
        does not retrain itself on every request — see
        ``docs/reports/phase5_interface_report.md``, "what changes for real
        data".
    """
    if method in ("rules_as_authored", "rules_phrase_deleaked"):
        if rule_config is None:
            raise ValueError(f"{method} requires rule_config")
        if method == "rules_as_authored":
            return RuleBaseline(schema, rule_config, variant="as_authored")
        if text_bank is None:
            raise ValueError("rules_phrase_deleaked requires text_bank")
        return RuleBaseline.de_leaked(
            schema, rule_config, text_bank, min_leak_words=2, variant="phrase_deleaked"
        )
    if method == "tfidf_logreg":
        if not training_instances:
            raise ValueError("tfidf_logreg requires training_instances")
        return TfidfBaseline(schema).fit(list(training_instances))
    raise ValueError(f"unknown detector method: {method!r}")


def assess_submission(
    record: Mapping[str, Any],
    *,
    schema: Schema,
    deidentifier: Deidentifier,
    validator: RecordValidator,
    detector: ContradictionDetector,
    review_config: Mapping[str, Any],
) -> AssessmentResult:
    """Run one raw submission through de-identify, validate, score, explain.

    Args:
        record: The raw submission. Field id -> value. May still contain
            personal data — that is what this function's first step removes.
        schema: The loaded data contract.
        deidentifier: Configured per constraint 3.
        validator: Configured per the data contract.
        detector: An already-fitted detector, from :func:`build_detector`.
        review_config: Parsed ``config/review.yaml``, for band cut-points.

    Returns:
        An :class:`AssessmentResult`. ``flagged`` contains only instances that
        cleared the detector's lowest configured band — a score with no band
        is, by the config's own definition, not worth a reviewer's time and
        is dropped rather than shown as a zero-confidence curiosity.
    """
    clean_record, deid_report = deidentifier.deidentify(record)
    validation = validator.validate(clean_record)

    instances = build_instances_single(schema, clean_record)
    bands = load_band_config(review_config, detector.name)

    flagged: list[FlaggedInstance] = []
    if instances:
        scores = detector.score(instances)
        for instance, score in zip(instances, scores):
            band = band_for_score(float(score), bands)
            if band is None:
                continue
            flag = detector.flag(instance, float(score), schema)
            flagged.append(FlaggedInstance(flag=flag, instance=instance, band=band))

    flagged.sort(key=lambda fi: fi.flag.score, reverse=True)

    return AssessmentResult(
        record_id=str(clean_record.get("record_id", "unknown")),
        method=detector.name,
        deid_report=deid_report,
        validation=validation,
        flagged=tuple(flagged),
        n_instances_scored=len(instances),
    )
