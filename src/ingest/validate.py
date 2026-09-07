"""Record validation against the ESQ data contract.

Runs after de-identification and before anything reads a record for modelling.
Two jobs:

1. **Contract conformance.** Structured answers must come from the value domain
   declared in ``schema.yaml``. Anything else means the config and the data have
   diverged, and every downstream assumption is void.
2. **Completeness, routing-aware.** The form says an ESQ "will be rejected if
   not all fields have been completed", so a missing justification matters. But
   the form also skips questions depending on earlier answers, and a blank field
   off the active route is correct rather than missing. Without the routing walk
   a completeness check reports noise on every single record, which is how a
   check ends up switched off.

Findings are returned, not raised. A validator that throws on the first problem
tells you about one field; a reviewer needs the whole picture, and Phase 5's UI
shows completeness findings alongside contradiction flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from src.generate.generator import walk_route
from src.ingest.schema_loader import FieldType, Schema

__all__ = ["Severity", "Finding", "ValidationResult", "RecordValidator"]


class Severity(str, Enum):
    """How much a finding should worry a reviewer."""

    ERROR = "error"      # the record breaks the data contract
    WARNING = "warning"  # the record is usable but incomplete or inconsistent
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One validation problem.

    Attributes:
        code: Stable machine-readable code, e.g. ``"value_not_in_domain"``.
        field_id: Field the finding attaches to, if any.
        severity: How serious.
        message: Human-readable description for a reviewer.
    """

    code: str
    field_id: str | None
    severity: Severity
    message: str


@dataclass
class ValidationResult:
    """The outcome for one record."""

    record_id: str
    findings: list[Finding]

    @property
    def is_valid(self) -> bool:
        """True if nothing breaks the data contract. Warnings are allowed."""
        return not any(f.severity is Severity.ERROR for f in self.findings)

    @property
    def errors(self) -> list[Finding]:
        """Contract-breaking findings only."""
        return [f for f in self.findings if f.severity is Severity.ERROR]

    def as_log_meta(self) -> dict[str, Any]:
        """Metadata-only summary, safe to log under constraint 4."""
        return {
            "record_id": self.record_id,
            "is_valid": self.is_valid,
            "n_findings": len(self.findings),
            "codes": sorted({f.code for f in self.findings}),
        }


class RecordValidator:
    """Validates ESQ records against the loaded schema."""

    def __init__(self, schema: Schema, config: Mapping[str, Any] | None = None) -> None:
        """Create a validator.

        Args:
            schema: The loaded data contract.
            config: The ``validation`` block of ``config/generation.yaml``.
        """
        cfg = dict(config or {})
        self.schema = schema
        self.enforce_domains = bool(cfg.get("enforce_value_domains", True))
        self.enforce_routing = bool(cfg.get("enforce_routing", True))
        self.require_justification = bool(
            cfg.get("require_justification_for_routed_fields", True)
        )
        self.min_free_text_chars = int(cfg.get("min_free_text_chars", 15))
        self.section_prefix = str(cfg.get("section_four_prefix", "4."))

    def validate(self, record: Mapping[str, Any]) -> ValidationResult:
        """Validate one record.

        Args:
            record: Field id -> value.

        Returns:
            A ``ValidationResult`` listing every finding.
        """
        findings: list[Finding] = []
        answers = {
            f.id: record[f.id]
            for f in self.schema.structured_answer_fields
            if record.get(f.id) not in (None, "")
        }
        reachable = (
            set(walk_route(self.schema, answers, self.section_prefix))
            if self.enforce_routing
            else None
        )

        if self.enforce_domains:
            findings.extend(self._check_domains(record))
        if reachable is not None:
            findings.extend(self._check_routing(record, reachable))
        if self.require_justification:
            findings.extend(self._check_justifications(record, reachable))
        findings.extend(self._check_no_outcome_leakage(record))

        return ValidationResult(
            record_id=str(record.get("record_id", "unknown")), findings=findings
        )

    def _check_domains(self, record: Mapping[str, Any]) -> Iterable[Finding]:
        """Every structured answer must be one of its declared options."""
        for field in self.schema.structured_answer_fields:
            value = record.get(field.id)
            if value in (None, ""):
                continue
            if field.value_domain and value not in field.value_domain:
                yield Finding(
                    code="value_not_in_domain",
                    field_id=field.id,
                    severity=Severity.ERROR,
                    message=(
                        f"{field.form_ref}: {value!r} is not one of the "
                        f"{len(field.value_domain)} options declared for this question."
                    ),
                )

    def _check_routing(
        self, record: Mapping[str, Any], reachable: set[str]
    ) -> Iterable[Finding]:
        """Answers off the active route, and gaps on it."""
        for field in self.schema.structured_answer_fields:
            if not str(field.form_ref or "").startswith(self.section_prefix):
                continue
            answered = record.get(field.id) not in (None, "")
            if answered and field.id not in reachable:
                yield Finding(
                    code="answered_off_route",
                    field_id=field.id,
                    severity=Severity.WARNING,
                    message=(
                        f"{field.form_ref} was answered but the routing from earlier "
                        "answers skips it. One of the two is wrong."
                    ),
                )
            elif not answered and field.id in reachable:
                yield Finding(
                    code="missing_on_route",
                    field_id=field.id,
                    severity=Severity.WARNING,
                    message=f"{field.form_ref} is on the active route but has no answer.",
                )

    def _check_justifications(
        self, record: Mapping[str, Any], reachable: set[str] | None
    ) -> Iterable[Finding]:
        """A routed question with an answer needs a usable rationale.

        The form states an ESQ will be rejected if not all fields are completed,
        so this is a real finding for a reviewer rather than a nicety — and a
        justification too short to say anything is the same problem as a blank
        one wearing a hat.
        """
        for structured_id, free_text_id in self.schema.justification_map.items():
            if record.get(structured_id) in (None, ""):
                continue
            if reachable is not None and structured_id not in reachable:
                continue
            text = record.get(free_text_id)
            if text in (None, ""):
                yield Finding(
                    code="missing_justification",
                    field_id=free_text_id,
                    severity=Severity.WARNING,
                    message=(
                        f"{self.schema[structured_id].form_ref} is answered but its "
                        "rationale box is empty."
                    ),
                )
            elif len(str(text).strip()) < self.min_free_text_chars:
                yield Finding(
                    code="justification_too_short",
                    field_id=free_text_id,
                    severity=Severity.WARNING,
                    message=(
                        f"{self.schema[structured_id].form_ref}: rationale is "
                        f"{len(str(text).strip())} characters, below the "
                        f"{self.min_free_text_chars}-character threshold for a "
                        "usable justification."
                    ),
                )

    def _check_no_outcome_leakage(self, record: Mapping[str, Any]) -> Iterable[Finding]:
        """The determination must never travel with a record into modelling.

        Declared as a check rather than relying on ingest having stripped it,
        because leakage is silent: the model just gets better and nobody asks why.
        """
        for field in self.schema.outcome_fields:
            if record.get(field.id) not in (None, ""):
                yield Finding(
                    code="outcome_leakage",
                    field_id=field.id,
                    severity=Severity.ERROR,
                    message=(
                        f"{field.id} carries the IR35 team's determination and must be "
                        "stripped before a record reaches any model."
                    ),
                )

    def validate_many(
        self, records: Iterable[Mapping[str, Any]]
    ) -> list[ValidationResult]:
        """Validate a sequence of records, preserving order."""
        return [self.validate(record) for record in records]
