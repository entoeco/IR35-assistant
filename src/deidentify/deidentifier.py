"""De-identification. Runs first in the pipeline, before anything else.

Constraint 3 of the build brief: this is a real, tested component, not a
comment. It is exercised on every record including synthetic ones — the point
being that the day it stops being a formality must not be the day it is first
run.

Pipeline position
-----------------
``deidentify -> validate -> features -> model -> UI``. Nothing upstream of the
model sees an un-scrubbed record, and the scrubber is the only component that
ever holds the original values.

What it does
------------
1. **Builds a gazetteer** from the record's own direct-identifier fields, before
   scrubbing them. This is what lets free-text scrubbing find the worker's name
   in a justification box.
2. **Scrubs free text** using that gazetteer plus the pattern detectors.
3. **Applies field-level policy** by ``pii_class``: replace, pseudonymise, mask
   or keep.
4. **Returns a report** of what was removed — counts by field and detector,
   never the values. The report is metadata, safe to log under constraint 4.

What it does not do
-------------------
It is not a guarantee. Regex-and-gazetteer de-identification misses names it has
never seen in text it cannot parse, and it cannot recognise that "the chap who
did the Falmer roof last summer" identifies someone. Before this touches real
submissions the University's Information Governance team need to see a
validation run against real text with measured recall, and to decide whether
residual re-identification risk is acceptable for the intended use. That
conversation is a Phase 6 deliverable and a hard gate, not a formality.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping, MutableMapping, Sequence

from src.ingest.schema_loader import FieldType, PIIClass, Schema
from src.utils.logging import StructuredLogger, get_logger

from .detectors import Span, detect_all, gazetteer_spans, resolve_overlaps

__all__ = ["DeidentificationReport", "Deidentifier"]

_DEFAULT_DETECTORS = (
    "email",
    "uk_phone",
    "uk_postcode",
    "money",
    "company_number",
    "ni_number",
    "url",
    "date_of_birth",
    "person_name",
)


@dataclass
class DeidentificationReport:
    """What was removed from one record. Metadata only — never values.

    Attributes:
        record_id: Identifier of the record processed.
        field_actions: Field id -> policy applied (``replace``, ``pseudonymise``,
            ``mask``, ``keep``).
        entity_counts: Detector name -> number of spans replaced across all free
            text in the record.
        fields_with_entities: Free-text field ids that contained at least one
            detected entity. This is the interesting signal in practice: it says
            *where* managers are writing personal data into boxes that are not
            meant to hold it.
        errors: Field ids that raised during scrubbing.
    """

    record_id: str
    field_actions: dict[str, str] = dc_field(default_factory=dict)
    entity_counts: Counter[str] = dc_field(default_factory=Counter)
    fields_with_entities: list[str] = dc_field(default_factory=list)
    errors: list[str] = dc_field(default_factory=list)

    @property
    def total_entities(self) -> int:
        """Total number of free-text entities replaced."""
        return sum(self.entity_counts.values())

    @property
    def is_clean(self) -> bool:
        """True if no free-text entity was found and nothing errored.

        On a well-formed synthetic record this is usually ``False``, because the
        generator writes invented names and rates into free text on purpose —
        that is how the component gets exercised rather than assumed.
        """
        return self.total_entities == 0 and not self.errors

    def as_log_meta(self) -> dict[str, Any]:
        """Render as loggable metadata under constraint 4."""
        return {
            "record_id": self.record_id,
            "entities_removed": self.total_entities,
            "entity_counts": dict(self.entity_counts),
            "fields_with_entities": self.fields_with_entities,
            "errors": self.errors,
        }


class Deidentifier:
    """Removes identifying content from an ESQ record.

    Example:
        >>> from src.ingest.schema_loader import load_schema
        >>> deid = Deidentifier(load_schema())
        >>> record = {
        ...     "record_id": "SYN-0001",
        ...     "q1_02_first_name": "Alis",
        ...     "q1_03_surname": "Ashwood",
        ...     "q4_05_rationale": "Alis Ashwood is the only person who can do this.",
        ... }
        >>> clean, report = deid.deidentify(record)
        >>> clean["q4_05_rationale"]
        '[PERSON_1] is the only person who can do this.'
        >>> report.entity_counts["person_name"]
        1
    """

    def __init__(
        self,
        schema: Schema,
        config: Mapping[str, Any] | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Create a de-identifier.

        Args:
            schema: The loaded data contract. Field policy is driven by
                ``pii_class``, so pointing at a revised form is a config edit.
            config: The ``deidentification`` block of ``pipeline.yaml``.
            logger: Structured logger. One is created if not supplied.
        """
        self.schema = schema
        cfg = dict(config or {})
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.on_error: str = str(cfg.get("on_error", "raise"))
        self.field_policy: Mapping[str, str] = cfg.get(
            "field_policy",
            {
                "direct_identifier": "replace",
                "quasi_identifier": "pseudonymise",
                "sensitive_commercial": "mask",
                "none": "keep",
            },
        )
        entity_cfg = cfg.get("entity_scrubbing", {}) or {}
        self.entity_scrubbing: bool = bool(entity_cfg.get("enabled", True))
        self.detectors: Sequence[str] = tuple(
            entity_cfg.get("detectors", _DEFAULT_DETECTORS)
        )
        self._name_groups: Sequence[Sequence[str]] = tuple(
            tuple(group) for group in cfg.get("name_groups", ())
        )
        self._salt = os.environ.get(
            str(cfg.get("salt_env_var", "IR35_DEID_SALT")),
            str(cfg.get("development_salt", "phase1-synthetic-only")),
        )
        self.log = logger or get_logger("deidentify")

    # -- surrogates -------------------------------------------------------
    def _pseudonym(self, value: str) -> str:
        """Salted, truncated hash. Stable for equal inputs, so cohort analysis
        still works; unlinkable to the original without the salt.

        Rotating the salt breaks linkage across runs by design.
        """
        digest = hashlib.sha256(f"{self._salt}|{value}".encode("utf-8")).hexdigest()
        return f"qi_{digest[:10]}"

    # Detector name -> surrogate prefix. Kept short and readable, because a
    # reviewer has to be able to read the scrubbed justification and still
    # follow the argument being made.
    _TOKEN_NAMES = {
        "person_name": "PERSON",
        "email": "EMAIL",
        "uk_phone": "PHONE",
        "uk_postcode": "POSTCODE",
        "company_number": "COMPANY_NO",
        "ni_number": "NINO",
        "url": "URL",
        "date_of_birth": "DOB",
    }

    @classmethod
    def _surrogate(cls, kind: str, index: int) -> str:
        """Surrogate token for a free-text entity.

        Surrogates are numbered within a record so that coreference survives —
        "Dr Ashwood ... Ashwood later confirmed" both become ``[PERSON_1]`` and
        the text stays readable to a reviewer. They are deliberately **not**
        stable across records: a corpus-wide stable surrogate would rebuild a
        linkable identifier, which is the thing being removed.

        Money is unnumbered: ``[AMOUNT]`` rather than ``[MONEY_1]``. Numbering
        amounts would let a reader infer that two figures were equal, which is
        the commercially sensitive fact we are removing.
        """
        if kind == "money":
            return "[AMOUNT]"
        return f"[{cls._TOKEN_NAMES.get(kind, kind.upper())}_{index}]"

    # -- free text --------------------------------------------------------
    def scrub_text(
        self,
        text: str,
        known_values: Sequence[str] = (),
        surrogate_index: MutableMapping[str, dict[str, int]] | None = None,
    ) -> tuple[str, Counter[str]]:
        """Remove entities from one free-text value.

        Args:
            text: The value to scrub.
            known_values: Direct-identifier values from the same record, used as
                a gazetteer.
            surrogate_index: Per-record surrogate numbering, shared across the
                record's fields so the same name gets the same token everywhere.

        Returns:
            The scrubbed text and a count of replacements by detector.
        """
        counts: Counter[str] = Counter()
        if not text or not self.entity_scrubbing:
            return text, counts

        spans: list[Span] = list(detect_all(text, self.detectors))
        if "person_name" in self.detectors:
            spans.extend(gazetteer_spans(text, known_values))
        resolved = resolve_overlaps(spans)
        if not resolved:
            return text, counts

        index = surrogate_index if surrogate_index is not None else {}
        result = text
        # Right to left, so earlier offsets stay valid as we rewrite.
        for span in sorted(resolved, key=lambda s: s.start, reverse=True):
            original = text[span.start : span.end]
            seen = index.setdefault(span.kind, {})
            key = original.lower()
            if key not in seen:
                seen[key] = len(seen) + 1
            token = self._surrogate(span.kind, seen[key])
            result = result[: span.start] + token + result[span.end :]
            counts[span.kind] += 1
        return result, counts

    # -- records ----------------------------------------------------------
    def _known_values(self, record: Mapping[str, Any]) -> list[str]:
        """Gazetteer terms taken from this record's own identifier fields.

        Also splits multi-word values, so an engaging officer recorded as
        "Bera Kirkwell" is found in free text as "Kirkwell" alone.
        """
        terms: list[str] = []
        for group in self._name_groups:
            parts = [str(record[fid]).strip() for fid in group if record.get(fid)]
            if len(parts) > 1:
                terms.append(" ".join(parts))
        for f in self.schema.pii_fields:
            if f.pii_class is PIIClass.QUASI_IDENTIFIER:
                continue
            value = record.get(f.id)
            if not isinstance(value, str) or not value.strip():
                continue
            terms.append(value.strip())
            terms.extend(part for part in value.split() if len(part) > 2)
        # Longest first so "Alis Ashwood" is matched before "Ashwood".
        return sorted(set(terms), key=len, reverse=True)

    def deidentify(
        self, record: Mapping[str, Any]
    ) -> tuple[dict[str, Any], DeidentificationReport]:
        """De-identify one ESQ record.

        Args:
            record: Field id -> value. Unknown keys are passed through
                untouched, which keeps generator bookkeeping (``record_id``,
                ``archetype``) intact.

        Returns:
            A new record dict and a report of what was removed.

        Raises:
            Exception: Re-raised from a failing scrub when ``on_error`` is
                ``"raise"`` (the default). Failing closed is the right posture:
                a record we could not scrub must not flow onward.
        """
        report = DeidentificationReport(record_id=str(record.get("record_id", "unknown")))
        if not self.enabled:
            self.log.warning("deidentify.disabled", meta={"record_id": report.record_id})
            return dict(record), report

        known = self._known_values(record)
        surrogate_index: dict[str, dict[str, int]] = {}
        out: dict[str, Any] = dict(record)

        # 1. Free text first, while the identifier fields still hold the values
        #    the gazetteer needs.
        for f in self.schema.of_type(FieldType.FREE_TEXT):
            value = record.get(f.id)
            if not isinstance(value, str) or not value:
                continue
            try:
                scrubbed, counts = self.scrub_text(value, known, surrogate_index)
            except Exception:
                report.errors.append(f.id)
                if self.on_error == "raise":
                    raise
                continue
            out[f.id] = scrubbed
            if counts:
                report.entity_counts.update(counts)
                report.fields_with_entities.append(f.id)

        # 2. Field-level policy.
        for f in self.schema.fields.values():
            if f.id not in out:
                continue
            action = self.field_policy.get(f.pii_class.value, "keep")
            report.field_actions[f.id] = action
            value = out[f.id]
            if value in (None, "") or action == "keep":
                continue
            if action == "replace":
                out[f.id] = f"[REDACTED:{f.id}]"
            elif action == "pseudonymise":
                out[f.id] = self._pseudonym(str(value))
            elif action == "mask":
                out[f.id] = f"[MASKED:{f.id}]"

        self.log.event("deidentify.completed", meta=report.as_log_meta())
        return out, report

    def deidentify_many(
        self, records: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[DeidentificationReport]]:
        """De-identify a sequence of records.

        Args:
            records: Records to process.

        Returns:
            Scrubbed records and their reports, in the same order.
        """
        cleaned: list[dict[str, Any]] = []
        reports: list[DeidentificationReport] = []
        for record in records:
            out, report = self.deidentify(record)
            cleaned.append(out)
            reports.append(report)
        self.log.event(
            "deidentify.batch_completed",
            meta={
                "n_records": len(records),
                "total_entities": sum(r.total_entities for r in reports),
                "records_with_entities": sum(1 for r in reports if r.total_entities),
                "records_with_errors": sum(1 for r in reports if r.errors),
            },
        )
        return cleaned, reports
