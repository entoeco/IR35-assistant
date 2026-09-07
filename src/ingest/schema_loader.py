"""Typed access to the ESQ data contract.

Constraint 5 of the build brief: **no hardcoded column names outside the
config.** Everything downstream — the generator, the de-identifier, the rule
engine, the baselines, the NLI framing and the UI — reaches the form's
structure through this module, and this module reads nothing but
``config/schema.yaml`` and ``config/contradiction_pairs.yaml``.

The practical test of that constraint is: pointing the system at a revised ESQ
should be a config edit. If a field name appears in a ``.py`` file, that has
failed. ``tests/test_no_hardcoded_fields.py`` enforces it mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

__all__ = ["FieldType", "PIIClass", "Field", "Pair", "Schema", "load_schema"]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class FieldType(str, Enum):
    """How a field behaves, not where it sits on the form."""

    STRUCTURED = "structured"
    FREE_TEXT = "free_text"
    METADATA = "metadata"


class PIIClass(str, Enum):
    """Drives de-identification. See ``src/deidentify``."""

    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    SENSITIVE_COMMERCIAL = "sensitive_commercial"
    NONE = "none"


@dataclass(frozen=True)
class Field:
    """One addressable field on the ESQ.

    Attributes:
        id: Stable key. The contract. Never derived from cell position.
        form_ref: Question number as printed, e.g. ``"4.5"``. Provenance only.
        label: The question text as it appears on the form.
        type: Structured, free text or metadata.
        pii_class: De-identification treatment.
        value_domain: Closed option list for structured fields, else ``None``.
        ir35_test: Which employment-status test this field speaks to.
        justification_for: For a rationale box, the id of the structured field
            it justifies.
        routing: Where the form sends the reader after each option value.
        role: ``"outcome"`` marks the IR35 team's determination block, which is
            stripped before any model sees a record.
        cell: Position in the June 2026 template. Documentation and a hint for
            the ``.xlsx`` adapter; not a stable contract.
        raw: The underlying mapping, for keys this dataclass does not model.
    """

    id: str
    form_ref: str | None
    label: str | None
    type: FieldType
    pii_class: PIIClass
    value_domain: tuple[str, ...] | None
    ir35_test: str | None
    justification_for: str | None
    routing: Mapping[str, Any] | None
    role: str | None
    cell: str | None
    raw: Mapping[str, Any]

    @property
    def is_outcome(self) -> bool:
        """True if this field records the determination rather than an input."""
        return self.role == "outcome"

    @property
    def needs_scrubbing(self) -> bool:
        """True if the field carries identifying or commercially sensitive data."""
        return self.pii_class is not PIIClass.NONE

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Field":
        """Build a Field from one entry of ``schema.yaml``'s ``fields`` list."""
        domain = raw.get("value_domain")
        return cls(
            id=raw["id"],
            form_ref=raw.get("form_ref"),
            label=raw.get("label"),
            type=FieldType(raw["type"]),
            pii_class=PIIClass(raw.get("pii_class", "none")),
            value_domain=tuple(domain) if domain else None,
            ir35_test=raw.get("ir35_test"),
            justification_for=raw.get("justification_for"),
            routing=raw.get("routing"),
            role=raw.get("role"),
            cell=raw.get("cell"),
            raw=raw,
        )


@dataclass(frozen=True)
class Pair:
    """A (structured answer, free-text justification) pair from one IR35 test.

    This is the detector's unit of work. Both halves must address the same test
    or the pair is meaningless — enforced in ``tests/test_schema_contract.py``.

    Attributes:
        id: Stable pair key, e.g. ``"p_4_05_right_to_reject"``.
        structured: Field id of the dropdown.
        free_text: Field id of the adjacent rationale box.
        ir35_test: The shared test.
        outside_leaning: The option value that points away from employment.
            Contradictions that undermine this value are the audit risk.
        priority: ``"high"`` for the pairs reported separately in Phase 4.
        contradiction: Plain-English description of what a contradiction is here.
        plantable_types: Contradiction types the generator may inject.
        cues: Lexical cues for the Phase 2 rule baseline.
    """

    id: str
    structured: str
    free_text: str
    ir35_test: str
    outside_leaning: str
    priority: str | None
    contradiction: str
    plantable_types: tuple[str, ...]
    cues: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def is_high_priority(self) -> bool:
        """True for the pairs whose contradictions most directly flip a status."""
        return self.priority == "high"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Pair":
        """Build a Pair from one entry of ``contradiction_pairs.yaml``."""
        return cls(
            id=raw["id"],
            structured=raw["structured"],
            free_text=raw["free_text"],
            ir35_test=raw["ir35_test"],
            outside_leaning=raw["outside_leaning"],
            priority=raw.get("priority"),
            contradiction=raw.get("contradiction", "").strip(),
            plantable_types=tuple(raw.get("plantable_types", ())),
            cues=raw.get("cues", {}),
            raw=raw,
        )


class Schema:
    """The loaded data contract: fields, pairs and their relationships."""

    def __init__(
        self,
        schema_doc: Mapping[str, Any],
        pairs_doc: Mapping[str, Any],
    ) -> None:
        """Build a Schema from two already-parsed YAML documents.

        Args:
            schema_doc: Parsed ``schema.yaml``.
            pairs_doc: Parsed ``contradiction_pairs.yaml``.
        """
        self._schema_doc = schema_doc
        self._pairs_doc = pairs_doc
        self.fields: dict[str, Field] = {
            f["id"]: Field.from_mapping(f) for f in schema_doc["fields"]
        }
        self.pairs: dict[str, Pair] = {
            p["id"]: Pair.from_mapping(p) for p in pairs_doc["pairs"]
        }
        self.version: str = schema_doc.get("schema_version", "unknown")
        self.form_version: str = schema_doc.get("form_version", "unknown")

    # -- lookups ----------------------------------------------------------
    def __getitem__(self, field_id: str) -> Field:
        return self.fields[field_id]

    def __contains__(self, field_id: object) -> bool:
        return field_id in self.fields

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields.values())

    def of_type(self, field_type: FieldType) -> list[Field]:
        """All fields of a given type, in template order."""
        return [f for f in self.fields.values() if f.type is field_type]

    def for_test(self, ir35_test: str) -> list[Field]:
        """All fields speaking to one IR35 test."""
        return [f for f in self.fields.values() if f.ir35_test == ir35_test]

    def pairs_for_test(self, ir35_test: str) -> list[Pair]:
        """All contradiction pairs within one IR35 test."""
        return [p for p in self.pairs.values() if p.ir35_test == ir35_test]

    # -- derived views ----------------------------------------------------
    @cached_property
    def input_fields(self) -> list[Field]:
        """Every field a model may see: everything except the outcome block.

        This is the leakage guard. Ingest strips by this list, so widening the
        outcome block in config removes fields from model view automatically.
        """
        return [f for f in self.fields.values() if not f.is_outcome]

    @cached_property
    def outcome_fields(self) -> list[Field]:
        """The IR35 team's determination block. Never visible at inference."""
        return [f for f in self.fields.values() if f.is_outcome]

    @cached_property
    def structured_answer_fields(self) -> list[Field]:
        """Structured, non-outcome fields with a closed value domain."""
        return [
            f
            for f in self.fields.values()
            if f.type is FieldType.STRUCTURED and not f.is_outcome and f.value_domain
        ]

    @cached_property
    def justification_map(self) -> dict[str, str]:
        """Structured field id -> its rationale field id."""
        return {
            f.justification_for: f.id
            for f in self.fields.values()
            if f.justification_for
        }

    @cached_property
    def pii_fields(self) -> list[Field]:
        """Fields the de-identifier treats at field level."""
        return [f for f in self.fields.values() if f.needs_scrubbing]

    @cached_property
    def ir35_tests(self) -> dict[str, Mapping[str, Any]]:
        """The test vocabulary, with labels and interpretation notes."""
        return dict(self._schema_doc.get("ir35_tests", {}))

    @cached_property
    def contradiction_types(self) -> dict[str, str]:
        """Taxonomy used for ground-truth labels and per-type evaluation."""
        return dict(self._schema_doc.get("contradiction_types", {}))

    @cached_property
    def subtlety_levels(self) -> tuple[int, ...]:
        """Permitted planted-contradiction subtlety levels (1 = blatant)."""
        return tuple(self._schema_doc.get("subtlety_levels", (1, 2, 3)))

    @cached_property
    def cross_field_checks(self) -> list[Mapping[str, Any]]:
        """Rule-based checks that are not (premise, hypothesis) pairs."""
        return list(self._pairs_doc.get("cross_field_checks", []))

    @cached_property
    def pair_defaults(self) -> Mapping[str, Any]:
        """Defaults applied across all pairs, e.g. the premise template."""
        return self._pairs_doc.get("defaults", {})

    def premise_for(self, pair: Pair, value: str) -> str:
        """Render a structured answer as an NLI premise.

        The template lives in config so that its wording can be tuned without
        touching model code — the phrasing of the premise measurably affects
        zero-shot NLI performance, and that is a Phase 3 experiment.

        Args:
            pair: The pair being evaluated.
            value: The selected option value.

        Returns:
            A natural-language sentence stating the manager's structured answer.
        """
        template = self.pair_defaults.get(
            "premise_template",
            "The engaging manager stated, in answer to '{label}': {value}",
        )
        return template.format(label=self.fields[pair.structured].label, value=value)

    def routed_fields(self, answers: Mapping[str, Any]) -> set[str]:
        """Field ids that the form's branching logic actually asks for.

        The ESQ skips questions depending on earlier answers, so a blank field
        off the active route is legitimate rather than missing. Anything that
        consumes completeness needs this distinction or it reports noise.

        Args:
            answers: Structured answers keyed by field id.

        Returns:
            The set of structured field ids reachable given those answers.
        """
        reachable: set[str] = set()
        for field in self.structured_answer_fields:
            routing = field.routing or {}
            targets = [v for k, v in routing.items() if k != "note"]
            selected = answers.get(field.id)
            reachable.add(field.id)
            if selected is not None and isinstance(routing.get(selected), str):
                reachable.add(routing[selected])
            elif not targets:
                continue
        return reachable


def load_schema(
    config_dir: Path | str | None = None,
    *,
    schema_file: str = "schema.yaml",
    pairs_file: str = "contradiction_pairs.yaml",
) -> Schema:
    """Load the data contract from disk.

    Args:
        config_dir: Directory holding the config files. Defaults to the repo's
            ``config/``.
        schema_file: Filename of the field schema.
        pairs_file: Filename of the contradiction-pair definitions.

    Returns:
        A ``Schema``.

    Raises:
        FileNotFoundError: If either config file is missing.
    """
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    schema_path = directory / schema_file
    pairs_path = directory / pairs_file
    for path in (schema_path, pairs_path):
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
    schema_doc = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    pairs_doc = yaml.safe_load(pairs_path.read_text(encoding="utf-8"))
    return Schema(schema_doc, pairs_doc)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file into a plain dict.

    Args:
        path: File to read.

    Returns:
        The parsed document, or an empty dict for an empty file.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_pipeline_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load ``config/pipeline.yaml`` (logging and de-identification settings)."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    return load_yaml(directory / "pipeline.yaml")


def load_generation_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Load ``config/generation.yaml`` (the reproducibility contract)."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    return load_yaml(directory / "generation.yaml")
