"""Tick-box vs tick-box consistency checks.

Everything else in ``src/models`` compares a structured answer against the
free text sitting next to it. This module compares structured answers
against *each other* — the same idea the rest of the project applies to text,
turned on the form's own tick-boxes. A manager who ticks "the work has
started" and, two questions later, ticks "not applicable, work has not
started" has contradicted themselves without writing a word of free text, and
that is just as real a "reasonable care" problem as a tick-box that disagrees
with its own justification.

WHERE THE CHECKS COME FROM
``config/contradiction_pairs.yaml``'s ``cross_field_checks`` block, authored
back in Phase 0, described six of these in plain English before any of this
project's detection code existed. Three had a shape simple enough to state as
a small, honest boolean condition — evaluated here. The other three did not,
and are left as documentation of a known gap rather than forced into a rule
that would not really be checking what its description claims — see the
comments in that config file for which is which and why.

THE CONDITION LANGUAGE IS DELIBERATELY SMALL
Three clause types — equals, is-one-of, at-least-one-of-these-is-filled-in —
ANDed together. Not a general expression language: the point is that a
non-programmer reading ``contradiction_pairs.yaml`` can see exactly what each
check does, and that adding a seventh check some day is a config edit, not a
Python change (constraint 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.ingest.schema_loader import Schema

__all__ = ["CrossFieldFinding", "evaluate_cross_field_checks"]


@dataclass(frozen=True)
class CrossFieldFinding:
    """One fired cross-field check, for one record.

    Attributes:
        check_id: The check's id in ``contradiction_pairs.yaml``.
        description: The human-readable description from config.
        severity: ``"low"``, ``"medium"`` or ``"high"``, as authored in
            config — how serious a data-quality problem this looks like, not
            a claim about employment status.
        fields: The field ids the condition actually inspected, in the order
            they were checked — what a reviewer should look at to see the
            clash for themselves.
    """

    check_id: str
    description: str
    severity: str
    fields: tuple[str, ...]


def _clause_holds(clause: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """Evaluate one clause of a ``when_all`` list against a record."""
    if "any_not_empty" in clause:
        return any(
            str(record.get(fid, "") or "").strip() for fid in clause["any_not_empty"]
        )
    field = clause["field"]
    value = record.get(field)
    if "eq" in clause:
        return value == clause["eq"]
    if "in" in clause:
        return value in clause["in"]
    raise ValueError(f"cross-field clause has no recognised operator: {clause!r}")


def _clause_fields(clause: Mapping[str, Any]) -> tuple[str, ...]:
    """Field ids a clause touches, for the finding's ``fields``."""
    if "any_not_empty" in clause:
        return tuple(clause["any_not_empty"])
    return (clause["field"],)


def evaluate_cross_field_checks(
    schema: Schema, record: Mapping[str, Any]
) -> list[CrossFieldFinding]:
    """Run every configured, condition-bearing cross-field check on one record.

    Args:
        schema: The loaded data contract.
        record: One (de-identified, validated) record. Field id -> value.

    Returns:
        One finding per check whose ``when_all`` clauses all held. Checks
        with no ``condition`` in config (see the comments in
        ``contradiction_pairs.yaml``) are silently skipped, not errored on —
        they are documented gaps, not broken checks.
    """
    findings: list[CrossFieldFinding] = []
    for check in schema.cross_field_checks:
        condition = check.get("condition")
        if not condition:
            continue
        clauses: Sequence[Mapping[str, Any]] = condition.get("when_all", [])
        if clauses and all(_clause_holds(clause, record) for clause in clauses):
            fields = tuple(
                dict.fromkeys(fid for clause in clauses for fid in _clause_fields(clause))
            )
            findings.append(
                CrossFieldFinding(
                    check_id=str(check["id"]),
                    description=" ".join(str(check["description"]).split()),
                    severity=str(check.get("severity", "medium")),
                    fields=fields,
                )
            )
    return findings
