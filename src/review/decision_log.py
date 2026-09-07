"""Reviewer decision capture.

Phase 5's gate requires accept/dismiss decisions to be captured with reviewer
id, timestamp, flag id and an optional reason. Two things make this more than
a form-to-file dump:

1. **The decision is on the flag, not on the record's status.** There is no
   field anywhere in this module for an employment-status outcome. A reviewer
   accepts or dismisses *this specific inconsistency*; the questionnaire's
   status is decided by the IR35 team through their own process, untouched by
   this tool. That is constraint 6 (UK GDPR Art.22 — no automated decision)
   made concrete in the data model, not just in the UI copy.
2. **The reason is content, not metadata**, under the same logic as
   constraint 4 (``config/pipeline.yaml``: ``logging.log_content``). A
   reviewer's free-text reason can describe a real person's circumstances
   just as easily as a manager's justification box can. ``log_reason_content``
   defaults to ``False``, so only a digest of the reason is retained unless a
   deployment has explicitly decided otherwise — the same digest scheme as
   ``src.utils.logging.content_digest``, for the same reasons.

WHY A LOG AND NOT A DATABASE
A flat, append-only JSONL file is the right amount of engineering for a
synthetic-data prototype: it is trivially inspectable, diffable, and needs no
running service. ``docs/phase_plan.md`` Phase 6 notes that a real deployment
needs a proper retention policy and almost certainly a database with access
controls — this file format is explicitly a placeholder for that, not a
proposal for production.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.utils.logging import content_digest

__all__ = ["ReviewDecision", "DecisionLog"]

Decision = Literal["accept", "dismiss"]


@dataclass(frozen=True)
class ReviewDecision:
    """One reviewer action on one flag.

    Attributes:
        decision_id: Stable id for this decision record.
        record_id: The submission the flag belongs to.
        pair_id: Which contradiction pair the flag was raised on.
        field_id: The rationale field the reviewer looked at.
        method: Which detector raised the flag.
        score: The raw score at the time of decision, for audit.
        band_label: The confidence band shown to the reviewer at the time.
        decision: ``"accept"`` (this is a genuine inconsistency worth
            following up) or ``"dismiss"`` (it is not). Never a status.
        reviewer_id: Free-typed identifier for who made the call. Treated as
            an identifier, not content — logged in full, same as an author
            field, per constraint 4's identifier/content distinction.
        reason: Optional free-text note. Content — see module docstring.
        timestamp_utc: When the decision was made.
    """

    decision_id: str
    record_id: str
    pair_id: str
    field_id: str
    method: str
    score: float
    band_label: str | None
    decision: Decision
    reviewer_id: str
    reason: str | None
    timestamp_utc: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_stored_dict(self, *, log_reason_content: bool, reason_max_chars: int) -> dict[str, Any]:
        """Serialise for the decision log, applying the reason-content policy.

        Args:
            log_reason_content: When ``False`` (the default), ``reason`` is
                replaced with a digest summary rather than stored verbatim.
            reason_max_chars: Truncation applied when content logging is on.

        Returns:
            A JSON-serialisable mapping.
        """
        out = asdict(self)
        reason = out.pop("reason")
        if log_reason_content:
            text = "" if reason is None else str(reason)
            out["reason"] = text[:reason_max_chars] + (
                "…" if len(text) > reason_max_chars else ""
            )
        else:
            out["reason_digest"] = content_digest(reason)
        out["reason_logged"] = log_reason_content
        return out


class DecisionLog:
    """Append-only store of reviewer decisions.

    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "decisions.jsonl"
        >>> log = DecisionLog(path)
        >>> decision = ReviewDecision(
        ...     decision_id="d1", record_id="SYN-0001", pair_id="p_4_05",
        ...     field_id="q4_05_rationale", method="rules_as_authored",
        ...     score=0.42, band_label="Priority", decision="accept",
        ...     reviewer_id="j.reviewer", reason="Matches the phone call notes.",
        ... )
        >>> log.record(decision)
        >>> len(log.load())
        1
    """

    def __init__(
        self,
        path: Path | str,
        *,
        log_reason_content: bool = False,
        reason_max_chars: int = 500,
    ) -> None:
        """Create a decision log.

        Args:
            path: JSONL file to append to. Parent directories are created.
            log_reason_content: See :meth:`ReviewDecision.as_stored_dict`.
            reason_max_chars: See :meth:`ReviewDecision.as_stored_dict`.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_reason_content = log_reason_content
        self.reason_max_chars = reason_max_chars

    def record(self, decision: ReviewDecision) -> dict[str, Any]:
        """Append one decision.

        Args:
            decision: The decision to store.

        Returns:
            The dict actually written, after the reason-content policy is
            applied — useful for tests and for echoing back to the UI.
        """
        row = decision.as_stored_dict(
            log_reason_content=self.log_reason_content,
            reason_max_chars=self.reason_max_chars,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=False) + "\n")
        return row

    def load(self) -> list[dict[str, Any]]:
        """Read every decision recorded so far, oldest first.

        Returns:
            The stored rows. An empty list if nothing has been recorded yet.
        """
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def for_record(self, record_id: str) -> list[dict[str, Any]]:
        """Decisions already made for one submission, oldest first.

        Used by the app to show a reviewer which flags on this record they
        have already actioned, so re-opening a submission does not ask them
        to decide the same flag twice without telling them they already did.
        """
        return [row for row in self.load() if row.get("record_id") == record_id]
