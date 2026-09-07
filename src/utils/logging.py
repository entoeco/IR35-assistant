"""Structured logging for the IR35 assistant.

Constraint 4 of the build brief: **log identifiers and metadata, not free-text
content, by default.** Content logging is a config flag defaulting to off.

The design point is that redaction is not something a caller has to remember.
`StructuredLogger.event()` takes two separate keyword groups — `meta` (always
emitted) and `content` (emitted only when `log_content` is enabled) — so the
only way to log a manager's free text is to put it in the `content` group,
where the flag governs it. There is no code path that quietly leaks a
justification into the log because someone interpolated it into a message
string; `event()` takes an event *name*, not a formatted message.

When content logging is off, each content value is still summarised as
`{"sha256": "<12 hex chars>", "chars": <int>}`. That is enough to correlate the
same text across pipeline stages, to prove a value was present, and to detect
that it changed, without retaining the text itself. It is a pseudonym, not
anonymisation — the hash of a short, predictable string is reversible by
brute force — so the digest is treated as personal data in its own right and
the docs say so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

__all__ = ["LoggingConfig", "StructuredLogger", "content_digest", "get_logger"]

_DIGEST_CHARS = 12


def content_digest(value: object) -> dict[str, Any]:
    """Summarise a value without retaining it.

    Args:
        value: Any value. Non-strings are coerced with ``str`` first.

    Returns:
        A mapping with a truncated SHA-256 hex digest and the character length.
        ``None`` and the empty string are reported as ``{"sha256": None,
        "chars": 0}`` so that "absent" is distinguishable from "present but
        redacted" in the log.

    Note:
        The digest is a pseudonym, not anonymisation. Free-text answers drawn
        from a small space (for example "Yes") hash to a predictable value, so
        digests must be handled under the same retention rules as the content.
    """
    if value is None or value == "":
        return {"sha256": None, "chars": 0}
    text = value if isinstance(value, str) else str(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return {"sha256": digest, "chars": len(text)}


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behaviour, normally loaded from ``config/pipeline.yaml``.

    Attributes:
        level: Standard library logging level name.
        log_content: When ``False`` (the default), free-text content passed to
            the ``content`` group is replaced by a digest summary. Setting this
            to ``True`` causes personal data to be written to the log and must
            not be enabled outside a synthetic-data environment.
        content_max_chars: Truncation applied when ``log_content`` is ``True``.
            A guard rail, not a privacy control.
        stream: Where to write. Defaults to stderr so that log output never
            contaminates piped data output.
    """

    level: str = "INFO"
    log_content: bool = False
    content_max_chars: int = 500
    stream: Any = field(default=None, repr=False)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "LoggingConfig":
        """Build a config from the ``logging`` block of a loaded YAML file."""
        cfg = cfg or {}
        return cls(
            level=str(cfg.get("level", "INFO")).upper(),
            log_content=bool(cfg.get("log_content", False)),
            content_max_chars=int(cfg.get("content_max_chars", 500)),
        )


class StructuredLogger:
    """Emits one JSON object per event.

    Example:
        >>> log = StructuredLogger("generate", LoggingConfig())
        >>> log.event(
        ...     "record.generated",
        ...     meta={"record_id": "SYN-0001", "archetype": "it_contractor"},
        ...     content={"q4_05_rationale": "we specifically need this person"},
        ... )  # doctest: +SKIP
        {"ts": "...", "logger": "generate", "event": "record.generated",
         "record_id": "SYN-0001", "archetype": "it_contractor",
         "content": {"q4_05_rationale": {"sha256": "3f2a...", "chars": 31}},
         "content_logged": false}
    """

    def __init__(self, name: str, config: LoggingConfig | None = None) -> None:
        """Create a logger.

        Args:
            name: Logger name, conventionally the pipeline stage.
            config: Logging configuration. Defaults to content logging off.
        """
        self.config = config or LoggingConfig()
        self._log = logging.getLogger(f"ir35.{name}")
        self.name = name
        if not self._log.handlers:
            handler = logging.StreamHandler(self.config.stream or sys.stderr)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(handler)
            self._log.propagate = False
        self._log.setLevel(getattr(logging, self.config.level, logging.INFO))

    def _render_content(self, content: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the content-logging policy to the content group."""
        if not self.config.log_content:
            return {k: content_digest(v) for k, v in content.items()}
        limit = self.config.content_max_chars
        rendered: dict[str, Any] = {}
        for key, value in content.items():
            text = "" if value is None else str(value)
            rendered[key] = text[:limit] + ("…" if len(text) > limit else "")
        return rendered

    def event(
        self,
        event: str,
        *,
        meta: Mapping[str, Any] | None = None,
        content: Mapping[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> dict[str, Any]:
        """Emit one structured event.

        Args:
            event: Dotted event name, e.g. ``"deidentify.completed"``.
            meta: Identifiers and metadata. Always emitted verbatim. Do not put
                free text here — it bypasses the content policy.
            content: Free-text values. Emitted only when ``log_content`` is
                enabled; otherwise replaced by digest summaries.
            level: Standard library logging level for this event.

        Returns:
            The record that was emitted, so tests can assert on it without
            capturing the stream.
        """
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "logger": self.name,
            "event": event,
        }
        record.update(dict(meta or {}))
        if content:
            record["content"] = self._render_content(content)
            record["content_logged"] = self.config.log_content
        self._log.log(level, json.dumps(record, default=str, sort_keys=False))
        return record

    def warning(self, event: str, **kwargs: Any) -> dict[str, Any]:
        """Emit an event at WARNING level."""
        return self.event(event, level=logging.WARNING, **kwargs)

    def error(self, event: str, **kwargs: Any) -> dict[str, Any]:
        """Emit an event at ERROR level."""
        return self.event(event, level=logging.ERROR, **kwargs)


def get_logger(name: str, config: LoggingConfig | None = None) -> StructuredLogger:
    """Convenience constructor mirroring ``logging.getLogger``."""
    return StructuredLogger(name, config)
