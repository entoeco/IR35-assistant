"""Constraint 4: content must not reach the log unless explicitly enabled.

These tests exist because the failure they guard against is silent. A logger
that leaks free text does not raise, does not fail a run, and is only noticed
when someone reads the log — by which time the personal data is already at rest.
"""

from __future__ import annotations

import io
import json

import pytest

from src.utils.logging import LoggingConfig, StructuredLogger, content_digest

SENSITIVE = "Dr Alis Ashwood cannot be replaced; nobody else has the expertise."


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


def make_logger(stream: io.StringIO, *, log_content: bool, name: str) -> StructuredLogger:
    return StructuredLogger(name, LoggingConfig(log_content=log_content, stream=stream))


def test_content_is_redacted_by_default(stream: io.StringIO) -> None:
    log = make_logger(stream, log_content=False, name="t_default")
    record = log.event(
        "deidentify.completed",
        meta={"record_id": "SYN-0001"},
        content={"q4_05_rationale": SENSITIVE},
    )
    assert SENSITIVE not in stream.getvalue()
    assert record["content"]["q4_05_rationale"]["chars"] == len(SENSITIVE)
    assert record["content_logged"] is False


def test_default_config_does_not_log_content() -> None:
    """The flag defaults to off with no config file present at all."""
    assert LoggingConfig().log_content is False
    assert LoggingConfig.from_mapping({}).log_content is False
    assert LoggingConfig.from_mapping(None).log_content is False


def test_metadata_is_always_emitted(stream: io.StringIO) -> None:
    log = make_logger(stream, log_content=False, name="t_meta")
    log.event("record.generated", meta={"record_id": "SYN-0007", "archetype": "it_contractor"})
    emitted = json.loads(stream.getvalue().strip())
    assert emitted["record_id"] == "SYN-0007"
    assert emitted["archetype"] == "it_contractor"
    assert emitted["event"] == "record.generated"
    assert "content" not in emitted


def test_content_is_emitted_when_explicitly_enabled(stream: io.StringIO) -> None:
    log = make_logger(stream, log_content=True, name="t_enabled")
    log.event("debug.text", content={"q4_05_rationale": SENSITIVE})
    emitted = json.loads(stream.getvalue().strip())
    assert emitted["content"]["q4_05_rationale"] == SENSITIVE
    assert emitted["content_logged"] is True


def test_content_is_truncated_when_enabled(stream: io.StringIO) -> None:
    log = StructuredLogger(
        "t_trunc", LoggingConfig(log_content=True, content_max_chars=10, stream=stream)
    )
    log.event("debug.text", content={"note": "x" * 100})
    emitted = json.loads(stream.getvalue().strip())
    assert emitted["content"]["note"] == "x" * 10 + "…"


def test_digest_is_stable_and_length_bearing() -> None:
    first = content_digest(SENSITIVE)
    second = content_digest(SENSITIVE)
    assert first == second
    assert first["chars"] == len(SENSITIVE)
    assert len(first["sha256"]) == 12


def test_digest_distinguishes_absent_from_redacted() -> None:
    """A blank justification is a completeness failure; the log must show it."""
    assert content_digest(None) == {"sha256": None, "chars": 0}
    assert content_digest("") == {"sha256": None, "chars": 0}
    assert content_digest("x")["sha256"] is not None


def test_digest_changes_when_content_changes() -> None:
    """Enough to detect that a pipeline stage altered a value."""
    before = content_digest("we would need to approve any substitute")
    after = content_digest("we would need to approve any substitute.")
    assert before["sha256"] != after["sha256"]


def test_every_line_is_valid_json(stream: io.StringIO) -> None:
    log = make_logger(stream, log_content=False, name="t_json")
    log.event("a", meta={"i": 1})
    log.warning("b", meta={"i": 2})
    log.error("c", meta={"i": 3})
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert {"ts", "logger", "event"} <= parsed.keys()
