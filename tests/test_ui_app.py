"""Phase 5 gate: "decision capture tested."

``streamlit.testing.v1.AppTest`` runs the real ``src/ui/app.py`` script — the
same code path ``streamlit run`` uses — without a browser, so these tests
catch the class of bug a browser screenshot cannot be relied on to catch
either: an uncached config re-read blowing up on the second rerun, a widget
key collision between two flags, a decision silently not reaching the log.

What this file does NOT replace: an actual keyboard-only walkthrough and a
contrast check against rendered CSS, both real WCAG 2.2 AA gate items, done
by hand and recorded in ``docs/reports/phase5_interface_report.md``. AppTest
has no concept of a browser, focus order or colour, so it cannot see those
failure modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(REPO_ROOT / "src" / "ui" / "app.py")
DECISION_LOG_PATH = REPO_ROOT / "data" / "reviewer_decisions" / "decisions.jsonl"


@pytest.fixture(autouse=True)
def clean_decision_log():
    """The app writes to the real config-declared path (there is no test
    override wired up for it — see the module docstring on why that is an
    acceptable trade-off here). Keep the working tree clean around each test
    rather than leaving decisions from a test run sitting in the repo."""
    if DECISION_LOG_PATH.exists():
        DECISION_LOG_PATH.unlink()
    yield
    if DECISION_LOG_PATH.exists():
        DECISION_LOG_PATH.unlink()


def _load_decisions() -> list[dict]:
    if not DECISION_LOG_PATH.exists():
        return []
    return [json.loads(line) for line in DECISION_LOG_PATH.read_text().splitlines() if line.strip()]


def test_app_loads_with_no_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    assert not at.exception
    # The scope banner must be present on first load, before any submission
    # is picked — a reviewer must see the Art.22 framing before anything else.
    assert any("not a determination" in md.value for md in at.markdown)


def test_scope_banner_never_mentions_a_status():
    """Belt and braces alongside test_flags_never_carry_a_status_determination
    in test_review.py: the framing text itself must not slip into naming a
    status either."""
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    banner = [md.value for md in at.markdown if "not a determination" in md.value][0]
    lowered = banner.lower()
    assert "is inside ir35" not in lowered
    assert "is outside ir35" not in lowered


def test_loading_a_sample_produces_flags_and_no_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.text_input[0].set_value("j.reviewer")
    at.button(key="load_sample").click()
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Accept — needs follow-up" in labels
    assert "Dismiss — not a concern" in labels


def test_decision_buttons_disabled_without_a_reviewer_id():
    """A decision with no reviewer id would be useless for the audit trail
    the decision log exists to provide, so the control is disabled until a
    name is entered — checked here as a fact about the rendered button, not
    just documented as an intention."""
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.button(key="load_sample").click()  # no reviewer id set
    at.run()
    accept = [b for b in at.button if b.label == "Accept — needs follow-up"][0]
    assert accept.disabled is True


def test_accept_click_writes_a_decision_and_shows_it_back():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.text_input[0].set_value("j.reviewer")
    at.button(key="load_sample").click()
    at.run()

    accept = [b for b in at.button if b.label == "Accept — needs follow-up"][0]
    accept.click()
    at.run()
    assert not at.exception

    decisions = _load_decisions()
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "accept"
    assert decisions[0]["reviewer_id"] == "j.reviewer"
    # No reason was typed. Constraint-4-style default: content redacted.
    assert "reason" not in decisions[0]
    assert decisions[0]["reason_digest"] == {"sha256": None, "chars": 0}

    # The app re-renders the decision inline rather than re-showing controls.
    assert any("Accepted by j.reviewer" in md.value for md in at.markdown)
    assert "Change this decision" in [b.label for b in at.button]


def test_dismiss_with_a_reason_is_redacted_by_default():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.text_input[0].set_value("j.reviewer")
    at.button(key="load_sample").click()
    at.run()

    reason_inputs = [ti for ti in at.text_input if ti.label.startswith("Optional note")]
    assert reason_inputs, "expected a reason field next to the decision buttons"
    reason_inputs[0].set_value("Confirmed with the department by phone.")

    dismiss = [b for b in at.button if b.label == "Dismiss — not a concern"][0]
    dismiss.click()
    at.run()
    assert not at.exception

    decisions = _load_decisions()
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "dismiss"
    assert "reason" not in decisions[0]
    assert decisions[0]["reason_digest"]["chars"] == len("Confirmed with the department by phone.")
    assert "Confirmed" not in json.dumps(decisions[0])


def test_switching_detector_method_rescoring_does_not_crash():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.text_input[0].set_value("j.reviewer")
    at.button(key="load_sample").click()
    at.run()

    at.radio(key="detector_method").set_value("rules_phrase_deleaked")
    at.run()
    assert not at.exception


def test_pasting_invalid_json_shows_an_error_not_a_crash():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.tabs[1].text_area[0].set_value("{not valid json")
    at.button(key="load_pasted").click()
    at.run()
    assert not at.exception
    assert any("Could not read that as a submission" in e.value for e in at.error)
