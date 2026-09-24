"""Tests for scripts/convert_esq_xlsx_to_json.py — the reusable converter
for turning a completed ESQ Excel submission into the schema-field-id-keyed
JSON the reviewer app expects.

These build a small synthetic workbook that mimics the real template's
label-then-answer layout (rather than depending on a real submission file,
which is never committed to the repo) and check the extraction survives the
same quirks the real template has: a label search that must not be fooled by
its own sub-label column, Excel storing a date as a datetime and an
identifier-like number as a float, and a blank cell staying absent from the
record rather than becoming an empty string.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import openpyxl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import convert_esq_xlsx_to_json as conv  # noqa: E402

SHEET_NAME = conv.SHEET_NAME


def _blank_sheet():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET_NAME
    return wb, ws


def test_find_label_row_and_value_in_row():
    wb, ws = _blank_sheet()
    ws.cell(row=14, column=2, value=1.1)
    ws.cell(row=14, column=3, value="Title")
    ws.cell(row=14, column=5, value="Mr")

    row = conv.find_label_row(ws, "title", (12, 20))
    assert row == 14
    assert conv.value_in_row(ws, row) == "Mr"


def test_value_in_row_skips_routing_and_note_text():
    wb, ws = _blank_sheet()
    ws.cell(row=39, column=3, value="Is the Engagement of work through a third party?")
    ws.cell(row=39, column=9, value="No")
    ws.cell(row=40, column=9, value="If Yes, please complete the following questions")

    row = conv.find_label_row(ws, "third party", (37, 44))
    assert conv.value_in_row(ws, row) == "No"


def test_extract_address_does_not_pick_up_its_own_sub_label():
    """Regression test: an earlier version of this extraction returned the
    literal text "1st line" as the address value, because it started
    scanning at the same column the sub-label itself lives in."""
    wb, ws = _blank_sheet()
    ws.cell(row=20, column=2, value=1.4)
    ws.cell(row=20, column=3, value="Registered Address")
    ws.cell(row=20, column=4, value="1st line")
    ws.cell(row=20, column=5, value="1 Apple Street")
    ws.cell(row=21, column=4, value="2nd line")
    ws.cell(row=21, column=5, value="Big Hill")
    ws.cell(row=22, column=4, value="3rd line")
    ws.cell(row=22, column=5, value="Coventry")
    ws.cell(row=23, column=4, value="Post code")
    ws.cell(row=23, column=5, value="C1 9BB")

    result = conv.extract_address(ws)
    assert result == {
        "q1_04_address_line1": "1 Apple Street",
        "q1_04_address_line2": "Big Hill",
        "q1_04_address_line3": "Coventry",
        "q1_04_postcode": "C1 9BB",
    }


def test_extract_engagement_dates():
    wb, ws = _blank_sheet()
    ws.cell(row=30, column=2, value=1.7)
    ws.cell(row=30, column=3, value="Length of Engagement")
    ws.cell(row=30, column=6, value="From")
    ws.cell(row=30, column=7, value=datetime.datetime(2026, 9, 1))
    ws.cell(row=30, column=9, value="To")
    ws.cell(row=30, column=10, value=datetime.datetime(2027, 5, 1))

    result = conv.extract_engagement_dates(ws)
    assert result["q1_07_engagement_from"] == datetime.datetime(2026, 9, 1)
    assert result["q1_07_engagement_to"] == datetime.datetime(2027, 5, 1)


def test_coerce_types_dates_to_iso_strings_and_strips_whitespace():
    record = {
        "q1_07_engagement_from": datetime.datetime(2026, 9, 1),
        "q4_01_already_started": "No ",  # trailing space, as the real template has
        "q2_03_company_number": 12151323,
    }
    out = conv.coerce_types(record)
    assert out["q1_07_engagement_from"] == "2026-09-01"
    assert out["q4_01_already_started"] == "No"
    assert out["q2_03_company_number"] == "12151323"  # identifier, not a quantity


def test_extract_section4_pairs_answers_with_their_own_rationale():
    """A minimal 2-question Section 4, mirroring the real template's layout:
    question number in column B marks the start of a block, the structured
    answer is the first non-empty cell in the block, and -- when a rationale
    label is present -- the rationale is the *last* non-empty cell, since the
    real template sometimes wraps the rationale text one row above its own
    label."""
    wb, ws = _blank_sheet()
    fields = ["q4_01_already_started", "q4_02_office_holder"]

    ws.cell(row=70, column=2, value=4.01)
    ws.cell(row=70, column=3, value="Has the worker already started?")
    ws.cell(row=70, column=5, value="No")

    ws.cell(row=75, column=2, value=4.02)
    ws.cell(row=75, column=3, value="Will the worker be an office holder?")
    ws.cell(row=75, column=5, value="No")
    ws.cell(row=76, column=5, value="No office to hold for this role")  # answer above its label
    ws.cell(row=77, column=3, value="Please explain the reason/rationale for your answer in 4.2")

    result = conv.extract_section4(ws, row_bounds=(60, 80), fields=fields)
    assert result == {
        "q4_01_already_started": "No",
        "q4_02_office_holder": "No",
        "q4_02_rationale": "No office to hold for this role",
    }


def test_extract_section4_leaves_an_unanswered_question_absent_not_blank():
    wb, ws = _blank_sheet()
    fields = ["q4_01_already_started", "q4_02_office_holder"]

    ws.cell(row=70, column=2, value=4.01)
    ws.cell(row=70, column=3, value="Has the worker already started?")
    ws.cell(row=70, column=5, value="No")

    ws.cell(row=75, column=2, value=4.02)
    ws.cell(row=75, column=3, value="Will the worker be an office holder?")
    # left entirely blank -- e.g. routed around by an earlier answer

    result = conv.extract_section4(ws, row_bounds=(60, 80), fields=fields)
    assert result == {"q4_01_already_started": "No"}
    assert "q4_02_office_holder" not in result


def test_extract_section4_raises_a_clear_error_if_the_row_count_does_not_match():
    wb, ws = _blank_sheet()
    ws.cell(row=70, column=2, value=4.01)
    ws.cell(row=75, column=2, value=4.02)

    with pytest.raises(ValueError, match="Expected 3 Section 4 question rows, found 2"):
        conv.extract_section4(ws, row_bounds=(60, 80), fields=["a", "b", "c"])
