"""Tests for scripts/convert_esq_msforms_xlsx_to_json.py — the converter for
the Microsoft Forms ESQ export HR moved to in Nov 2026 (see the "LIVE FORM
MIGRATION NOTE" in config/schema.yaml).

These build small synthetic workbooks that mimic the real export's shape
(one header row, one row per response, one column per question) rather than
depending on a real submission file, which is never committed to the repo.
They check the specific quirks that shape has: branching questions that
split one rationale box into several option-specific columns with at most
one populated, telephone/company-number columns MS Forms coerces to a
number even though they are identifiers, wording drift between the old
template's value_domain strings and the live form's, and multiple responses
in a single export.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import openpyxl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import convert_esq_msforms_xlsx_to_json as conv  # noqa: E402

from src.ingest.schema_loader import load_schema  # noqa: E402

SHEET_NAME = conv.SHEET_NAME


def _sheet_with_rows(headers: list[str], rows: list[list]):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET_NAME
    for c, h in enumerate(headers, start=1):
        ws.cell(row=1, column=c, value=h)
    for r, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            ws.cell(row=r, column=c, value=v)
    return wb, ws


def test_map_headers_matches_known_questions_and_ignores_forms_metadata():
    headers = [
        "Id", "Start time", "Completion time", "Email", "Name",
        "1.1 Title", "1.2 First Name",
    ]
    col_to_field, unmapped = conv.map_headers(headers)
    assert col_to_field[5] == "q1_01_title"
    assert col_to_field[6] == "q1_02_first_name"
    # the five Forms-added identity/metadata columns are recognised and
    # silently dropped, not reported as unmapped
    assert unmapped == []
    assert 0 not in col_to_field and 1 not in col_to_field


def test_map_headers_reports_a_genuinely_unknown_column():
    col_to_field, unmapped = conv.map_headers(["1.1 Title", "Some new question nobody has seen before"])
    assert col_to_field == {0: "q1_01_title"}
    assert unmapped == ["Some new question nobody has seen before"]


def test_map_headers_normalises_nbsp_and_whitespace():
    """The real export uses \\xa0 (non-breaking space) instead of a plain
    space in several headers. Matching must not be fooled by that."""
    col_to_field, unmapped = conv.map_headers(["4.3\xa0Would the worker's business send someone else (a substitute)..."])
    assert col_to_field == {0: "q4_03_substitute_sent"}
    assert unmapped == []


def test_map_headers_merges_branching_rationale_columns_to_one_field_id():
    headers = [
        "4.3 Would the worker's business send someone else (a substitute)...",
        "Please explain the reason/rationale for your answer 'Yes - you accepted them' in 4.3",
        "Please explain the reason/rationale for your answer 'Yes - you did not accept them' in 4.3",
        "Please explain the reason/rationale for your answer 'No - it has not happened' in 4.3",
    ]
    col_to_field, unmapped = conv.map_headers(headers)
    assert col_to_field == {0: "q4_03_substitute_sent", 1: "q4_03_rationale", 2: "q4_03_rationale", 3: "q4_03_rationale"}
    assert unmapped == []


def test_extract_responses_picks_the_one_populated_branch_column(tmp_path):
    headers = [
        "Id",
        "4.3 Would the worker's business send someone else (a substitute)...",
        "Please explain the reason/rationale for your answer 'Yes - you accepted them' in 4.3",
        "Please explain the reason/rationale for your answer 'Yes - you did not accept them' in 4.3",
        "Please explain the reason/rationale for your answer 'No - it has not happened' in 4.3",
    ]
    rows = [[1, "Yes - you did not accept them", None, "we declined the offer", None]]
    wb, ws = _sheet_with_rows(headers, rows)
    path = tmp_path / "msforms_branch_test.xlsx"
    wb.save(path)
    records, ids, unmapped = conv.extract_responses(path)
    assert unmapped == []
    assert records == [{
        "q4_03_substitute_sent": "Yes - you did not accept them",
        "q4_03_rationale": "we declined the offer",
    }]
    assert ids == {0: 1}


def test_normalize_domain_value_tolerates_whitespace_and_trailing_period():
    schema = load_schema()
    domain = schema["q4_17_put_right"].value_domain
    value, note = conv.normalize_domain_value(
        "No -  the work is time-specific or for a single event.", list(domain)
    )
    assert value == "No - the work is time-specific or for a single event"
    assert note is not None


def test_normalize_domain_value_leaves_an_unrecognisable_value_untouched():
    """No silent guessing: a value with no close match is passed through
    unchanged so RecordValidator can flag it for a human."""
    value, note = conv.normalize_domain_value("Not even close to an option", ["Yes", "No"])
    assert value == "Not even close to an option"
    assert note is None


def test_normalize_domain_value_is_a_noop_for_an_exact_match():
    value, note = conv.normalize_domain_value("Yes", ["Yes", "No"])
    assert value == "Yes"
    assert note is None


def test_coerce_types_casts_phone_and_company_number_to_strings_not_floats():
    schema = load_schema()
    record = {
        "q1_05_telephone": 7890123456,
        "q5_04_phone": 367892,
        "q2_03_company_number": 12151323.0,
        "q1_07_engagement_from": datetime.datetime(2026, 10, 5),
    }
    out, notes = conv.coerce_types(record, schema)
    assert out["q1_05_telephone"] == "7890123456"
    assert out["q5_04_phone"] == "367892"
    assert out["q2_03_company_number"] == "12151323"
    assert out["q1_07_engagement_from"] == "2026-10-05"
    assert notes == []


def test_convert_all_derives_record_id_from_the_ms_forms_id_column_not_pii(tmp_path):
    headers = ["Id", "Email", "Name", "1.1 Title"]
    rows = [[42, "manager@sussex.ac.uk", "A Manager", "Mr"]]
    wb, ws = _sheet_with_rows(headers, rows)
    path = tmp_path / "msforms_id_test.xlsx"
    wb.save(path)

    converted = conv.convert_all(path)
    assert len(converted) == 1
    record, notes = converted[0]
    assert record["record_id"] == "ESQ-MSFORMS-42"
    assert "manager@sussex.ac.uk" not in record.values()
    assert "A Manager" not in record.values()


def test_convert_all_handles_multiple_responses_in_one_export(tmp_path):
    headers = ["Id", "1.1 Title", "1.2 First Name"]
    rows = [
        [1, "Mr", "Isaac"],
        [2, "Ms", "Jo"],
    ]
    wb, ws = _sheet_with_rows(headers, rows)
    path = tmp_path / "msforms_multi_test.xlsx"
    wb.save(path)

    converted = conv.convert_all(path)
    assert [r["record_id"] for r, _ in converted] == ["ESQ-MSFORMS-1", "ESQ-MSFORMS-2"]
    assert converted[0][0]["q1_02_first_name"] == "Isaac"
    assert converted[1][0]["q1_02_first_name"] == "Jo"


def test_retired_fields_are_never_populated_because_no_column_maps_to_them():
    """q2_04_entity_relationship, q2_05_years_in_business, q3_01_engaged_directly
    and q5_02_engaging_officer exist in schema.yaml only for the synthetic
    generator's sake and have no column on the live form -- confirm nothing
    in QUESTION_FIELDS maps to them, so a real export can never populate
    them by accident."""
    retired = {
        "q2_04_entity_relationship", "q2_05_years_in_business",
        "q3_01_engaged_directly", "q5_02_engaging_officer",
    }
    mapped_targets = {field_id for field_id, _ in conv.QUESTION_FIELDS}
    mapped_targets |= set(conv.SPECIAL_RATIONALE_FIELDS.values())
    mapped_targets |= set(conv.SLOT_TO_RATIONALE_FIELD.values())
    assert not (retired & mapped_targets)


def test_an_all_blank_trailing_row_is_skipped(tmp_path):
    headers = ["Id", "1.1 Title"]
    rows = [[1, "Mr"], [None, None]]
    wb, ws = _sheet_with_rows(headers, rows)
    path = tmp_path / "msforms_blank_row_test.xlsx"
    wb.save(path)

    converted = conv.convert_all(path)
    assert len(converted) == 1
