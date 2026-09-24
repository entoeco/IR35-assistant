#!/usr/bin/env python3
"""Convert a completed ESQ (Employment Status Questionnaire) Excel file into
the schema-field-id-keyed JSON the reviewer app's "paste submission" and
"upload a file" tabs expect (see ``config/schema.yaml``).

Usage:

    python scripts/convert_esq_xlsx_to_json.py path/to/completed_esq.xlsx
    python scripts/convert_esq_xlsx_to_json.py path/to/completed_esq.xlsx -o out.json
    python scripts/convert_esq_xlsx_to_json.py path/to/completed_esq.xlsx --record-id MY-ID-01

Why this exists, and what it assumes
-------------------------------------
Every real submission is a filled-in copy of the *same* ESQ Excel template
(sheet name "Employment Status Questionnaire"), so the row a label sits on
does not move between submissions -- only the answer next to it does. This
script finds each answer by searching for its known label text rather than a
fixed row number, so it survives the small formatting differences (extra
blank rows, merged cells) that show up between real copies of the template.
It was built by hand-converting a real test submission once (see
``docs/reports`` for how that first conversion was verified) and generalising
what worked; if a future revision of the template moves a *label's wording*
rather than just its position, the ``LABEL_FIELDS`` table below is the one
place to update.

Two things this script deliberately does NOT do:

- It never reads or writes the four "For IR35 Team use only" fields
  (``ir35_assessment_date``, ``ir35_assessor_name``, ``ir35_hr_authorisation``,
  ``ir35_outcome``). A genuine manager-submitted ESQ would never carry them --
  they are filled in later by the IR35 team -- and ``ir35_outcome``
  specifically would trip the app's own outcome-leakage check.
- It never guesses a value for a blank field. A blank cell in the workbook
  becomes a field that is simply absent from the JSON, not an empty string --
  that is what ``src/ingest/validate.py``'s routing-aware completeness check
  expects, and it is how a legitimately-skipped question (routed around by
  an earlier answer) is told apart from one that should have been answered.

After extraction, the record is validated against the real data contract
(``src.ingest.validate.RecordValidator``, the same class the app uses) and
any findings are printed -- errors (a value outside the declared domain, for
example a stray leading/trailing space that survived) are worth fixing before
using the record; warnings (an unanswered rationale box, a field answered off
the routed path) usually just reflect what the manager actually wrote, and
are printed for a human to look over, not treated as failures.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

import openpyxl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest.schema_loader import load_generation_config, load_schema  # noqa: E402
from src.ingest.validate import RecordValidator, Severity  # noqa: E402

SHEET_NAME = "Employment Status Questionnaire"

# Section 4's 31 structured questions, in form order. Extracted positionally
# (see extract_section4 below) rather than by label, because their rationale
# boxes share near-identical label text ("Please explain the reason/rationale
# for your answer in 4.N") that is easiest to tell apart by row order.
Q4_FIELDS = [
    "q4_01_already_started", "q4_02_office_holder", "q4_03_substitute_sent",
    "q4_04_worker_pays_substitute", "q4_05_right_to_reject", "q4_06_paid_another_person",
    "q4_07_would_pay_substitute", "q4_08_moved_from_task", "q4_09_decide_how",
    "q4_10_decide_schedule", "q4_11_decide_where", "q4_12_buys_equipment",
    "q4_13_vehicle_costs", "q4_14_materials_unreimbursed", "q4_15_other_costs",
    "q4_16_payment_basis", "q4_17_put_right", "q4_18_benefits",
    "q4_19_manages_staff", "q4_20_identifies_as", "q4_21_exclusivity_clause",
    "q4_22_permission_required", "q4_23_ownership_rights", "q4_24_rights_to_university",
    "q4_25_option_to_buy_rights", "q4_26_previous_contract", "q4_27_first_in_series",
    "q4_28_starts_immediately_after", "q4_29_extendable", "q4_30_majority_of_time",
    "q4_31_other_clients",
]

# (field_id, label substring to search for in column C, (row_start, row_end)
# to search within). Row ranges are generous bands around where each section
# lives in the template, not exact rows -- see the module docstring.
LABEL_FIELDS: list[tuple[str, str, tuple[int, int]]] = [
    ("q1_01_title", "title", (12, 20)),
    ("q1_02_first_name", "first name", (12, 20)),
    ("q1_03_surname", "surname", (12, 20)),
    ("q1_05_telephone", "contact telephone number", (20, 30)),
    ("q1_06_internet_presence", "internet presence", (20, 32)),
    ("q1_08_contract_in_place", "is there, or will there be, a contract in place", (30, 37)),
    ("q1_08_contract_form", "please give details of what form the contract takes", (30, 37)),
    ("q2_01_via_third_party", "is the engagement of work through a third party", (37, 44)),
    ("q2_02_company_name", "company name", (37, 55)),
    ("q2_03_company_number", "reg company number", (37, 55)),
    ("q2_04_entity_relationship", "what is the relationship between the contracting entity", (37, 55)),
    ("q2_05_years_in_business", "how long have they been in business", (37, 55)),
    ("q3_01_engaged_directly", "is the worker being engaged directly", (55, 60)),
    ("q4_00_role_title", "contract or role title", (60, 65)),
    ("q4_00_duties_narrative", "description of duties of engagement", (60, 70)),
    ("q5_01_school_division", "school/division", (290, 330)),
    ("q5_02_engaging_officer", "name of the engaging officer", (290, 330)),
    ("q5_03_contact_name", "contact name (if different)", (290, 330)),
    ("q5_04_phone", "phone number", (290, 330)),
    ("q5_05_email", "email address", (290, 330)),
]


def _is_routing_or_note(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("if you answered") or t.startswith("go to") or t.startswith("(")


def find_label_row(ws, label_substring: str, row_range: tuple[int, int], col: int = 3) -> int | None:
    lowered = label_substring.lower()
    for r in range(*row_range):
        v = ws.cell(row=r, column=col).value
        if isinstance(v, str) and lowered in v.lower():
            return r
    return None


def value_in_row(ws, row: int, col_start: int = 4, col_end: int = 13) -> Any:
    for col in range(col_start, col_end):
        v = ws.cell(row=row, column=col).value
        if v not in (None, "") and not (isinstance(v, str) and _is_routing_or_note(v)):
            return v
    return None


def extract_labelled_fields(ws) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_id, label, row_range in LABEL_FIELDS:
        row = find_label_row(ws, label, row_range)
        if row is None:
            continue
        value = value_in_row(ws, row)
        if value is not None:
            out[field_id] = value
    return out


def extract_address(ws, row_range: tuple[int, int] = (12, 30)) -> dict[str, Any]:
    """The address is four sub-rows under one "Registered Address" label,
    each with its own sub-label ("1st line", "2nd line", "3rd line", "Post
    code") in column D rather than column C."""
    anchor = find_label_row(ws, "registered address", row_range)
    if anchor is None:
        return {}
    sub_labels = {
        "1st line": "q1_04_address_line1",
        "2nd line": "q1_04_address_line2",
        "3rd line": "q1_04_address_line3",
        "post code": "q1_04_postcode",
    }
    out: dict[str, Any] = {}
    for r in range(anchor, anchor + 8):
        d_val = ws.cell(row=r, column=4).value
        if not isinstance(d_val, str):
            continue
        for sub_label, field_id in sub_labels.items():
            if sub_label in d_val.lower():
                # Column D on this row is the sub-label itself ("1st line",
                # "Post code", ...) -- start the scan one column later so
                # that label text is never mistaken for the answer.
                value = value_in_row(ws, r, col_start=5)
                if value is not None:
                    out[field_id] = value
    return out


def extract_engagement_dates(ws, row_range: tuple[int, int] = (25, 37)) -> dict[str, Any]:
    row = find_label_row(ws, "length of engagement", row_range)
    if row is None:
        return {}
    out: dict[str, Any] = {}
    cells = [(c, ws.cell(row=row, column=c).value) for c in range(4, 13)]
    for i, (col, val) in enumerate(cells):
        if isinstance(val, str) and val.strip().lower() == "from" and i + 1 < len(cells):
            out["q1_07_engagement_from"] = cells[i + 1][1]
        if isinstance(val, str) and val.strip().lower() == "to" and i + 1 < len(cells):
            out["q1_07_engagement_to"] = cells[i + 1][1]
    return out


def extract_section4(
    ws, row_bounds: tuple[int, int] = (60, 290), fields: list[str] = Q4_FIELDS
) -> dict[str, Any]:
    start, end = row_bounds
    qrows = [r for r in range(start, end) if isinstance(ws.cell(row=r, column=2).value, (int, float))]
    if len(qrows) != len(fields):
        raise ValueError(
            f"Expected {len(fields)} Section 4 question rows, found {len(qrows)}. "
            "The template layout may have changed -- check row_bounds and Q4_FIELDS."
        )

    def rationale_label_row_in(row_start: int, row_end: int) -> int | None:
        for r in range(row_start, row_end):
            c_val = ws.cell(row=r, column=3).value
            if isinstance(c_val, str) and c_val.strip().lower().startswith("please explain the reason/rationale"):
                return r
        return None

    def all_candidates(row_start: int, row_end: int) -> list[tuple[int, int, Any]]:
        out = []
        for r in range(row_start, row_end):
            for col in range(5, 13):  # E..L
                v = ws.cell(row=r, column=col).value
                if v not in (None, "") and not (isinstance(v, str) and _is_routing_or_note(v)):
                    out.append((r, col, v))
        return out

    section4: dict[str, Any] = {}
    for i, row_start in enumerate(qrows):
        row_end = qrows[i + 1] if i + 1 < len(qrows) else end
        field_id = fields[i]
        rat_row = rationale_label_row_in(row_start, row_end)
        cands = all_candidates(row_start, row_end)

        if cands:
            section4[field_id] = cands[0][2]
        if rat_row is not None and len(cands) >= 2:
            m = re.match(r"(q4_\d\d)_", field_id)
            assert m is not None
            section4[m.group(1) + "_rationale"] = cands[-1][2]
    return section4


def coerce_types(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (datetime.datetime, datetime.date)):
            out[key] = value.date().isoformat() if isinstance(value, datetime.datetime) else value.isoformat()
        elif isinstance(value, str):
            out[key] = value.strip()
        elif isinstance(value, float) and value.is_integer():
            # Excel stores "12151323" (a company number) as a float; keep it
            # as the string a reviewer would recognise, not "12151323.0".
            out[key] = str(int(value))
        elif key == "q2_03_company_number" and isinstance(value, int):
            # Sometimes read back as a plain int instead -- same reasoning:
            # a company number is an identifier, not a quantity, so it goes
            # into the JSON as a string.
            out[key] = str(value)
        else:
            out[key] = value
    return out


def convert(xlsx_path: Path, record_id: str | None = None) -> dict[str, Any]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[SHEET_NAME]

    record: dict[str, Any] = {}
    record.update(extract_labelled_fields(ws))
    record.update(extract_address(ws))
    record.update(extract_engagement_dates(ws))
    record.update(extract_section4(ws))
    record = coerce_types(record)

    record = {"record_id": record_id or f"ESQ-{xlsx_path.stem}", **record}
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xlsx_path", type=Path, help="Path to the completed ESQ .xlsx file")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output JSON path (default: alongside the input file)")
    parser.add_argument("--record-id", default=None, help="record_id to embed (default: derived from the filename)")
    args = parser.parse_args()

    if not args.xlsx_path.exists():
        print(f"error: {args.xlsx_path} does not exist", file=sys.stderr)
        return 1

    record = convert(args.xlsx_path, record_id=args.record_id)

    schema = load_schema()
    gen_cfg = load_generation_config()
    validator = RecordValidator(schema, gen_cfg.get("validation", {}))
    result = validator.validate(record)

    out_path = args.out or args.xlsx_path.with_suffix(".json")
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(record)} fields to {out_path}")
    print()
    if result.errors:
        print(f"{len(result.errors)} error(s) -- worth fixing before using this record:")
        for f in result.errors:
            print(f"  ERROR   {f.field_id}: {f.message}")
    warnings = [f for f in result.findings if f.severity is Severity.WARNING]
    if warnings:
        print(f"{len(warnings)} warning(s) -- usually just reflect what was actually filled in:")
        for f in warnings:
            print(f"  warning {f.field_id}: {f.message}")
    if not result.findings:
        print("No validation findings -- the record is complete and every answer is in its declared domain.")

    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
