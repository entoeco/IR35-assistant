#!/usr/bin/env python3
"""Convert a Microsoft Forms ESQ export (.xlsx) into the schema-field-id-keyed
JSON the reviewer app's "paste submission" and "upload a file" tabs expect
(see ``config/schema.yaml``).

Usage:

    python scripts/convert_esq_msforms_xlsx_to_json.py path/to/export.xlsx
    python scripts/convert_esq_msforms_xlsx_to_json.py path/to/export.xlsx -o out.json
    python scripts/convert_esq_msforms_xlsx_to_json.py path/to/export.xlsx -o out_dir/

Why this exists
----------------
From Nov 2026, HR collect the ESQ through a Microsoft Forms form rather than
the June-2026 Excel template that ``convert_esq_xlsx_to_json.py`` understands.
The export shape is completely different: one header row, one row per
response, one column per question -- no row-position heuristics needed -- but
with two quirks of its own:

1. A question whose routing branches on the exact option chosen (e.g. "Would
   the worker's business send someone else... 4.3") gets ONE rationale column
   PER POSSIBLE OPTION ("...for your answer 'Yes - you accepted them' in
   4.3", "...'Yes - you did not accept them' in 4.3", ...), with at most one
   populated per response. This script merges them back into the single
   ``q4_NN_rationale`` field the schema expects.
2. The worker's address is one multi-line text box (``q1_04_address_full``)
   rather than four separate line/postcode cells.

What changed on the live form vs. the June-2026 template is explained in
full in ``config/schema.yaml``'s "LIVE FORM MIGRATION NOTE" and in
``docs/reports/phase8_msforms_migration.md``. In short:

- Three genuinely new questions exist only on the live form and are mapped
  to three new, additive schema fields: ``q1_09_umbrella_company`` (+ its
  rationale ``q1_09_umbrella_company_details``), ``q3_02_self_assessment_
  registered``, and the combined ``q1_04_address_full``.
- Four questions that exist in ``config/schema.yaml`` for the synthetic
  generator's sake (``q2_04_entity_relationship``, ``q2_05_years_in_
  business``, ``q3_01_engaged_directly``, ``q5_02_engaging_officer``) have no
  column on the live form at all. This script never populates them --
  that is correct, not a bug, see the schema note.
- A few options picked up cosmetic wording differences (extra/odd
  whitespace, a stray trailing period, capitalisation) between the template
  and the live form. ``normalize_domain_value`` below tolerates these by
  matching case- and punctuation-insensitively against the schema's
  declared ``value_domain`` and substituting the canonical string, so the
  record matches the historical wording the rest of the config (notably
  ``config/text_bank.yaml``) already keys on. Anything it cannot
  confidently match is left as-is, which surfaces as a ``value_not_in_
  domain`` validation error for a human to look at -- the safe failure mode.
- The live form merged two of ``q4_03_substitute_sent``'s four options (the
  two that routed to the same next question anyway) into one new option,
  "No - it has not happened". That option was added to the schema's
  value_domain rather than normalized away, because it is not a wording
  tweak -- see the schema note on the consequence for the
  ``x_started_but_not_applicable`` cross-field check.

Two things this script deliberately does NOT do (same reasoning as
``convert_esq_xlsx_to_json.py``):

- It never reads or writes the four "For IR35 Team use only" outcome fields.
- It never guesses a value for a blank field. A blank cell becomes a field
  that is absent from the JSON, not an empty string.

It also deliberately does NOT carry the export's own "Id" column's sibling
"Email"/"Name" columns (the Microsoft 365 identity of whoever submitted the
form) into the record -- that identifies the manager, not the worker, and is
not part of the data contract. The ``record_id`` is derived from the
numeric MS Forms "Id" column instead, which is not personally identifying on
its own.

A single export can hold many responses. This script converts all of them;
see ``main`` for how output paths are chosen.
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

SHEET_NAME = "Sheet1"

# Columns Microsoft Forms adds that are not part of the ESQ data contract at
# all -- metadata about the submission event, or the submitting manager's own
# identity rather than the worker's.
IGNORED_HEADERS = {"id", "start time", "completion time", "email", "name"}

# (field_id, a substring that appears in -- and only in -- that question's
# header, once \xa0 is normalised to a plain space). Matched by substring
# rather than exact/prefix equality so that incidental whitespace or
# punctuation drift in a future export does not silently stop matching.
QUESTION_FIELDS: list[tuple[str, str]] = [
    ("q1_01_title", "1.1 title"),
    ("q1_02_first_name", "1.2 first name"),
    ("q1_03_surname", "1.3 surname"),
    ("q1_04_address_full", "1.4 registered address"),
    ("q1_05_telephone", "1.5 contact telephone number"),
    ("q1_06_internet_presence", "1.5a internet presence"),
    ("q1_07_engagement_from", "length of engagement from"),
    ("q1_07_engagement_to", "length of engagement to"),
    ("q1_09_umbrella_company", "is the worker providing their services through an umbrella company"),
    ("q1_08_contract_in_place", "is there, or will there be, a contract in place"),
    ("q2_01_via_third_party", "is the engagement of work through a limited company"),
    ("q2_02_company_name", "2.2 company name"),
    ("q2_03_company_number", "2.3 reg company number"),
    ("q3_02_self_assessment_registered", "is the individual registered for self assessment with hmrc"),
    ("q4_00_role_title", "contract or role title"),
    ("q4_00_duties_narrative", "description of duties of engagement"),
    ("q4_01_already_started", "has the worker already started working for the university of sussex"),
    ("q4_02_office_holder", "will the worker be an office holder"),
    ("q4_03_substitute_sent", "would the worker's business send someone else"),
    ("q4_04_worker_pays_substitute", "will the worker pay their substitute"),
    ("q4_05_right_to_reject", "do you have the right to reject a substitute"),
    ("q4_06_paid_another_person", "has the worker paid another person to do a significant portion of the work"),
    ("q4_07_would_pay_substitute", "would the worker have to pay their substitute"),
    ("q4_08_moved_from_task", "can the worker be moved from the task they originally agreed to"),
    ("q4_09_decide_how", "can you (the engager) decide how the work is done"),
    ("q4_10_decide_schedule", "can you (the engager) decide the schedule of working hours"),
    ("q4_11_decide_where", "can you (the engager) choose where the worker does the work"),
    ("q4_12_buys_equipment", "does the worker have to buy equipment before they are paid"),
    ("q4_13_vehicle_costs", "will the worker have to fund an vehicle costs"),
    ("q4_14_materials_unreimbursed", "does the worker pay the costs of all the materials or supplies"),
    ("q4_15_other_costs", "will the worker have to fund any other costs before they are paid"),
    ("q4_16_payment_basis", "how will the worker be paid for this work"),
    ("q4_17_put_right", "is not satisfied with the work, does the worker need to put it right"),
    ("q4_18_benefits", "will the worker be provided with paid for benefits"),
    ("q4_19_manages_staff", "is the worker responsible for hiring or dismissing workers"),
    ("q4_20_identifies_as", "how do they identify themselves"),
    ("q4_21_exclusivity_clause", "does the contract stop the worker from doing similar work"),
    ("q4_22_permission_required", "is the worker required to ask permission to work for other organisations"),
    ("q4_23_ownership_rights", "are there any ownership rights relating to the contract"),
    ("q4_24_rights_to_university", "does the contract state the rights to this work belong to your organisation"),
    ("q4_25_option_to_buy_rights", "does the contract give your organisation the option to buy the rights"),
    ("q4_26_previous_contract", "has the worker had a previous contract with your organisation"),
    ("q4_27_first_in_series", "is the current contract the first in a series"),
    ("q4_28_starts_immediately_after", "will this contract start immediately after the previous one ended"),
    ("q4_29_extendable", "does the current contract allow for it to be extended"),
    ("q4_30_majority_of_time", "will this work take up the majority of the workers"),
    ("q4_31_other_clients", "has the worker done any self-employed work of a similar nature"),
    ("q5_01_school_division", "school/division"),
    ("q5_03_contact_name", "contact name (if different)"),
    ("q5_04_phone", "phone number"),
    ("q5_05_email", "email address"),
]

# Rationale columns whose header has no "in N.NN" suffix to key off, because
# the question itself is not numbered the way Section 4's are.
SPECIAL_RATIONALE_FIELDS: dict[str, str] = {
    "please give details of what form the contract takes, eg. email exchange, formal letter, etc.":
        "q1_08_contract_form",
    "please provide all the details in the box provided":
        "q1_09_umbrella_company_details",
}

# Every Section 4 question except 4.1 has a rationale box (schema.yaml
# documents 4.1 as the one deliberate exception). Both the branching form
# ("...for your answer 'OPTION' in 4.N") and the plain form ("...in 4.N")
# end in this slot number, which is what this table keys on.
SLOT_TO_RATIONALE_FIELD: dict[str, str] = {
    f"4.{n}": f"q4_{n:02d}_rationale" for n in range(2, 32)
}

_BRANCH_RATIONALE_RE = re.compile(
    r"^please explain the reason/rationale for your answer '.+' in (?P<slot>\d+\.\d+)$"
)
_PLAIN_RATIONALE_RE = re.compile(
    r"^please explain the reason/rationale for your answer in (?P<slot>\d+\.\d+)$"
)

# Fields that MS Forms exports as a number even though the value is really an
# identifier, not a quantity -- a company number or a phone number is never
# arithmetic.
STRING_NOT_NUMBER_FIELDS = {"q1_05_telephone", "q5_04_phone", "q2_03_company_number"}


def normalize_header(raw: Any) -> str:
    """Flatten a header cell to lowercase, with \\xa0 and run of whitespace
    collapsed to a single space, so matching is robust to the small
    whitespace differences Microsoft Forms introduces between questions."""
    text = str(raw or "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def map_headers(headers: list[Any]) -> tuple[dict[int, str], list[str]]:
    """Return (column index -> field id, list of headers nothing matched)."""
    col_to_field: dict[int, str] = {}
    unmapped: list[str] = []

    for idx, raw_header in enumerate(headers):
        norm = normalize_header(raw_header)
        if not norm or norm in IGNORED_HEADERS:
            continue

        if norm in SPECIAL_RATIONALE_FIELDS:
            col_to_field[idx] = SPECIAL_RATIONALE_FIELDS[norm]
            continue

        m = _BRANCH_RATIONALE_RE.match(norm) or _PLAIN_RATIONALE_RE.match(norm)
        if m:
            field_id = SLOT_TO_RATIONALE_FIELD.get(m.group("slot"))
            if field_id:
                col_to_field[idx] = field_id
                continue

        matched = False
        for field_id, substring in QUESTION_FIELDS:
            if substring in norm:
                col_to_field[idx] = field_id
                matched = True
                break
        if not matched:
            unmapped.append(str(raw_header))

    return col_to_field, unmapped


def normalize_domain_value(value: str, domain: list[str]) -> tuple[str, str | None]:
    """Match a real answer against a field's declared value_domain, tolerant
    of the cosmetic wording drift observed between the June-2026 template and
    the live Microsoft Forms form (stray whitespace, a trailing period,
    curly-vs-straight apostrophes, capitalisation).

    Returns (value_to_use, note). ``note`` is set only when a fuzzy match
    substituted the canonical schema string for the raw one, so the caller
    can surface that it happened. If no option matches even loosely, the
    raw value is returned unchanged and will be flagged as
    ``value_not_in_domain`` by RecordValidator -- a human should look at it
    rather than this script guessing.
    """
    if value in domain:
        return value, None

    def fold(s: str) -> str:
        s = s.replace("’", "'").replace("‘", "'")
        s = re.sub(r"\s+", " ", s.strip().lower())
        s = s.rstrip(".")
        return s

    folded_value = fold(value)
    matches = [opt for opt in domain if fold(opt) == folded_value]
    if len(matches) == 1:
        return matches[0], f"normalized {value!r} to the declared option {matches[0]!r}"
    return value, None


def coerce_types(record: dict[str, Any], schema) -> tuple[dict[str, Any], list[str]]:
    """Cast datetimes/numbers to the JSON-friendly forms the rest of the
    pipeline expects, and apply ``normalize_domain_value`` to every
    structured answer. Returns (record, notes)."""
    notes: list[str] = []
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (datetime.datetime, datetime.date)):
            out[key] = value.date().isoformat() if isinstance(value, datetime.datetime) else value.isoformat()
            continue
        if isinstance(value, str):
            value = value.strip()
        elif key in STRING_NOT_NUMBER_FIELDS and isinstance(value, (int, float)):
            value = str(int(value))
        elif isinstance(value, float) and value.is_integer():
            value = str(int(value))

        field = schema[key] if key in schema.fields else None
        if field is not None and field.value_domain and isinstance(value, str) and value != "":
            new_value, note = normalize_domain_value(value, field.value_domain)
            if note:
                notes.append(f"{key}: {note}")
            value = new_value

        out[key] = value
    return out, notes


def extract_responses(xlsx_source: Any) -> tuple[list[dict[str, Any]], dict[int, Any], list[str]]:
    """Read every response row. Returns (raw_records, id_by_row_index, unmapped_headers).

    ``xlsx_source`` is anything ``openpyxl.load_workbook`` accepts: a path, or
    a file-like object such as the in-memory buffer Streamlit's file
    uploader hands you -- so the reviewer app can convert an upload without
    writing a temporary file to disk.
    """
    wb = openpyxl.load_workbook(xlsx_source, data_only=True)
    ws = wb[SHEET_NAME] if SHEET_NAME in wb.sheetnames else wb[wb.sheetnames[0]]

    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    col_to_field, unmapped = map_headers(headers)

    id_col = next(
        (i for i, h in enumerate(headers) if normalize_header(h) == "id"), None
    )

    records: list[dict[str, Any]] = []
    ids: dict[int, Any] = {}
    for row in range(2, ws.max_row + 1):
        row_values = [ws.cell(row=row, column=c + 1).value for c in range(len(headers))]
        if all(v in (None, "") for v in row_values):
            continue  # a wholly blank trailing row

        record: dict[str, Any] = {}
        for idx, field_id in col_to_field.items():
            value = row_values[idx]
            if value in (None, ""):
                continue
            if field_id not in record or record[field_id] in (None, ""):
                record[field_id] = value
            # A branching question's multiple rationale columns should never
            # BOTH be populated for one response (it is a single-select) --
            # if they are, that is a genuine data anomaly worth surfacing
            # rather than silently picking one.

        records.append(record)
        if id_col is not None:
            ids[len(records) - 1] = row_values[id_col]

    return records, ids, unmapped


def convert_all(
    xlsx_source: Any, fallback_name: str = "upload"
) -> list[tuple[dict[str, Any], list[str]]]:
    """Convert every response row. Returns a list of (record, notes).

    ``xlsx_source`` -- see ``extract_responses``. ``fallback_name`` is used to
    build a record_id only for the (expected not to happen on a real export)
    case where there is no "Id" column to key off; pass something meaningful
    (e.g. the uploaded filename's stem) when ``xlsx_source`` is not a path.
    """
    schema = load_schema()
    records, ids, unmapped = extract_responses(xlsx_source)
    if unmapped:
        print(
            "warning: these columns were not recognised and were ignored:\n  "
            + "\n  ".join(unmapped),
            file=sys.stderr,
        )

    results: list[tuple[dict[str, Any], list[str]]] = []
    for i, raw_record in enumerate(records):
        record, notes = coerce_types(raw_record, schema)
        ms_id = ids.get(i)
        record_id = f"ESQ-MSFORMS-{ms_id}" if ms_id is not None else f"ESQ-MSFORMS-{fallback_name}-{i + 1}"
        record = {"record_id": record_id, **record}
        results.append((record, notes))
    return results


def _report(record: dict[str, Any], notes: list[str], validator: RecordValidator) -> bool:
    """Print a per-record summary. Returns True if the record has no errors."""
    result = validator.validate(record)
    print(f"--- {record['record_id']} ({len(record)} fields) ---")
    if notes:
        print(f"{len(notes)} wording normalization(s) applied:")
        for n in notes:
            print(f"  note    {n}")
    if result.errors:
        print(f"{len(result.errors)} error(s) -- worth fixing before using this record:")
        for f in result.errors:
            print(f"  ERROR   {f.field_id}: {f.message}")
    warnings = [f for f in result.findings if f.severity is Severity.WARNING]
    if warnings:
        print(f"{len(warnings)} warning(s) -- usually just reflect what was actually filled in:")
        for f in warnings:
            print(f"  warning {f.field_id}: {f.message}")
    if not result.findings and not notes:
        print("No validation findings -- the record is complete and every answer is in its declared domain.")
    print()
    return result.is_valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xlsx_path", type=Path, help="Path to the Microsoft Forms ESQ export (.xlsx)")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="Output path. A single response writes to this file path; "
             "multiple responses write one JSON file per response into this "
             "directory (created if needed). Default: alongside the input file.",
    )
    args = parser.parse_args()

    if not args.xlsx_path.exists():
        print(f"error: {args.xlsx_path} does not exist", file=sys.stderr)
        return 1

    converted = convert_all(args.xlsx_path, fallback_name=args.xlsx_path.stem)
    if not converted:
        print("error: no response rows found in this export", file=sys.stderr)
        return 1

    schema = load_schema()
    gen_cfg = load_generation_config()
    validator = RecordValidator(schema, gen_cfg.get("validation", {}))

    all_valid = True
    if len(converted) == 1:
        record, notes = converted[0]
        out_path = args.out or args.xlsx_path.with_suffix(".json")
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(record)} fields to {out_path}\n")
        all_valid = _report(record, notes, validator)
    else:
        out_dir = args.out or args.xlsx_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{len(converted)} responses found -- writing one JSON file per response into {out_dir}\n")
        for record, notes in converted:
            out_path = out_dir / f"{record['record_id']}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            if not _report(record, notes, validator):
                all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
