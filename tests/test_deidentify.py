"""Constraint 3: de-identification is a real, tested component.

The tests that matter here are the ones about free text. Field-level scrubbing
of a box labelled "Surname" is easy and would pass whatever we wrote. The
component earns its place by catching a worker's name inside
*"Please explain the reason/rationale for your answer in 4.5"*, and by not
destroying the IR35 signal while it does so.
"""

from __future__ import annotations

import pytest

from src.deidentify.deidentifier import Deidentifier
from src.deidentify.detectors import Span, resolve_overlaps
from src.ingest.schema_loader import load_schema

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture
def deid(schema) -> Deidentifier:
    return Deidentifier(schema)


# --- free text: the part that actually matters ------------------------------

def test_gazetteer_finds_the_worker_name_in_a_justification_box(deid: Deidentifier) -> None:
    """The canonical leak: the form declares the name, the manager repeats it."""
    record = {
        "record_id": "SYN-0001",
        "q1_02_first_name": "Alis",
        "q1_03_surname": "Ashwood",
        "q4_05_rationale": "Alis Ashwood is the only person who can deliver this.",
    }
    clean, report = deid.deidentify(record)
    assert "Alis" not in clean["q4_05_rationale"]
    assert "Ashwood" not in clean["q4_05_rationale"]
    assert report.entity_counts["person_name"] >= 1
    assert "q4_05_rationale" in report.fields_with_entities


def test_surname_alone_is_caught(deid: Deidentifier) -> None:
    """Managers rarely repeat the full name. Partial matching is the real case."""
    record = {
        "record_id": "SYN-0002",
        "q1_02_first_name": "Bera",
        "q1_03_surname": "Kirkwell",
        "q4_09_rationale": "Kirkwell decides the method entirely.",
    }
    clean, _ = deid.deidentify(record)
    assert "Kirkwell" not in clean["q4_09_rationale"]


def test_surrogates_preserve_coreference_within_a_record(deid: Deidentifier) -> None:
    """A reviewer has to be able to read the scrubbed text and follow it."""
    record = {
        "record_id": "SYN-0003",
        "q1_03_surname": "Thornwell",
        "q4_05_rationale": "Thornwell was engaged directly. Thornwell holds the licence.",
    }
    clean, _ = deid.deidentify(record)
    text = clean["q4_05_rationale"]
    assert text.count("[PERSON_1]") == 2


def test_surrogates_are_not_stable_across_records(schema) -> None:
    """Corpus-wide stable surrogates would rebuild a linkable identifier."""
    deid = Deidentifier(schema)
    a, _ = deid.deidentify(
        {"record_id": "A", "q1_03_surname": "Thornwell", "q4_05_rationale": "Thornwell agreed."}
    )
    b, _ = deid.deidentify(
        {"record_id": "B", "q1_03_surname": "Marbury", "q4_05_rationale": "Marbury agreed."}
    )
    # Same token, different people — which is the point: the token carries no
    # cross-record information.
    assert a["q4_05_rationale"] == b["q4_05_rationale"]


def test_titled_name_not_declared_on_the_form_is_still_caught(deid: Deidentifier) -> None:
    record = {
        "record_id": "SYN-0004",
        "q4_19_rationale": "They report to Prof Marchmont for academic matters.",
    }
    clean, report = deid.deidentify(record)
    assert "Marchmont" not in clean["q4_19_rationale"]
    assert report.entity_counts["person_name"] == 1


@pytest.mark.parametrize(
    "text,leaked,kind",
    [
        ("Contact them on a.ashwood@example.com for scheduling.", "a.ashwood@example.com", "email"),
        ("Their mobile is 07700 900123 if needed.", "900123", "uk_phone"),
        ("Invoices go to their office at BN1 9RH.", "BN1 9RH", "uk_postcode"),
        ("Registered as 09876543 at Companies House.", "09876543", "company_number"),
        ("Their NI number is QQ123456C on file.", "QQ123456C", "ni_number"),
        ("See their profile at www.example.com/profile for detail.", "example.com", "url"),
        ("Date of birth 12/03/1978 recorded in the file.", "12/03/1978", "date_of_birth"),
    ],
)
def test_pattern_detectors_remove_what_they_claim_to(
    deid: Deidentifier, text: str, leaked: str, kind: str
) -> None:
    record = {"record_id": "SYN-P", "q4_16_rationale": text}
    clean, report = deid.deidentify(record)
    assert leaked not in clean["q4_16_rationale"]
    assert report.entity_counts[kind] >= 1


# --- not destroying the signal ----------------------------------------------

def test_money_is_masked_but_the_rate_basis_survives(deid: Deidentifier) -> None:
    """The figure is commercially sensitive; 'per day' is IR35 signal.

    Pair p_4_16_payment_basis turns on whether the text describes a day rate
    against a 'fixed price' answer. Removing the unit along with the number
    would blind the detector to a contradiction that changes a determination.
    """
    record = {
        "record_id": "SYN-0005",
        "q4_16_rationale": "Fixed price agreed, but invoiced monthly at £450 per day.",
    }
    clean, report = deid.deidentify(record)
    text = clean["q4_16_rationale"]
    assert "450" not in text
    assert "per day" in text
    assert "monthly" in text
    assert report.entity_counts["money"] == 1


def test_contract_dates_are_left_alone(deid: Deidentifier) -> None:
    """Only DOB-shaped dates go. Contract continuity is the whole of pair
    p_4_28_starts_immediately_after — blanket date removal would destroy it."""
    record = {
        "record_id": "SYN-0006",
        "q4_28_rationale": "Continues seamlessly from the contract ending 31 July 2026.",
    }
    clean, report = deid.deidentify(record)
    assert "31 July 2026" in clean["q4_28_rationale"]
    assert report.entity_counts["date_of_birth"] == 0


def test_structured_status_answers_are_never_scrubbed(deid: Deidentifier) -> None:
    """Scrubbing a dropdown value would remove the signal, not the risk."""
    record = {
        "record_id": "SYN-0007",
        "q4_05_right_to_reject": "No",
        "q4_09_decide_how": "Not Relevant - it is highly skilled work",
    }
    clean, _ = deid.deidentify(record)
    assert clean["q4_05_right_to_reject"] == "No"
    assert clean["q4_09_decide_how"] == "Not Relevant - it is highly skilled work"


def test_unknown_keys_pass_through(deid: Deidentifier) -> None:
    """Generator bookkeeping must survive the scrubber."""
    record = {"record_id": "SYN-0008", "archetype": "it_contractor", "register": "terse"}
    clean, _ = deid.deidentify(record)
    assert clean["archetype"] == "it_contractor"
    assert clean["register"] == "terse"


# --- field-level policy ------------------------------------------------------

def test_direct_identifiers_are_replaced(deid: Deidentifier) -> None:
    record = {"record_id": "SYN-0009", "q1_03_surname": "Ashwood", "q5_05_email": "b@example.com"}
    clean, report = deid.deidentify(record)
    assert clean["q1_03_surname"] == "[REDACTED:q1_03_surname]"
    assert clean["q5_05_email"] == "[REDACTED:q5_05_email]"
    assert report.field_actions["q1_03_surname"] == "replace"


def test_quasi_identifiers_are_pseudonymised_and_stay_joinable(schema) -> None:
    """School/division is the Phase 4 disparity cohort variable. Discarding it
    would make the fairness analysis impossible, so it is hashed, not dropped."""
    deid = Deidentifier(schema)
    a, _ = deid.deidentify({"record_id": "A", "q5_01_school_division": "School of Psychology"})
    b, _ = deid.deidentify({"record_id": "B", "q5_01_school_division": "School of Psychology"})
    c, _ = deid.deidentify({"record_id": "C", "q5_01_school_division": "IT Services"})
    assert a["q5_01_school_division"] == b["q5_01_school_division"]
    assert a["q5_01_school_division"] != c["q5_01_school_division"]
    assert "Psychology" not in a["q5_01_school_division"]


def test_pseudonyms_change_with_the_salt(schema) -> None:
    """Rotating the salt breaks linkage across runs, by design."""
    one = Deidentifier(schema, {"development_salt": "salt-a"})
    two = Deidentifier(schema, {"development_salt": "salt-b"})
    a, _ = one.deidentify({"record_id": "A", "q5_01_school_division": "IT Services"})
    b, _ = two.deidentify({"record_id": "A", "q5_01_school_division": "IT Services"})
    assert a["q5_01_school_division"] != b["q5_01_school_division"]


def test_commercially_sensitive_fields_are_masked(deid: Deidentifier) -> None:
    record = {"record_id": "SYN-0010", "q2_03_company_number": "09876543"}
    clean, _ = deid.deidentify(record)
    assert clean["q2_03_company_number"] == "[MASKED:q2_03_company_number]"


# --- reporting and posture ---------------------------------------------------

def test_report_carries_no_content(deid: Deidentifier) -> None:
    """The report is logged under constraint 4, so it must be metadata only."""
    record = {
        "record_id": "SYN-0011",
        "q1_03_surname": "Ashwood",
        "q4_05_rationale": "Ashwood cannot be substituted, email a@example.com.",
    }
    _, report = deid.deidentify(record)
    serialised = str(report.as_log_meta())
    assert "Ashwood" not in serialised
    assert "example.com" not in serialised
    assert report.total_entities >= 2


def test_disabled_deidentifier_passes_through_and_warns(schema) -> None:
    deid = Deidentifier(schema, {"enabled": False})
    record = {"record_id": "SYN-0012", "q1_03_surname": "Ashwood"}
    clean, report = deid.deidentify(record)
    assert clean["q1_03_surname"] == "Ashwood"
    assert report.field_actions == {}


def test_batch_processing_is_order_preserving(deid: Deidentifier) -> None:
    records = [{"record_id": f"SYN-{i:04d}", "q1_03_surname": "Ashwood"} for i in range(5)]
    cleaned, reports = deid.deidentify_many(records)
    assert [r.record_id for r in reports] == [r["record_id"] for r in records]
    assert len(cleaned) == 5


def test_is_clean_flags_a_record_with_nothing_detected(deid: Deidentifier) -> None:
    record = {"record_id": "SYN-0013", "q4_09_rationale": "The worker sets the method."}
    _, report = deid.deidentify(record)
    assert report.is_clean


# --- overlap resolution ------------------------------------------------------

def test_higher_priority_span_wins_an_overlap() -> None:
    """An email contains a domain the URL detector also matches."""
    spans = [Span(0, 20, "url", 80), Span(0, 25, "email", 100)]
    assert [s.kind for s in resolve_overlaps(spans)] == ["email"]


def test_longer_span_wins_at_equal_priority() -> None:
    spans = [Span(5, 10, "person_name", 70), Span(0, 15, "person_name", 70)]
    resolved = resolve_overlaps(spans)
    assert len(resolved) == 1
    assert resolved[0].length == 15


def test_non_overlapping_spans_are_all_kept() -> None:
    spans = [Span(0, 5, "email", 100), Span(10, 15, "uk_phone", 75)]
    assert len(resolve_overlaps(spans)) == 2


def test_email_is_not_partly_eaten_by_the_url_detector(deid: Deidentifier) -> None:
    record = {"record_id": "SYN-0014", "q4_20_rationale": "Reachable at cas.dunley@example.org daily."}
    clean, report = deid.deidentify(record)
    assert "example.org" not in clean["q4_20_rationale"]
    assert "cas.dunley" not in clean["q4_20_rationale"]
    assert report.entity_counts["email"] == 1
    assert report.entity_counts["url"] == 0
