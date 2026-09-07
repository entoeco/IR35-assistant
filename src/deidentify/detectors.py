"""Entity detectors for free-text de-identification.

Field-level policy (driven by ``pii_class`` in ``schema.yaml``) handles the
boxes that are *designed* to hold personal data. It is not sufficient on its
own, because the residual risk in this form lives somewhere else: a manager
answering *"Please explain the reason/rationale for your answer in 4.5"* writes
**"Dr Ashwood is the only person who can do this"** into a box the schema
classifies as carrying no personal data. Everything in this module exists to
catch that.

Design notes
------------
**Spans, not sequential substitution.** Each detector returns character spans.
The scrubber collects every span, resolves overlaps by priority and length, then
rewrites the string once from right to left. Sequential regex substitution would
let a later detector match inside an earlier detector's replacement token
(``[EMAIL_1]`` contains a capitalised word), which is exactly the kind of bug
that shows up on real data and not on the fixtures.

**The gazetteer is the strongest name detector we have.** The names most likely
to appear in a justification box are the ones already declared elsewhere on the
same form — the worker, the engaging officer, the named contact. So the
scrubber is handed those values and matches them directly. That is high
precision and needs no model. The pattern-based detectors below are the
fallback for names the form does not already know, and they are the weakest
part of the component; ``docs/`` records that honestly rather than claiming
coverage the regexes do not have.

**Money keeps its unit.** ``£450 per day`` becomes ``£[AMOUNT] per day``, not
``[AMOUNT]``. The commercially sensitive part is the figure; the *rate basis*
is IR35 signal and one of the highest-value contradiction cues on the form
(pair ``p_4_16_payment_basis``). Destroying "per day" to remove "450" would
protect nothing extra and would blind the detector to a contradiction that
directly affects a determination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Pattern

__all__ = ["Span", "DETECTORS", "detect_all", "resolve_overlaps", "gazetteer_spans"]


@dataclass(frozen=True, order=True)
class Span:
    """A detected entity occupying ``text[start:end]``.

    Attributes:
        start: Inclusive start offset.
        end: Exclusive end offset.
        kind: Detector name, e.g. ``"email"``. Becomes the surrogate prefix.
        priority: Higher wins when two spans overlap. Specific formats beat
            general ones, so an email is not partly eaten by a URL detector.
        keep_suffix: Text appended after the surrogate, used to preserve a unit
            such as ``" per day"`` while removing the value.
    """

    start: int
    end: int
    kind: str
    priority: int = 0
    keep_suffix: str = ""

    @property
    def length(self) -> int:
        """Number of characters covered."""
        return self.end - self.start


def _regex_detector(
    name: str,
    pattern: Pattern[str],
    priority: int,
    *,
    group: int = 0,
) -> Callable[[str], Iterator[Span]]:
    """Build a detector that yields one span per regex match.

    Args:
        name: Detector name, used as the surrogate prefix.
        pattern: Compiled pattern.
        priority: Overlap-resolution priority.
        group: Capture group to redact. Group 0 redacts the whole match; a
            higher group redacts only part of it, which is how the money
            detector keeps its unit.
    """

    def detect(text: str) -> Iterator[Span]:
        for match in pattern.finditer(text):
            if match.group(group) is None:
                continue
            yield Span(match.start(group), match.end(group), name, priority)

    detect.__name__ = f"detect_{name}"
    return detect


# --- Patterns ---------------------------------------------------------------
# Deliberately conservative. A false positive costs a reviewer a moment of
# confusion; a false negative leaves personal data in a corpus. Where the two
# trade off, these lean towards over-matching.

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

URL = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"']+|"
    r"\b(?:[\w-]+\.)+(?:com|org|net|ac\.uk|co\.uk|io|gov\.uk)\b(?:/[^\s<>\"']*)?"
)

# UK NI number: two prefix letters, six digits, one suffix letter A-D.
#
# Deliberately looser than the real prefix rules (which exclude D, F, I, Q, U, V
# in first position and O in second). Validating strictly would leak the
# widely-used documentation example "QQ123456C" and, worse, any mistyped real
# number — and a mistyped NINO still identifies a person. The cost of the loose
# pattern is that a NINO-shaped string that is not a NINO gets labelled
# `ni_number` in the report. That is a labelling inaccuracy in metadata; the
# strict version's cost is personal data left in the corpus. Not a close call.
NI_NUMBER = re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")

# UK postcode, outward + inward, optional space.
UK_POSTCODE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE
)

UK_PHONE = re.compile(
    r"(?:\+44\s?|\b0)(?:\d\s?){9,10}\b"
)

# Companies House numbers: 8 digits, or 2 letters + 6 digits (SC, NI, OC…).
COMPANY_NUMBER = re.compile(r"\b(?:[A-Z]{2}\d{6}|\d{8})\b")

# Money: the figure is masked, any unit that follows is kept. Group 1 is the
# numeric part only.
MONEY = re.compile(
    r"£\s?(\d[\d,]*(?:\.\d{2})?)|"
    r"\b(\d[\d,]*(?:\.\d{2})?)\s?(?:GBP|pounds)\b",
    re.IGNORECASE,
)

# Date of birth, only where the context word makes it a DOB. General dates are
# left alone: contract start/end dates are IR35 signal (mutuality of
# obligation), and blanket date removal would blind pair p_4_28.
DATE_OF_BIRTH = re.compile(
    r"\b(?:d\.?o\.?b\.?|date of birth|born)\b[:\s]*"
    r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"|\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE,
)

TITLE_NAME = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof(?:essor)?|Sir|Dame)\.?\s+"
    r"[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2}"
)


DETECTORS: dict[str, Callable[[str], Iterator[Span]]] = {
    # Priority order matters where formats can overlap: an email address
    # contains a domain that the URL detector would also match.
    "email": _regex_detector("email", EMAIL, priority=100),
    "ni_number": _regex_detector("ni_number", NI_NUMBER, priority=95),
    "uk_postcode": _regex_detector("uk_postcode", UK_POSTCODE, priority=90),
    "date_of_birth": _regex_detector("date_of_birth", DATE_OF_BIRTH, priority=88, group=1),
    "url": _regex_detector("url", URL, priority=80),
    "uk_phone": _regex_detector("uk_phone", UK_PHONE, priority=75),
    "person_name": _regex_detector("person_name", TITLE_NAME, priority=70),
    "company_number": _regex_detector("company_number", COMPANY_NUMBER, priority=60),
    "money": _regex_detector("money", MONEY, priority=50),
}


def _money_spans(text: str) -> Iterator[Span]:
    """Money needs custom handling: mask the figure, keep the currency and unit."""
    for match in MONEY.finditer(text):
        group = 1 if match.group(1) is not None else 2
        yield Span(match.start(group), match.end(group), "money", 50)


DETECTORS["money"] = _money_spans


def gazetteer_spans(text: str, known_values: Iterable[str]) -> Iterator[Span]:
    """Find values already declared elsewhere on the same form.

    This is the highest-precision name detector available here, because the form
    tells us who the people are before we ever look at the free text.

    Args:
        text: Free-text value to search.
        known_values: Values taken from this record's direct-identifier fields
            (worker forename and surname, engaging officer, named contact,
            company name). Values shorter than three characters are skipped —
            matching a two-letter initial would shred ordinary prose.

    Yields:
        Spans with kind ``"person_name"`` and a priority above the pattern
        detectors, so a gazetteer hit wins any overlap.
    """
    for value in known_values:
        if not value or len(value.strip()) < 3:
            continue
        for match in re.finditer(rf"\b{re.escape(value.strip())}\b", text, re.IGNORECASE):
            yield Span(match.start(), match.end(), "person_name", priority=200)


def detect_all(text: str, enabled: Iterable[str]) -> list[Span]:
    """Run the enabled pattern detectors over a string.

    Args:
        text: The value to scan.
        enabled: Detector names from ``pipeline.yaml``.

    Returns:
        All spans found, unresolved and possibly overlapping.
    """
    spans: list[Span] = []
    for name in enabled:
        detector = DETECTORS.get(name)
        if detector is None:
            continue
        spans.extend(detector(text))
    return spans


def resolve_overlaps(spans: Iterable[Span]) -> list[Span]:
    """Reduce overlapping spans to a non-overlapping set.

    Resolution order is priority first, then length, then position — a specific
    format beats a general one, and a longer match beats a shorter one inside it.

    Args:
        spans: Candidate spans, in any order.

    Returns:
        Non-overlapping spans sorted by start offset.
    """
    ranked = sorted(spans, key=lambda s: (-s.priority, -s.length, s.start))
    kept: list[Span] = []
    for span in ranked:
        if any(span.start < k.end and k.start < span.end for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)
