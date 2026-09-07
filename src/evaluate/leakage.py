"""Cue leakage: measuring how much the rule baseline was told the answer.

THE PROBLEM
The lexical cues in ``contradiction_pairs.yaml`` were authored in Phase 0. The
contradiction text in ``text_bank.yaml`` was authored in Phase 1. Both by the
same process, a week apart, from the same understanding of what an IR35
contradiction looks like. Unsurprisingly, they overlap: 32% of planted
contradiction units contain a cue string verbatim, and in the worst case the cue
*is* the sentence — cue ``"we specifically need this individual"`` against the
planted text ``"we specifically need this individual, nobody else has the
expertise"``.

A rule baseline evaluated against that data is partly being graded on a
memorised answer key. Reporting its recall as a single number would overstate
what a keyword approach achieves on text it has not seen, and would understate
the transformer's advantage in Phase 3 — in both directions, the wrong
conclusion.

THE TREATMENT
Not all overlap is cheating. ``"trustee"``, ``"unpaid"`` and ``"mileage"`` are
domain vocabulary any analyst would list independently of this corpus.
``"we specifically need this individual"`` is a copied sentence. Rather than
draw an arbitrary line between them by word count, this module computes the
overlap and the rule baseline is run twice:

* **as-authored** — every cue. An upper bound: what a keyword list achieves when
  its phrasing happens to match the data.
* **de-leaked** — every cue that appears verbatim anywhere in the text bank is
  removed, including the legitimate domain terms. A lower bound, deliberately
  conservative.

True performance on real submissions lies between the two, and the width of that
interval is itself the finding: it is a direct measure of how much a lexical
method depends on having seen the phrasing before.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Mapping

from src.ingest.schema_loader import Schema

__all__ = ["CueLeakage", "analyse_cue_leakage", "deleaked_cues", "cue_list"]


def cue_list(cues: Mapping[str, Any] | None) -> list[str]:
    """Flatten a pair's ``cues`` block into a list of lexical strings.

    The block holds several named groups (``inside_signals``, ``outside_signals``)
    plus regex entries such as ``rate_pattern``. Only the string lists are
    lexical cues; regexes are handled separately by the rule baseline.

    Args:
        cues: The ``cues`` mapping from ``contradiction_pairs.yaml``.

    Returns:
        Lower-cased cue strings.
    """
    out: list[str] = []
    for value in (cues or {}).values():
        if isinstance(value, list):
            out.extend(str(v).lower().strip() for v in value if str(v).strip())
    return out


@dataclass
class CueLeakage:
    """Overlap between the Phase 0 cue lists and the Phase 1 text bank.

    Attributes:
        n_units: Contradiction content units examined.
        n_units_with_cue: Units containing at least one cue verbatim.
        n_cues: Distinct cues across all pairs.
        n_leaked_cues: Cues appearing verbatim in some contradiction unit.
        leaked_by_pair: Pair id -> the cues of that pair found in its own text.
        worst_examples: A few (pair, cue, unit) triples with the longest match,
            for the report — the longest matches are the most damning.
    """

    n_units: int = 0
    n_units_with_cue: int = 0
    n_cues: int = 0
    n_leaked_cues: int = 0
    leaked_by_pair: dict[str, list[str]] = dc_field(default_factory=dict)
    worst_examples: list[tuple[str, str, str]] = dc_field(default_factory=list)

    @property
    def unit_leakage_rate(self) -> float:
        """Share of contradiction units containing a cue verbatim."""
        return self.n_units_with_cue / self.n_units if self.n_units else 0.0

    @property
    def cue_leakage_rate(self) -> float:
        """Share of cues that appear verbatim in the generated text."""
        return self.n_leaked_cues / self.n_cues if self.n_cues else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the report."""
        return {
            "n_units": self.n_units,
            "n_units_with_cue": self.n_units_with_cue,
            "unit_leakage_rate": round(self.unit_leakage_rate, 4),
            "n_cues": self.n_cues,
            "n_leaked_cues": self.n_leaked_cues,
            "cue_leakage_rate": round(self.cue_leakage_rate, 4),
            "pairs_affected": len(self.leaked_by_pair),
        }


def analyse_cue_leakage(schema: Schema, text_bank: Mapping[str, Any]) -> CueLeakage:
    """Measure how far the Phase 0 cues match the Phase 1 contradiction text.

    Args:
        schema: The loaded data contract, carrying each pair's cues.
        text_bank: Parsed ``text_bank.yaml``.

    Returns:
        A ``CueLeakage`` summary.
    """
    result = CueLeakage()
    all_cues: set[str] = set()
    leaked_cues: set[str] = set()
    examples: list[tuple[int, str, str, str]] = []

    for pair in schema.pairs.values():
        cues = cue_list(pair.cues)
        all_cues.update(cues)
        entry = text_bank["pairs"].get(pair.id, {})
        for units in (entry.get("contradicting") or {}).values():
            for unit in units:
                text = str(unit["text"]).lower()
                result.n_units += 1
                hits = [c for c in cues if c in text]
                if hits:
                    result.n_units_with_cue += 1
                    leaked_cues.update(hits)
                    result.leaked_by_pair.setdefault(pair.id, [])
                    result.leaked_by_pair[pair.id].extend(
                        h for h in hits if h not in result.leaked_by_pair[pair.id]
                    )
                    longest = max(hits, key=len)
                    examples.append((len(longest), pair.id, longest, str(unit["text"])))

    result.n_cues = len(all_cues)
    result.n_leaked_cues = len(leaked_cues)
    examples.sort(reverse=True)
    result.worst_examples = [(p, c, t) for _, p, c, t in examples[:5]]
    return result


def contradiction_corpus(text_bank: Mapping[str, Any]) -> str:
    """All planted contradiction text, concatenated and lower-cased."""
    return " || ".join(
        str(unit["text"]).lower()
        for entry in text_bank["pairs"].values()
        for units in (entry.get("contradicting") or {}).values()
        for unit in units
    )


def deleaked_cues(
    schema: Schema, text_bank: Mapping[str, Any], min_leak_words: int = 3
) -> dict[str, list[str]]:
    """Build a cue list with text-bank-matching cues removed.

    ``min_leak_words`` sets what counts as leakage, and the choice matters:

    * ``min_leak_words=3`` (the default) removes multi-word phrases that appear
      verbatim — copied phrasing such as *"we specifically need this
      individual"* — while keeping one- and two-word domain vocabulary such as
      *"trustee"*, *"reimbursed"* and *"core hours"*. An independent analyst
      building a cue list for IR35 would write those short terms without ever
      seeing this corpus, so removing them would not measure leakage; it would
      just remove the domain.
    * ``min_leak_words=1`` removes every matching cue including the vocabulary.
      That was the first version tried, and it drops the baseline's recall to
      near zero — which is close to tautological ("remove every word that occurs
      in the data and you find nothing") rather than informative. It is retained
      as a pessimistic floor and reported as such, not as an estimate.

    The three-word line is a judgement, stated here so a reader can disagree with
    it and recompute. It is not a claim that three words is where copying begins.

    Args:
        schema: The loaded data contract.
        text_bank: Parsed ``text_bank.yaml``.
        min_leak_words: Minimum word count for a verbatim match to count as
            leakage and be removed.

    Returns:
        Pair id -> surviving cues.
    """
    corpus_text = contradiction_corpus(text_bank)
    surviving: dict[str, list[str]] = {}
    for pair in schema.pairs.values():
        surviving[pair.id] = [
            c
            for c in cue_list(pair.cues)
            if not (len(c.split()) >= min_leak_words and c in corpus_text)
        ]
    return surviving


def as_authored_cues(schema: Schema) -> dict[str, list[str]]:
    """The cue lists exactly as written in Phase 0.

    Args:
        schema: The loaded data contract.

    Returns:
        Pair id -> cues.
    """
    return {pair.id: cue_list(pair.cues) for pair in schema.pairs.values()}
