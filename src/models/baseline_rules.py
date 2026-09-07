"""Rule and keyword baseline.

Reads the free text, compares what it says against the polarity of the answer
sitting beside it, and scores the mismatch. Three sources of evidence:

1. **Pair cues** from ``contradiction_pairs.yaml`` — phrases authored for one
   specific question.
2. **Family lexicons** from ``rule_baseline.yaml`` — cross-cutting vocabulary
   grouped by contradiction type and restricted to the IR35 tests where it makes
   sense. These are what give the baseline any ability to fire on phrasing its
   per-pair cue list does not cover.
3. **Structural regexes** — where the contradiction is a shape rather than a
   word, most obviously a day rate described alongside a "fixed price" answer.

Negation is handled: a cue preceded by a negator within a short window is
suppressed *and* penalised, because "we do not require them on site" is evidence
*for* the answer, not against it. Skipping this is the single most common way a
keyword baseline becomes a strawman.

WHAT THIS DELIBERATELY DOES NOT DO
It never reads the structured answers as a group and never scores employment
status. That is the labelling engine's job, and sharing logic between the two
would mean Phase 2 grading the generator against itself.

WHAT IT CANNOT DO, BY CONSTRUCTION
It can only fire where someone has written down the vocabulary. It has no way to
detect ``scope_qualification`` expressed in unfamiliar words, and no way at all
to detect ``hedged_non_support``, where the contradiction is the *absence* of
commitment rather than the presence of a phrase. Those two types are where the
Phase 3 comparison should earn its keep — and if it does not, that is the
finding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluate.leakage import (
    as_authored_cues,
    contradiction_corpus,
    cue_list,
    deleaked_cues,
)
from src.features.dataset import PairInstance
from src.ingest.schema_loader import Schema
from src.models.base import ContradictionDetector

__all__ = ["RuleHit", "RuleBaseline"]

_WORD = re.compile(r"[a-z0-9£\[\]'-]+")


@dataclass(frozen=True)
class RuleHit:
    """One piece of evidence found in a justification.

    Attributes:
        source: ``"pair_cue"``, ``"family_lexicon"`` or ``"regex"``.
        term: The phrase or pattern id that matched.
        negated: True if a negator suppressed it.
        detail: Extra context for the reviewer-facing explanation.
    """

    source: str
    term: str
    negated: bool = False
    detail: str = ""


class RuleBaseline(ContradictionDetector):
    """Lexical contradiction detector.

    Example:
        >>> from src.ingest.schema_loader import load_schema, load_yaml, CONFIG_DIR
        >>> detector = RuleBaseline(load_schema(), load_yaml(CONFIG_DIR / "rule_baseline.yaml"))
        >>> detector.name
        'rules_as_authored'
    """

    def __init__(
        self,
        schema: Schema,
        config: Mapping[str, Any],
        cues: Mapping[str, Sequence[str]] | None = None,
        variant: str = "as_authored",
    ) -> None:
        """Create the baseline.

        Args:
            schema: The loaded data contract.
            config: Parsed ``rule_baseline.yaml``.
            cues: Pair id -> cue list. Defaults to the Phase 0 cues as authored.
                The de-leaked variant passes a reduced list here.
            variant: Names the run in reports. ``"as_authored"``,
                ``"phrase_deleaked"`` or ``"vocab_stripped"`` — the three points
                on the cue-contamination interval described in
                ``src.evaluate.leakage``.
        """
        self.schema = schema
        self.config = config
        self.variant = variant
        self.name = f"rules_{variant}"
        self.description = (
            "Keyword and rule baseline: pair cues, contradiction-family lexicons, "
            "structural regexes, with negation handling"
            + ("" if variant == "as_authored" else f" [{variant.replace('_', ' ')}]")
        )
        self.cues: dict[str, list[str]] = {
            k: list(v) for k, v in (cues or as_authored_cues(schema)).items()
        }

        scoring = config["scoring"]
        self._w = scoring["weights"]
        self._offset = float(scoring["offset"])
        self._priority_boost = float(scoring["priority_boost"])
        self._max_score = float(scoring["max_score"])

        negation = config["negation"]
        self._neg_window = int(negation["window"])
        self._neg_markers = {str(m).lower() for m in negation["markers"]}

        self._families = config["families"]
        self._regexes = [
            {**entry, "compiled": re.compile(entry["pattern"], re.IGNORECASE)}
            for entry in config.get("regexes", [])
        ]

    # -- construction helpers ---------------------------------------------
    @classmethod
    def de_leaked(
        cls,
        schema: Schema,
        config: Mapping[str, Any],
        text_bank: Mapping[str, Any],
        min_leak_words: int = 3,
        variant: str | None = None,
    ) -> "RuleBaseline":
        """Build a variant with text-bank-matching cues removed.

        The same rule is applied to the family lexicons as to the pair cues, so
        the whole lexical surface is treated consistently.

        Args:
            schema: The loaded data contract.
            config: Parsed ``rule_baseline.yaml``.
            text_bank: Parsed ``text_bank.yaml``.
            min_leak_words: Minimum word count for a verbatim match to count as
                leakage. Three keeps domain vocabulary and removes copied
                phrasing; one removes everything and gives a pessimistic floor.
                See ``src.evaluate.leakage.deleaked_cues``.
            variant: Name for reports. Defaults to a name derived from the rule.

        Returns:
            A de-leaked ``RuleBaseline``.
        """
        corpus = contradiction_corpus(text_bank)

        def keep(term: str) -> bool:
            lowered = str(term).lower()
            return not (len(lowered.split()) >= min_leak_words and lowered in corpus)

        clean_config = {
            **config,
            "families": {
                name: {**family, "terms": [t for t in family["terms"] if keep(t)]}
                for name, family in config["families"].items()
            },
        }
        return cls(
            schema,
            clean_config,
            cues=deleaked_cues(schema, text_bank, min_leak_words),
            variant=variant or ("phrase_deleaked" if min_leak_words >= 2 else "vocab_stripped"),
        )

    # -- matching ---------------------------------------------------------
    def _is_negated(self, tokens: Sequence[str], start: int) -> bool:
        """True if a negator sits within the window before a match."""
        window = tokens[max(0, start - self._neg_window) : start]
        return any(
            token in self._neg_markers or token.endswith("n't") for token in window
        )

    def _find(self, text: str, term: str) -> tuple[bool, bool]:
        """Look for a term and decide whether it is negated.

        Args:
            text: Lower-cased justification.
            term: Lower-cased phrase.

        Returns:
            ``(found, negated)``.
        """
        position = text.find(term)
        if position < 0:
            return False, False
        tokens = _WORD.findall(text[:position])
        return True, self._is_negated(tokens, len(tokens))

    def hits(self, instance: PairInstance) -> list[RuleHit]:
        """Collect every piece of evidence for one instance.

        Args:
            instance: The pair instance.

        Returns:
            The hits found, negated ones included and marked.
        """
        found: list[RuleHit] = []
        text = instance.free_text.lower()
        if not text.strip():
            return found

        pair = self.schema.pairs[instance.pair_id]

        # 1. Pair cues. Only meaningful when the answer leans the way the cue
        #    argues against: an "inside" phrase is evidence of contradiction
        #    only where the manager selected an outside-leaning option.
        if instance.is_outside_leaning:
            for term in self.cues.get(instance.pair_id, []):
                seen, negated = self._find(text, term)
                if seen:
                    found.append(RuleHit("pair_cue", term, negated))

        # 2. Family lexicons, restricted to the tests where they mean something.
        for name, family in self._families.items():
            if instance.ir35_test not in family.get("applies_to_tests", []):
                continue
            wants_outside = family.get("contradicts_polarity", "outside") == "outside"
            if wants_outside and not instance.is_outside_leaning:
                continue
            for term in family["terms"]:
                seen, negated = self._find(text, str(term).lower())
                if seen:
                    found.append(RuleHit("family_lexicon", str(term), negated, name))

        # 3. Structural regexes.
        for entry in self._regexes:
            if instance.pair_id not in entry.get("applies_to_pairs", []):
                continue
            gate = entry.get("only_when_answer_contains")
            if gate and gate.lower() not in instance.structured_value.lower():
                continue
            if entry["compiled"].search(instance.free_text):
                found.append(
                    RuleHit("regex", entry["id"], False, str(entry.get("explanation", "")))
                )
        return found

    # -- detector interface -----------------------------------------------
    def fit(self, instances: Sequence[PairInstance]) -> "RuleBaseline":
        """No-op. The rules come from config, not from data.

        Present so the evaluation harness can treat every method identically —
        and so that the rule baseline is never accidentally advantaged by
        skipping the cross-validation loop.
        """
        return self

    def _raw_score(self, instance: PairInstance) -> tuple[float, list[RuleHit]]:
        """Weighted evidence before the logistic."""
        found = self.hits(instance)
        raw = 0.0
        for hit in found:
            weight = float(self._w.get(hit.source, 1.0))
            raw += -float(self._w["negation_penalty"]) if hit.negated else weight
        if self.schema.pairs[instance.pair_id].is_high_priority:
            raw += self._priority_boost
        return raw, found

    def score(self, instances: Sequence[PairInstance]) -> np.ndarray:
        """Score instances.

        Args:
            instances: Instances to score.

        Returns:
            Probabilities in [0, 1].
        """
        out = np.zeros(len(instances), dtype=float)
        for index, instance in enumerate(instances):
            raw, found = self._raw_score(instance)
            if not any(not h.negated for h in found):
                out[index] = 0.0
                continue
            out[index] = min(
                self._max_score, 1.0 / (1.0 + math.exp(-(raw - self._offset)))
            )
        return out

    def explain(self, instance: PairInstance) -> str:
        """Say which phrases fired and why they conflict with the answer.

        Args:
            instance: The instance.

        Returns:
            A sentence a reviewer can check against the form in front of them.
            It describes an inconsistency; it never asserts a status.
        """
        found = [h for h in self.hits(instance) if not h.negated]
        if not found:
            return "No lexical evidence of inconsistency."
        pair = self.schema.pairs[instance.pair_id]
        form_ref = self.schema[instance.structured_field].form_ref
        phrases = ", ".join(f"“{h.term}”" for h in found[:3] if h.source != "regex")
        structural = [h.detail for h in found if h.source == "regex" and h.detail]

        parts = [
            f"Question {form_ref} was answered “{instance.structured_value}”, "
            f"but the justification "
        ]
        if phrases:
            parts.append(f"uses {phrases}")
            if structural:
                parts.append(f", and {structural[0]}")
        elif structural:
            parts.append(structural[0])
        parts.append(
            f". For the {pair.ir35_test.replace('_', ' ')} test these point in "
            "opposite directions. Worth a reviewer's eye."
        )
        return "".join(parts)
