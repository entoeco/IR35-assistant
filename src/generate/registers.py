"""Slot filling and register realisation.

The generator separates *what a justification says* (a content unit) from *how
it is written* (a register). This module owns the second half.

Why the separation matters for Phase 4
--------------------------------------
The style-invariance test re-renders one saved record state in all three
registers and asserts the detector produces the same flags. That test is only
meaningful if the three renderings are known to carry identical semantics. If
verbose, terse and hedged variants had each been authored by hand, a difference
in the detector's output could be a difference in what they said rather than a
sensitivity to surface form, and the test would prove nothing.

The realiser therefore only ever *wraps* a proposition. It adds openers,
closers and elaborating clauses, and it softens modal verbs. It never negates,
never drops a clause, and never introduces a new claim. A planted contradiction
lives in the content unit and survives all three renderings by construction.

Hedged register vs hedged_non_support contradiction
---------------------------------------------------
These are different things and the generator keeps them apart. The *register*
wraps any proposition in epistemic markers — that is style, and it is never
labelled a contradiction. The ``hedged_non_support`` contradiction *type* has
the hedging baked into the content unit: the manager selected a definite option
and then wrote something that fails to support it. A hedged-register rendering
of a supporting unit is not a contradiction; a terse rendering of a
``hedged_non_support`` unit still is.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = ["VocabContext", "RegisterRealiser"]

_SLOT = re.compile(r"\{(\w+)\}")
_MAX_SLOT_PASSES = 4


@dataclass(frozen=True)
class VocabContext:
    """Slot fillers chosen once per record.

    Fixed for the life of a record so that the same engagement is described
    consistently across all 30 justification boxes. A form where every answer
    names a different deliverable reads as noise, and would let a model learn
    "mentions the same project twice" as a spurious coherence cue.
    """

    values: Mapping[str, str]

    def fill(self, text: str) -> str:
        """Resolve ``{slot}`` placeholders, including slots nested in fillers.

        Args:
            text: A content unit, possibly containing placeholders.

        Returns:
            The text with every known placeholder replaced. Unknown placeholders
            are left intact so they surface in tests rather than silently
            producing odd prose.
        """
        result = text
        for _ in range(_MAX_SLOT_PASSES):
            replaced = _SLOT.sub(
                lambda m: str(self.values.get(m.group(1), m.group(0))), result
            )
            if replaced == result:
                break
            result = replaced
        return result

    @classmethod
    def sample(
        cls,
        rng: random.Random,
        common: Mapping[str, Sequence[str]],
        archetype_vocab: Mapping[str, Sequence[str]],
    ) -> "VocabContext":
        """Pick one filler per slot for this record.

        Args:
            rng: Seeded generator, so the context is reproducible.
            common: Slots shared across archetypes.
            archetype_vocab: Slots specific to this engagement type. Overrides
                ``common`` on a name clash.
        """
        values: dict[str, str] = {}
        for source in (common, archetype_vocab):
            for slot, options in source.items():
                if options:
                    values[slot] = rng.choice(list(options))
        return cls(values)


class RegisterRealiser:
    """Renders a content unit in one of three writing registers.

    Example:
        >>> import random
        >>> cfg = {"terse": {"openers": [""], "closers": [""],
        ...                  "add_second_sentence_rate": 0.0}}
        >>> r = RegisterRealiser(cfg)
        >>> r.realise("the worker sets the method", "terse", random.Random(0),
        ...           VocabContext({}))
        'The worker sets the method.'
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Create a realiser.

        Args:
            config: The ``registers`` block of ``text_bank.yaml``.
        """
        self.config = {k: dict(v) for k, v in config.items()}

    @staticmethod
    def _capitalise(text: str) -> str:
        """Upper-case the first alphabetic character, leaving the rest alone.

        ``str.capitalize`` would lower-case the remainder and destroy proper
        nouns and acronyms ("VPN", "Falmer", "FTE").
        """
        for i, ch in enumerate(text):
            if ch.isalpha():
                return text[:i] + ch.upper() + text[i + 1 :]
        return text

    @staticmethod
    def _terminate(text: str) -> str:
        """Ensure the sentence ends with punctuation."""
        stripped = text.rstrip()
        if stripped and stripped[-1] not in ".!?":
            return stripped + "."
        return stripped

    @staticmethod
    def _soften(text: str, softeners: Mapping[str, str]) -> str:
        """Replace definite modals with tentative ones.

        Purely a surface change: "must" becomes "would be expected to", which
        weakens the tone without altering which proposition is asserted.
        """
        result = f" {text} "
        for definite, tentative in softeners.items():
            # First occurrence only. Softening every copula in a sentence
            # produces "this would generally be specialist work and we are not
            # in a position to direct how it would generally be done" — which
            # reads as broken rather than tentative, and would give a model an
            # artefact of the generator to latch onto rather than a register.
            result = result.replace(definite, tentative, 1)
        return result.strip()

    def realise(
        self,
        core: str,
        register: str,
        rng: random.Random,
        vocab: VocabContext,
    ) -> str:
        """Render one content unit in the given register.

        Args:
            core: The content unit, with placeholders unresolved.
            register: ``"verbose"``, ``"terse"`` or ``"hedged"``.
            rng: Seeded generator.
            vocab: This record's slot fillers.

        Returns:
            A rendered justification. Semantics are preserved: the realiser adds
            framing but never negates, drops or introduces a claim.
        """
        cfg = self.config.get(register, {})
        body = vocab.fill(core)

        if register == "hedged":
            body = self._soften(body, cfg.get("softeners", {}))

        opener = rng.choice(cfg.get("openers") or [""])
        opener = vocab.fill(opener).strip()

        elaborations = cfg.get("elaborations") or []
        if elaborations and rng.random() < 0.5:
            body = f"{body}, {vocab.fill(rng.choice(elaborations))}"

        if opener:
            sentence = f"{opener} {body}"
        else:
            sentence = body
            if register == "terse" and rng.random() < cfg.get("lowercase_start_rate", 0.0):
                # Managers writing tersely often do not capitalise a fragment.
                sentence = self._terminate(sentence)
                return sentence
        sentence = self._terminate(self._capitalise(sentence))

        if rng.random() < cfg.get("add_second_sentence_rate", 0.0):
            closer = vocab.fill(rng.choice(cfg.get("closers") or [""])).strip()
            if closer:
                sentence = f"{sentence} {self._capitalise(closer)}"
                sentence = self._terminate(sentence)
        return sentence

    def realise_narrative(
        self,
        parts: Sequence[str],
        register: str,
        rng: random.Random,
        vocab: VocabContext,
    ) -> str:
        """Render the document-level duties narrative from several parts.

        The duties box is the largest on the form and is not the justification
        for any single answer, so it is built from independent sentences rather
        than from one proposition.

        Args:
            parts: Sentence templates.
            register: Writing register.
            rng: Seeded generator.
            vocab: This record's slot fillers.

        Returns:
            The rendered narrative.
        """
        sentences = [self._terminate(self._capitalise(vocab.fill(p))) for p in parts]
        if register == "terse":
            sentences = sentences[: max(1, len(sentences) - 1)]
        return " ".join(sentences)
