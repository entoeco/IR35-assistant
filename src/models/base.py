"""The detector interface every method implements.

One interface for the rule baseline, the TF-IDF baseline, the Phase 3 NLI model
and the few-shot LLM. Two reasons that matters here beyond tidiness:

* **The method comparison has to be honest.** If each method were evaluated
  through its own code path, differences in scoring, thresholding or handling of
  blank text would contaminate the comparison. One interface, one evaluation
  harness, one set of metrics.
* **Phase 5 has to show a reason, not a number.** Every detector must produce a
  human-readable explanation for every flag it raises, because the tool is
  decision support: a reviewer who cannot see why a field was flagged cannot
  accept or dismiss the flag on any informed basis, and a confidence score on
  its own is not a reason.

Scores are probabilities of contradiction in [0, 1]. Thresholding is the
evaluation harness's job, not the detector's — thresholds are chosen for recall
against a stated reviewer-burden budget, and that choice belongs in one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.features.dataset import PairInstance

__all__ = ["Flag", "ContradictionDetector"]


@dataclass(frozen=True)
class Flag:
    """One raised flag, as a reviewer sees it.

    Attributes:
        record_id: Submission.
        pair_id: Which contradiction pair fired.
        field_id: The rationale field to show the reviewer.
        form_ref: The question number as printed on the form, e.g. ``"4.5"``.
        ir35_test: Which employment-status test this bears on.
        score: Model confidence in [0, 1].
        explanation: Why this was flagged, in a sentence a non-specialist can
            act on. Never a bare score.
    """

    record_id: str
    pair_id: str
    field_id: str
    form_ref: str
    ir35_test: str
    score: float
    explanation: str


class ContradictionDetector(ABC):
    """Base class for every contradiction-detection method.

    Subclasses implement ``fit``, ``score`` and ``explain``. A detector that
    needs no training (the rule baseline) still implements ``fit`` as a no-op,
    so the evaluation harness can treat all methods identically.
    """

    #: Short identifier used in reports and result tables.
    name: str = "detector"

    #: One-line description of the method, printed in the comparison table.
    description: str = ""

    @abstractmethod
    def fit(self, instances: Sequence[PairInstance]) -> "ContradictionDetector":
        """Train on labelled instances.

        Args:
            instances: Training instances, each carrying its ``label``.

        Returns:
            ``self``, so training can be chained.
        """

    @abstractmethod
    def score(self, instances: Sequence[PairInstance]) -> np.ndarray:
        """Score instances for contradiction.

        Args:
            instances: Instances to score.

        Returns:
            An array of probabilities in [0, 1], aligned with ``instances``.
            Higher means more likely to be a contradiction.
        """

    @abstractmethod
    def explain(self, instance: PairInstance) -> str:
        """Say why this instance would be flagged.

        Args:
            instance: The instance.

        Returns:
            A sentence a reviewer can act on. Must not be a bare score, and must
            not assert a determination — the tool never issues one.
        """

    def flag(
        self, instance: PairInstance, score: float, schema
    ) -> Flag:
        """Render a scored instance as a reviewer-facing flag.

        Args:
            instance: The instance.
            score: Its score.
            schema: The loaded data contract, for the question's form reference.

        Returns:
            A ``Flag``.
        """
        return Flag(
            record_id=instance.record_id,
            pair_id=instance.pair_id,
            field_id=instance.free_text_field,
            form_ref=str(schema[instance.structured_field].form_ref or ""),
            ir35_test=instance.ir35_test,
            score=float(score),
            explanation=self.explain(instance),
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
