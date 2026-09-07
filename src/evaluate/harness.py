"""Cross-validated scoring, so every method is measured the same way.

One code path for every detector. Each instance is scored by a model that never
saw its group — its record on the ``record`` split, its content unit on the
``unit`` split — and the out-of-fold scores are assembled into a single vector
covering the whole corpus.

The reason for out-of-fold scoring rather than a single holdout is arithmetic:
a 30% holdout leaves about twenty positive instances, and dividing those across
seven contradiction types and three subtlety levels leaves cells of two or three.
Cross-validation predicts all seventy out of sample, which is still not a large
number but is enough for the per-type comparison the phase exists to make.

The rule baseline has nothing to fit, so it scores identically inside or outside
the loop. It is run through the loop anyway — a method that skips the harness is
a method being compared on different terms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from src.features.dataset import PairInstance
from src.features.splits import SplitKind, make_folds
from src.models.base import ContradictionDetector
from src.utils.logging import StructuredLogger, get_logger

__all__ = ["CrossValScores", "cross_val_scores"]


@dataclass
class CrossValScores:
    """Out-of-fold scores for one method on one split.

    Attributes:
        method: Detector name.
        split: Split kind used to form the folds.
        scores: Out-of-fold probability for every instance, in corpus order.
        n_folds: Folds used.
        fit_seconds: Total time spent fitting, for the runtime comparison the
            local-only deployment assumption makes relevant.
        fold_positives: Positives held out in each fold, so a fold with none is
            visible rather than silently deflating the average.
        detector: The last fitted detector, kept for producing example
            explanations. Not used for scoring.
    """

    method: str
    split: str
    scores: np.ndarray
    n_folds: int
    fit_seconds: float
    fold_positives: list[int]
    detector: ContradictionDetector | None = None

    def as_dict(self) -> dict[str, Any]:
        """Metadata-only summary, safe to log under constraint 4."""
        return {
            "method": self.method,
            "split": self.split,
            "n_folds": self.n_folds,
            "fit_seconds": round(self.fit_seconds, 2),
            "fold_positives": self.fold_positives,
            "mean_score": round(float(self.scores.mean()), 5),
            "nonzero_scores": int((self.scores > 0).sum()),
        }


def cross_val_scores(
    factory: Callable[[], ContradictionDetector],
    instances: Sequence[PairInstance],
    kind: SplitKind = "record",
    n_folds: int = 5,
    seed: int = 20260907,
    logger: StructuredLogger | None = None,
) -> CrossValScores:
    """Produce an out-of-fold score for every instance.

    Args:
        factory: Callable returning a fresh, unfitted detector. A factory rather
            than an instance so that each fold trains from scratch — reusing one
            object would leak the previous fold's fit into the next.
        instances: All pair instances, in corpus order.
        kind: What the folds keep together.
        n_folds: Number of folds.
        seed: Reproducibility seed.
        logger: Structured logger.

    Returns:
        A ``CrossValScores``.
    """
    log = logger or get_logger("evaluate")
    folds = make_folds(instances, kind, n_folds, seed)
    scores = np.zeros(len(instances), dtype=float)
    fold_positives: list[int] = []
    elapsed = 0.0
    detector: ContradictionDetector | None = None

    for fold, (train_idx, test_idx) in enumerate(folds):
        train = [instances[i] for i in train_idx]
        test = [instances[i] for i in test_idx]
        fold_positives.append(sum(i.label for i in test))

        detector = factory()
        started = time.perf_counter()
        detector.fit(train)
        elapsed += time.perf_counter() - started
        scores[test_idx] = detector.score(test)

        log.event(
            "evaluate.fold_completed",
            meta={
                "method": detector.name,
                "split": kind,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "test_positives": fold_positives[-1],
            },
        )

    result = CrossValScores(
        method=detector.name if detector else "unknown",
        split=kind,
        scores=scores,
        n_folds=n_folds,
        fit_seconds=elapsed,
        fold_positives=fold_positives,
        detector=detector,
    )
    log.event("evaluate.cross_val_completed", meta=result.as_dict())
    return result
