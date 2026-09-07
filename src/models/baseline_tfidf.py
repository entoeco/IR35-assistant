"""TF-IDF + logistic regression baseline.

An honest attempt. The method comparison is only worth reporting if the
statistical baseline is built the way someone would build it if this were the
whole project, so it gets:

* **Word and character n-grams.** Character n-grams matter because the terse
  register drops function words and writes fragments; word n-grams alone are
  brittle across registers, and register robustness is a Phase 4 deliverable.
* **Explicit interaction features.** A contradiction is not a property of the
  text alone — *"we would need to approve any substitute"* is perfectly
  consistent with "Yes, we have a right to reject" and contradicts "No". So
  free-text tokens are crossed with the answer's polarity, giving the linear
  model the conjunction it cannot otherwise represent. Without this the model
  can only learn "these words are suspicious", which is the wrong hypothesis
  class for the task.
* **Class weighting.** At a 0.7% base rate an unweighted logistic regression
  predicts the majority class everywhere. Weights are balanced, and the
  threshold is then chosen by the evaluation harness against a reviewer-burden
  budget rather than left at 0.5.

WHY POOLED AND NOT PER-PAIR
The Phase 1 corpus has 70 positive instances across 31 pairs. Fitting a model
per pair would mean two positives each. One pooled model with pair identity as a
feature is the only defensible choice at this scale, and the report says so.

WHAT THIS MEASURES ON SYNTHETIC DATA
On a record-disjoint split, most test propositions have been seen verbatim in
training, so a strong score partly reflects template memorisation. That is why
``src/features/splits.py`` also provides a content-unit-disjoint split, and why
both are reported. The gap between them is the size of the memorisation effect.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src.features.dataset import PairInstance
from src.ingest.schema_loader import Schema
from src.models.base import ContradictionDetector

__all__ = ["TfidfBaseline"]

_TOKEN = re.compile(r"[a-z][a-z'-]+")


class TfidfBaseline(ContradictionDetector):
    """Pooled logistic regression over TF-IDF and interaction features."""

    name = "tfidf_logreg"
    description = (
        "TF-IDF (word 1-2 grams + char_wb 3-5 grams) with answer-polarity "
        "interaction features, pooled logistic regression, balanced class weights"
    )

    def __init__(
        self,
        schema: Schema,
        *,
        C: float = 1.0,
        max_word_features: int = 30_000,
        max_char_features: int = 30_000,
        seed: int = 20260907,
    ) -> None:
        """Create the baseline.

        Args:
            schema: The loaded data contract. Used only for pair metadata in
                explanations; no field name is hardcoded.
            C: Inverse regularisation strength.
            max_word_features: Cap on the word n-gram vocabulary.
            max_char_features: Cap on the character n-gram vocabulary.
            seed: Reproducibility seed.
        """
        self.schema = schema
        self.C = C
        self.seed = seed
        self._word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_word_features,
            sublinear_tf=True,
            lowercase=True,
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=max_char_features,
            sublinear_tf=True,
            lowercase=True,
        )
        self._interaction = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            sublinear_tf=True,
            lowercase=False,
            token_pattern=r"\S+",
        )
        self._categorical = OneHotEncoder(handle_unknown="ignore")
        self._model = LogisticRegression(
            class_weight="balanced",
            C=C,
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )
        self._fitted = False

    # -- feature construction ---------------------------------------------
    @staticmethod
    def _answer_and_text(instance: PairInstance) -> str:
        """The answer and its justification as one string.

        Keeping the answer in the same field as the text lets word bigrams
        straddle the boundary, which is a cheap way to capture some of the
        conjunction the interaction block handles explicitly.
        """
        return f"ANSWER {instance.structured_value} [SEP] {instance.free_text}"

    @staticmethod
    def _interaction_tokens(instance: PairInstance) -> str:
        """Free-text tokens crossed with the answer's polarity and IR35 test.

        This is the feature that lets a linear model represent "this phrase
        contradicts *this* answer" rather than "this phrase is suspicious". The
        same words carry opposite evidence depending on which way the answer
        leans, and without the cross the model cannot express that at all.
        """
        polarity = "OUT" if instance.is_outside_leaning else "IN"
        tokens = _TOKEN.findall(instance.free_text.lower())
        crossed = [f"{polarity}~{instance.ir35_test}~{token}" for token in tokens]
        return " ".join(crossed)

    def _dense_block(self, instances: Sequence[PairInstance]) -> csr_matrix:
        """Small numeric block: polarity, usability and length.

        Text length is included because register correlates with length, and an
        explicit feature is preferable to the model inferring it from n-gram
        counts where it would be entangled with content.
        """
        rows = []
        for instance in instances:
            length = len(instance.free_text)
            rows.append(
                [
                    1.0 if instance.is_outside_leaning else 0.0,
                    1.0 if instance.has_usable_text else 0.0,
                    min(length, 400) / 400.0,
                    1.0 if instance.anomaly else 0.0,
                ]
            )
        return csr_matrix(np.asarray(rows, dtype=float))

    def _categorical_block(self, instances: Sequence[PairInstance]) -> np.ndarray:
        """Pair identity, IR35 test and register as categorical columns."""
        return np.asarray(
            [[i.pair_id, i.ir35_test, i.structured_value] for i in instances], dtype=object
        )

    def _features(self, instances: Sequence[PairInstance], fit: bool) -> csr_matrix:
        """Build the design matrix.

        Args:
            instances: Instances to featurise.
            fit: True to fit the vectorisers, False to transform only.

        Returns:
            A sparse feature matrix.
        """
        combined = [self._answer_and_text(i) for i in instances]
        texts = [i.free_text for i in instances]
        crossed = [self._interaction_tokens(i) for i in instances]
        categorical = self._categorical_block(instances)

        if fit:
            word = self._word.fit_transform(combined)
            char = self._char.fit_transform(texts)
            inter = self._interaction.fit_transform(crossed)
            cats = self._categorical.fit_transform(categorical)
        else:
            word = self._word.transform(combined)
            char = self._char.transform(texts)
            inter = self._interaction.transform(crossed)
            cats = self._categorical.transform(categorical)

        return hstack([word, char, inter, cats, self._dense_block(instances)]).tocsr()

    # -- detector interface -----------------------------------------------
    def fit(self, instances: Sequence[PairInstance]) -> "TfidfBaseline":
        """Fit vectorisers and the classifier.

        Args:
            instances: Training instances carrying labels.

        Returns:
            ``self``.

        Raises:
            ValueError: If the training set contains only one class, which at a
                0.7% base rate is a real possibility for a small fold and should
                fail loudly rather than silently produce constant scores.
        """
        labels = np.asarray([i.label for i in instances])
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "training fold contains a single class; the split is too small "
                "or not stratified"
            )
        features = self._features(instances, fit=True)
        self._model.fit(features, labels)
        self._fitted = True
        return self

    def score(self, instances: Sequence[PairInstance]) -> np.ndarray:
        """Predicted probability of contradiction.

        Args:
            instances: Instances to score.

        Returns:
            Probabilities in [0, 1].

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("TfidfBaseline.score called before fit")
        features = self._features(instances, fit=False)
        return self._model.predict_proba(features)[:, 1]

    def top_features(self, instance: PairInstance, k: int = 4) -> list[tuple[str, float]]:
        """The features present in this instance that push hardest towards a flag.

        A linear model can say exactly why it fired, which is worth exploiting:
        Phase 5 needs a reason a reviewer can check, not a number.

        Args:
            instance: The instance.
            k: How many features to return.

        Returns:
            ``(feature name, coefficient)`` pairs, most positive first.
        """
        if not self._fitted:
            return []
        names = np.concatenate(
            [
                np.asarray(self._word.get_feature_names_out(), dtype=object),
                np.asarray(self._char.get_feature_names_out(), dtype=object),
                np.asarray(self._interaction.get_feature_names_out(), dtype=object),
                np.asarray(self._categorical.get_feature_names_out(), dtype=object),
                np.asarray(["is_outside", "usable_text", "length", "anomaly"], dtype=object),
            ]
        )
        row = self._features([instance], fit=False).tocoo()
        coefficients = self._model.coef_[0]
        contributions = [
            (str(names[col]), float(coefficients[col] * value))
            for col, value in zip(row.col, row.data)
        ]
        contributions.sort(key=lambda item: -item[1])
        # Character n-grams are unreadable in an explanation; drop them.
        readable = [
            (name, weight)
            for name, weight in contributions
            if weight > 0 and " " not in name[:1] and not name.startswith(" ")
        ]
        return readable[:k]

    def explain(self, instance: PairInstance) -> str:
        """Say which learned features drove the flag.

        Args:
            instance: The instance.

        Returns:
            A reviewer-facing sentence. Describes an inconsistency; never
            asserts a status.
        """
        form_ref = self.schema[instance.structured_field].form_ref
        features = self.top_features(instance)
        if not features:
            return "No learned evidence of inconsistency."
        readable = ", ".join(
            f"“{name.split('~')[-1]}”" for name, _ in features if "~" in name or " " in name or name
        )
        return (
            f"Question {form_ref} was answered “{instance.structured_value}”. "
            f"The strongest signals in the justification for a mismatch on the "
            f"{instance.ir35_test.replace('_', ' ')} test were {readable}. "
            "Worth a reviewer's eye."
        )
