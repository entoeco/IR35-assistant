"""Train/test splits, and why this project needs two of them.

THE PROBLEM WITH ONE SPLIT
The corpus is template-generated. Each contradiction pair has a handful of
authored content units, and 350 records reuse them. Split the corpus by record
and almost every proposition in the test set has already been seen, verbatim,
in training. A TF-IDF model then scores well by memorising template strings, and
the number means nothing about whether it would catch a contradiction phrased in
a way it has not seen — which is the only thing that matters on real
submissions.

This is a limitation of synthetic data, not something to design away. So it gets
measured instead:

* ``record`` split — records disjoint, content units shared. The conventional
  split, and the optimistic one. Reported as the upper bound.
* ``unit`` split — content units disjoint, records may straddle. Test text is
  guaranteed unseen. Reported as the honest generalisation estimate.

The gap between the two is itself a result: it quantifies how much of a model's
apparent performance is template memorisation. A method whose score barely moves
between the splits is generalising; one that collapses was reading the template.

Both splits are stratified so that positives are not concentrated on one side —
at a 0.7% base rate an unstratified split can easily produce a test set with too
few positives to estimate recall from.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from src.features.dataset import PairInstance

__all__ = [
    "Split",
    "split_by_record",
    "split_by_unit",
    "make_split",
    "make_folds",
    "group_key",
]

SplitKind = Literal["record", "unit"]


@dataclass(frozen=True)
class Split:
    """A train/test partition of pair instances.

    Attributes:
        kind: ``"record"`` or ``"unit"``.
        train: Training instances.
        test: Held-out instances.
        seed: Seed used, so the split is reproducible.
        note: One-line statement of what this split does and does not control.
    """

    kind: str
    train: list[PairInstance]
    test: list[PairInstance]
    seed: int
    note: str

    @property
    def summary(self) -> dict[str, Any]:
        """Sizes and positive counts, for the report."""
        return {
            "kind": self.kind,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "train_positives": sum(i.label for i in self.train),
            "test_positives": sum(i.label for i in self.test),
            "test_positive_rate": round(
                sum(i.label for i in self.test) / max(len(self.test), 1), 5
            ),
            "train_records": len({i.record_id for i in self.train}),
            "test_records": len({i.record_id for i in self.test}),
            "shared_units": len(
                {i.unit_key for i in self.train} & {i.unit_key for i in self.test}
            ),
        }


def _stratified_groups(
    groups: dict[str, list[PairInstance]], test_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Assign whole groups to train or test, balancing positives across both.

    Groups carrying at least one positive are shuffled and split separately from
    groups carrying none. Without that, a 0.7% base rate makes it easy to draw a
    test set with too few positives to estimate recall from at all.

    Args:
        groups: Group key -> its instances.
        test_fraction: Share of each stratum to hold out.
        seed: Seed for the shuffle.

    Returns:
        Train keys and test keys.
    """
    rng = random.Random(seed)
    with_positive = sorted(k for k, v in groups.items() if any(i.label for i in v))
    without = sorted(k for k, v in groups.items() if not any(i.label for i in v))

    test: set[str] = set()
    for stratum in (with_positive, without):
        shuffled = list(stratum)
        rng.shuffle(shuffled)
        cut = round(len(shuffled) * test_fraction)
        test.update(shuffled[:cut])
    train = (set(groups) - test)
    return train, test


def split_by_record(
    instances: Sequence[PairInstance], test_fraction: float = 0.3, seed: int = 20260907
) -> Split:
    """Split with records disjoint. The conventional, optimistic split.

    Every pair from a submission lands on the same side, so nothing leaks
    between train and test at the record level. Content units are shared, so
    most test text has been seen in training — which is why this is reported as
    an upper bound rather than as the result.

    Args:
        instances: All pair instances.
        test_fraction: Share of records to hold out.
        seed: Reproducibility seed.

    Returns:
        A ``Split``.
    """
    groups: dict[str, list[PairInstance]] = defaultdict(list)
    for instance in instances:
        groups[instance.record_id].append(instance)
    train_keys, test_keys = _stratified_groups(groups, test_fraction, seed)
    return Split(
        kind="record",
        train=[i for i in instances if i.record_id in train_keys],
        test=[i for i in instances if i.record_id in test_keys],
        seed=seed,
        note=(
            "Records disjoint; content units shared. Test text has mostly been seen "
            "in training, so scores here are an upper bound inflated by template reuse."
        ),
    )


def split_by_unit(
    instances: Sequence[PairInstance], test_fraction: float = 0.3, seed: int = 20260907
) -> Split:
    """Split with content units disjoint. The honest generalisation test.

    Every instance rendered from a given proposition lands on the same side, so
    no test justification's underlying content unit appears in training. A model
    must generalise from the propositions it saw to ones it did not, which is
    the situation on real submissions.

    Records straddle the split. That is the trade-off: record-level style
    (archetype vocabulary, register) can appear on both sides, which could
    flatter a model that learns register rather than meaning. Phase 4's
    style-invariance test is what catches that, and the two analyses are meant
    to be read together.

    Args:
        instances: All pair instances.
        test_fraction: Share of content units to hold out.
        seed: Reproducibility seed.

    Returns:
        A ``Split``.
    """
    groups: dict[str, list[PairInstance]] = defaultdict(list)
    for instance in instances:
        groups[instance.unit_key or f"__empty__{instance.pair_id}"].append(instance)
    train_keys, test_keys = _stratified_groups(groups, test_fraction, seed)

    def key_of(instance: PairInstance) -> str:
        return instance.unit_key or f"__empty__{instance.pair_id}"

    return Split(
        kind="unit",
        train=[i for i in instances if key_of(i) in train_keys],
        test=[i for i in instances if key_of(i) in test_keys],
        seed=seed,
        note=(
            "Content units disjoint; no test proposition appears in training. Records "
            "straddle the split, so record-level style is not controlled — read "
            "alongside the Phase 4 style-invariance results."
        ),
    )


def make_split(
    instances: Sequence[PairInstance],
    kind: SplitKind = "record",
    test_fraction: float = 0.3,
    seed: int = 20260907,
) -> Split:
    """Build a split by name.

    Args:
        instances: All pair instances.
        kind: ``"record"`` or ``"unit"``.
        test_fraction: Share to hold out.
        seed: Reproducibility seed.

    Returns:
        A ``Split``.

    Raises:
        ValueError: If ``kind`` is not a known split.
    """
    if kind == "record":
        return split_by_record(instances, test_fraction, seed)
    if kind == "unit":
        return split_by_unit(instances, test_fraction, seed)
    raise ValueError(f"unknown split kind: {kind!r}")


def group_key(instance: PairInstance, kind: SplitKind) -> str:
    """The value a split keeps together.

    Args:
        instance: The instance.
        kind: ``"record"`` groups by submission; ``"unit"`` groups by the
            content unit the justification was rendered from.

    Returns:
        The group key.

    Raises:
        ValueError: If ``kind`` is not a known split.
    """
    if kind == "record":
        return instance.record_id
    if kind == "unit":
        return instance.unit_key or f"__empty__{instance.pair_id}"
    raise ValueError(f"unknown split kind: {kind!r}")


def make_folds(
    instances: Sequence[PairInstance],
    kind: SplitKind = "record",
    n_folds: int = 5,
    seed: int = 20260907,
) -> list[tuple[list[int], list[int]]]:
    """Build grouped, stratified cross-validation folds.

    WHY CROSS-VALIDATION RATHER THAN ONE HOLDOUT
    A single 30% holdout leaves roughly 20 positive instances. Split those
    across seven contradiction types and three subtlety levels and every cell
    has two or three cases, which is not enough to say anything. Cross-validation
    gives an out-of-fold prediction for **every** instance in the corpus, so the
    per-type and per-subtlety breakdowns are computed over all 70 positives
    rather than a fifth of them.

    Each instance is still predicted by a model that never saw its group, so the
    guarantee that matters — no leakage from train to test — is unchanged.

    Args:
        instances: All pair instances.
        kind: What to keep together, ``"record"`` or ``"unit"``.
        n_folds: Number of folds.
        seed: Reproducibility seed.

    Returns:
        ``n_folds`` pairs of (train indices, test indices) into ``instances``.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        groups[group_key(instance, kind)].append(index)

    labels = {
        key: any(instances[i].label for i in idx) for key, idx in groups.items()
    }
    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    # Deal positive-bearing groups round-robin first so every fold gets some,
    # then the rest. At a 0.7% base rate, random assignment produces folds with
    # no positives at all often enough to matter.
    for stratum in (True, False):
        keys = sorted(k for k, has_positive in labels.items() if has_positive is stratum)
        rng.shuffle(keys)
        for position, key in enumerate(keys):
            assignment[key] = position % n_folds

    folds: list[tuple[list[int], list[int]]] = []
    for fold in range(n_folds):
        test = [i for key, idx in groups.items() if assignment[key] == fold for i in idx]
        train = [i for key, idx in groups.items() if assignment[key] != fold for i in idx]
        folds.append((sorted(train), sorted(test)))
    return folds
