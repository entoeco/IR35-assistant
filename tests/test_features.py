"""Pair instances, splits and folds.

The split guarantees are what make every downstream number meaningful, so they
are asserted rather than assumed: nothing may leak from train to test, and the
unit-disjoint split must really hold out the phrasing it claims to.
"""

from __future__ import annotations

import pytest

from src.features.dataset import build_instances, instance_summary, load_corpus
from src.features.splits import group_key, make_folds, make_split
from src.ingest.schema_loader import load_schema

DATA = "data/synthetic"


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def instances(schema):
    records, truth, states = load_corpus(DATA)
    return build_instances(schema, records, truth, states)


# --- dataset -----------------------------------------------------------------

def test_corpus_files_are_aligned() -> None:
    """A silent misalignment would attach every label to the wrong record."""
    records, truth, states = load_corpus(DATA)
    assert len(records) == len(truth) == len(states)
    assert [r["record_id"] for r in records] == [t["record_id"] for t in truth]


def test_missing_corpus_raises_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)


def test_one_instance_per_answered_pair(instances, schema) -> None:
    """Skipped questions produce no instance — a reviewer never sees them, and
    counting them would deflate every rate reported."""
    for instance in instances[:200]:
        assert instance.structured_value
        assert instance.pair_id in schema.pairs


def test_labels_match_the_ground_truth_count(instances) -> None:
    records, truth, states = load_corpus(DATA)
    expected = sum(t["n_contradictions"] for t in truth)
    assert sum(i.label for i in instances) == expected


def test_positive_rate_is_the_operating_regime(instances) -> None:
    """Roughly one contradiction per thirty completed boxes. Every metric in
    Phase 2 is chosen because this number is small."""
    summary = instance_summary(instances)
    assert 0.003 < summary["positive_rate"] < 0.02
    assert summary["instances_per_record"] > 20


def test_positive_instances_carry_type_and_subtlety(instances) -> None:
    for instance in instances:
        if instance.label == 1:
            assert instance.contradiction_type
            assert instance.subtlety in (1, 2, 3)
        else:
            assert instance.contradiction_type is None


def test_no_outcome_field_leaks_into_an_instance(instances) -> None:
    """The status label is carried for reporting, never as a model input.

    Detectors receive ``structured_value`` and ``free_text``; ``ir35_label`` is
    on the instance for breakdowns only. This test pins the fields a detector
    can reach.
    """
    instance = instances[0]
    model_visible = {"structured_value", "free_text", "is_outside_leaning", "pair_id"}
    assert model_visible <= set(vars(instance))
    assert instance.ir35_label in ("inside", "outside", "undetermined")


def test_blank_justifications_are_kept_and_flagged(instances) -> None:
    """Dropping them would remove the hardest negatives from the evaluation."""
    blanks = [i for i in instances if i.anomaly == "blank"]
    assert blanks
    for instance in blanks:
        assert not instance.has_usable_text


# --- splits ------------------------------------------------------------------

def test_record_split_keeps_submissions_whole(instances) -> None:
    split = make_split(instances, "record")
    assert not ({i.record_id for i in split.train} & {i.record_id for i in split.test})


def test_unit_split_holds_out_the_phrasing(instances) -> None:
    """The guarantee the honest evaluation rests on: no test justification's
    underlying proposition appears anywhere in training."""
    split = make_split(instances, "unit")
    train_units = {i.unit_key for i in split.train if i.unit_key}
    test_units = {i.unit_key for i in split.test if i.unit_key}
    assert not (train_units & test_units)


def test_both_splits_have_positives_on_each_side(instances) -> None:
    """At a 0.7% base rate an unstratified split can leave a test set with too
    few positives to estimate recall from at all."""
    for kind in ("record", "unit"):
        split = make_split(instances, kind)
        assert split.summary["train_positives"] > 20
        assert split.summary["test_positives"] > 5


def test_unknown_split_kind_raises(instances) -> None:
    with pytest.raises(ValueError):
        make_split(instances, "something_else")  # type: ignore[arg-type]


def test_splits_are_reproducible(instances) -> None:
    first = make_split(instances, "record", seed=7)
    second = make_split(instances, "record", seed=7)
    assert [i.record_id for i in first.test] == [i.record_id for i in second.test]


# --- folds -------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["record", "unit"])
def test_folds_cover_every_instance_exactly_once(instances, kind) -> None:
    """Out-of-fold scoring only works if each instance is held out once."""
    folds = make_folds(instances, kind)
    held_out = [index for _, test in folds for index in test]
    assert sorted(held_out) == list(range(len(instances)))


@pytest.mark.parametrize("kind", ["record", "unit"])
def test_no_group_spans_train_and_test_within_a_fold(instances, kind) -> None:
    folds = make_folds(instances, kind)
    for train, test in folds:
        train_keys = {group_key(instances[i], kind) for i in train}
        test_keys = {group_key(instances[i], kind) for i in test}
        assert not (train_keys & test_keys)


@pytest.mark.parametrize("kind", ["record", "unit"])
def test_every_fold_has_positives_on_both_sides(instances, kind) -> None:
    """A fold with no held-out positives contributes nothing to recall and
    silently drags the average; a fold with no training positives cannot fit."""
    for train, test in make_folds(instances, kind):
        assert sum(instances[i].label for i in test) > 0
        assert sum(instances[i].label for i in train) > 0


def test_folds_are_reproducible(instances) -> None:
    assert make_folds(instances, "record", seed=3) == make_folds(instances, "record", seed=3)
