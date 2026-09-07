"""Turn ESQ records into the detector's unit of prediction.

THE UNIT OF PREDICTION
A ``PairInstance`` is one (record, contradiction-pair) combination: a structured
answer, the free-text justification sitting beside it, and whether a
contradiction was planted there. This is the right granularity because it is
what a reviewer acts on — they accept or dismiss a flag on *one field of one
submission*, not a verdict on the whole form.

THE CLASS BALANCE IS THE PROBLEM
A manager fills roughly thirty justification boxes. In the Phase 1 corpus, 70 of
them across 350 submissions carry a contradiction — a base rate near 0.7%. That
is not a flaw in the data; it is the operating regime. Any metric that ignores
it flatters the system: a detector that never fires scores 99.3% accuracy.

Two consequences run through the rest of the phase:

* Accuracy is not reported anywhere. Precision, recall, PR-AUC and — most
  usefully — **flags per submission** are.
* Thresholds are chosen for recall, and the cost of that choice is expressed as
  reviewer burden rather than as a precision number, because "0.08 precision"
  means nothing to the IR35 team and "you will see four flags per form, three of
  which you will dismiss" means everything.

Blank and non-responsive justifications are kept as instances, flagged rather
than dropped. A blank box cannot contain a contradiction, but it is a
completeness failure the form itself cares about, and dropping those rows would
quietly remove the hardest negatives from the evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.ingest.schema_loader import Schema

__all__ = [
    "PairInstance",
    "build_instances",
    "build_instances_single",
    "load_corpus",
    "instance_summary",
]


@dataclass(frozen=True)
class PairInstance:
    """One (structured answer, justification) pair from one submission.

    Attributes:
        record_id: Submission identifier.
        pair_id: Contradiction pair, from ``contradiction_pairs.yaml``.
        structured_field: Field id of the dropdown.
        free_text_field: Field id of the rationale box.
        ir35_test: The test both halves address.
        structured_value: The option the manager selected.
        free_text: The justification as written (post de-identification).
        is_outside_leaning: True when the selected option points away from
            employment. Contradictions here are the audit risk under HMRC's
            reasonable-care duty.
        label: 1 if a contradiction was planted on this pair, else 0.
        contradiction_type: Ground-truth type, or ``None`` for negatives.
        subtlety: Ground-truth subtlety 1-3, or ``None`` for negatives.
        unit_key: The content unit the text was rendered from. This is what the
            unit-disjoint split partitions on, so that test text can be
            guaranteed unseen during training.
        anomaly: ``"blank"`` or ``"non_responsive"`` where one was injected.
        archetype: Engagement type, a Phase 4 cohort variable.
        register: Writing register, the Phase 4 style-invariance variable.
        ir35_label: The record's status label. Never a model input — carried so
            evaluation can break results down by determination.
    """

    record_id: str
    pair_id: str
    structured_field: str
    free_text_field: str
    ir35_test: str
    structured_value: str
    free_text: str
    is_outside_leaning: bool
    label: int
    contradiction_type: str | None
    subtlety: int | None
    unit_key: str
    anomaly: str | None
    archetype: str
    register: str
    ir35_label: str

    @property
    def has_usable_text(self) -> bool:
        """False for blank or near-blank justifications."""
        return len(self.free_text.strip()) >= 15

    def as_dict(self) -> dict[str, Any]:
        """Serialise for reporting."""
        return asdict(self)


def load_corpus(
    data_dir: Path | str,
    *,
    records_file: str = "esq_synthetic_v1.jsonl",
    truth_file: str = "ground_truth_v1.jsonl",
    state_file: str = "generator_state_v1.jsonl",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read the three Phase 1 output files.

    Args:
        data_dir: Directory holding the corpus.
        records_file: De-identified records — the modelling input.
        truth_file: Labels and planted contradictions.
        state_file: Generator state, needed for the content-unit split.

    Returns:
        Records, ground truth and states, index-aligned.

    Raises:
        FileNotFoundError: If any file is missing.
        ValueError: If the three files disagree on length or record order.
    """
    directory = Path(data_dir)

    def read(name: str) -> list[dict]:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"corpus file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    records, truth, states = read(records_file), read(truth_file), read(state_file)
    if not (len(records) == len(truth) == len(states)):
        raise ValueError(
            f"corpus files disagree on length: {len(records)}, {len(truth)}, {len(states)}"
        )
    for rec, tru, sta in zip(records, truth, states):
        if not (rec["record_id"] == tru["record_id"] == sta["record_id"]):
            raise ValueError(f"corpus files are not aligned at {rec['record_id']}")
    return records, truth, states


def build_instances(
    schema: Schema,
    records: Sequence[Mapping[str, Any]],
    truth: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
) -> list[PairInstance]:
    """Build one instance per answered pair per record.

    Args:
        schema: The loaded data contract.
        records: De-identified records.
        truth: Ground-truth rows, aligned with records.
        states: Generator states, aligned with records. Supply the content-unit
            key each justification was rendered from.

    Returns:
        Every (record, pair) instance, in corpus order.

    Note:
        A pair produces an instance only when its structured question was
        actually answered — that is, when the form's routing asked it. Emitting
        instances for skipped questions would add thousands of rows the reviewer
        would never see and would deflate every rate we report.
    """
    instances: list[PairInstance] = []
    for record, row, state in zip(records, truth, states):
        planted = {c["pair_id"]: c for c in row["contradictions"]}
        anomalies = {a["field_id"]: a["kind"] for a in row["anomalies"]}
        units: Mapping[str, str] = state.get("units", {})

        for pair in schema.pairs.values():
            value = record.get(pair.structured)
            if value in (None, ""):
                continue
            contradiction = planted.get(pair.id)
            instances.append(
                PairInstance(
                    record_id=str(record["record_id"]),
                    pair_id=pair.id,
                    structured_field=pair.structured,
                    free_text_field=pair.free_text,
                    ir35_test=pair.ir35_test,
                    structured_value=str(value),
                    free_text=str(record.get(pair.free_text, "") or ""),
                    is_outside_leaning=(value == pair.outside_leaning),
                    label=1 if contradiction else 0,
                    contradiction_type=(
                        contradiction["contradiction_type"] if contradiction else None
                    ),
                    subtlety=contradiction["subtlety"] if contradiction else None,
                    unit_key=str(units.get(pair.id, "")),
                    anomaly=anomalies.get(pair.free_text),
                    archetype=str(row["archetype"]),
                    register=str(row["register"]),
                    ir35_label=str(row["ir35_label"]),
                )
            )
    return instances


def build_instances_single(schema: Schema, record: Mapping[str, Any]) -> list[PairInstance]:
    """Build pair instances for one live record with no ground truth.

    Phase 5's reviewer app scores a submission nobody has labelled — there is
    no ``truth`` row and no generator ``state``, because this record was not
    generated, it was submitted. This is the same per-pair walk as
    :func:`build_instances`, with the fields that only ever come from Phase 1's
    synthetic pipeline (label, contradiction type, subtlety, unit key,
    archetype, register, ir35 label) filled with honest "unknown" values
    rather than omitted, so a detector fitted on the corpus and a live
    submission produce the same shape of instance.

    Args:
        schema: The loaded data contract.
        record: One de-identified, validated record. Field id -> value.

    Returns:
        One instance per pair whose structured question was answered, in
        schema order.
    """
    instances: list[PairInstance] = []
    for pair in schema.pairs.values():
        value = record.get(pair.structured)
        if value in (None, ""):
            continue
        instances.append(
            PairInstance(
                record_id=str(record.get("record_id", "unknown")),
                pair_id=pair.id,
                structured_field=pair.structured,
                free_text_field=pair.free_text,
                ir35_test=pair.ir35_test,
                structured_value=str(value),
                free_text=str(record.get(pair.free_text, "") or ""),
                is_outside_leaning=(value == pair.outside_leaning),
                label=0,
                contradiction_type=None,
                subtlety=None,
                unit_key="",
                anomaly=None,
                archetype=str(record.get("archetype", "unknown")),
                register=str(record.get("register", "unknown")),
                ir35_label="unknown",
            )
        )
    return instances


def instance_summary(instances: Iterable[PairInstance]) -> dict[str, Any]:
    """Descriptive counts for the dataset, for the report and for logging.

    Args:
        instances: The built instances.

    Returns:
        Counts and rates. Metadata only — no free text — so it is safe to log
        under constraint 4.
    """
    items = list(instances)
    positives = [i for i in items if i.label == 1]
    total = len(items) or 1
    by_type: dict[str, int] = {}
    by_subtlety: dict[int, int] = {}
    for item in positives:
        if item.contradiction_type:
            by_type[item.contradiction_type] = by_type.get(item.contradiction_type, 0) + 1
        if item.subtlety:
            by_subtlety[item.subtlety] = by_subtlety.get(item.subtlety, 0) + 1
    return {
        "n_instances": len(items),
        "n_records": len({i.record_id for i in items}),
        "n_pairs": len({i.pair_id for i in items}),
        "n_positive": len(positives),
        "positive_rate": round(len(positives) / total, 5),
        "instances_per_record": round(len(items) / max(len({i.record_id for i in items}), 1), 1),
        "n_blank_or_non_responsive": sum(1 for i in items if i.anomaly),
        "n_unusable_text": sum(1 for i in items if not i.has_usable_text),
        "by_contradiction_type": dict(sorted(by_type.items())),
        "by_subtlety": dict(sorted(by_subtlety.items())),
        "n_distinct_units": len({i.unit_key for i in items if i.unit_key}),
    }


def iter_texts(instances: Iterable[PairInstance]) -> Iterator[str]:
    """Yield the free text of each instance. Convenience for vectorisers."""
    for instance in instances:
        yield instance.free_text
