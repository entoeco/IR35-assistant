#!/usr/bin/env python3
"""Generate the Phase 1 synthetic corpus.

Runs the pipeline in its real order — generate, **de-identify**, validate,
write — so that the de-identification stage is exercised on every record rather
than assumed to work (constraint 3).

The corpus written to ``data/synthetic/`` is the DE-IDENTIFIED one. The raw
generator output contains invented names, addresses and rates; keeping the
de-identified version as the canonical modelling input means no downstream
phase can accidentally train on identity fields, and means the pipeline order is
demonstrated rather than described.

Alongside it, ``generator_state_v1.jsonl`` records the state each record was
rendered from — archetype, register, answers, content-unit choices and planted
contradictions. That is what Phase 4 re-renders in three registers for the
style-invariance test, and it is what makes the corpus reproducible without
shipping three copies of every record.

Usage:
    python scripts/generate_dataset.py [--n 350] [--out data/synthetic]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deidentify.deidentifier import Deidentifier  # noqa: E402
from src.generate.generator import EsqGenerator  # noqa: E402
from src.ingest.schema_loader import (  # noqa: E402
    CONFIG_DIR,
    load_generation_config,
    load_pipeline_config,
    load_schema,
    load_yaml,
)
from src.ingest.validate import RecordValidator  # noqa: E402
from src.models.cest_rules import CestRuleEngine  # noqa: E402
from src.utils.logging import LoggingConfig, get_logger  # noqa: E402


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows as newline-delimited JSON.

    Keys are sorted so that a regenerated file is byte-identical when the data
    is identical — which is what makes the reproducibility test a real test
    rather than a comparison of dict ordering.

    Args:
        path: Destination file.
        rows: Records to write.

    Returns:
        The number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_report(
    records: list[Mapping[str, Any]],
    truth: list[Mapping[str, Any]],
    validations: list[Any],
    deid_reports: list[Any],
    gen_cfg: Mapping[str, Any],
    schema: Any,
) -> str:
    """Render the dataset report as Markdown.

    Args:
        records: De-identified records.
        truth: Ground-truth rows.
        validations: Validation results, index-aligned with records.
        deid_reports: De-identification reports, index-aligned.
        gen_cfg: The generation config actually used.
        schema: The loaded data contract.

    Returns:
        Markdown text.
    """
    n = len(records)
    labels = collections.Counter(t["ir35_label"] for t in truth)
    archetypes = collections.Counter(t["archetype"] for t in truth)
    registers = collections.Counter(t["register"] for t in truth)
    contradictions = [c for t in truth for c in t["contradictions"]]
    ctypes = collections.Counter(c["contradiction_type"] for c in contradictions)
    subtlety = collections.Counter(c["subtlety"] for c in contradictions)
    tests = collections.Counter(c["ir35_test"] for c in contradictions)
    pairs_hit = collections.Counter(c["pair_id"] for c in contradictions)
    anomalies = collections.Counter(a["kind"] for t in truth for a in t["anomalies"])
    n_with = sum(1 for t in truth if t["has_contradiction"])
    targets = gen_cfg["targets"]

    free_text_lengths = [
        len(str(record.get(f.id, "")))
        for record in records
        for f in schema.of_type(__import__("src.ingest.schema_loader", fromlist=["FieldType"]).FieldType.FREE_TEXT)
        if record.get(f.id)
    ]

    def pct(count: int) -> str:
        return f"{count / n:.1%}"

    def in_range(value: float, bounds: list[float]) -> str:
        return "PASS" if bounds[0] <= value <= bounds[1] else "OUT OF RANGE"

    lines: list[str] = []
    add = lines.append
    add("# Phase 1 — Synthetic Dataset Report")
    add("")
    add(f"- **Dataset version:** `{gen_cfg['dataset_version']}`")
    add(f"- **Seed:** `{gen_cfg['seed']}`")
    add(f"- **Records:** {n}")
    add("- **Provenance:** generated from `config/generation.yaml`, "
        "`config/text_bank.yaml` and `config/ir35_weights.yaml`. "
        "Regenerating from the same configs reproduces this corpus byte-for-byte.")
    add("")
    add("All content is synthetic (constraint 1). Every name, address, company, "
        "rate and email is drawn from the invented pools in the generation config.")
    add("")

    add("## Acceptance against targets")
    add("")
    add("| Measure | Target | Actual | |")
    add("|---|---|---|---|")
    inside = labels["inside"] / n
    undet = labels["undetermined"] / n
    contra = n_with / n
    add(f"| Inside rate | {targets['inside_rate'][0]:.0%}–{targets['inside_rate'][1]:.0%} | {inside:.1%} | {in_range(inside, targets['inside_rate'])} |")
    add(f"| Undetermined rate | {targets['undetermined_rate'][0]:.0%}–{targets['undetermined_rate'][1]:.0%} | {undet:.1%} | {in_range(undet, targets['undetermined_rate'])} |")
    add(f"| Records with a contradiction | {targets['contradiction_record_rate'][0]:.0%}–{targets['contradiction_record_rate'][1]:.0%} | {contra:.1%} | {in_range(contra, targets['contradiction_record_rate'])} |")
    add("")

    add("## Status labels")
    add("")
    add("| Label | n | share |")
    add("|---|---|---|")
    for label in ("inside", "outside", "undetermined"):
        add(f"| {label} | {labels[label]} | {pct(labels[label])} |")
    add("")
    add("`undetermined` is the rule engine abstaining, not a failure: it is the "
        "'unable to determine' outcome CEST itself returns. These records are "
        "excluded from status-classifier training and kept for reviewer-facing "
        "evaluation.")
    add("")

    add("## Engagement archetypes")
    add("")
    add("| Archetype | n | inside | outside | undetermined |")
    add("|---|---|---|---|---|")
    for name in gen_cfg["archetype_mix"]:
        subset = [t for t in truth if t["archetype"] == name]
        counts = collections.Counter(t["ir35_label"] for t in subset)
        size = len(subset) or 1
        add(
            f"| {name} | {len(subset)} | {counts['inside'] / size:.1%} | "
            f"{counts['outside'] / size:.1%} | {counts['undetermined'] / size:.1%} |"
        )
    add("")
    add("The spread is the point. Visiting lecturers skew inside — timetabled, "
        "on-site, repeated year on year — and are expected to be where the "
        "detector performs worst. That is a Phase 4 finding to report, not a "
        "defect to design away.")
    add("")

    add("## Writing registers")
    add("")
    add("| Register | n | share |")
    add("|---|---|---|")
    for name, count in sorted(registers.items()):
        add(f"| {name} | {count} | {pct(count)} |")
    add("")
    if free_text_lengths:
        add(f"Free-text length: median {int(statistics.median(free_text_lengths))} "
            f"characters, 95th percentile "
            f"{int(sorted(free_text_lengths)[int(0.95 * len(free_text_lengths))])}, "
            f"max {max(free_text_lengths)}.")
        add("")

    add("## Planted contradictions")
    add("")
    add(f"{len(contradictions)} contradictions across {n_with} records "
        f"({pct(n_with)} of the corpus).")
    add("")
    add("### By type")
    add("")
    add("| Contradiction type | n |")
    add("|---|---|")
    for name, count in sorted(ctypes.items(), key=lambda kv: -kv[1]):
        add(f"| {name} | {count} |")
    add("")
    add("**The type mix is stratified by design, not sampled.** Independent "
        "per-record sampling produced `named_person_dependency` once in 350 "
        "records, which makes per-type recall unmeasurable — and per-type "
        "precision and recall is Phase 4's headline deliverable. The corpus "
        "therefore deals a balanced pool of target types across the records "
        "selected to carry contradictions.")
    add("")
    add("The trade-off must be stated in the evaluation: per-type recall is "
        "estimable, but the type *prior* is a property of the experiment design "
        "and is not evidence about how contradictions are distributed in real "
        "submissions. No aggregate detection rate from this corpus should be "
        "reported as a field estimate.")
    add("")
    add("### By subtlety")
    add("")
    add("| Level | Meaning | n |")
    add("|---|---|---|")
    meanings = {
        1: "blatant — the text plainly says the opposite",
        2: "requires reading — a condition, veto or reimbursement flips it",
        3: "requires domain knowledge — the form's own exclusions, or case law",
    }
    for level in (1, 2, 3):
        add(f"| {level} | {meanings[level]} | {subtlety[level]} |")
    add("")
    add("### By IR35 test")
    add("")
    add("| Test | n |")
    add("|---|---|")
    for name, count in sorted(tests.items(), key=lambda kv: -kv[1]):
        add(f"| {name} | {count} |")
    add("")
    undermining = sum(1 for c in contradictions if c["undermines_outside_leaning"])
    add(f"{undermining} of {len(contradictions)} contradictions undermine an "
        "outside-leaning answer. Those are the audit risk under HMRC's "
        "reasonable-care duty — a justification that quietly guts the answer "
        "supporting an *outside* determination — and Phase 4 reports recall on "
        "them separately.")
    add("")
    add(f"Pair coverage: {len(pairs_hit)} of {len(schema.pairs)} pairs carry at "
        "least one contradiction. Pairs with fewer than three instances are not "
        "separately evaluable; Phase 4 reports per *type* and per *test*, which "
        "have workable counts, rather than per pair.")
    add("")

    add("## Anomalies (not contradictions)")
    add("")
    add("| Kind | n |")
    add("|---|---|")
    for name, count in sorted(anomalies.items()):
        add(f"| {name} | {count} |")
    add("")
    add("Kept separately labelled. A blank rationale is a completeness failure "
        "the form itself cares about — 'Your ESQ will be rejected if not all "
        "fields have been completed' — and a reviewer needs to see it, but "
        "scoring it as a contradiction would flatter the detector.")
    add("")

    add("## Pipeline verification")
    add("")
    invalid = [v for v in validations if not v.is_valid]
    warnings = collections.Counter(
        f.code for v in validations for f in v.findings if f.severity.value == "warning"
    )
    add(f"- Schema validation: **{n - len(invalid)}/{n} records valid** "
        "(no value outside its declared domain, no outcome leakage).")
    if warnings:
        add("- Warnings raised (expected — these are the injected anomalies):")
        for code, count in sorted(warnings.items(), key=lambda kv: -kv[1]):
            add(f"  - `{code}`: {count}")
    entities = sum(r.total_entities for r in deid_reports)
    with_entities = sum(1 for r in deid_reports if r.total_entities)
    add(f"- De-identification: **{entities} entities removed** from free text "
        f"across {with_entities} records, plus field-level treatment on every "
        "record. The stage is exercised on the whole corpus rather than assumed.")
    add("")

    add("## Files")
    add("")
    add("| File | Contents |")
    add("|---|---|")
    add("| `esq_synthetic_v1.jsonl` | De-identified records. The canonical modelling input. |")
    add("| `ground_truth_v1.jsonl` | Status labels with full score breakdown, planted contradictions, anomalies. |")
    add("| `generator_state_v1.jsonl` | Per-record generator state. Re-render in any register for the Phase 4 style-invariance test. |")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Generate, de-identify, validate and write the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="record count override")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    schema = load_schema()
    gen_cfg = load_generation_config()
    pipeline_cfg = load_pipeline_config()
    text_bank = load_yaml(CONFIG_DIR / "text_bank.yaml")

    log = get_logger("pipeline", LoggingConfig.from_mapping(pipeline_cfg.get("logging")))
    engine = CestRuleEngine(schema)
    generator = EsqGenerator(schema, gen_cfg, text_bank, engine)
    deidentifier = Deidentifier(schema, pipeline_cfg.get("deidentification"), log)
    validator = RecordValidator(schema, gen_cfg.get("validation"))

    states, raw_records, truth = generator.generate(args.n)

    # Pipeline order: de-identify FIRST, then validate, then write.
    records, deid_reports = deidentifier.deidentify_many(raw_records)
    validations = validator.validate_many(records)

    out_dir = args.out or (REPO_ROOT / "data" / "synthetic")
    write_jsonl(out_dir / "esq_synthetic_v1.jsonl", records)
    write_jsonl(out_dir / "ground_truth_v1.jsonl", truth)
    write_jsonl(out_dir / "generator_state_v1.jsonl", [s.as_dict() for s in states])

    report = build_report(records, truth, validations, deid_reports, gen_cfg, schema)
    report_path = REPO_ROOT / gen_cfg["output"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    invalid = [v for v in validations if not v.is_valid]
    log.event(
        "pipeline.completed",
        meta={
            "n_records": len(records),
            "invalid_records": len(invalid),
            "output_dir": str(out_dir),
            "report": str(report_path),
        },
    )
    if invalid and gen_cfg.get("validation", {}).get("fail_on_error", True):
        for result in invalid[:5]:
            print(f"INVALID {result.record_id}: {[f.code for f in result.errors]}")
        return 1
    print(f"Wrote {len(records)} records to {out_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
