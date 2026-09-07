#!/usr/bin/env python3
"""Phase 3: evaluate the NLI and few-shot detectors against the Phase 2 baselines.

Runs through the same harness, the same splits and the same metrics as Phase 2,
so the comparison is like-for-like. Which backend runs is decided by
``config/model.yaml`` (constraint 2) and can be overridden here with
``--backend``.

If the selected backend cannot run — no cached weights, no API key — this script
says so and stops. It does not fall back to the mock and present the result as
an evaluation, because a plumbing check dressed up as a finding is worse than no
finding at all. Run it with ``--backend mock`` deliberately to verify the
pipeline; the output is then labelled as a verification run and carries no
metrics table.

Usage:
    python scripts/run_phase3.py                          # backend from config
    python scripts/run_phase3.py --backend mock           # plumbing verification
    python scripts/run_phase3.py --limit-records 60       # cheap smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate.harness import cross_val_scores  # noqa: E402
from src.evaluate.metrics import EvaluationResult, evaluate  # noqa: E402
from src.features.dataset import build_instances, instance_summary, load_corpus  # noqa: E402
from src.ingest.schema_loader import (  # noqa: E402
    CONFIG_DIR,
    load_pipeline_config,
    load_schema,
    load_yaml,
)
from src.models.backend import BackendUnavailable, get_backend  # noqa: E402
from src.models.base import ContradictionDetector  # noqa: E402
from src.models.baseline_rules import RuleBaseline  # noqa: E402
from src.models.baseline_tfidf import TfidfBaseline  # noqa: E402
from src.models.fewshot import FewShotDetector  # noqa: E402
from src.models.nli import NliDetector  # noqa: E402
from src.utils.logging import LoggingConfig, get_logger  # noqa: E402

SPLITS = ("record", "unit")


def fmt(value: float, places: int = 3) -> str:
    """Format a float, or an em dash for NaN."""
    return "—" if value != value else f"{value:.{places}f}"


def recall_row(name: str, breakdown: Mapping[str, Mapping[str, Any]], keys: list[str]) -> str:
    """One table row of per-level recall."""
    cells = []
    for key in keys:
        entry = breakdown.get(key)
        cells.append(f"{entry['recall']:.2f} ({entry['caught']}/{entry['n']})" if entry else "—")
    return f"| {name} | " + " | ".join(cells) + " |"


def build_report(
    results: dict[tuple[str, str], EvaluationResult],
    phase2: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
    backend_name: str,
    backend_describe: Mapping[str, Any],
    runtimes: Mapping[str, float],
    target_recall: float,
    is_verification: bool,
) -> str:
    """Render the Phase 3 report as Markdown."""
    lines: list[str] = []
    add = lines.append
    methods = sorted({m for _, m in results})

    add("# Phase 3 — Transformer and Few-Shot Comparison")
    add("")
    if is_verification:
        add("> **VERIFICATION RUN, NOT A RESULT.** This run used the deterministic "
            "mock backend. It proves the pipeline end to end — premise construction, "
            "batching, score mapping, thresholding, reporting — but the mock returns "
            "canned distributions keyed on trigger phrases, so **no number below is "
            "evidence about model performance.** See `docs/phase3_status.md` for why "
            "the real run has not happened yet and how to produce it.")
        add("")
    add(f"Backend: `{backend_name}` — {backend_describe.get('model') or 'n/a'} "
        f"({'local' if backend_describe.get('is_local') else 'hosted'}).")
    add("")
    add("**This is decision support.** Every method produces a human-readable reason "
        "with every flag, and no method is asked for, or capable of producing, an "
        "employment status determination.")
    add("")

    add("## Results")
    add("")
    for split in SPLITS:
        rows = [(m, results[(split, m)]) for m in methods if (split, m) in results]
        if not rows:
            continue
        add(f"### `{split}` split")
        add("")
        add("| Method | PR-AUC | ROC-AUC | Macro-F1 | Brier | Threshold | Recall | "
            "Precision | Flags/form | Missed |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for name, result in rows:
            point = result.chosen
            add(
                f"| `{name}` | {fmt(result.pr_auc)} | {fmt(result.roc_auc)} | "
                f"{fmt(result.macro_f1)} | {fmt(result.brier, 4)} | "
                f"{point.threshold:.3f} | **{point.recall:.2f}** | "
                f"{point.precision:.3f} | {point.flags_per_record:.2f} | "
                f"{point.n_false_negatives} |"
            )
        add("")
        for name, result in rows:
            for note in result.notes:
                add(f"- `{name}`: {note}")
        add("")

    if phase2:
        add("### Phase 2 baselines, for comparison")
        add("")
        add("| Method | Split | PR-AUC | Recall | Flags/form |")
        add("|---|---|---|---|---|")
        for row in phase2.get("results", []):
            chosen = row.get("chosen") or {}
            add(
                f"| `{row['method']}` | `{row['split']}` | {fmt(row['pr_auc'])} | "
                f"{chosen.get('recall', float('nan')):.2f} | "
                f"{chosen.get('flags_per_record', float('nan')):.2f} |"
            )
        add("")
        add("The comparison that matters is the `unit` split: TF-IDF collapsed there "
            "because it had memorised the templates, while the rule baseline was "
            "unaffected because it does not learn. Both methods added in this phase are "
            "also split-independent — zero-shot NLI and few-shot prompting fit nothing — "
            "so any gap between them and the rule baseline on the `unit` split is a real "
            "difference in ability to generalise, not a difference in what was memorised.")
        add("")

    types = sorted({k for r in results.values() for k in r.by_type})
    if types:
        add("## Recall by contradiction type (`record` split)")
        add("")
        add("| Method | " + " | ".join(t.replace("_", " ") for t in types) + " |")
        add("|" + "---|" * (len(types) + 1))
        for name in methods:
            result = results.get(("record", name))
            if result:
                add(recall_row(f"`{name}`", result.by_type, types))
        add("")
        add("The two types to watch are `hedged_non_support` and "
            "`named_person_dependency`, where the Phase 2 rule baseline scored 0.20 and "
            "0.40. Those are the specific claims made for the transformer approach: that "
            "a model reasoning over the *relation* between two statements can detect a "
            "justification that fails to support a definite answer, where a keyword list "
            "has no phrase to match.")
        add("")

    subtleties = sorted({k for r in results.values() for k in r.by_subtlety})
    if subtleties:
        add("## Recall by subtlety (`record` split)")
        add("")
        add("| Method | " + " | ".join(f"level {s}" for s in subtleties) + " |")
        add("|" + "---|" * (len(subtleties) + 1))
        for name in methods:
            result = results.get(("record", name))
            if result:
                add(recall_row(f"`{name}`", result.by_subtlety, subtleties))
        add("")

    add("## Runtime")
    add("")
    add("| Method | Total inference time |")
    add("|---|---|")
    for name, seconds in runtimes.items():
        add(f"| `{name}` | {seconds:.1f}s |")
    add("")
    add("Recorded because the deployment assumption is local-only. A method that needs a "
        "GPU, or that costs a hosted API call per field per submission, is a different "
        "proposition for the University than one that runs on a laptop — and that is a "
        "procurement and Information Governance question as much as an accuracy one.")
    add("")

    add("## Dataset")
    add("")
    add(f"- {summary['n_instances']:,} instances across {summary['n_records']} submissions")
    add(f"- {summary['n_positive']} planted contradictions ({summary['positive_rate']:.2%})")
    add(f"- Recall target {target_recall:.0%}")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run Phase 3 and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None, help="override config/model.yaml")
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--limit-records", type=int, default=None,
        help="score only the first N submissions — for a cheap smoke run",
    )
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "synthetic")
    parser.add_argument(
        "--skip-fewshot", action="store_true",
        help="NLI only; the few-shot arm is the expensive one",
    )
    args = parser.parse_args()

    schema = load_schema()
    model_config = load_yaml(CONFIG_DIR / "model.yaml")
    pipeline_config = load_pipeline_config()
    log = get_logger("phase3", LoggingConfig.from_mapping(pipeline_config.get("logging")))

    try:
        backend = get_backend(model_config, override=args.backend, logger=log)
    except (KeyError, BackendUnavailable) as exc:
        print(f"Cannot build backend: {exc}", file=sys.stderr)
        return 2

    ok, message = backend.health_check()
    is_verification = backend.name == "mock"
    if not ok:
        print(f"Backend '{backend.name}' is not usable: {message}", file=sys.stderr)
        print(
            "\nRun `python scripts/fetch_models.py` where huggingface.co is reachable, "
            "or run this script with `--backend mock` for a pipeline verification "
            "(which produces no evidence about model performance).",
            file=sys.stderr,
        )
        return 2
    print(f"Backend '{backend.name}': {message}")

    records, truth, states = load_corpus(args.data)
    if args.limit_records:
        records = records[: args.limit_records]
        truth = truth[: args.limit_records]
        states = states[: args.limit_records]
    instances = build_instances(schema, records, truth, states)
    summary = instance_summary(instances)
    log.event("phase3.dataset_built", meta=summary)

    rule_config = load_yaml(CONFIG_DIR / "rule_baseline.yaml")
    factories: dict[str, Callable[[], ContradictionDetector]] = {
        "nli_cross_encoder": lambda: NliDetector(schema, backend, model_config["nli"]),
    }
    if not args.skip_fewshot:
        factories["fewshot_llm"] = lambda: FewShotDetector(
            schema, backend, model_config["fewshot"]
        )
    # Re-run the rule baseline in the same process so the comparison table is
    # generated from one run rather than stitched from two.
    factories["rules_as_authored"] = lambda: RuleBaseline(schema, rule_config)
    factories["tfidf_logreg"] = lambda: TfidfBaseline(schema)

    results: dict[tuple[str, str], EvaluationResult] = {}
    runtimes: dict[str, float] = {}
    for split in SPLITS:
        for name, factory in factories.items():
            scored = cross_val_scores(factory, instances, split, args.folds, logger=log)
            result = evaluate(name, split, instances, scored.scores, args.target_recall)
            results[(split, name)] = result
            runtimes[name] = runtimes.get(name, 0.0) + scored.fit_seconds
            print(
                f"{split:7s} {name:22s} PR-AUC={fmt(result.pr_auc)} "
                f"recall={result.chosen.recall:.2f} "
                f"flags/form={result.chosen.flags_per_record:.2f}"
            )

    phase2_path = REPO_ROOT / "docs" / "reports" / "phase2_results.json"
    phase2 = json.loads(phase2_path.read_text()) if phase2_path.exists() else None

    reports_dir = REPO_ROOT / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_verification" if is_verification else ""
    report_path = reports_dir / f"phase3_report{suffix}.md"
    report_path.write_text(
        build_report(
            results, phase2, summary, backend.name, backend.describe(),
            runtimes, args.target_recall, is_verification,
        ),
        encoding="utf-8",
    )
    (reports_dir / f"phase3_results{suffix}.json").write_text(
        json.dumps(
            {
                "backend": backend.describe(),
                "is_verification_run": is_verification,
                "dataset": summary,
                "target_recall": args.target_recall,
                "results": [r.as_dict() for r in results.values()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.event("phase3.completed", meta={"report": str(report_path)})
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
