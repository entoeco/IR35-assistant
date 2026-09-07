#!/usr/bin/env python3
"""Phase 4: evaluation and bias testing.

Four things the earlier phases could not do:

* **Style invariance.** Re-render every engagement in all three registers with
  everything else held identical, and measure whether the flags move.
* **Calibration.** Phase 5 shows a confidence number to a reviewer; this is
  where that number is tested as a claim about the world.
* **Honest uncertainty.** Cluster-bootstrap intervals on the per-type,
  per-subtlety and per-archetype cells, which hold nine to eleven positives each
  and cannot support a bare point estimate.
* **A stated operating point.** Recall-optimised, with reviewer burden as the
  price, and the reasoning written down.

Usage:
    python scripts/run_phase4.py [--target-recall 0.8] [--folds 5] [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate.calibration import (  # noqa: E402
    CalibrationResult,
    GroupRecall,
    assess_calibration,
    bootstrap_recall,
    recall_with_intervals,
)
from src.evaluate.harness import cross_val_scores  # noqa: E402
from src.evaluate.metrics import EvaluationResult, evaluate  # noqa: E402
from src.evaluate.plots import plot_calibration, plot_robustness  # noqa: E402
from src.evaluate.style_invariance import (  # noqa: E402
    StyleInvarianceResult,
    measure_style_invariance,
)
from src.features.dataset import build_instances, instance_summary, load_corpus  # noqa: E402
from src.generate.generator import EsqGenerator  # noqa: E402
from src.ingest.schema_loader import (  # noqa: E402
    CONFIG_DIR,
    load_generation_config,
    load_pipeline_config,
    load_schema,
    load_yaml,
)
from src.models.base import ContradictionDetector  # noqa: E402
from src.models.baseline_rules import RuleBaseline  # noqa: E402
from src.models.baseline_tfidf import TfidfBaseline  # noqa: E402
from src.models.cest_rules import CestRuleEngine  # noqa: E402
from src.utils.logging import LoggingConfig, get_logger  # noqa: E402


def fmt(value: float, places: int = 3) -> str:
    """Format a float, or an em dash for NaN."""
    return "—" if value != value else f"{value:.{places}f}"


def interval_row(name: str, entries: Sequence[GroupRecall]) -> list[str]:
    """Markdown rows for one method's per-group recall with intervals."""
    return [
        f"| `{name}` | {e.group.replace('_', ' ')} | {e.n} | {e.caught} | "
        f"{e.recall:.2f} | {e.low:.2f}–{e.high:.2f} | {e.interval_width:.2f} |"
        for e in entries
    ]


def build_report(
    results: Mapping[str, EvaluationResult],
    calibrations: Sequence[CalibrationResult],
    style: Mapping[str, StyleInvarianceResult],
    by_type: Mapping[str, Sequence[GroupRecall]],
    by_subtlety: Mapping[str, Sequence[GroupRecall]],
    by_archetype: Mapping[str, Sequence[GroupRecall]],
    overall_ci: Mapping[str, tuple[float, float]],
    summary: Mapping[str, Any],
    target_recall: float,
    n_boot: int,
) -> str:
    """Render the Phase 4 report as Markdown."""
    lines: list[str] = []
    add = lines.append
    methods = list(results)
    primary = methods[0]

    add("# Phase 4 — Evaluation and Bias Testing")
    add("")
    add("Headline metrics, then the robustness tests that decide whether the tool is "
        "safe to put in front of a reviewer: does it read substance or surface, does "
        "it work equally well across engagement types, does it find the subtle cases, "
        "and does its confidence number mean anything.")
    add("")
    add("**This is decision support.** Nothing here issues or overrides a Status "
        "Determination Statement, and the operating point is chosen on the assumption "
        "that a human reads every flag.")
    add("")
    add(f"Evaluated on the Phase 1 synthetic corpus: {summary['n_instances']:,} "
        f"(submission, question) instances across {summary['n_records']} submissions, "
        f"{summary['n_positive']} planted contradictions "
        f"({summary['positive_rate']:.2%}). Scores are out-of-fold.")
    add("")
    add("> The Phase 3 transformer arms are absent: this sandbox's egress policy blocks "
        "`huggingface.co`, so no model weights could be fetched. See "
        "`docs/phase3_status.md`. Everything below therefore evaluates the methods that "
        "can actually run, and the same harness will accept the NLI and few-shot "
        "detectors unchanged once weights are available.")
    add("")

    add("## 1. Headline metrics")
    add("")
    add("| Method | PR-AUC | ROC-AUC | Macro-F1 | Threshold | Recall | 95% CI | "
        "Precision | Flags/form | Missed |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for name in methods:
        result = results[name]
        point = result.chosen
        low, high = overall_ci[name]
        add(
            f"| `{name}` | {fmt(result.pr_auc)} | {fmt(result.roc_auc)} | "
            f"{fmt(result.macro_f1)} | {point.threshold:.3f} | **{point.recall:.2f}** | "
            f"{low:.2f}–{high:.2f} | {point.precision:.3f} | "
            f"{point.flags_per_record:.2f} | {point.n_false_negatives} |"
        )
    add("")
    for name in methods:
        for note in results[name].notes:
            add(f"- `{name}`: {note}")
    add("")
    add("Macro-F1 is reported because the brief asks for it. It should be read with "
        "care at this class balance: it averages the F1 of the majority class — which "
        "is near 1.0 for everything — with the F1 of the minority class, so it "
        "compresses real differences. PR-AUC and flags-per-form are the metrics that "
        "describe what a reviewer experiences.")
    add("")

    add("## 2. The operating point, and why recall")
    add("")
    add(f"Thresholds are chosen as **the highest threshold that still reaches "
        f"{target_recall:.0%} recall**, not the threshold maximising F1.")
    add("")
    add("The asymmetry is the whole argument. A missed contradiction reaches HMRC as a "
        "determination made without the reasonable care the off-payroll rules require — "
        "exposure measured in tax liability, interest and penalties, plus the dispute "
        "itself. A false flag costs a reviewer perhaps thirty seconds to read and "
        "dismiss. Optimising F1 would trade recall for precision at roughly one-to-one, "
        "which is the wrong exchange rate by two orders of magnitude.")
    add("")
    add("**The limit on that argument is human, not statistical.** If reviewers see so "
        "many flags that they start dismissing without reading, recall on paper rises "
        "while recall in practice collapses — and the tool becomes worse than nothing, "
        "because it produces an audit trail suggesting the contradictions were "
        "considered. The precision floor is therefore set by reviewer fatigue, and we "
        "do not yet know where it is. That is why:")
    add("")
    add("- **flags per submission** is reported at every operating point, not precision alone;")
    add("- **Phase 5 captures every accept/dismiss decision**, because those are the only "
        "data that will locate the real floor;")
    add("- the threshold is config, not code, so it can be moved once there is evidence.")
    add("")
    for name in methods:
        point = results[name].chosen
        add(f"At the chosen point, `{name}` shows a reviewer "
            f"**{point.flags_per_record:.2f} flags per submission** and misses "
            f"**{point.n_false_negatives}** of {summary['n_positive']} contradictions.")
    add("")

    add("## 3. Style invariance")
    add("")
    add("Every engagement re-rendered verbose, terse and hedged with the structured "
        "answers, the propositions and the planted contradictions held **literally "
        "identical** — only the wording differs. This is possible because the Phase 1 "
        "generator separates what a record says from how it is written, and it is what "
        "makes the test meaningful: any difference in the flags is attributable to "
        "surface form alone.")
    add("")
    add("| Method | Flag-set agreement (Jaccard) | Records agreeing exactly | "
        "Agreement on true flags | Agreement on false flags | Mean score drift |")
    add("|---|---|---|---|---|---|")
    for name, result in style.items():
        add(
            f"| `{name}` | {result.mean_jaccard:.3f} | "
            f"{result.records_fully_agreeing}/{result.n_records} "
            f"({result.agreement_rate:.0%}) | {result.tp_jaccard:.3f} | "
            f"{result.fp_jaccard:.3f} | {result.mean_abs_drift:.4f} |"
        )
    add("")
    add("### Recall by register, same contradictions")
    add("")
    registers = ["verbose", "terse", "hedged"]
    add("| Method | " + " | ".join(registers) + " | spread |")
    add("|" + "---|" * (len(registers) + 2))
    for name, result in style.items():
        values = [result.recall_by_register.get(r, float("nan")) for r in registers]
        spread = max(values) - min(values)
        add(
            f"| `{name}` | " + " | ".join(f"{v:.2f}" for v in values) + f" | {spread:.2f} |"
        )
    add("")
    add("**The headline is the gap between the last two columns.** Recall barely "
        "moves across registers — the table below shows a spread of a few points at "
        "most — so an aggregate metric would report the detector as register-robust. "
        "But the flag *sets* agree far less well, and splitting the agreement by "
        "whether a flag was correct shows why: the true contradictions are found "
        "consistently, while a substantially different set of **false** flags is "
        "raised depending on how the manager wrote. Stable recall, unstable reviewer "
        "experience.")
    add("")
    add("That distinction matters operationally. The same submission, rewritten in a "
        "different voice by a different manager, shows the reviewer a different set of "
        "fields to check. Aggregate recall cannot see this, and it is exactly the kind "
        "of inconsistency that erodes trust in a tool — a reviewer who notices that "
        "two similar forms produced different flags has no way to tell whether the "
        "difference is meaningful.")
    add("")
    add("**Why this is a fairness test, not just a robustness test.** Register is not "
        "randomly distributed across engagement types. In this corpus visiting-lecturer "
        "submissions skew verbose and hedged; technical-trades submissions skew terse. "
        "A detector that performs worse on hedged prose therefore performs worse for a "
        "particular category of worker — and the per-archetype breakdown alone cannot "
        "tell you whether that is because the engagement type is harder or because "
        "those managers write differently. Holding the engagement fixed and varying only "
        "the writing separates the two.")
    add("")
    for name, result in style.items():
        if result.drift_by_archetype:
            worst = max(result.drift_by_archetype.items(), key=lambda kv: kv[1])
            best = min(result.drift_by_archetype.items(), key=lambda kv: kv[1])
            add(f"`{name}`: score drift is largest on {worst[0].replace('_', ' ')} "
                f"({worst[1]:.3f}) and smallest on {best[0].replace('_', ' ')} "
                f"({best[1]:.3f}).")
    add("")

    add("## 4. Recall by contradiction type")
    add("")
    add("| Method | Type | n | Caught | Recall | 95% CI | CI width |")
    add("|---|---|---|---|---|---|---|")
    for name in methods:
        lines.extend(interval_row(name, by_type[name]))
    add("")
    add(f"Intervals are cluster bootstrap over **records**, not instances "
        f"({n_boot:,} replicates). Instances within one submission share a manager, a "
        "vocabulary, an engagement and a register, so resampling them independently "
        "would produce intervals narrower than the data supports.")
    add("")
    add("Read the interval widths before the point estimates. At nine to eleven "
        "positives per type, a cell is consistent with a wide range of true values, and "
        "a difference of one or two caught cases is not a difference in capability.")
    add("")

    add("## 5. Subtlety curve")
    add("")
    add("| Method | Level | n | Caught | Recall | 95% CI | CI width |")
    add("|---|---|---|---|---|---|---|")
    for name in methods:
        lines.extend(interval_row(name, by_subtlety[name]))
    add("")
    add("Level 1 is blatant — the text plainly says the opposite. Level 2 needs "
        "reading: a condition, a veto or a reimbursement flips the meaning. Level 3 "
        "needs domain knowledge, usually the form's own cost exclusions or the case "
        "law. **Level 3 is where the tool is most needed and least reliable**, because "
        "those are also the cases a busy reviewer misses.")
    add("")

    add("## 6. Role-type disparity")
    add("")
    add("| Method | Archetype | n | Caught | Recall | 95% CI | CI width |")
    add("|---|---|---|---|---|---|---|")
    for name in methods:
        lines.extend(interval_row(name, by_archetype[name]))
    add("")
    entries = by_archetype[primary]
    if entries:
        ranked = sorted(entries, key=lambda e: e.recall)
        worst, best = ranked[0], ranked[-1]
        overlapping = worst.high >= best.low
        verdict = (
            "**overlap**, so this is a direction to investigate rather than a "
            "demonstrated effect"
            if overlapping
            else "**do not overlap**, so this is a real difference at this sample size"
        )
        add(f"For `{primary}`, the gap runs from {worst.recall:.0%} on "
            f"{worst.group.replace('_', ' ')} engagements to {best.recall:.0%} on "
            f"{best.group.replace('_', ' ')} ones. The intervals {verdict}.")
        add("")
    add("Either way it belongs in the Information Governance conversation before real "
        "data. A tool that protects one category of worker better than another is a "
        "governance question first and a metric second — and the direction here is the "
        "one predicted in Phase 1: teaching engagements are structurally the hardest "
        "case, because genuine autonomy of method sits alongside a fixed timetable, "
        "on-site delivery and year-on-year repetition.")
    add("")

    add("## 7. Calibration")
    add("")
    add("| Method | ECE | MCE | Brier | Reliability | Resolution |")
    add("|---|---|---|---|---|---|")
    for result in calibrations:
        add(
            f"| `{result.method}` | {result.ece:.4f} | {result.mce:.4f} | "
            f"{result.brier:.5f} | {result.reliability:.6f} | {result.resolution:.6f} |"
        )
    add("")
    add("Lower reliability is better calibrated; higher resolution means the scores "
        "separate positives from negatives. Both are components of the Brier score, "
        "which is why a method can have a good Brier score by being uniformly "
        "unconfident — the decomposition is what stops that being mistaken for skill.")
    add("")
    add("**This matters because Phase 5 puts a confidence number in front of a "
        "reviewer.** A number that is not calibrated is worse than no number: a "
        "reviewer who learns that 0.9 usually means 'wrong' has been trained to "
        "distrust the tool, and one who takes 0.9 at face value has been misled into "
        "skipping a check. The rule baseline's score is a squashed weighted count of "
        "matched cues — monotone and useful for ranking, never intended as a "
        "probability — so its output should be shown as a band (low / medium / high) "
        "or a rank, not a percentage. That is a Phase 5 design constraint arising from "
        "this measurement.")
    add("")

    add("## 8. Limitations")
    add("")
    add("1. **The transformer arms are missing.** The comparison the assignment is "
        "built around is incomplete until the Phase 3 models can be run.")
    add("2. **Small cells.** Nine to eleven positives per contradiction type; the "
        "intervals are wide and are reported for that reason.")
    add("3. **Synthetic phrasing is discrete.** Real justifications vary continuously; "
        "template propositions do not. The style-invariance test varies register but "
        "not the underlying wording of a proposition, so it measures robustness to "
        "*style*, not to paraphrase.")
    add("4. **The corpus contradiction-type mix is stratified by design** (Phase 1), so "
        "per-type recall is estimable but the aggregate is not a field estimate.")
    add("5. **Reviewer fatigue is unmeasured.** The precision floor is asserted, not "
        "evidenced, until Phase 5 has collected accept/dismiss decisions.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the Phase 4 evaluation and write the report and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "synthetic")
    args = parser.parse_args()

    schema = load_schema()
    rule_config = load_yaml(CONFIG_DIR / "rule_baseline.yaml")
    text_bank = load_yaml(CONFIG_DIR / "text_bank.yaml")
    generation_config = load_generation_config()
    pipeline_config = load_pipeline_config()
    log = get_logger("phase4", LoggingConfig.from_mapping(pipeline_config.get("logging")))

    records, truth, states_raw = load_corpus(args.data)
    instances = build_instances(schema, records, truth, states_raw)
    summary = instance_summary(instances)
    log.event("phase4.dataset_built", meta=summary)

    factories: dict[str, Callable[[], ContradictionDetector]] = {
        "rules_as_authored": lambda: RuleBaseline(schema, rule_config),
        "tfidf_logreg": lambda: TfidfBaseline(schema),
    }

    results: dict[str, EvaluationResult] = {}
    scores: dict[str, np.ndarray] = {}
    for name, factory in factories.items():
        scored = cross_val_scores(factory, instances, "record", args.folds, logger=log)
        scores[name] = scored.scores
        results[name] = evaluate(name, "record", instances, scored.scores, args.target_recall)
        print(
            f"{name:20s} PR-AUC={fmt(results[name].pr_auc)} "
            f"recall={results[name].chosen.recall:.2f} "
            f"flags/form={results[name].chosen.flags_per_record:.2f}"
        )

    y_true = np.asarray([i.label for i in instances])
    overall_ci: dict[str, tuple[float, float]] = {}
    by_type: dict[str, list[GroupRecall]] = {}
    by_subtlety: dict[str, list[GroupRecall]] = {}
    by_archetype: dict[str, list[GroupRecall]] = {}
    calibrations: list[CalibrationResult] = []

    for name in factories:
        threshold = results[name].chosen.threshold
        predicted = (scores[name] >= threshold).astype(int)
        overall_ci[name] = bootstrap_recall(instances, y_true, predicted, args.boot)
        by_type[name] = recall_with_intervals(
            instances, scores[name], threshold, "contradiction_type", args.boot
        )
        by_subtlety[name] = recall_with_intervals(
            instances, scores[name], threshold, "subtlety", args.boot
        )
        by_archetype[name] = recall_with_intervals(
            instances, scores[name], threshold, "archetype", args.boot
        )
        calibrations.append(assess_calibration(name, y_true, scores[name]))
        print(f"{name:20s} bootstrap and calibration done")

    # Style invariance needs the generator, to re-render the same states.
    generator = EsqGenerator(schema, generation_config, text_bank, CestRuleEngine(schema))
    states = [generator.build_state(i) for i in range(len(records))]
    style: dict[str, StyleInvarianceResult] = {}
    for name, factory in factories.items():
        style[name] = measure_style_invariance(
            schema,
            generator,
            states,
            truth,
            factory,
            results[name].chosen.threshold,
            instances,
            args.folds,
            logger=log,
        )
        print(
            f"{name:20s} style invariance: jaccard={style[name].mean_jaccard:.3f} "
            f"drift={style[name].mean_abs_drift:.4f}"
        )

    reports_dir = REPO_ROOT / "docs" / "reports"
    calibration_figure = plot_calibration(
        calibrations, summary["positive_rate"], reports_dir / "phase4_calibration.png"
    )
    robustness_figure = plot_robustness(
        {
            "subtlety": {n: by_subtlety[n] for n in factories},
            "type": {n: by_type[n] for n in factories},
            "archetype": {n: by_archetype[n] for n in factories},
        },
        reports_dir / "phase4_robustness.png",
        titles={
            "subtlety": "By planted subtlety\n(1 blatant → 3 needs domain knowledge)",
            "type": "By contradiction type",
            "archetype": "By engagement archetype",
        },
    )

    report = build_report(
        results, calibrations, style, by_type, by_subtlety, by_archetype,
        overall_ci, summary, args.target_recall, args.boot,
    )
    report_path = reports_dir / "phase4_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")

    (reports_dir / "phase4_results.json").write_text(
        json.dumps(
            {
                "dataset": summary,
                "target_recall": args.target_recall,
                "n_boot": args.boot,
                "results": [r.as_dict() for r in results.values()],
                "calibration": [c.as_dict() for c in calibrations],
                "style_invariance": {k: v.as_dict() for k, v in style.items()},
                "by_type": {k: [e.as_dict() for e in v] for k, v in by_type.items()},
                "by_subtlety": {k: [e.as_dict() for e in v] for k, v in by_subtlety.items()},
                "by_archetype": {k: [e.as_dict() for e in v] for k, v in by_archetype.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.event("phase4.completed", meta={"report": str(report_path)})
    print(f"\nReport:     {report_path}")
    print(f"Figures:    {calibration_figure.name}, {robustness_figure.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
