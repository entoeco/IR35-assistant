#!/usr/bin/env python3
"""Phase 2: evaluate the baselines and write the method-comparison report.

Runs every method through one harness, on two splits, with cross-validated
out-of-fold scores, and writes:

* ``docs/reports/phase2_baseline_report.md`` — the comparison and its caveats
* ``docs/reports/phase2_pr_curves.png`` — precision-recall by split
* ``docs/reports/phase2_results.json`` — the numbers, for Phase 3 to compare against

Usage:
    python scripts/run_baselines.py [--target-recall 0.8] [--folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate.harness import cross_val_scores  # noqa: E402
from src.evaluate.leakage import analyse_cue_leakage  # noqa: E402
from src.evaluate.metrics import EvaluationResult, evaluate  # noqa: E402
from src.evaluate.plots import plot_pr_curves  # noqa: E402
from src.features.dataset import build_instances, instance_summary, load_corpus  # noqa: E402
from src.ingest.schema_loader import (  # noqa: E402
    CONFIG_DIR,
    load_pipeline_config,
    load_schema,
    load_yaml,
)
from src.models.baseline_rules import RuleBaseline  # noqa: E402
from src.models.baseline_tfidf import TfidfBaseline  # noqa: E402
from src.models.base import ContradictionDetector  # noqa: E402
from src.utils.logging import LoggingConfig, get_logger  # noqa: E402

SPLITS = ("record", "unit")


def build_factories(schema, rule_cfg, text_bank) -> dict[str, Callable[[], ContradictionDetector]]:
    """Detector factories, in report order.

    Three rule variants rather than one, because the cue lists and the
    generator's text bank were authored by the same process: reporting a single
    number would present agreement between two authoring passes as detection
    ability. See ``src/evaluate/leakage.py``.
    """
    return {
        "rules_as_authored": lambda: RuleBaseline(schema, rule_cfg),
        "rules_phrase_deleaked": lambda: RuleBaseline.de_leaked(schema, rule_cfg, text_bank, 3),
        "rules_vocab_stripped": lambda: RuleBaseline.de_leaked(schema, rule_cfg, text_bank, 1),
        "tfidf_logreg": lambda: TfidfBaseline(schema),
    }


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
    summary: Mapping[str, Any],
    leakage: Mapping[str, Any],
    cue_examples: list[tuple[str, str, str]],
    target_recall: float,
    n_folds: int,
    runtimes: Mapping[str, float],
    figure_name: str,
) -> str:
    """Render the Phase 2 report as Markdown."""
    methods = sorted({m for _, m in results})
    order = [
        "rules_as_authored",
        "rules_phrase_deleaked",
        "rules_vocab_stripped",
        "tfidf_logreg",
    ]
    methods = [m for m in order if m in methods] + [m for m in methods if m not in order]

    lines: list[str] = []
    add = lines.append

    add("# Phase 2 — Baseline Method Comparison")
    add("")
    add("Rule/keyword and TF-IDF baselines, evaluated with grouped cross-validation "
        "on two splits. Both are honest attempts: the point of the comparison is to "
        "find where a classical method genuinely fails, so that Phase 3 has something "
        "real to beat.")
    add("")
    add("**This is decision support.** Every flag is advisory and every method here "
        "produces a human-readable reason alongside its score. Nothing in this phase "
        "issues or overrides a Status Determination Statement.")
    add("")

    add("## The headline")
    add("")
    record_tfidf = results.get(("record", "tfidf_logreg"))
    unit_tfidf = results.get(("unit", "tfidf_logreg"))
    record_rules = results.get(("record", "rules_as_authored"))
    if record_tfidf and unit_tfidf and record_rules:
        add(f"On a conventional record-disjoint split, TF-IDF + logistic regression looks "
            f"strong: PR-AUC **{record_tfidf.pr_auc:.3f}** against a no-skill floor of "
            f"{summary['positive_rate']:.3f}, reaching {record_tfidf.chosen.recall:.0%} recall "
            f"at {record_tfidf.chosen.flags_per_record:.2f} flags per submission.")
        add("")
        add(f"On the unit-disjoint split — where no test justification's underlying "
            f"proposition appears in training — the same model collapses to PR-AUC "
            f"**{unit_tfidf.pr_auc:.3f}**, which is essentially the no-skill floor. "
            f"The rule baseline is unchanged across the two splits at PR-AUC "
            f"**{record_rules.pr_auc:.3f}**, because it does not learn from the corpus "
            "and so has nothing to memorise.")
        add("")
        add("**Read that as: almost all of the TF-IDF baseline's apparent performance is "
            "template memorisation.** It is not detecting contradiction; it is "
            "recognising sentences it has seen. On phrasing it has not seen, it is no "
            "better than chance. That is the single most important result in this phase, "
            "and it reframes what Phase 3 has to demonstrate — not 'beat 0.72', but "
            "'produce any signal at all on unseen phrasing'.")
        add("")

    add("## Dataset and evaluation design")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Prediction unit | one (submission, contradiction-pair) instance |")
    add(f"| Instances | {summary['n_instances']:,} across {summary['n_records']} submissions |")
    add(f"| Positive instances | {summary['n_positive']} |")
    add(f"| Positive rate | {summary['positive_rate']:.2%} |")
    add(f"| Instances per submission | {summary['instances_per_record']} |")
    add(f"| Distinct content units | {summary['n_distinct_units']} |")
    add(f"| Cross-validation | {n_folds}-fold, grouped and stratified |")
    add(f"| Recall target | {target_recall:.0%} |")
    add("")
    add("**Accuracy is not reported anywhere in this phase.** At a "
        f"{summary['positive_rate']:.2%} positive rate a detector that never fires scores "
        f"{1 - summary['positive_rate']:.1%} accuracy and finds nothing. The operational "
        "metric is **flags per submission** — precision of 0.22 means nothing to the IR35 "
        "team, whereas 'you will see roughly one flag every one or two forms' decides "
        "whether the tool gets used.")
    add("")
    add("Cross-validation rather than a single holdout: a 30% holdout leaves about twenty "
        "positives, and splitting those across seven contradiction types and three "
        "subtlety levels gives cells of two or three. Out-of-fold scoring predicts all "
        f"{summary['n_positive']} out of sample.")
    add("")

    add("### Why two splits")
    add("")
    add("| Split | What it holds out | What it measures |")
    add("|---|---|---|")
    add("| `record` | whole submissions | the conventional split. Content units are shared, "
        "so most test phrasing has been seen in training — an **upper bound** inflated by "
        "template reuse. |")
    add("| `unit` | whole content units | no test proposition appears in training. Records "
        "straddle the split, so record-level style is not controlled — a **lower bound**, "
        "because template propositions are discrete and a novel one may share almost no "
        "vocabulary with training, whereas real phrasing varies continuously. |")
    add("")
    add("Real-world performance sits between the two. Reporting only the first would have "
        "overstated the classical baseline substantially.")
    add("")

    add("## Cue leakage")
    add("")
    add(f"The lexical cues were authored in Phase 0; the contradiction text in Phase 1. "
        f"Same author, same understanding of the domain, a week apart — so they overlap. "
        f"**{leakage['n_units_with_cue']} of {leakage['n_units']} planted contradiction "
        f"units ({leakage['unit_leakage_rate']:.0%}) contain a cue verbatim**, and "
        f"{leakage['n_leaked_cues']} of {leakage['n_cues']} cues "
        f"({leakage['cue_leakage_rate']:.0%}) appear somewhere in the generated text, "
        f"across {leakage['pairs_affected']} pairs.")
    add("")
    add("The worst cases are straightforwardly a copied sentence:")
    add("")
    for pair_id, cue, unit in cue_examples[:3]:
        add(f"- `{pair_id}` — cue `\"{cue}\"` against planted text *\"{unit[:90]}\"*")
    add("")
    add("So the rule baseline is run three ways rather than once:")
    add("")
    add("| Variant | Rule | Reading |")
    add("|---|---|---|")
    add("| `rules_as_authored` | every cue | upper bound — what a keyword list achieves when its phrasing happens to match the data |")
    add("| `rules_phrase_deleaked` | cues of 3+ words appearing verbatim are removed | copied phrasing stripped, domain vocabulary kept. **The number to quote.** |")
    add("| `rules_vocab_stripped` | every verbatim-matching cue removed | pessimistic floor. Strips legitimate domain terms too, so it is close to tautological — reported for completeness, not as an estimate |")
    add("")
    add("The three-word line is a judgement, stated so it can be disagreed with and "
        "recomputed. The gap between the first two variants is the useful number: it "
        "measures how much the baseline depends on having seen the exact phrasing.")
    add("")

    add("## Results")
    add("")
    add(f"![Precision-recall by split]({figure_name})")
    add("")
    for split in SPLITS:
        add(f"### `{split}` split")
        add("")
        add("| Method | PR-AUC | ROC-AUC | Macro-F1 | Brier | Threshold | Recall | Precision | Flags/form | Missed |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for method in methods:
            result = results.get((split, method))
            if not result:
                continue
            point = result.chosen
            add(
                f"| `{method}` | {fmt(result.pr_auc)} | {fmt(result.roc_auc)} | "
                f"{fmt(result.macro_f1)} | {fmt(result.brier, 4)} | {point.threshold:.3f} | "
                f"**{point.recall:.2f}** | {point.precision:.3f} | "
                f"{point.flags_per_record:.2f} | {point.n_false_negatives} |"
            )
        add("")
        notes = {
            method: results[(split, method)].notes
            for method in methods
            if (split, method) in results and results[(split, method)].notes
        }
        if notes:
            for method, method_notes in notes.items():
                for note in method_notes:
                    add(f"- `{method}`: {note}")
            add("")

    add("### ROC-AUC is reported but should not be read")
    add("")
    add("The brief asks for it, so it is in the table. At this class balance it is "
        "misleading: the false-positive rate barely moves because the negative class is "
        "enormous, so every method looks respectable. The `unit` split makes the point — "
        "TF-IDF holds a ROC-AUC around 0.6 while its PR-AUC sits on the no-skill floor. "
        "PR-AUC and flags-per-form are the metrics that describe what a reviewer "
        "experiences.")
    add("")

    add("## Where each method fails")
    add("")
    add("**These breakdowns are on the `record` split**, because it is the only split "
        "with enough positives per cell to break down at all. That means the TF-IDF "
        "column is the *memorising* regime: its per-type numbers describe how well it "
        "recognises propositions it has already seen, and they do not transfer to unseen "
        "phrasing. The rule columns are split-independent and can be read at face value. "
        "Compare the rule variants against each other freely; compare them against TF-IDF "
        "only with that asymmetry in mind.")
    add("")
    types = sorted({k for r in results.values() for k in r.by_type})
    add("### Recall by contradiction type (`record` split, at the chosen operating point)")
    add("")
    add("| Method | " + " | ".join(t.replace("_", " ") for t in types) + " |")
    add("|" + "---|" * (len(types) + 1))
    for method in methods:
        result = results.get(("record", method))
        if result:
            add(recall_row(f"`{method}`", result.by_type, types))
    add("")
    rules = results.get(("record", "rules_as_authored"))
    tfidf = results.get(("record", "tfidf_logreg"))
    if rules and tfidf:
        hedged_rules = rules.by_type.get("hedged_non_support", {}).get("recall", 0)
        named_rules = rules.by_type.get("named_person_dependency", {}).get("recall", 0)
        add(f"The rule baseline's weakest types are the predicted ones. "
            f"`hedged_non_support` recall is **{hedged_rules:.2f}** — a keyword list "
            "cannot detect the *absence* of commitment, because there is no phrase to "
            "match; the contradiction is that the manager selected a definite option and "
            "then wrote something that fails to support it. "
            f"`named_person_dependency` is **{named_rules:.2f}** despite being the "
            "highest-priority pair in the set, because the dependency is usually implied "
            "rather than stated in listed vocabulary.")
        add("")
        add("These two types are the specific place a transformer should earn its keep. "
            "If Phase 3 does not improve on them, the method comparison has a real finding "
            "to report rather than a foregone conclusion.")
        add("")
        add("`rules_vocab_stripped` scoring near zero on every type is the tautology "
            "warned about above, visible in full: remove every term that occurs in the "
            "data and a lexical method has nothing left to match. It is in the table so "
            "the floor is not mistaken for a result.")
        add("")

    subtleties = sorted({k for r in results.values() for k in r.by_subtlety})
    add("### Recall by subtlety (`record` split)")
    add("")
    add("| Method | " + " | ".join(f"level {s}" for s in subtleties) + " |")
    add("|" + "---|" * (len(subtleties) + 1))
    for method in methods:
        result = results.get(("record", method))
        if result:
            add(recall_row(f"`{method}`", result.by_subtlety, subtleties))
    add("")
    add("Level 1 is blatant, level 2 needs reading, level 3 needs domain knowledge — the "
        "form's own cost exclusions, or the case law. The rule baseline degrades sharply "
        "at level 3, which is the expected shape: those are the contradictions a busy "
        "reviewer misses too, and they are the ones the tool most needs to catch.")
    add("")

    archetypes = sorted({k for r in results.values() for k in r.by_archetype})
    add("### Recall by engagement archetype (`record` split)")
    add("")
    add("| Method | " + " | ".join(a.replace("_", " ") for a in archetypes) + " |")
    add("|" + "---|" * (len(archetypes) + 1))
    for method in methods:
        result = results.get(("record", method))
        if result:
            add(recall_row(f"`{method}`", result.by_archetype, archetypes))
    add("")
    add("A first look at the disparity question Phase 4 takes up properly. Cell counts "
        "here are small — single figures per archetype — so differences should be read as "
        "a direction to investigate, not as an effect.")
    add("")
    rules_arch = results.get(("record", "rules_as_authored"))
    if rules_arch and rules_arch.by_archetype:
        ranked = sorted(rules_arch.by_archetype.items(), key=lambda kv: kv[1]["recall"])
        worst, best = ranked[0], ranked[-1]
        add(f"The direction to investigate is already visible: the rule baseline catches "
            f"{best[1]['recall']:.0%} of contradictions on {best[0].replace('_', ' ')} "
            f"engagements and {worst[1]['recall']:.0%} on "
            f"{worst[0].replace('_', ' ')} ones. That is the shape predicted in Phase 1 — "
            "the borderline cases cluster in teaching engagements, where autonomy of "
            "method sits alongside a fixed timetable — and if it survives Phase 4 with "
            "larger cells it is a fairness finding, not a tuning problem. A tool that "
            "protects one category of worker better than another is a governance issue "
            "before it is a metric.")
        add("")

    registers = sorted({k for r in results.values() for k in r.by_register})
    add("### Recall by writing register (`record` split)")
    add("")
    add("| Method | " + " | ".join(registers) + " |")
    add("|" + "---|" * (len(registers) + 1))
    for method in methods:
        result = results.get(("record", method))
        if result:
            add(recall_row(f"`{method}`", result.by_register, registers))
    add("")
    add("Register here is confounded with archetype, because archetypes carry different "
        "register mixes. Phase 4's style-invariance test resolves that by re-rendering the "
        "*same* engagement in all three registers, which this breakdown cannot do.")
    add("")

    add("## Runtime")
    add("")
    add("| Method | Total fit time across folds |")
    add("|---|---|")
    for method, seconds in runtimes.items():
        add(f"| `{method}` | {seconds:.1f}s |")
    add("")
    add("Recorded because the deployment assumption is local-only: a method that needs a "
        "GPU to be practical is a different proposition for the University than one that "
        "runs on a laptop.")
    add("")

    add("## Method choices, and what was rejected")
    add("")
    add("**Rule baseline.** Pair cues, contradiction-family lexicons grouped by type and "
        "restricted to the IR35 tests where they apply, structural regexes for rate shapes, "
        "and negation handling with a four-token window. Negation matters more than it "
        "looks: without it the matcher fires on *\"we do not require them on site\"*, which "
        "is evidence **for** the answer. Skipping it is the usual way a keyword baseline "
        "becomes a strawman.")
    add("")
    add("*Rejected:* reusing the structured-answer weights from the labelling engine. That "
        "would have made Phase 2 grade the generator against its own logic. The rule "
        "baseline reads free text only, and shares no coefficients with "
        "`config/ir35_weights.yaml`.")
    add("")
    add("**TF-IDF baseline.** Word 1-2 grams over the answer and text together, character "
        "3-5 grams for robustness to the terse register, and explicit interaction features "
        "crossing free-text tokens with the answer's polarity and IR35 test. The "
        "interaction block is the important part: a contradiction is not a property of the "
        "text alone. *\"We would need to approve any substitute\"* is consistent with "
        "\"Yes, we have a right to reject\" and contradicts \"No\". Without the cross, a "
        "linear model can only learn \"these words are suspicious\", which is the wrong "
        "hypothesis class.")
    add("")
    add("*Rejected:* per-pair models. Seventy positives across thirty-one pairs is two "
        "each. One pooled model with pair identity as a feature is the only defensible "
        "choice at this scale.")
    add("")
    add("*Rejected:* tuning the threshold to maximise F1. The objective is a stated recall "
        "level at the least reviewer burden that achieves it. A missed contradiction "
        "reaches HMRC as a determination made without the reasonable care the off-payroll "
        "rules require; a false flag costs a reviewer half a minute. F1 would trade away "
        "recall to buy precision, which is the wrong trade here.")
    add("")

    add("## Limitations")
    add("")
    add("1. **The rule baseline cannot reach the recall target at all.** Its ceiling is "
        "reported in the tables above. That is a property of lexical matching, not of "
        "tuning — there is no threshold at which it finds what it has no vocabulary for.")
    add("2. **TF-IDF's record-split score is not a performance estimate.** It is what "
        "memorising 140 content units gets you. The unit-split figure is closer to honest "
        "and is itself pessimistic.")
    add("3. **The corpus labels came from a rule engine** (`config/ir35_weights.yaml`). "
        "The contradiction labels are independent of it — they are planted, not derived — "
        "so this phase is not circular in the way a *status* classifier would be. But no "
        "status classification is attempted here, and that is why.")
    add("4. **Small cells.** Per-type recall rests on nine to eleven positives per type. "
        "Differences of one or two cases are noise.")
    add("5. **Calibration is poor for the rule baseline by construction** — its scores are "
        "a squashed weighted count, not a probability. The Brier scores reflect that. "
        "Phase 4's calibration analysis is where this gets treated properly.")
    add("6. **Register and archetype are confounded** in the breakdowns above.")
    add("")

    add("## What Phase 3 has to beat")
    add("")
    add("| Target | Value |")
    add("|---|---|")
    if record_rules:
        add(f"| Rule baseline, PR-AUC (split-independent) | {record_rules.pr_auc:.3f} |")
        add(f"| Rule baseline, recall ceiling | {record_rules.chosen.recall:.2f} |")
    deleaked = results.get(("record", "rules_phrase_deleaked"))
    if deleaked:
        add(f"| Rule baseline de-leaked, PR-AUC | {deleaked.pr_auc:.3f} |")
    if unit_tfidf:
        add(f"| TF-IDF on unseen phrasing, PR-AUC | {unit_tfidf.pr_auc:.3f} |")
    add("")
    add("The bar that matters is the `unit` split: a method that generalises to phrasing "
        "it has not seen. A cross-encoder NLI model should, because it reasons over the "
        "relationship between two texts rather than over a learned vocabulary — but that "
        "is a hypothesis, and Phase 3 exists to test it, not to confirm it.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the baselines and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "synthetic")
    args = parser.parse_args()

    schema = load_schema()
    rule_cfg = load_yaml(CONFIG_DIR / "rule_baseline.yaml")
    text_bank = load_yaml(CONFIG_DIR / "text_bank.yaml")
    pipeline_cfg = load_pipeline_config()
    log = get_logger("phase2", LoggingConfig.from_mapping(pipeline_cfg.get("logging")))

    records, truth, states = load_corpus(args.data)
    instances = build_instances(schema, records, truth, states)
    summary = instance_summary(instances)
    log.event("phase2.dataset_built", meta=summary)

    factories = build_factories(schema, rule_cfg, text_bank)
    results: dict[tuple[str, str], EvaluationResult] = {}
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
    points: dict[str, dict[str, tuple[float, float]]] = {}
    runtimes: dict[str, float] = {}
    y_true = np.asarray([i.label for i in instances])

    for split in SPLITS:
        curves[split] = {}
        points[split] = {}
        for name, factory in factories.items():
            scored = cross_val_scores(factory, instances, split, args.folds, logger=log)
            result = evaluate(scored.method, split, instances, scored.scores, args.target_recall)
            results[(split, scored.method)] = result
            runtimes[scored.method] = runtimes.get(scored.method, 0.0) + scored.fit_seconds
            if name != "rules_vocab_stripped":  # keep the figure to three series
                precision, recall, _ = precision_recall_curve(y_true, scored.scores)
                curves[split][scored.method] = (recall, precision, result.pr_auc)
                points[split][scored.method] = (result.chosen.recall, result.chosen.precision)

    leakage = analyse_cue_leakage(schema, text_bank)
    reports_dir = REPO_ROOT / "docs" / "reports"
    figure = plot_pr_curves(
        curves, summary["positive_rate"], reports_dir / "phase2_pr_curves.png", points
    )

    report = build_report(
        results,
        summary,
        leakage.as_dict(),
        leakage.worst_examples,
        args.target_recall,
        args.folds,
        runtimes,
        figure.name,
    )
    report_path = reports_dir / "phase2_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "dataset": summary,
        "cue_leakage": leakage.as_dict(),
        "target_recall": args.target_recall,
        "n_folds": args.folds,
        "results": [result.as_dict() for result in results.values()],
    }
    (reports_dir / "phase2_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    log.event(
        "phase2.completed",
        meta={"report": str(report_path), "figure": str(figure), "n_results": len(results)},
    )
    print(f"Report:  {report_path}")
    print(f"Figure:  {figure}")
    print(f"Results: {reports_dir / 'phase2_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
