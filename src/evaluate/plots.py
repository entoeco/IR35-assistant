"""Figures for the method comparison.

Precision-recall rather than ROC. At a 0.7% positive rate ROC curves look
flattering for everything — the false-positive rate barely moves because the
negative class is enormous — while PR curves show what a reviewer would actually
experience. Both numbers are reported in the tables; the picture is PR.

Both panels share one y axis on purpose. The unit-split panel is supposed to
look flat against the floor: that flatness is the finding, and rescaling the
panel to make its curves legible would hide it.

Palette: categorical slots 1-3 of the validated default (blue, orange, aqua),
which clear the all-pairs CVD and normal-vision floors. Aqua sits below 3:1 on
the light surface, so the relief rule applies and every series carries a direct
label as well as a legend entry — identity is never colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["plot_pr_curves", "plot_calibration", "plot_robustness"]

# Validated categorical slots 1-3 (light mode) plus text and surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e7e6e2"


def plot_pr_curves(
    curves: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray, float]]],
    prevalence: float,
    output_path: Path | str,
    operating_points: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
) -> Path:
    """Draw precision-recall curves, one panel per split.

    Args:
        curves: split -> method -> (recall, precision, pr_auc).
        prevalence: Positive rate, drawn as the no-skill reference line.
        output_path: Where to write the PNG.
        operating_points: split -> method -> (recall, precision) of the chosen
            threshold, marked on the curve.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(curves)
    fig, axes = plt.subplots(
        1, len(splits), figsize=(11.0, 4.8), dpi=170, sharey=True, facecolor=SURFACE
    )
    if len(splits) == 1:
        axes = [axes]

    titles = {
        "record": "Record-disjoint split\n(content units shared — optimistic)",
        "unit": "Unit-disjoint split\n(test phrasing unseen — honest)",
    }

    for axis, split in zip(axes, splits):
        axis.set_facecolor(SURFACE)
        axis.axhline(
            prevalence,
            color=TEXT_MUTED,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        axis.annotate(
            f"no-skill floor ({prevalence:.1%})",
            xy=(0.02, prevalence),
            xytext=(0, 7),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color=TEXT_MUTED,
        )

        for index, (method, (recall, precision, pr_auc)) in enumerate(curves[split].items()):
            colour = SERIES[index % len(SERIES)]
            axis.step(
                recall,
                precision,
                where="post",
                color=colour,
                linewidth=2.0,
                label=method,
                zorder=3,
            )
            point = (operating_points or {}).get(split, {}).get(method)
            if point:
                axis.plot(
                    [point[0]],
                    [point[1]],
                    marker="o",
                    markersize=8,
                    color=colour,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.8,
                    zorder=4,
                )

        # Direct labels anchored to the operating-point markers rather than to
        # the curve maxima. The rule curves both peak at precision 1.0 near zero
        # recall, so labelling the maximum stacks every label in the same few
        # pixels. The markers are spread across the panel, and they are the
        # points a reader cares about. Offsets alternate to keep two markers at
        # similar coordinates from colliding.
        for index, method in enumerate(curves[split]):
            point = (operating_points or {}).get(split, {}).get(method)
            if not point:
                continue
            axis.annotate(
                method,
                xy=point,
                xytext=(9, 10 if index % 2 == 0 else -16),
                textcoords="offset points",
                fontsize=8,
                color=TEXT_SECONDARY,
                zorder=5,
            )

        axis.set_xlim(0, 1.0)
        axis.set_ylim(0, 1.0)
        axis.set_xlabel("Recall", fontsize=9, color=TEXT_SECONDARY)
        axis.set_title(
            titles.get(split, split), fontsize=10, color=TEXT_PRIMARY, pad=22, loc="left"
        )
        # PR-AUC per panel, not in the shared legend: the two splits give
        # different values for the same method, and one legend cannot carry both
        # without mislabelling a panel.
        axis.annotate(
            "PR-AUC   "
            + "   ·   ".join(f"{m} {a:.3f}" for m, (_, _, a) in curves[split].items()),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=7.5,
            color=TEXT_MUTED,
        )
        axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)

    axes[0].set_ylabel("Precision", fontsize=9, color=TEXT_SECONDARY)
    # One shared legend below the panels rather than inside one of them: an
    # in-panel legend sits on top of the curves at this aspect ratio, and the
    # two panels share a series set so two legends would be redundant.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.suptitle(
        "Contradiction detection: precision-recall by evaluation split",
        fontsize=12,
        color=TEXT_PRIMARY,
        x=0.011,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.011,
        0.012,
        "Markers show the operating point chosen for the recall target. "
        "Synthetic corpus, 350 submissions, 70 planted contradictions.",
        fontsize=7.5,
        color=TEXT_MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.93))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_calibration(
    results: Sequence[Any],
    prevalence: float,
    output_path: Path | str,
) -> Path:
    """Reliability diagram: predicted score against observed positive rate.

    Phase 5 shows a confidence number beside every flag, so that number is a
    claim about the world and this is where the claim is tested. A method whose
    curve sits well below the diagonal is over-confident; one far above is
    under-confident. Either way the number should be presented to reviewers as a
    band or a rank rather than a percentage.

    Args:
        results: ``CalibrationResult`` objects, one per method.
        prevalence: Base rate, drawn as a reference.
        output_path: Where to write the PNG.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.4, 5.2), dpi=170, facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    axis.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    axis.annotate(
        "perfect calibration",
        xy=(0.62, 0.62),
        xytext=(4, -14),
        textcoords="offset points",
        fontsize=8,
        color=TEXT_MUTED,
        rotation=38,
        rotation_mode="anchor",
    )
    axis.axhline(prevalence, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (1, 3)), zorder=1)
    # Right-hand side: the low-score end of every curve sits on top of the base
    # rate line, so a left-anchored label lands underneath a series.
    axis.annotate(
        f"base rate {prevalence:.1%}",
        xy=(0.99, prevalence),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=TEXT_MUTED,
    )

    for index, result in enumerate(results):
        if not result.bin_centres:
            continue
        colour = SERIES[index % len(SERIES)]
        axis.plot(
            result.bin_centres,
            result.bin_observed,
            marker="o",
            markersize=8,
            linewidth=2.0,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.8,
            label=f"{result.method} (ECE {result.ece:.3f})",
            zorder=3,
        )
        axis.annotate(
            result.method,
            xy=(result.bin_centres[-1], result.bin_observed[-1]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_SECONDARY,
            zorder=5,
        )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Predicted score", fontsize=9, color=TEXT_SECONDARY)
    axis.set_ylabel("Observed positive rate", fontsize=9, color=TEXT_SECONDARY)
    axis.set_title(
        "Calibration: does the confidence number mean anything?",
        fontsize=11,
        color=TEXT_PRIMARY,
        loc="left",
        pad=10,
    )
    axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axis.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.text(
        0.012,
        0.015,
        "Bins holding fewer than 20 instances are omitted: at a 0.7% base rate they "
        "carry no information.",
        fontsize=7.5,
        color=TEXT_MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_robustness(
    panels: Mapping[str, Mapping[str, Sequence[Any]]],
    output_path: Path | str,
    titles: Mapping[str, str] | None = None,
) -> Path:
    """Recall by group, with cluster-bootstrap intervals, one panel per cut.

    Dot-and-interval rather than bars. Bars would imply a meaningful zero
    baseline and, worse, would draw the eye to differences between point
    estimates that the intervals show are not there — which is exactly the
    over-reading these cells invite at nine to eleven positives each.

    Args:
        panels: Panel key -> method -> sequence of ``GroupRecall``.
        output_path: Where to write the PNG.
        titles: Panel key -> display title.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(panels)
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.6 * len(keys), 5.0), dpi=170, facecolor=SURFACE
    )
    if len(keys) == 1:
        axes = [axes]

    for axis, key in zip(axes, keys):
        axis.set_facecolor(SURFACE)
        methods = list(panels[key])
        groups: list[str] = []
        for method in methods:
            for entry in panels[key][method]:
                if entry.group not in groups:
                    groups.append(entry.group)
        positions = {group: i for i, group in enumerate(groups)}

        for m_index, method in enumerate(methods):
            colour = SERIES[m_index % len(SERIES)]
            # Offset each method slightly so overlapping intervals stay legible.
            offset = (m_index - (len(methods) - 1) / 2) * 0.18
            for entry in panels[key][method]:
                y = positions[entry.group] + offset
                axis.plot(
                    [entry.low, entry.high],
                    [y, y],
                    color=colour,
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=2,
                )
                axis.plot(
                    [entry.recall],
                    [y],
                    marker="o",
                    markersize=8,
                    color=colour,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.8,
                    zorder=3,
                    label=method if entry is panels[key][method][0] else None,
                )
        # Sample size goes in the tick label, not inside the plot. An in-plot
        # annotation collides with any interval whose upper bound reaches the
        # axis edge, which at these cell sizes is most of them.
        counts = {
            entry.group: entry.n
            for method in methods
            for entry in panels[key][method]
        }
        axis.set_yticks(range(len(groups)))
        axis.set_yticklabels(
            [f"{g.replace('_', ' ')}  (n={counts.get(g, 0)})" for g in groups], fontsize=8
        )
        axis.set_xlim(0, 1.0)
        axis.set_ylim(-0.6, len(groups) - 0.4)
        axis.invert_yaxis()
        axis.set_xlabel("Recall (95% cluster-bootstrap interval)", fontsize=9, color=TEXT_SECONDARY)
        axis.set_title(
            (titles or {}).get(key, key), fontsize=10, color=TEXT_PRIMARY, loc="left", pad=8
        )
        axis.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)

    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()),
        list(seen),
        loc="lower center",
        ncol=len(seen),
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(
        "Robustness: where does the detector fail, and by how much",
        fontsize=12,
        color=TEXT_PRIMARY,
        x=0.011,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
