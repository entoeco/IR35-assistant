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
from typing import Mapping, Sequence

import numpy as np

__all__ = ["plot_pr_curves"]

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
