#!/usr/bin/env python3
"""Generate DR vs FPR scatter plot for HALLMARK.

Shows the recall-precision spectrum across all citation verification tools.
LLMs use circle markers, API/rule-based tools use square markers.
Partial-coverage tools use hollow (unfilled) markers.

Usage:
    python scripts/generate_dr_fpr_scatter.py
    python scripts/generate_dr_fpr_scatter.py --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Publication-quality settings (matches generate_figures.py)
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Colorblind-safe palette (IBM Design Library, matches generate_figures.py)
IBM_BLUE = "#648FFF"
IBM_PURPLE = "#785EF0"
IBM_MAGENTA = "#DC267F"
IBM_ORANGE = "#FE6100"
IBM_YELLOW = "#FFB000"

# --------------------------------------------------------------------------- #
# Data points from evaluation results
# --------------------------------------------------------------------------- #

TOOLS: list[dict] = [
    # API / rule-based tools (square markers)
    {
        "name": "DOI-only",
        "dr": 0.256,
        "fpr": 0.195,
        "kind": "api",
        "partial": False,
        "color": IBM_BLUE,
    },
    {
        "name": "HaRC",
        "dr": 0.155,
        "fpr": 0.000,
        "kind": "api",
        "partial": True,
        "coverage": 1.8,
        "color": IBM_PURPLE,
    },
    {
        "name": "verify-citations",
        "dr": 0.042,
        "fpr": 0.024,
        "kind": "api",
        "partial": True,
        "coverage": 6.3,
        "color": IBM_MAGENTA,
    },
    # LLM tools (circle markers)
    {
        "name": "GPT-5.1",
        "dr": 0.797,
        "fpr": 0.171,
        "kind": "llm",
        "partial": False,
        "color": IBM_ORANGE,
    },
    {
        "name": "DeepSeek-R1",
        "dr": 0.871,
        "fpr": 0.640,
        "kind": "llm",
        "partial": False,
        "color": IBM_YELLOW,
    },
    {
        "name": "DeepSeek-V3.2",
        "dr": 0.880,
        "fpr": 0.730,
        "kind": "llm",
        "partial": False,
        "color": "#785EF0",
    },
    {
        "name": "Qwen3-235B",
        "dr": 0.832,
        "fpr": 0.551,
        "kind": "llm",
        "partial": False,
        "color": "#DC267F",
    },
    {
        "name": "Mistral Large",
        "dr": 0.691,
        "fpr": 0.258,
        "kind": "llm",
        "partial": False,
        "color": "#648FFF",
    },
    {
        "name": "Gemini 2.5 Flash",
        "dr": 0.482,
        "fpr": 0.101,
        "kind": "llm",
        "partial": False,
        "color": "#2CA02C",
    },
]

# Label offsets (dx, dy in points) — hand-tuned to avoid overlap
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "DOI-only": (8, -8),
    "HaRC": (8, 6),
    "verify-citations": (8, 6),
    "GPT-5.1": (-10, -10),
    "DeepSeek-R1": (8, 6),
    "DeepSeek-V3.2": (8, -8),
    "Qwen3-235B": (-10, -10),
    "Mistral Large": (8, -8),
    "Gemini 2.5 Flash": (8, 6),
}


def generate_dr_fpr_scatter(output_dir: Path) -> None:
    """Create the DR vs FPR scatter plot."""
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # Random baseline diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="0.6",
        linewidth=0.8,
        zorder=1,
        label="Random baseline",
    )

    # Plot each tool
    for tool in TOOLS:
        marker = "o" if tool["kind"] == "llm" else "s"
        facecolor = "none" if tool["partial"] else tool["color"]
        edgecolor = tool["color"]
        linewidth = 1.5 if tool["partial"] else 0.6

        ax.scatter(
            tool["fpr"],
            tool["dr"],
            s=55,
            marker=marker,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=3,
        )

        # Label placement
        dx, dy = LABEL_OFFSETS.get(tool["name"], (8, 4))
        fontsize = 7
        ax.annotate(
            tool["name"],
            (tool["fpr"], tool["dr"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=fontsize,
            color="0.15",
            ha="left" if dx > 0 else "right",
            va="center",
        )

    # Axes
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Detection Rate (DR)")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.03, 1.0)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Build legend with proxy artists
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.4",
            markeredgecolor="0.4",
            markersize=6,
            label="LLM-based",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="0.4",
            markeredgecolor="0.4",
            markersize=6,
            label="API / rule-based",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="none",
            markeredgecolor="0.4",
            markeredgewidth=1.5,
            markersize=6,
            label="Partial coverage*",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="0.6",
            linewidth=0.8,
            label="Random baseline",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=7,
        framealpha=0.9,
        edgecolor="0.8",
    )

    # Footnote for partial coverage
    fig.text(
        0.12,
        -0.02,
        "* Partial coverage: HaRC 1.8%, verify-citations 6.3% of entries resolved",
        fontsize=6,
        color="0.4",
        ha="left",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dr_fpr_scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DR vs FPR scatter plot for HALLMARK")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/figures",
        help="Output directory for the figure (default: paper/figures)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    generate_dr_fpr_scatter(Path(args.output_dir))


if __name__ == "__main__":
    main()
