#!/usr/bin/env python3
"""Generate evaluation figures for HALLMARK.  [evaluation]

Creates publication-quality figures:
1. Tier-wise detection rates (grouped bar chart)
2. Per-type detection heatmap
3. Cost-accuracy tradeoff
4. Temporal robustness comparison

Usage:
    python scripts/generate_figures.py \
        --results-dir results/ \
        --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)

# Publication-quality settings
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

# Colorblind-safe palette (IBM Design Library)
COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

# Display names for tools in figures
DISPLAY_NAMES = {
    "doi_only": "DOI-only",
    "harc": "HaRC",
    "verify_citations": "verify-citations",
    "llm_openai": "GPT-5.1",
    "bibtexupdater": "bibtex-updater",
    "ensemble": "Ensemble",
    "doi_presence_heuristic": "DOI-heuristic",
}


def _display_name(tool_name: str) -> str:
    return DISPLAY_NAMES.get(tool_name, tool_name)


def load_results(results_dir: Path) -> list[dict]:
    """Load all evaluation result JSONs (skips non-evaluation files)."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        # Only include standard evaluation results (must have tool_name at top level)
        if "tool_name" in data:
            results.append(data)
    return results


def fig_tier_detection_rates(results: list[dict], output_dir: Path) -> None:
    """Bar chart: detection rate per tier, grouped by tool."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    tools = [_display_name(r["tool_name"]) for r in results]
    tiers = [1, 2, 3]
    tier_labels = ["Tier 1\n(Easy)", "Tier 2\n(Medium)", "Tier 3\n(Hard)"]

    x = np.arange(len(tiers))
    width = 0.8 / max(len(tools), 1)

    for i, result in enumerate(results):
        tier_metrics = result.get("per_tier_metrics", {})
        rates = []
        for t in tiers:
            m = tier_metrics.get(str(t), {})
            rates.append(m.get("detection_rate", 0.0))

        offset = (i - len(tools) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width * 0.9,
            label=_display_name(result["tool_name"]),
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels
        for bar, rate in zip(bars, rates, strict=True):
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{rate:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Difficulty Tier")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Hallucination Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "tier_detection_rates.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_per_type_heatmap(results: list[dict], output_dir: Path) -> None:
    """Heatmap: detection rate per hallucination type per tool."""
    if not results:
        return

    # Collect all types across results
    all_types = set()
    for r in results:
        all_types.update(r.get("per_type_metrics", {}).keys())
    all_types.discard("valid")
    types = sorted(all_types)

    if not types:
        logger.warning("No per-type metrics found")
        return

    tools = [_display_name(r["tool_name"]) for r in results]
    matrix = np.zeros((len(tools), len(types)))

    for i, result in enumerate(results):
        type_metrics = result.get("per_type_metrics", {})
        for j, t in enumerate(types):
            m = type_metrics.get(t, {})
            matrix[i, j] = m.get("detection_rate", 0.0)

    fig, ax = plt.subplots(figsize=(8, max(2, len(tools) * 0.6 + 1)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", "\n") for t in types], rotation=0, fontsize=7)
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools)

    # Add text annotations
    for i in range(len(tools)):
        for j in range(len(types)):
            val = matrix[i, j]
            color = "white" if val < 0.45 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Detection Rate by Hallucination Type")
    fig.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)

    path = output_dir / "per_type_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_cost_accuracy(results: list[dict], output_dir: Path) -> None:
    """Scatter plot: F1 vs cost (entries/second)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for i, result in enumerate(results):
        f1 = result.get("f1_hallucination", 0)
        cost = result.get("cost_efficiency", None)
        if cost is None or cost == 0:
            continue

        ax.scatter(
            cost,
            f1,
            s=100,
            color=COLORS[i % len(COLORS)],
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            _display_name(result["tool_name"]),
            (cost, f1),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax.set_xlabel("Throughput (entries/sec)")
    ax.set_ylabel("F1 (Hallucination)")
    ax.set_title("Cost-Accuracy Tradeoff")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    path = output_dir / "cost_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_overall_comparison(results: list[dict], output_dir: Path) -> None:
    """Grouped bar chart: primary metrics comparison across tools."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))

    tools = [_display_name(r["tool_name"]) for r in results]
    metrics = ["detection_rate", "f1_hallucination", "tier_weighted_f1"]
    metric_labels = ["Detection Rate", "F1", "Tier-weighted F1"]

    x = np.arange(len(tools))
    width = 0.8 / len(metrics)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels, strict=True)):
        values = [r.get(metric, 0) for r in results]
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width * 0.9,
            label=label,
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values, strict=True):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_ylabel("Score")
    ax.set_title("Baseline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tools)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "overall_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_temporal_robustness(results_dir: Path, output_dir: Path) -> None:
    """Two-panel figure: DR and FPR (baseline vs probe) across all models."""
    # Discover all temporal_probe_*.json files
    probe_files = sorted(results_dir.glob("temporal_probe_*.json"))
    # Exclude the probe set JSONL
    probe_files = [p for p in probe_files if p.suffix == ".json"]
    if not probe_files:
        # Fallback: try legacy single-model file
        legacy = results_dir / "temporal_probe.json"
        if legacy.exists():
            probe_files = [legacy]
        else:
            logger.warning("No temporal_probe_*.json found, skipping temporal figure")
            return

    # Load all model results
    models: list[dict] = []
    for path in probe_files:
        with open(path) as f:
            data = json.load(f)
        probe = data.get("probe_metrics", {})
        baseline = data.get("full_baseline", {})
        # Derive display name
        display = data.get("display_name", None)
        if display is None:
            # Legacy format: extract from filename
            stem = path.stem.replace("temporal_probe_", "").replace("temporal_probe", "GPT-5.1")
            display = stem
        models.append(
            {
                "name": display,
                "dr_base": baseline.get("detection_rate", 0),
                "dr_probe": probe.get("detection_rate", 0),
                "fpr_base": baseline.get("false_positive_rate", 0),
                "fpr_probe": probe.get("false_positive_rate", 0),
                "ece_base": baseline.get("ece", 0),
                "ece_probe": probe.get("ece", 0),
            }
        )

    # Sort by baseline FPR (ascending)
    models.sort(key=lambda m: m["fpr_base"])

    n = len(models)
    names = [m["name"] for m in models]

    # Colorblind-safe viridis-derived palette
    cmap = plt.colormaps.get_cmap("viridis").resampled(max(n, 2))
    model_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, (ax_dr, ax_fpr) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    x = np.arange(n)
    width = 0.35

    # ── Left panel: Detection Rate ────────────────────────────────────
    ax_dr.bar(
        x - width / 2,
        [m["dr_base"] for m in models],
        width,
        label="Baseline (2021\u20132023)",
        color=[(*c[:3], 0.5) for c in model_colors],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_probe = ax_dr.bar(
        x + width / 2,
        [m["dr_probe"] for m in models],
        width,
        label="Probe (2024\u20132026)",
        color=model_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars_probe, [m["dr_probe"] for m in models], strict=True):
        if val > 0:
            ax_dr.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_dr.set_ylabel("Detection Rate")
    ax_dr.set_title("(a) Detection Rate")
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_dr.set_ylim(0, 1.15)
    ax_dr.legend(loc="lower right", fontsize=8)
    ax_dr.spines["top"].set_visible(False)
    ax_dr.spines["right"].set_visible(False)

    # ── Right panel: False Positive Rate ──────────────────────────────
    ax_fpr.bar(
        x - width / 2,
        [m["fpr_base"] for m in models],
        width,
        label="Baseline (2021\u20132023)",
        color=[(*c[:3], 0.5) for c in model_colors],
        edgecolor="white",
        linewidth=0.5,
    )
    ax_fpr.bar(
        x + width / 2,
        [m["fpr_probe"] for m in models],
        width,
        label="Probe (2024\u20132026)",
        color=model_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    # Annotate FPR multiplier
    for i, m in enumerate(models):
        base_fpr = m["fpr_base"]
        probe_fpr = m["fpr_probe"]
        if base_fpr > 0.01:
            mult = probe_fpr / base_fpr
            ax_fpr.text(
                i + width / 2,
                probe_fpr + 0.02,
                f"{mult:.1f}\u00d7",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="red" if mult > 2.0 else "black",
            )
        elif probe_fpr > 0:
            ax_fpr.text(
                i + width / 2,
                probe_fpr + 0.02,
                f"{probe_fpr:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax_fpr.set_ylabel("False Positive Rate")
    ax_fpr.set_title("(b) False Positive Rate")
    ax_fpr.set_xticks(x)
    ax_fpr.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_fpr.set_ylim(0, 1.15)
    ax_fpr.legend(loc="upper left", fontsize=8)
    ax_fpr.spines["top"].set_visible(False)
    ax_fpr.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / "temporal_robustness.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument(
        "--tools",
        type=str,
        help="Comma-separated list of tool names to include (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Only include results from this split (e.g., dev_public)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        logger.error(f"No results found in {results_dir}")
        return

    if args.tools:
        tool_set = {t.strip() for t in args.tools.split(",")}
        results = [r for r in results if r.get("tool_name") in tool_set]
    if args.split:
        results = [r for r in results if r.get("split_name") == args.split]

    logger.info(f"Loaded {len(results)} evaluation results")

    fig_tier_detection_rates(results, output_dir)
    fig_per_type_heatmap(results, output_dir)
    fig_cost_accuracy(results, output_dir)
    fig_overall_comparison(results, output_dir)
    fig_temporal_robustness(results_dir, output_dir)

    print(f"\nGenerated 5 figures in {output_dir}/")


if __name__ == "__main__":
    main()
