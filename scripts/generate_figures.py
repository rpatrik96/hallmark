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
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Colorblind-safe palette (IBM Design Library, extended)
COLORS = [
    "#648FFF",  # blue
    "#785EF0",  # purple
    "#DC267F",  # magenta
    "#FE6100",  # orange
    "#FFB000",  # gold
    "#44AA99",  # teal (Tol)
    "#882255",  # wine (Tol)
    "#DDCC77",  # sand (Tol)
    "#117733",  # green (Tol)
    "#999933",  # olive (Tol)
]

# Display names for tools in figures
DISPLAY_NAMES = {
    "doi_only": "DOI-only",
    "harc": "HaRC",
    "verify_citations": "verify-citations",
    "llm_openai": "GPT-5.1",
    "llm_anthropic": "Claude Sonnet 4.5",
    "llm_openrouter_deepseek_r1": "DeepSeek-R1",
    "llm_openrouter_deepseek_v3": "DeepSeek-V3.2",
    "llm_openrouter_qwen": "Qwen3-235B",
    "llm_openrouter_mistral": "Mistral Large",
    "llm_openrouter_gemini_flash": "Gemini Flash",
    "bibtexupdater": "bibtex-updater",
    "ensemble": "Ensemble",
    "doi_presence_heuristic": "DOI-heuristic",
}

# Tools excluded from tier detection rate chart (partial coverage, metrics not meaningful)
_PARTIAL_COVERAGE_TOOLS = {"harc", "verify_citations"}

# Tools shown in main results table (Table 5) — used to filter figures
_MAIN_TABLE_TOOLS = {
    "doi_only",
    "harc",
    "verify_citations",
    "llm_openai",
    "llm_openrouter_deepseek_r1",
    "llm_openrouter_deepseek_v3",
    "llm_openrouter_qwen",
    "llm_openrouter_mistral",
    "llm_openrouter_gemini_flash",
}


def _display_name(tool_name: str) -> str:
    return DISPLAY_NAMES.get(tool_name, tool_name)


def load_results(results_dir: Path) -> list[dict]:
    """Load dev_public evaluation result JSONs (skips CI, test, no-prescreening variants)."""
    results = []
    for path in sorted(results_dir.glob("*_dev_public.json")):
        # Skip CI bootstrap and no-prescreening variants
        if "_ci." in path.name or "_no_prescreening" in path.name:
            continue
        with open(path) as f:
            data = json.load(f)
        # Only include standard evaluation results (must have tool_name at top level)
        if "tool_name" in data:
            results.append(data)
    return results


def fig_tier_detection_rates(results: list[dict], output_dir: Path) -> None:
    """Bar chart: detection rate per tier, grouped by tool.

    Excludes partial-coverage tools (HaRC, verify-citations) whose per-tier
    metrics on a small subset are not meaningful.
    """
    # Filter to main-table full-coverage tools only
    _tier_tools = _MAIN_TABLE_TOOLS - _PARTIAL_COVERAGE_TOOLS
    filtered = [r for r in results if r["tool_name"] in _tier_tools]
    if not filtered:
        logger.warning("No full-coverage results for tier chart")
        return

    fig, ax = plt.subplots(figsize=(7, 3.8))

    tools = [_display_name(r["tool_name"]) for r in filtered]
    tiers = [1, 2, 3]
    tier_labels = ["Tier 1\n(Easy)", "Tier 2\n(Medium)", "Tier 3\n(Hard)"]

    n_tools = len(tools)
    x = np.arange(len(tiers))
    # Wider group span (0.85) and wider bars for readability
    group_width = 0.85
    width = group_width / max(n_tools, 1)

    for i, result in enumerate(filtered):
        tier_metrics = result.get("per_tier_metrics", {})
        rates = []
        for t in tiers:
            m = tier_metrics.get(str(t), {})
            rates.append(m.get("detection_rate", 0.0))

        offset = (i - n_tools / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width * 0.88,
            label=_display_name(result["tool_name"]),
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels only when few enough tools to avoid clutter
        if n_tools <= 5:
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
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "tier_detection_rates.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_per_type_heatmap(results: list[dict], output_dir: Path) -> None:
    """Heatmap: detection rate per hallucination type per tool.

    Excludes partial-coverage tools (HaRC, verify-citations) whose per-type
    metrics on a small subset are not meaningful — consistent with the tier chart.
    """
    _heatmap_tools = _MAIN_TABLE_TOOLS - _PARTIAL_COVERAGE_TOOLS
    results = [r for r in results if r["tool_name"] in _heatmap_tools]
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

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(tools) * 0.55 + 1.2)))
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

    fig.tight_layout()
    path = output_dir / "per_type_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_cost_accuracy(results: list[dict], output_dir: Path) -> None:
    """Scatter plot: F1 vs throughput (entries/second)."""
    results = [r for r in results if r["tool_name"] in _MAIN_TABLE_TOOLS]
    fig, ax = plt.subplots(figsize=(6, 4))

    # Collect points for label adjustment
    points = []
    for i, result in enumerate(results):
        f1 = result.get("f1_hallucination", 0)
        cost = result.get("cost_efficiency", None)
        if cost is None or cost == 0:
            continue

        name = _display_name(result["tool_name"])
        ax.scatter(
            cost,
            f1,
            s=140,
            color=COLORS[i % len(COLORS)],
            edgecolors="black",
            linewidth=0.7,
            zorder=3,
        )
        points.append((cost, f1, name, i))

    # Place labels with manual offset logic to avoid overlaps
    # Sort by x to detect horizontal crowding
    points.sort(key=lambda p: (p[0], p[1]))
    placed: list[tuple[float, float]] = []
    for cost, f1, name, _idx in points:
        # Default offset
        dx, dy = 10, 6
        # Check for nearby labels and adjust
        for px, py in placed:
            if abs(cost - px) < 0.15 and abs(f1 - py) < 0.08:
                dy = -14  # shift below
                break
        placed.append((cost, f1))
        ax.annotate(
            name,
            (cost, f1),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5) if len(points) > 6 else None,
        )

    ax.set_xlabel("Throughput (entries/sec)")
    ax.set_ylabel("F1 (Hallucination)")
    ax.set_title("Cost\u2013Accuracy Tradeoff")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "cost_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_overall_comparison(results: list[dict], output_dir: Path) -> None:
    """Horizontal grouped bar chart: primary metrics comparison across tools.

    Excludes partial-coverage tools (HaRC, verify-citations) whose metrics
    on a small subset are not meaningful for visual comparison.
    """
    _comparison_tools = _MAIN_TABLE_TOOLS - _PARTIAL_COVERAGE_TOOLS
    results = [r for r in results if r["tool_name"] in _comparison_tools]
    if not results:
        return

    tools = [_display_name(r["tool_name"]) for r in results]
    metrics = ["detection_rate", "f1_hallucination", "tier_weighted_f1"]
    metric_labels = ["Detection Rate", "F1", "Tier-weighted F1"]

    n_tools = len(tools)
    fig, ax = plt.subplots(figsize=(7, max(3.5, n_tools * 0.55 + 1.2)))

    y = np.arange(n_tools)
    height = 0.8 / len(metrics)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels, strict=True)):
        values = [r.get(metric, 0) for r in results]
        offset = (i - len(metrics) / 2 + 0.5) * height
        bars = ax.barh(
            y + offset,
            values,
            height * 0.88,
            label=label,
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values, strict=True):
            if val > 0:
                ax.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7,
                )

    ax.set_xlabel("Score")
    ax.set_title("Baseline Comparison")
    ax.set_yticks(y)
    ax.set_yticklabels(tools)
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # best tool at top

    fig.tight_layout()
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
