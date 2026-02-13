#!/usr/bin/env python3
"""Analyze baseline performance stratified by generation method.

This script evaluates how tools perform on different types of benchmark entries:
- perturbation: systematic perturbations of valid entries
- llm_generated: LLM-generated hallucinations
- real_world: actual hallucinations from published papers
- adversarial: hand-crafted to fool specific strategies
- scraped: valid entries from proceedings

Usage:
    python scripts/analyze_by_generation_method.py --split dev_public
    python scripts/analyze_by_generation_method.py --split dev_public --latex
    python scripts/analyze_by_generation_method.py --split dev_public --baselines harc,doi_only
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from hallmark.baselines.registry import check_available, run_baseline
from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import build_confusion_matrix

logger = logging.getLogger(__name__)


def compute_metrics_for_group(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, float]:
    """Compute detection rate, F1, and FPR for a group of entries."""
    cm = build_confusion_matrix(entries, predictions)

    return {
        "detection_rate": cm.detection_rate,
        "f1": cm.f1,
        "fpr": cm.false_positive_rate,
        "count": len(entries),
    }


def analyze_by_generation_method(
    entries: list[BenchmarkEntry],
    baseline_results: dict[str, dict[str, Prediction]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Group entries by generation_method and compute metrics for each baseline.

    Returns:
        Dict mapping baseline_name -> generation_method -> metrics
    """
    # Group entries by generation method
    method_groups: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        method_groups[entry.generation_method].append(entry)

    # Compute metrics for each baseline and generation method
    results: dict[str, dict[str, dict[str, float]]] = {}

    for baseline_name, predictions in baseline_results.items():
        results[baseline_name] = {}

        for method, method_entries in sorted(method_groups.items()):
            metrics = compute_metrics_for_group(method_entries, predictions)
            results[baseline_name][method] = metrics

    return results


def print_text_table(
    results: dict[str, dict[str, dict[str, float]]],
    baselines: list[str],
) -> None:
    """Print results as a formatted text table."""
    # Determine all generation methods present
    all_methods: set[str] = set()
    for baseline_metrics in results.values():
        all_methods.update(baseline_metrics.keys())

    methods = sorted(all_methods)

    # Print header
    print("\nPerformance by Generation Method")
    print("=" * 100)

    for method in methods:
        print(f"\n{method.upper()}")
        print("-" * 100)
        print(f"{'Baseline':<20} {'Count':>8} {'DR':>8} {'F1':>8} {'FPR':>8}")
        print("-" * 100)

        for baseline in baselines:
            if method not in results[baseline]:
                continue

            metrics = results[baseline][method]
            count = int(metrics["count"])
            dr = metrics["detection_rate"]
            f1 = metrics["f1"]
            fpr = metrics["fpr"]

            print(f"{baseline:<20} {count:>8} {dr:>8.3f} {f1:>8.3f} {fpr:>8.3f}")

    print("=" * 100)


def print_latex_table(
    results: dict[str, dict[str, dict[str, float]]],
    baselines: list[str],
) -> None:
    """Print results as a LaTeX table fragment."""
    # Determine all generation methods present (exclude scraped if all valid)
    all_methods: set[str] = set()
    for baseline_metrics in results.values():
        all_methods.update(baseline_metrics.keys())

    # Filter out "scraped" if it's present (typically all valid entries)
    methods = sorted([m for m in all_methods if m != "scraped"])

    if not methods:
        print("% No hallucinated generation methods found")
        return

    # Build LaTeX table
    num_baselines = len(baselines)
    col_spec = "l|" + "|".join(["rrr"] * num_baselines)

    print("\\begin{tabular}{" + col_spec + "}")
    print("\\toprule")

    # Multi-column header for baselines
    header_parts = []
    for baseline in baselines:
        header_parts.append(
            f"\\multicolumn{{3}}{{c{'|' if baseline != baselines[-1] else ''}}}{{{baseline}}}"
        )
    print("& " + " & ".join(header_parts) + " \\\\")

    # Sub-header with metric names
    subheader_parts = []
    for _ in baselines:
        subheader_parts.extend(["DR", "F1", "FPR"])
    print("Generation Method & " + " & ".join(subheader_parts) + " \\\\")

    print("\\midrule")

    # Data rows
    for method in methods:
        row_values = [method.replace("_", "\\_")]

        for baseline in baselines:
            if method in results[baseline]:
                metrics = results[baseline][method]
                dr = metrics["detection_rate"]
                f1 = metrics["f1"]
                fpr = metrics["fpr"]
                row_values.extend([f"{dr:.3f}", f"{f1:.3f}", f"{fpr:.3f}"])
            else:
                row_values.extend(["--", "--", "--"])

        print(" & ".join(row_values) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze baseline performance by generation method"
    )
    parser.add_argument(
        "--split",
        default="dev_public",
        help="Benchmark split to analyze (default: dev_public)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output LaTeX table fragment instead of text table",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="doi_only,verify_citations",
        help="Comma-separated list of baselines to run (default: doi_only,verify_citations)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load benchmark entries
    try:
        entries = load_split(
            args.split,
            version="v1.0",
            data_dir=str(args.data_dir) if args.data_dir else None,
        )
        logger.info(f"Loaded {len(entries)} entries from {args.split}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse baseline names
    baseline_names = [b.strip() for b in args.baselines.split(",")]

    # Check availability and run each baseline
    baseline_results: dict[str, dict[str, Prediction]] = {}

    for baseline_name in baseline_names:
        is_available, message = check_available(baseline_name)

        if not is_available:
            logger.warning(f"Skipping '{baseline_name}': {message}")
            continue

        logger.info(f"Running {baseline_name}...")
        try:
            predictions = run_baseline(baseline_name, entries)
            baseline_results[baseline_name] = {p.bibtex_key: p for p in predictions}
            logger.info(f"Got {len(predictions)} predictions from {baseline_name}")
        except Exception as e:
            logger.error(f"Failed to run baseline '{baseline_name}': {e}")
            continue

    if not baseline_results:
        print("Error: No baselines could be run", file=sys.stderr)
        sys.exit(1)

    # Analyze by generation method
    results = analyze_by_generation_method(entries, baseline_results)

    # Output results
    if args.latex:
        print_latex_table(results, list(baseline_results.keys()))
    else:
        print_text_table(results, list(baseline_results.keys()))


if __name__ == "__main__":
    main()
