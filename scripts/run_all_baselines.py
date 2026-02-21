#!/usr/bin/env python3
"""Main orchestrator script for running all HALLMARK baselines and producing a leaderboard.  [evaluation]

This script runs all available baselines (or a subset) on a specified dataset split,
evaluates their performance, and generates a leaderboard with detailed metrics.

Usage:
    python scripts/run_all_baselines.py --split dev_public --output-dir results/
    python scripts/run_all_baselines.py --baselines random,llm_judge --parallel
    python scripts/run_all_baselines.py --baselines free --skip-unavailable
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hallmark.baselines.registry import check_available, list_baselines, run_baseline
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HALLMARK baselines and generate leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev_public",
        choices=["dev_public", "test_public", "test_hidden"],
        help="Dataset split to evaluate on (default: dev_public)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for result JSONs (default: results/)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="free",
        help='Comma-separated list, "all", or "free" (default: free)',
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run baselines concurrently using ThreadPoolExecutor",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        default=True,
        help="Skip baselines whose dependencies aren't installed (default: True)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Dataset version (default: v1.0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Run each baseline N times and report mean/std of metrics (default: 1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def determine_baselines(baseline_arg: str, skip_unavailable: bool) -> list[str]:
    """Determine which baselines to run based on CLI argument.

    Args:
        baseline_arg: CLI argument value (comma-separated, "all", or "free")
        skip_unavailable: Whether to skip baselines with missing dependencies

    Returns:
        List of baseline names to run
    """
    if baseline_arg.lower() == "all":
        candidates = list_baselines(free_only=False)
    elif baseline_arg.lower() == "free":
        candidates = list_baselines(free_only=True)
    else:
        candidates = [name.strip() for name in baseline_arg.split(",")]

    # Filter by availability
    available_baselines = []
    for name in candidates:
        is_available, message = check_available(name)
        if is_available:
            available_baselines.append(name)
            logger.debug(f"Baseline '{name}' is available")
        else:
            if skip_unavailable:
                logger.warning(f"Skipping '{name}': {message}")
            else:
                logger.error(f"Baseline '{name}' not available: {message}")
                sys.exit(1)

    if not available_baselines:
        logger.error("No available baselines to run")
        sys.exit(1)

    return available_baselines


def run_single_baseline(
    name: str,
    entries: list[Any],
    split: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Run a single baseline and save results.

    Args:
        name: Baseline name
        entries: Benchmark entries to evaluate on
        split: Dataset split name
        output_dir: Directory to save results

    Returns:
        Dictionary with baseline results or None if failed
    """
    try:
        logger.info(f"Running {name}...")

        # Run baseline
        predictions = run_baseline(name, entries)

        # Evaluate
        result = evaluate(entries, predictions, tool_name=name, split_name=split)

        # Save results
        output_file = output_dir / f"{name}_{split}.json"
        output_file.write_text(result.to_json())
        logger.debug(f"Saved results to {output_file}")

        # Print summary
        summary = (
            f"{name}: "
            f"F1={result.f1_hallucination:.3f} "
            f"DR={result.detection_rate:.3f} "
            f"FPR={result.false_positive_rate:.3f} "
            f"TW-F1={result.tier_weighted_f1:.3f}"
        )
        logger.info(summary)

        return {
            "name": name,
            "f1_hallucination": result.f1_hallucination,
            "detection_rate": result.detection_rate,
            "false_positive_rate": result.false_positive_rate,
            "tier_weighted_f1": result.tier_weighted_f1,
            "mcc": result.mcc,
            "predictions": predictions,
        }

    except Exception as e:
        logger.error(f"Failed to run baseline '{name}': {e}", exc_info=True)
        return None


def print_leaderboard(results: list[dict[str, Any]]) -> None:
    """Print formatted leaderboard table sorted by F1 score.

    Args:
        results: List of baseline result dictionaries
    """
    if not results:
        logger.warning("No results to display")
        return

    # Sort by F1 score (descending)
    sorted_results = sorted(results, key=lambda x: x["f1_hallucination"], reverse=True)

    # Print table
    print("\n" + "=" * 80)
    print("HALLMARK BASELINE LEADERBOARD")
    print("=" * 80)
    print(f"{'Rank':<6} {'Baseline':<25} {'F1':<8} {'DR':<8} {'FPR':<8} {'TW-F1':<8} {'MCC':<8}")
    print("-" * 88)

    for rank, result in enumerate(sorted_results, start=1):
        fpr_val = result.get("false_positive_rate")
        fpr_str = f"{fpr_val:<8.3f}" if fpr_val is not None else "N/A     "
        mcc_val = result.get("mcc")
        mcc_str = f"{mcc_val:<8.3f}" if mcc_val is not None else "N/A     "
        print(
            f"{rank:<6} "
            f"{result['name']:<25} "
            f"{result['f1_hallucination']:<8.3f} "
            f"{result['detection_rate']:<8.3f} "
            f"{fpr_str} "
            f"{result['tier_weighted_f1']:<8.3f} "
            f"{mcc_str}"
        )

    print("=" * 88)
    print(
        "Metrics: F1 = F1-Hallucination, DR = Detection Rate, "
        "FPR = False Positive Rate, TW-F1 = Tier-Weighted F1, "
        "MCC = Matthews Correlation Coefficient"
    )
    print("=" * 88 + "\n")


def save_results_matrix_if_available(
    entries: list[Any],
    results: list[dict[str, Any]],
    split: str,
    output_dir: Path,
) -> None:
    """Try to build and save results matrix if ranking module is available.

    Args:
        entries: Benchmark entries with ground truth
        results: List of baseline result dictionaries
        split: Dataset split name
        output_dir: Directory to save matrix
    """
    try:
        from hallmark.evaluation.ranking import (
            build_results_matrix,
            rank_tools_plackett_luce,
            save_results_matrix,
        )

        logger.info("Building results matrix...")

        # Collect tool predictions
        tool_predictions = {r["name"]: r["predictions"] for r in results}

        # Build and save matrix
        entry_keys, tool_names, matrix_data = build_results_matrix(entries, tool_predictions)
        matrix_file = output_dir / f"results_matrix_{split}.csv"
        save_results_matrix(entry_keys, tool_names, matrix_data, str(matrix_file))
        logger.info(f"Saved results matrix to {matrix_file}")

        # Rank tools with CIs
        ranked_with_ci = rank_tools_plackett_luce(
            entry_keys, tool_names, matrix_data, compute_ci=True
        )
        ranked, score_cis = ranked_with_ci

        print("\n" + "=" * 88)
        print("PLACKETT-LUCE RANKINGS")
        print("=" * 88)
        print(f"{'Rank':<6} {'Tool':<25} {'Score':<10} {'95% CI':<20}")
        print("-" * 88)
        for rank, (tool, score) in enumerate(ranked, start=1):
            ci = score_cis.get(tool)
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
            print(f"{rank:<6} {tool:<25} {score:<10.4f} {ci_str:<20}")
        print("=" * 88 + "\n")

    except ImportError:
        logger.debug("Ranking module not available, skipping matrix generation")
    except Exception as e:
        logger.warning(f"Failed to build results matrix: {e}")


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.verbose)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load entries once
    logger.info(f"Loading {args.split} split (version={args.version})...")
    try:
        entries = load_split(
            split=args.split,
            version=args.version,
            data_dir=str(args.data_dir) if args.data_dir else None,
        )
        logger.info(f"Loaded {len(entries)} entries")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Determine which baselines to run
    baselines = determine_baselines(args.baselines, args.skip_unavailable)
    logger.info(f"Running {len(baselines)} baselines: {', '.join(baselines)}")

    # Run baselines
    results = []
    if args.parallel:
        logger.info("Running baselines in parallel (max_workers=4)...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    run_single_baseline,
                    name,
                    entries,
                    args.split,
                    args.output_dir,
                ): name
                for name in baselines
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        logger.info("Running baselines sequentially...")
        for name in baselines:
            result = run_single_baseline(name, entries, args.split, args.output_dir)
            if result is not None:
                results.append(result)

    # Multi-run variance reporting
    if args.n_runs > 1 and results:
        logger.info(f"Running {args.n_runs - 1} additional runs for variance estimation...")
        from collections import defaultdict

        run_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        # Collect metrics from first run
        for r in results:
            run_metrics[r["name"]].append(
                {
                    "f1": r["f1_hallucination"],
                    "dr": r["detection_rate"],
                    "mcc": r.get("mcc", 0.0) or 0.0,
                }
            )
        # Additional runs
        for run_idx in range(1, args.n_runs):
            logger.info(f"  Run {run_idx + 1}/{args.n_runs}...")
            for name in baselines:
                try:
                    preds = run_baseline(name, entries)
                    res = evaluate(entries, preds, tool_name=name, split_name=args.split)
                    run_metrics[name].append(
                        {
                            "f1": res.f1_hallucination,
                            "dr": res.detection_rate,
                            "mcc": res.mcc or 0.0,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"  Run {run_idx + 1} failed for {name}: {exc}")

        # Print variance table
        import statistics

        print("\n" + "=" * 88)
        print("CROSS-RUN VARIANCE (mean +/- std)")
        print("=" * 88)
        print(f"{'Baseline':<25} {'F1':>14} {'DR':>14} {'MCC':>14}")
        print("-" * 88)
        for name in sorted(run_metrics):
            metrics_list = run_metrics[name]
            if len(metrics_list) < 2:
                continue
            f1s = [m["f1"] for m in metrics_list]
            drs = [m["dr"] for m in metrics_list]
            mccs = [m["mcc"] for m in metrics_list]
            print(
                f"{name:<25} "
                f"{statistics.mean(f1s):.3f}+/-{statistics.stdev(f1s):.3f}  "
                f"{statistics.mean(drs):.3f}+/-{statistics.stdev(drs):.3f}  "
                f"{statistics.mean(mccs):.3f}+/-{statistics.stdev(mccs):.3f}"
            )
        print("=" * 88 + "\n")

    # Print leaderboard
    if results:
        print_leaderboard(results)

        # Pairwise tool comparison with multiple comparison correction
        if len(results) >= 2:
            try:
                from hallmark.evaluation.metrics import compare_tools as compare_tools_fn

                tool_preds = {r["name"]: r["predictions"] for r in results}
                comparisons = compare_tools_fn(entries, tool_preds, n_bootstrap=1000, seed=42)
                if comparisons:
                    print("\n" + "=" * 88)
                    print("PAIRWISE TOOL COMPARISONS (F1, Holm-corrected)")
                    print("=" * 88)
                    print(f"{'Tool A':<20} {'Tool B':<20} {'Diff':>8} {'p-adj':>8} {'Sig':>5}")
                    print("-" * 88)
                    for c in comparisons:
                        sig_str = "*" if c["significant"] else ""
                        print(
                            f"{c['tool_a']:<20} {c['tool_b']:<20} "
                            f"{c['observed_diff']:>+8.3f} "
                            f"{c['p_value_adjusted']:>8.3f} {sig_str:>5}"
                        )
                    print("=" * 88 + "\n")
            except Exception as exc:
                logger.debug(f"Pairwise comparison skipped: {exc}")

        # Try to build results matrix and PL ranking with CIs
        save_results_matrix_if_available(entries, results, args.split, args.output_dir)

        logger.info(f"Successfully completed {len(results)}/{len(baselines)} baselines")
    else:
        logger.error("No baselines completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
