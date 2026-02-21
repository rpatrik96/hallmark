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
    print(f"{'Rank':<6} {'Baseline':<25} {'F1':<8} {'DR':<8} {'FPR':<8} {'TW-F1':<8}")
    print("-" * 80)

    for rank, result in enumerate(sorted_results, start=1):
        print(
            f"{rank:<6} "
            f"{result['name']:<25} "
            f"{result['f1_hallucination']:<8.3f} "
            f"{result['detection_rate']:<8.3f} "
            f"{result['false_positive_rate']:<8.3f} "
            f"{result['tier_weighted_f1']:<8.3f}"
        )

    print("=" * 80)
    print(
        "Metrics: F1 = F1-Hallucination, DR = Detection Rate, "
        "FPR = False Positive Rate, TW-F1 = Tier-Weighted F1"
    )
    print("=" * 80 + "\n")


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
            rank_tools,
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

        # Rank tools
        rankings = rank_tools(entries, tool_predictions)
        logger.info("Tool rankings:")
        for rank, (tool, score) in enumerate(rankings, start=1):
            logger.info(f"  {rank}. {tool}: {score:.3f}")

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

    # Print leaderboard
    if results:
        print_leaderboard(results)

        # Try to build results matrix
        save_results_matrix_if_available(entries, results, args.split, args.output_dir)

        logger.info(f"Successfully completed {len(results)}/{len(baselines)} baselines")
    else:
        logger.error("No baselines completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
