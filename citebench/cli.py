"""CiteBench CLI: evaluate, contribute, leaderboard.

Usage:
    citebench evaluate --split dev_public --baseline doi_only
    citebench contribute --file entries.jsonl --contributor "name"
    citebench leaderboard --split test_public
    citebench stats --split dev_public
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from citebench.dataset.loader import get_statistics, load_split
from citebench.dataset.schema import (
    BenchmarkEntry,
    Prediction,
    load_entries,
    load_predictions,
)
from citebench.evaluation.metrics import evaluate


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="citebench",
        description="CiteBench: Citation Hallucination Detection Benchmark",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a tool on a benchmark split"
    )
    eval_parser.add_argument(
        "--split",
        default="dev_public",
        choices=["dev_public", "test_public", "test_hidden"],
        help="Benchmark split to evaluate on",
    )
    eval_parser.add_argument(
        "--baseline",
        choices=["doi_only", "bibtexupdater", "llm_openai", "llm_anthropic"],
        help="Run a built-in baseline",
    )
    eval_parser.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions JSONL file (alternative to --baseline)",
    )
    eval_parser.add_argument(
        "--output", type=str, help="Path to write evaluation results JSON"
    )
    eval_parser.add_argument(
        "--data-dir", type=str, help="Override data directory"
    )
    eval_parser.add_argument(
        "--version", default="v1.0", help="Dataset version"
    )
    eval_parser.add_argument(
        "--tool-name", default="unknown", help="Name of the tool being evaluated"
    )

    # --- contribute ---
    contrib_parser = subparsers.add_parser(
        "contribute", help="Submit entries to the benchmark pool"
    )
    contrib_parser.add_argument(
        "--file", required=True, type=str, help="Path to JSONL file with entries"
    )
    contrib_parser.add_argument(
        "--contributor", required=True, type=str, help="Contributor name/identifier"
    )
    contrib_parser.add_argument(
        "--data-dir", type=str, help="Override data directory"
    )

    # --- stats ---
    stats_parser = subparsers.add_parser(
        "stats", help="Show statistics for a benchmark split"
    )
    stats_parser.add_argument(
        "--split",
        default="dev_public",
        choices=["dev_public", "test_public"],
    )
    stats_parser.add_argument("--data-dir", type=str, help="Override data directory")
    stats_parser.add_argument("--version", default="v1.0", help="Dataset version")

    # --- leaderboard ---
    lb_parser = subparsers.add_parser(
        "leaderboard", help="Show leaderboard for a split"
    )
    lb_parser.add_argument(
        "--split",
        default="test_public",
        choices=["dev_public", "test_public"],
    )
    lb_parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing evaluation result JSONs",
    )

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "evaluate":
        return _cmd_evaluate(args)
    elif args.command == "contribute":
        return _cmd_contribute(args)
    elif args.command == "stats":
        return _cmd_stats(args)
    elif args.command == "leaderboard":
        return _cmd_leaderboard(args)

    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    try:
        entries = load_split(
            split=args.split,
            version=args.version,
            data_dir=args.data_dir,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    logging.info(f"Loaded {len(entries)} entries from {args.split}")

    # Get predictions
    predictions: list[Prediction]
    tool_name = args.tool_name

    if args.predictions:
        predictions = load_predictions(args.predictions)
        logging.info(f"Loaded {len(predictions)} predictions from {args.predictions}")
    elif args.baseline:
        predictions = _run_baseline(args.baseline, entries)
        tool_name = args.baseline
    else:
        logging.error("Provide --predictions or --baseline")
        return 1

    # Evaluate
    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name=tool_name,
        split_name=args.split,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"  CiteBench Evaluation: {result.tool_name}")
    print(f"  Split: {result.split_name}")
    print(f"{'='*60}")
    print(f"  Entries:          {result.num_entries}")
    print(f"  Hallucinated:     {result.num_hallucinated}")
    print(f"  Valid:            {result.num_valid}")
    print(f"{'─'*60}")
    print(f"  Detection Rate:   {result.detection_rate:.3f}")
    print(f"  False Pos. Rate:  {result.false_positive_rate:.3f}")
    print(f"  F1 (Halluc.):     {result.f1_hallucination:.3f}")
    print(f"  Tier-weighted F1: {result.tier_weighted_f1:.3f}")
    if result.cost_efficiency:
        print(f"  Entries/sec:      {result.cost_efficiency:.1f}")
    if result.mean_api_calls:
        print(f"  Mean API calls:   {result.mean_api_calls:.1f}")
    print(f"{'─'*60}")

    if result.per_tier_metrics:
        print("  Per-tier breakdown:")
        for tier, metrics in sorted(result.per_tier_metrics.items()):
            if tier == 0:
                continue
            print(
                f"    Tier {tier}: DR={metrics['detection_rate']:.3f} "
                f"F1={metrics['f1']:.3f} (n={metrics['count']:.0f})"
            )
    print(f"{'='*60}\n")

    # Save results
    if args.output:
        Path(args.output).write_text(result.to_json())
        logging.info(f"Results written to {args.output}")

    return 0


def _cmd_contribute(args: argparse.Namespace) -> int:
    """Run contribute command."""
    from citebench.contribution.pool_manager import PoolManager
    from citebench.dataset.loader import DEFAULT_DATA_DIR

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    manager = PoolManager(data_dir)

    try:
        entries = load_entries(args.file)
    except Exception as e:
        logging.error(f"Failed to load entries: {e}")
        return 1

    path = manager.submit_contribution(entries, args.contributor)
    review = manager.review_contribution(path)

    print(f"\nContribution submitted: {review['total']} entries")
    print(f"  Valid: {review['valid']}")
    print(f"  Invalid: {review['invalid']}")

    for r in review["results"]:
        if not r["valid"]:
            print(f"  INVALID {r['key']}: {r['errors']}")
        elif r["warnings"]:
            print(f"  WARNING {r['key']}: {r['warnings']}")

    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Run stats command."""
    try:
        entries = load_split(
            split=args.split,
            version=args.version,
            data_dir=args.data_dir,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    stats = get_statistics(entries)

    print(f"\n{'='*50}")
    print(f"  CiteBench Statistics: {args.split}")
    print(f"{'='*50}")
    print(f"  Total entries:    {stats['total']}")
    print(f"  Valid:            {stats['valid']}")
    print(f"  Hallucinated:     {stats['hallucinated']}")
    print(f"  Hall. rate:       {stats['hallucination_rate']:.1%}")
    print(f"{'─'*50}")
    print("  Tier distribution:")
    for tier, count in sorted(stats["tier_distribution"].items()):
        print(f"    Tier {tier}: {count}")
    if stats["type_distribution"]:
        print("  Type distribution:")
        for h_type, count in sorted(stats["type_distribution"].items()):
            print(f"    {h_type}: {count}")
    print(f"{'='*50}\n")

    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Show leaderboard from saved evaluation results."""
    results_dir = Path(args.results_dir) if args.results_dir else Path("results")

    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        return 1

    results = []
    for path in results_dir.glob("*.json"):
        with open(path) as f:
            results.append(json.load(f))

    if not results:
        print("No evaluation results found.")
        return 0

    # Sort by F1
    results.sort(key=lambda r: r.get("f1_hallucination", 0), reverse=True)

    print(f"\n{'='*70}")
    print(f"  CiteBench Leaderboard: {args.split}")
    print(f"{'='*70}")
    print(f"  {'Rank':<6}{'Tool':<25}{'F1':<8}{'DR':<8}{'FPR':<8}{'TW-F1':<8}")
    print(f"  {'─'*62}")

    for i, r in enumerate(results, 1):
        print(
            f"  {i:<6}{r.get('tool_name', '?'):<25}"
            f"{r.get('f1_hallucination', 0):<8.3f}"
            f"{r.get('detection_rate', 0):<8.3f}"
            f"{r.get('false_positive_rate', 0):<8.3f}"
            f"{r.get('tier_weighted_f1', 0):<8.3f}"
        )

    print(f"{'='*70}\n")
    return 0


def _run_baseline(name: str, entries: list[BenchmarkEntry]) -> list[Prediction]:
    """Run a built-in baseline."""
    if name == "doi_only":
        from citebench.baselines.doi_only import run_doi_only

        return run_doi_only(entries)
    elif name == "bibtexupdater":
        from citebench.baselines.bibtexupdater import run_bibtex_check

        return run_bibtex_check(entries)
    elif name == "llm_openai":
        from citebench.baselines.llm_verifier import verify_with_openai

        return verify_with_openai(entries)
    elif name == "llm_anthropic":
        from citebench.baselines.llm_verifier import verify_with_anthropic

        return verify_with_anthropic(entries)
    else:
        raise ValueError(f"Unknown baseline: {name}")


if __name__ == "__main__":
    sys.exit(main())
