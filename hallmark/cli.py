"""HALLMARK CLI: evaluate, contribute, leaderboard.

Usage:
    hallmark evaluate --split dev_public --baseline doi_only
    hallmark contribute --file entries.jsonl --contributor "name"
    hallmark leaderboard --split test_public
    hallmark stats --split dev_public
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hallmark.dataset.loader import get_statistics, load_split
from hallmark.dataset.schema import (
    BenchmarkEntry,
    Prediction,
    load_entries,
    load_predictions,
)
from hallmark.evaluation.metrics import evaluate


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hallmark",
        description="HALLMARK: Citation Hallucination Detection Benchmark",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a tool on a benchmark split")
    eval_parser.add_argument(
        "--split",
        default="dev_public",
        choices=["dev_public", "test_public", "test_hidden", "stress_test"],
        help="Benchmark split to evaluate on",
    )
    eval_parser.add_argument(
        "--baseline",
        type=str,
        help="Run a registered baseline (see 'hallmark list-baselines')",
    )
    eval_parser.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions JSONL file (alternative to --baseline)",
    )
    eval_parser.add_argument("--output", type=str, help="Path to write evaluation results JSON")
    eval_parser.add_argument("--data-dir", type=str, help="Override data directory")
    eval_parser.add_argument("--version", default="v1.0", help="Dataset version")
    eval_parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Max entries to evaluate (0 = all). Selects a stratified sample preserving hallucination ratio.",
    )
    eval_parser.add_argument(
        "--tool-name", default="unknown", help="Name of the tool being evaluated"
    )
    eval_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-type metrics and subtest accuracy",
    )
    eval_parser.add_argument(
        "--ci",
        action="store_true",
        default=False,
        help="Compute bootstrap confidence intervals (requires numpy)",
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
    contrib_parser.add_argument("--data-dir", type=str, help="Override data directory")

    # --- stats ---
    stats_parser = subparsers.add_parser("stats", help="Show statistics for a benchmark split")
    stats_parser.add_argument(
        "--split",
        default="dev_public",
        choices=["dev_public", "test_public", "stress_test"],
    )
    stats_parser.add_argument("--data-dir", type=str, help="Override data directory")
    stats_parser.add_argument("--version", default="v1.0", help="Dataset version")

    # --- leaderboard ---
    lb_parser = subparsers.add_parser("leaderboard", help="Show leaderboard for a split")
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

    # --- list-baselines ---
    subparsers.add_parser("list-baselines", help="List available baselines and their status")

    # --- validate-results ---
    val_parser = subparsers.add_parser(
        "validate-results", help="Validate pre-computed reference results"
    )
    val_parser.add_argument(
        "--results-dir",
        type=str,
        default="data/v1.0/baseline_results",
        help="Directory containing manifest.json and result files",
    )
    val_parser.add_argument(
        "--metadata",
        type=str,
        help="Path to dataset metadata.json for cross-validation",
    )
    val_parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject results with F1=0.0 (likely failed runs)",
    )

    # --- diagnose ---
    diag_parser = subparsers.add_parser("diagnose", help="Per-entry error analysis")
    diag_parser.add_argument(
        "--split",
        required=True,
        choices=["dev_public", "test_public", "test_hidden", "stress_test"],
    )
    diag_parser.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    diag_parser.add_argument("--type", default=None, help="Filter to specific hallucination type")
    diag_parser.add_argument(
        "--errors-only", action="store_true", help="Show only misclassified entries"
    )
    diag_parser.add_argument("--data-dir", type=str, help="Override data directory")
    diag_parser.add_argument("--version", default="v1.0", help="Dataset version")

    # --- history-append ---
    hist_parser = subparsers.add_parser(
        "history-append", help="Append current results to history JSONL log"
    )
    hist_parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory containing result JSONs"
    )
    hist_parser.add_argument(
        "--output",
        type=str,
        help="Path to history JSONL file (default: <results-dir>/history.jsonl)",
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
    elif args.command == "list-baselines":
        return _cmd_list_baselines()
    elif args.command == "diagnose":
        return _cmd_diagnose(args)
    elif args.command == "history-append":
        return _cmd_history_append(args)
    elif args.command == "validate-results":
        return _cmd_validate_results(args)

    return 0


def _stratified_sample(entries: list[BenchmarkEntry], n: int) -> list[BenchmarkEntry]:
    """Return a stratified sample preserving the hallucination ratio."""
    import random

    hallucinated = [e for e in entries if e.label == "HALLUCINATED"]
    valid = [e for e in entries if e.label != "HALLUCINATED"]

    ratio = len(hallucinated) / len(entries) if entries else 0
    n_hall = max(1, round(n * ratio))
    n_valid = n - n_hall

    rng = random.Random(42)
    sampled_hall = rng.sample(hallucinated, min(n_hall, len(hallucinated)))
    sampled_valid = rng.sample(valid, min(n_valid, len(valid)))

    combined = sampled_hall + sampled_valid
    rng.shuffle(combined)
    return combined


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

    # Subsample if requested (stratified by label)
    if args.max_entries and 0 < args.max_entries < len(entries):
        entries = _stratified_sample(entries, args.max_entries)
        logging.info(f"Sampled {len(entries)} entries (--max-entries {args.max_entries})")

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
        compute_ci=args.ci,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  HALLMARK Evaluation: {result.tool_name}")
    print(f"  Split: {result.split_name}")
    print(f"{'=' * 60}")
    print(f"  Entries:          {result.num_entries}")
    print(f"  Hallucinated:     {result.num_hallucinated}")
    print(f"  Valid:            {result.num_valid}")
    if result.num_uncertain > 0:
        print(f"  Uncertain:        {result.num_uncertain}")
    print(f"{'─' * 60}")
    print(f"  Detection Rate:   {result.detection_rate:.3f}")
    if result.detection_rate_ci:
        print(f"    95% CI: [{result.detection_rate_ci[0]:.3f}, {result.detection_rate_ci[1]:.3f}]")
    fpr_str = (
        f"{result.false_positive_rate:.3f}" if result.false_positive_rate is not None else "N/A"
    )
    print(f"  False Pos. Rate:  {fpr_str}")
    if result.fpr_ci:
        print(f"    95% CI: [{result.fpr_ci[0]:.3f}, {result.fpr_ci[1]:.3f}]")
    print(f"  F1 (Halluc.):     {result.f1_hallucination:.3f}")
    if result.f1_hallucination_ci:
        print(
            f"    95% CI: [{result.f1_hallucination_ci[0]:.3f}, {result.f1_hallucination_ci[1]:.3f}]"
        )
    print(f"  Tier-weighted F1: {result.tier_weighted_f1:.3f}")
    if result.tier_weighted_f1_ci:
        print(
            f"    95% CI: [{result.tier_weighted_f1_ci[0]:.3f}, {result.tier_weighted_f1_ci[1]:.3f}]"
        )
    if result.mcc is not None:
        print(f"  MCC:              {result.mcc:.3f}")
        if result.mcc_ci:
            print(f"    95% CI: [{result.mcc_ci[0]:.3f}, {result.mcc_ci[1]:.3f}]")
    if result.macro_f1 is not None:
        print(f"  Macro-F1:         {result.macro_f1:.3f}")
    if result.ece is not None:
        print(f"  ECE:              {result.ece:.3f}")
        if result.ece_ci:
            print(f"    95% CI: [{result.ece_ci[0]:.3f}, {result.ece_ci[1]:.3f}]")
    if result.cost_efficiency:
        print(f"  Entries/sec:      {result.cost_efficiency:.1f}")
    if result.mean_api_calls:
        print(f"  Mean API calls:   {result.mean_api_calls:.1f}")
    print(f"{'─' * 60}")

    if result.per_tier_metrics:
        print("  Per-tier breakdown:")
        for tier, metrics in sorted(result.per_tier_metrics.items()):
            if tier == 0:
                continue
            print(
                f"    Tier {tier}: DR={metrics['detection_rate']:.3f} "
                f"F1={metrics['f1']:.3f} (n={metrics['count']:.0f})"
            )

    # Detailed output if requested
    if args.detailed:
        print(f"{'─' * 60}")
        if result.per_type_metrics:
            print("  Per-type detection rates:")
            for h_type, metrics in sorted(result.per_type_metrics.items()):
                count = int(metrics["count"])
                dr = metrics["detection_rate"]
                print(f"    {h_type:<30} {count:>3} entries  DR={dr:.3f}")

        # Compute and display subtest accuracy
        from hallmark.evaluation.metrics import subtest_accuracy_table

        pred_map = {p.bibtex_key: p for p in predictions}
        subtest_acc = subtest_accuracy_table(entries, pred_map)
        if subtest_acc:
            print(f"{'─' * 60}")
            print("  Subtest accuracy:")
            for subtest_name, metrics in sorted(subtest_acc.items()):
                acc = metrics["accuracy"]
                count = int(metrics["count"])
                print(f"    {subtest_name:<25} {acc:.3f} ({count} entries)")

    print(f"{'=' * 60}\n")

    # Save results
    if args.output:
        Path(args.output).write_text(result.to_json())
        logging.info(f"Results written to {args.output}")

    return 0


def _cmd_contribute(args: argparse.Namespace) -> int:
    """Run contribute command."""
    from hallmark.contribution.pool_manager import PoolManager
    from hallmark.dataset.loader import DEFAULT_DATA_DIR

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

    print(f"\n{'=' * 50}")
    print(f"  HALLMARK Statistics: {args.split}")
    print(f"{'=' * 50}")
    print(f"  Total entries:    {stats['total']}")
    print(f"  Valid:            {stats['valid']}")
    print(f"  Hallucinated:     {stats['hallucinated']}")
    print(f"  Hall. rate:       {stats['hallucination_rate']:.1%}")
    print(f"{'─' * 50}")
    print("  Tier distribution:")
    for tier, count in sorted(stats["tier_distribution"].items()):
        print(f"    Tier {tier}: {count}")
    if stats["type_distribution"]:
        print("  Type distribution:")
        for h_type, count in sorted(stats["type_distribution"].items()):
            print(f"    {h_type}: {count}")
    print(f"{'=' * 50}\n")

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
            data = json.load(f)
        if data.get("split_name") == args.split:
            results.append(data)

    if not results:
        print("No evaluation results found.")
        return 0

    # Sort by F1
    results.sort(key=lambda r: r.get("f1_hallucination", 0), reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  HALLMARK Leaderboard: {args.split}")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<6}{'Tool':<25}{'F1':<8}{'DR':<8}{'FPR':<8}{'TW-F1':<8}{'MCC':<8}")
    print(f"  {'─' * 70}")

    for i, r in enumerate(results, 1):
        fpr_val = r.get("false_positive_rate")
        fpr_str = f"{fpr_val:<8.3f}" if fpr_val is not None else "N/A     "
        mcc_val = r.get("mcc")
        mcc_str = f"{mcc_val:<8.3f}" if mcc_val is not None else "N/A     "
        print(
            f"  {i:<6}{r.get('tool_name', '?'):<25}"
            f"{r.get('f1_hallucination', 0):<8.3f}"
            f"{r.get('detection_rate', 0):<8.3f}"
            f"{fpr_str}"
            f"{r.get('tier_weighted_f1', 0):<8.3f}"
            f"{mcc_str}"
        )

    print(f"{'=' * 70}\n")
    return 0


def _cmd_list_baselines() -> int:
    """List all registered baselines and their availability."""
    from hallmark.baselines.registry import check_available, get_registry

    registry = get_registry()
    print(f"\n{'=' * 70}")
    print("  HALLMARK Registered Baselines")
    print(f"{'=' * 70}")
    print(f"  {'Name':<22}{'Available':<12}{'Free':<8}{'Description'}")
    print(f"  {'─' * 66}")

    for name, info in sorted(registry.items()):
        avail, _msg = check_available(name)
        avail_str = "yes" if avail else "NO"
        free_str = "yes" if info.is_free else "no"
        print(f"  {name:<22}{avail_str:<12}{free_str:<8}{info.description}")

    print(f"{'=' * 70}\n")
    return 0


def _cmd_history_append(args: argparse.Namespace) -> int:
    """Append current results to history JSONL log."""
    import datetime

    results_dir = Path(args.results_dir)
    history_file = Path(args.output) if args.output else results_dir / "history.jsonl"

    count = 0
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tool_name": data.get("tool_name"),
            "split_name": data.get("split_name"),
            "f1_hallucination": data.get("f1_hallucination"),
            "detection_rate": data.get("detection_rate"),
            "false_positive_rate": data.get("false_positive_rate"),
            "tier_weighted_f1": data.get("tier_weighted_f1"),
            "num_entries": data.get("num_entries"),
            "cost_efficiency": data.get("cost_efficiency"),
        }
        with open(history_file, "a") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count += 1

    logging.info(f"Appended {count} results to {history_file}")
    return 0


def _cmd_validate_results(args: argparse.Namespace) -> int:
    """Validate pre-computed reference results."""
    from hallmark.evaluation.validate import validate_reference_results

    vr = validate_reference_results(
        results_dir=args.results_dir,
        metadata_path=args.metadata,
        strict=args.strict,
    )

    if vr.warnings:
        for w in vr.warnings:
            logging.warning(w)

    if vr.errors:
        for e in vr.errors:
            logging.error(e)
        print(f"\nValidation FAILED: {len(vr.errors)} error(s)")
        return 1

    print("\nValidation passed.")
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Per-entry error analysis."""
    try:
        entries = load_split(
            split=args.split,
            version=args.version,
            data_dir=args.data_dir,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    predictions = load_predictions(args.predictions)
    pred_map = {p.bibtex_key: p for p in predictions}

    # Filter by hallucination type if requested
    if args.type:
        entries = [e for e in entries if e.hallucination_type == args.type]

    correct = 0
    wrong = 0
    by_type: dict[str, dict[str, int]] = {}

    header = (
        f"{'bibtex_key':<30} {'true':<14} {'pred':<14} {'type':<28} {'tier':<5} {'conf':<6} reason"
    )
    print(f"\n{'=' * 120}")
    print(f"  HALLMARK Diagnose: {args.split}")
    print(f"{'=' * 120}")
    print(f"  {header}")
    print(f"  {'─' * 116}")

    for entry in entries:
        pred = pred_map.get(entry.bibtex_key)
        pred_label = pred.label if pred else "MISSING"
        true_label = entry.label
        confidence = f"{pred.confidence:.2f}" if pred and pred.confidence is not None else "N/A"
        reason = (pred.reason or "") if pred else ""
        reason_trunc = reason[:80] if len(reason) > 80 else reason
        h_type = entry.hallucination_type or "valid"
        tier = str(entry.difficulty_tier) if entry.difficulty_tier is not None else "-"

        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        else:
            wrong += 1

        # Track by-type breakdown
        if h_type not in by_type:
            by_type[h_type] = {"correct": 0, "wrong": 0}
        if is_correct:
            by_type[h_type]["correct"] += 1
        else:
            by_type[h_type]["wrong"] += 1

        if args.errors_only and is_correct:
            continue

        print(
            f"  {entry.bibtex_key:<30} {true_label:<14} {pred_label:<14} "
            f"{h_type:<28} {tier:<5} {confidence:<6} {reason_trunc}"
        )

    print(f"\n{'─' * 120}")
    total = correct + wrong
    print(f"  Total: {total}  Correct: {correct}  Wrong: {wrong}")
    if total > 0:
        print(f"  Accuracy: {correct / total:.3f}")
    print("\n  By-type breakdown:")
    for h_type, counts in sorted(by_type.items()):
        t = counts["correct"] + counts["wrong"]
        acc = counts["correct"] / t if t > 0 else 0.0
        print(f"    {h_type:<30} correct={counts['correct']}/{t}  acc={acc:.3f}")
    print(f"{'=' * 120}\n")

    return 0


def _run_baseline(name: str, entries: list[BenchmarkEntry]) -> list[Prediction]:
    """Run a baseline via the registry."""
    from hallmark.baselines.registry import run_baseline

    return run_baseline(name, entries)


if __name__ == "__main__":
    sys.exit(main())
