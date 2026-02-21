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

from hallmark.dataset.loader import filter_by_tier, filter_by_type, get_statistics, load_split
from hallmark.dataset.schema import (
    BenchmarkEntry,
    Prediction,
    load_entries,
    load_predictions,
)
from hallmark.evaluation.metrics import (
    EvaluationResult,
    build_confusion_matrix,
    evaluate,
)

_SPLIT_CHOICES = ["dev_public", "test_public", "test_hidden", "stress_test"]


def _format_csv(result: EvaluationResult) -> str:
    """Format evaluation result as a CSV row with header."""
    fpr = result.false_positive_rate if result.false_positive_rate is not None else ""
    mcc = result.mcc if result.mcc is not None else ""
    ece = result.ece if result.ece is not None else ""
    auroc = result.auroc if result.auroc is not None else ""
    auprc = result.auprc if result.auprc is not None else ""
    macro_f1 = result.macro_f1 if result.macro_f1 is not None else ""
    header = (
        "tool,split,detection_rate,fpr,f1_hallucination,tier_weighted_f1,ece,mcc,"
        "auroc,auprc,macro_f1,coverage,coverage_adjusted_f1"
    )
    row = (
        f"{result.tool_name},{result.split_name},"
        f"{result.detection_rate:.3f},"
        f"{fpr if fpr == '' else f'{fpr:.3f}'},"
        f"{result.f1_hallucination:.3f},"
        f"{result.tier_weighted_f1:.3f},"
        f"{ece if ece == '' else f'{ece:.3f}'},"
        f"{mcc if mcc == '' else f'{mcc:.3f}'},"
        f"{auroc if auroc == '' else f'{auroc:.3f}'},"
        f"{auprc if auprc == '' else f'{auprc:.3f}'},"
        f"{macro_f1 if macro_f1 == '' else f'{macro_f1:.3f}'},"
        f"{result.coverage:.3f},"
        f"{result.coverage_adjusted_f1:.3f}"
    )
    return f"{header}\n{row}"


def _format_latex(result: EvaluationResult) -> str:
    """Format evaluation result as a LaTeX booktabs table row."""
    tool = result.tool_name.replace("_", r"\_")
    fpr = f"{result.false_positive_rate:.3f}" if result.false_positive_rate is not None else "--"
    mcc = f"{result.mcc:.3f}" if result.mcc is not None else "--"
    ece = f"{result.ece:.3f}" if result.ece is not None else "--"
    return (
        f"{tool} & {result.detection_rate:.3f} & {fpr} & "
        f"{result.f1_hallucination:.3f} & {result.tier_weighted_f1:.3f} & "
        f"{ece} & {mcc} \\\\"
    )


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
        choices=_SPLIT_CHOICES,
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
    eval_parser.add_argument("--tool-name", default=None, help="Name of the tool being evaluated")
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
    eval_parser.add_argument(
        "--filter-tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Filter entries to a specific difficulty tier (1, 2, or 3)",
    )
    eval_parser.add_argument(
        "--filter-type",
        type=str,
        default=None,
        metavar="TYPE",
        help='Filter entries to a specific hallucination type (e.g., "fabricated_doi")',
    )
    eval_parser.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "latex"],
        help="Output format: text (default), csv, or latex",
    )
    eval_parser.add_argument(
        "--strict",
        action="store_true",
        help="Require predictions for all entries; exit with error if any are missing",
    )
    eval_parser.add_argument(
        "--by-generation-method",
        action="store_true",
        help="Show per-generation-method metrics breakdown (perturbation, adversarial, etc.)",
    )
    eval_parser.add_argument(
        "--prescreening-breakdown",
        action="store_true",
        default=False,
        help=(
            "After main metrics, show a breakdown of pre-screening overrides vs. "
            "tool-only predictions (counts and per-group accuracy)"
        ),
    )
    eval_parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        metavar="PATH",
        help="Save predictions to a JSONL file at the given path",
    )
    eval_parser.add_argument(
        "--by-source",
        action="store_true",
        default=False,
        help="Show metrics broken down by API source combination",
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
        choices=_SPLIT_CHOICES,
    )
    stats_parser.add_argument("--data-dir", type=str, help="Override data directory")
    stats_parser.add_argument("--version", default="v1.0", help="Dataset version")

    # --- leaderboard ---
    lb_parser = subparsers.add_parser("leaderboard", help="Show leaderboard for a split")
    lb_parser.add_argument(
        "--split",
        default="test_public",
        choices=_SPLIT_CHOICES,
    )
    lb_parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing evaluation result JSONs",
    )
    lb_parser.add_argument(
        "--sort-by",
        default="cov_f1",
        choices=["f1", "cov_f1", "mcc", "tw_f1", "detection_rate", "ece"],
        help="Metric to sort leaderboard by (default: cov_f1)",
    )
    lb_parser.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "latex"],
        help="Output format: text (default), csv, or latex",
    )
    lb_parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help=(
            "Add significance cluster labels (A, B, C, ...) — tools sharing a letter are "
            "not statistically distinguishable (paired bootstrap, p > 0.05). "
            "Requires prediction JSONL files; see --predictions-dir."
        ),
    )
    lb_parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help=(
            "Directory containing prediction JSONL files named <tool>_<split>.jsonl. "
            "Defaults to --results-dir when --cluster is used."
        ),
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
        choices=_SPLIT_CHOICES,
    )
    diag_parser.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    diag_parser.add_argument("--type", default=None, help="Filter to specific hallucination type")
    diag_parser.add_argument(
        "--errors-only", action="store_true", help="Show only misclassified entries"
    )
    diag_parser.add_argument(
        "--full", action="store_true", help="Show full reason strings without truncation"
    )
    diag_parser.add_argument(
        "--gate",
        action="store_true",
        default=False,
        help="Exit with code 1 if any misclassifications are found",
    )
    diag_parser.add_argument("--data-dir", type=str, help="Override data directory")
    diag_parser.add_argument("--version", default="v1.0", help="Dataset version")

    # --- validate-predictions ---
    vp_parser = subparsers.add_parser(
        "validate-predictions", help="Validate a predictions JSONL file"
    )
    vp_parser.add_argument("--file", required=True, type=str, help="Path to predictions JSONL file")
    vp_parser.add_argument(
        "--split",
        default=None,
        choices=_SPLIT_CHOICES,
        help="Benchmark split to validate bibtex_keys against (optional)",
    )
    vp_parser.add_argument("--data-dir", type=str, help="Override data directory")
    vp_parser.add_argument("--version", default="v1.0", help="Dataset version")

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
    elif args.command == "validate-predictions":
        return _cmd_validate_predictions(args)

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
        if args.split == "test_hidden":
            print(
                "The test_hidden split is held out for official evaluation. "
                "Submit predictions at https://github.com/rpatrik96/hallmark .",
                file=sys.stderr,
            )
        else:
            logging.error(str(e))
        return 1

    logging.info(f"Loaded {len(entries)} entries from {args.split}")

    # Apply tier/type filters before sampling
    if args.filter_tier is not None:
        entries = filter_by_tier(entries, args.filter_tier)
        logging.info(f"Filtered to tier {args.filter_tier}: {len(entries)} entries remaining")

    if args.filter_type is not None:
        entries = filter_by_type(entries, args.filter_type)
        logging.info(f"Filtered to type '{args.filter_type}': {len(entries)} entries remaining")

    # Subsample if requested (stratified by label)
    if args.max_entries and 0 < args.max_entries < len(entries):
        entries = _stratified_sample(entries, args.max_entries)
        logging.info(f"Sampled {len(entries)} entries (--max-entries {args.max_entries})")

    # Get predictions
    predictions: list[Prediction]
    tool_name: str = args.tool_name or "unknown"

    if args.predictions:
        predictions = load_predictions(args.predictions)
        logging.info(f"Loaded {len(predictions)} predictions from {args.predictions}")
        # Infer tool name from filename if not explicitly provided
        if args.tool_name is None:
            tool_name = Path(args.predictions).stem
    elif args.baseline:
        try:
            predictions = _run_baseline(args.baseline, entries, split=args.split)
        except (ImportError, ValueError) as e:
            logging.error(
                f"Baseline '{args.baseline}' is not available: {e}\n"
                "Run 'hallmark list-baselines' to see available options."
            )
            return 1
        tool_name = args.baseline
    else:
        logging.error("Provide --predictions or --baseline")
        return 1

    # Filter predictions to match filtered entries
    if args.filter_tier is not None or args.filter_type is not None:
        entry_keys = {e.bibtex_key for e in entries}
        predictions = [p for p in predictions if p.bibtex_key in entry_keys]

    # Strict coverage check
    if args.strict:
        entry_keys = {e.bibtex_key for e in entries}
        pred_keys = {p.bibtex_key for p in predictions}
        missing = entry_keys - pred_keys
        if missing:
            print(
                f"ERROR: --strict mode: {len(missing)} entries have no predictions.",
                file=sys.stderr,
            )
            print(f"First 10 missing: {sorted(missing)[:10]}", file=sys.stderr)
            return 1

    # Evaluate
    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name=tool_name,
        split_name=args.split,
        compute_ci=args.ci,
    )

    if result.coverage < 1.0:
        logging.warning(
            "Coverage is %.1f%% (%d/%d entries). Missing predictions are treated as VALID.",
            result.coverage * 100,
            int(result.coverage * result.num_entries),
            result.num_entries,
        )

    # Build pred_map once (needed for confusion matrix and detailed sections)
    pred_map = {p.bibtex_key: p for p in predictions}

    # Dispatch on output format
    output_format: str = args.format

    if output_format == "csv":
        print(_format_csv(result))
    elif output_format == "latex":
        print(_format_latex(result))
    else:
        # --- text format (default) ---
        print(f"\n{'=' * 60}")
        print(f"  HALLMARK Evaluation: {result.tool_name}")
        print(f"  Split: {result.split_name}")
        print(f"{'=' * 60}")

        # Stress-test split notice: all entries are hallucinated by design
        if result.num_valid == 0:
            print("  Note: stress_test contains only hallucinated entries (no valid references).")
            print("  FPR and specificity are undefined. Use detection rate as the primary metric.")
            print(f"{'─' * 60}")

        prevalence = result.num_hallucinated / result.num_entries if result.num_entries else 0
        print(f"  Entries:          {result.num_entries}")
        print(f"  Hallucinated:     {result.num_hallucinated} ({prevalence:.1%})")
        print(f"  Valid:            {result.num_valid}")
        if result.num_uncertain > 0:
            print(f"  Uncertain:        {result.num_uncertain}")
        print("  NOTE: Prevalence varies across splits. Use MCC for cross-split comparison.")

        # Confusion matrix
        cm = build_confusion_matrix(entries, pred_map)
        print(f"{'─' * 60}")
        print("  Confusion Matrix:")
        print(f"    TP: {cm.tp:<6} FP: {cm.fp}")
        print(f"    FN: {cm.fn:<6} TN: {cm.tn}")

        print(f"{'─' * 60}")
        print(f"  Detection Rate:   {result.detection_rate:.3f}")
        if result.detection_rate_ci:
            print(
                f"    95% CI: [{result.detection_rate_ci[0]:.3f}, {result.detection_rate_ci[1]:.3f}]"
            )
        fpr_str = (
            f"{result.false_positive_rate:.3f}" if result.false_positive_rate is not None else "N/A"
        )
        print(f"  False Pos. Rate:  {fpr_str}")
        if result.fpr_ci:
            print(f"    95% CI: [{result.fpr_ci[0]:.3f}, {result.fpr_ci[1]:.3f}]")
        print(f"  F1 (Halluc.):     {result.f1_hallucination:.3f}")
        print(f"  Coverage:         {result.coverage:.1%}")
        if result.coverage < 1.0:
            print(f"  Cov-adj F1:       {result.coverage_adjusted_f1:.3f}")
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
        if result.auroc is not None:
            print(f"  AUROC:            {result.auroc:.3f}")
        if result.auprc is not None:
            print(f"  AUPRC:            {result.auprc:.3f}")
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

        # Generation method breakdown
        if args.by_generation_method:
            from hallmark.evaluation.metrics import per_generation_method_metrics

            gm_metrics = per_generation_method_metrics(entries, pred_map)
            if gm_metrics:
                print(f"{'─' * 60}")
                print("  Generation method breakdown:")
                print(f"    {'Method':<16} {'N':>5}  {'Det.Rate':>8}  {'F1':>6}")
                print(f"    {'─' * 42}")
                for method, m in sorted(gm_metrics.items()):
                    n = int(m["n"])
                    all_valid = all(
                        e.label == "VALID" for e in entries if e.generation_method == method
                    )
                    if all_valid:
                        dr_str = "       —"
                        f1_str = "     —"
                    else:
                        dr_str = f"{m['detection_rate']:8.3f}"
                        f1_str = f"{m['f1']:6.3f}"
                    print(f"    {method:<16} {n:>5}  {dr_str}  {f1_str}")

        # Detailed output if requested
        if args.detailed:
            print(f"{'─' * 60}")
            if result.per_type_metrics:
                # Re-compute per-type metrics with CIs if --ci is also set
                if args.ci:
                    from hallmark.evaluation.metrics import per_type_metrics as _per_type

                    type_metrics_ci = _per_type(entries, pred_map, compute_ci=True)
                else:
                    type_metrics_ci = result.per_type_metrics
                print("  Per-type detection rates:")
                for h_type, metrics in sorted(type_metrics_ci.items()):
                    count = int(metrics["count"])
                    dr = metrics["detection_rate"]
                    if args.ci and "dr_ci_lower" in metrics:
                        ci_lo = metrics["dr_ci_lower"]
                        ci_hi = metrics["dr_ci_upper"]
                        dr_str = f"{dr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
                    else:
                        dr_str = f"{dr:.3f}"
                    print(f"    {h_type:<30} {count:>3} entries  DR={dr_str}")

            # Compute and display subtest accuracy
            from hallmark.evaluation.metrics import subtest_accuracy_table

            subtest_acc = subtest_accuracy_table(entries, pred_map)
            if subtest_acc:
                print(f"{'─' * 60}")
                print("  Subtest accuracy:")
                for subtest_name, sub_metrics in sorted(subtest_acc.items()):
                    acc = sub_metrics["accuracy"]
                    count = int(sub_metrics["count"])
                    print(f"    {subtest_name:<25} {acc:.3f} ({count} entries)")

        # Pre-screening breakdown: auto-detect or explicit flag
        has_prescreening = any("[Pre-screening override]" in (p.reason or "") for p in predictions)
        if has_prescreening or args.prescreening_breakdown:
            from hallmark.baselines.prescreening import (
                compute_prescreening_breakdown,
                format_prescreening_breakdown,
            )

            true_labels = {e.bibtex_key: e.label for e in entries}
            breakdown = compute_prescreening_breakdown(predictions, true_labels)
            print(f"{'─' * 60}")
            for line in format_prescreening_breakdown(breakdown).splitlines():
                print(f"  {line}")

        print(f"{'=' * 60}\n")

    # By-source breakdown
    if args.by_source:
        from hallmark.evaluation import source_stratified_metrics

        src_metrics = source_stratified_metrics(entries, predictions)
        if src_metrics:
            print(f"\n{'─' * 60}")
            print("  Source-stratified metrics:")
            print(f"    {'Source':<30} {'N':>5}  {'Det.Rate':>8}  {'F1':>6}")
            print(f"    {'─' * 54}")
            for src_key, m in sorted(src_metrics.items()):
                n = int(m["count"])
                print(f"    {src_key:<30} {n:>5}  {m['detection_rate']:8.3f}  {m['f1']:6.3f}")

    # Save predictions
    if args.save_predictions:
        from hallmark.dataset.schema import save_predictions as _save_predictions

        _save_predictions(predictions, args.save_predictions)
        logging.info(f"Predictions written to {args.save_predictions}")

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

    if review["invalid"] > 0:
        return 1
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
    if "method_distribution" in stats:
        print("  Generation Method Distribution:")
        for method, count in sorted(stats["method_distribution"].items()):
            print(f"    {method}: {count}")
    print(f"{'=' * 50}\n")

    return 0


# Mapping from --sort-by choices to result dict field names
_LEADERBOARD_SORT_FIELDS: dict[str, str] = {
    "f1": "f1_hallucination",
    "cov_f1": "coverage_adjusted_f1",
    "mcc": "mcc",
    "tw_f1": "tier_weighted_f1",
    "detection_rate": "detection_rate",
    "ece": "ece",
}


def _compute_significance_clusters(
    tool_names: list[str],
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
) -> dict[str, str]:
    """Assign cluster labels (A, B, C, ...) via greedy graph coloring.

    Two tools share a cluster label when they are NOT significantly different
    (p_adjusted > 0.05 after paired bootstrap with Holm-Bonferroni correction).

    Tools are processed in rank order (the order of ``tool_names``).  Each tool
    gets the lowest letter not already used by any tool it IS significantly
    different from.

    Returns a dict mapping tool_name -> cluster letter.
    """
    from hallmark.evaluation.metrics import compare_tools

    comparisons = compare_tools(entries, tool_predictions)

    # Build the set of significantly-different pairs for O(1) lookup
    sig_pairs: set[frozenset[str]] = set()
    for c in comparisons:
        if c["significant"]:
            sig_pairs.add(frozenset([str(c["tool_a"]), str(c["tool_b"])]))

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cluster_labels: dict[str, str] = {}

    for tool in tool_names:
        # Collect letters already taken by tools this tool is significantly different from
        blocked: set[str] = set()
        for other, letter in cluster_labels.items():
            if frozenset([tool, other]) in sig_pairs:
                blocked.add(letter)
        # Assign the first available letter
        for letter in alphabet:
            if letter not in blocked:
                cluster_labels[tool] = letter
                break

    return cluster_labels


def _load_predictions_for_clustering(
    tool_names: list[str],
    split: str,
    predictions_dir: Path,
) -> dict[str, list[Prediction]]:
    """Load prediction JSONL files for a set of tools.

    Looks for files named ``<tool>_<split>.jsonl`` inside ``predictions_dir``.
    Logs a warning for any tool whose file is missing.
    """
    tool_predictions: dict[str, list[Prediction]] = {}
    for tool in tool_names:
        pred_file = predictions_dir / f"{tool}_{split}.jsonl"
        if pred_file.exists():
            tool_predictions[tool] = load_predictions(pred_file)
        else:
            logging.warning(
                f"No predictions file found for '{tool}' at {pred_file}; "
                "skipping this tool in significance clustering."
            )
    return tool_predictions


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

    # Sort by selected metric
    sort_field = _LEADERBOARD_SORT_FIELDS.get(args.sort_by, "f1_hallucination")
    # For ECE lower is better; for all others higher is better
    reverse = args.sort_by != "ece"

    def _sort_key(r: dict) -> float:  # type: ignore[type-arg]
        val = r.get(sort_field)
        if val is None:
            return float("inf") if not reverse else float("-inf")
        return float(val)

    results.sort(key=_sort_key, reverse=reverse)

    # Significance clustering (opt-in via --cluster)
    clusters: dict[str, str] = {}
    if args.cluster:
        predictions_dir = Path(args.predictions_dir) if args.predictions_dir else results_dir
        if not predictions_dir.exists():
            logging.error(f"Predictions directory not found: {predictions_dir}")
            return 1

        tool_names_ranked = [r.get("tool_name", "?") for r in results]
        try:
            entries = load_split(split=args.split)
        except FileNotFoundError as e:
            logging.error(f"Cannot load split for clustering: {e}")
            return 1

        tool_predictions = _load_predictions_for_clustering(
            tool_names_ranked, args.split, predictions_dir
        )
        if len(tool_predictions) < 2:
            logging.warning(
                "Fewer than 2 tools have prediction files; significance clustering requires "
                "at least 2. Skipping cluster column."
            )
        else:
            clusterable = [t for t in tool_names_ranked if t in tool_predictions]
            logging.info(
                f"Running significance clustering on {len(clusterable)} tools "
                f"({len(tool_predictions)} prediction files loaded)..."
            )
            clusters = _compute_significance_clusters(clusterable, entries, tool_predictions)

    output_format: str = args.format

    if output_format == "csv":
        header = (
            "rank,tool,split,f1_hallucination,detection_rate,fpr,tier_weighted_f1,mcc,ece,cov_f1"
        )
        if clusters:
            header += ",cluster"
        print(header)
        for i, r in enumerate(results, 1):
            tool = r.get("tool_name", "?")
            fpr_val = r.get("false_positive_rate")
            fpr_str = f"{fpr_val:.3f}" if fpr_val is not None else ""
            mcc_val = r.get("mcc")
            mcc_str = f"{mcc_val:.3f}" if mcc_val is not None else ""
            ece_val = r.get("ece")
            ece_str = f"{ece_val:.3f}" if ece_val is not None else ""
            cov_f1_val = r.get("coverage_adjusted_f1")
            cov_f1_str = f"{cov_f1_val:.3f}" if cov_f1_val is not None else ""
            row = (
                f"{i},{tool},{r.get('split_name', args.split)},"
                f"{r.get('f1_hallucination', 0):.3f},"
                f"{r.get('detection_rate', 0):.3f},"
                f"{fpr_str},{r.get('tier_weighted_f1', 0):.3f},{mcc_str},{ece_str},{cov_f1_str}"
            )
            if clusters:
                row += f",{clusters.get(tool, '')}"
            print(row)
    elif output_format == "latex":
        col_spec = "lrrrrrrr" + ("r" if clusters else "")
        print(r"\begin{tabular}{" + col_spec + r"}")
        print(r"\toprule")
        header_row = r"Tool & F1 & DR & FPR & TW-F1 & MCC & ECE & CovF1"
        if clusters:
            header_row += r" & Cluster"
        print(header_row + r" \\")
        print(r"\midrule")
        for r in results:
            tool = r.get("tool_name", "?")
            tool_tex = tool.replace("_", r"\_")
            fpr_val = r.get("false_positive_rate")
            fpr_str = f"{fpr_val:.3f}" if fpr_val is not None else "--"
            mcc_val = r.get("mcc")
            mcc_str = f"{mcc_val:.3f}" if mcc_val is not None else "--"
            ece_val = r.get("ece")
            ece_str = f"{ece_val:.3f}" if ece_val is not None else "--"
            cov_f1_val = r.get("coverage_adjusted_f1")
            cov_f1_str = f"{cov_f1_val:.3f}" if cov_f1_val is not None else "--"
            row = (
                f"{tool_tex} & {r.get('f1_hallucination', 0):.3f} & "
                f"{r.get('detection_rate', 0):.3f} & {fpr_str} & "
                f"{r.get('tier_weighted_f1', 0):.3f} & {mcc_str} & {ece_str} & {cov_f1_str}"
            )
            if clusters:
                row += f" & {clusters.get(tool, '--')}"
            print(row + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        if clusters:
            print(
                r"% Tools sharing a cluster letter are not statistically distinguishable"
                r" (paired bootstrap, p > 0.05, Holm-Bonferroni correction)."
            )
    else:
        width = 86 if clusters else 78
        print(f"\n{'=' * width}")
        print(f"  HALLMARK Leaderboard: {args.split}  (sorted by {args.sort_by})")
        print("  NOTE: Use MCC for cross-split comparison (prevalence-invariant).")
        print(f"{'=' * width}")
        if clusters:
            print(
                f"  {'Rank':<6}{'Tool':<25}{'F1':<8}{'DR':<8}{'FPR':<8}"
                f"{'TW-F1':<8}{'MCC':<8}{'CovF1':<8}{'Cluster':<8}"
            )
        else:
            print(
                f"  {'Rank':<6}{'Tool':<25}{'F1':<8}{'DR':<8}{'FPR':<8}"
                f"{'TW-F1':<8}{'MCC':<8}{'CovF1':<8}"
            )
        print(f"  {'─' * (width - 2)}")

        for i, r in enumerate(results, 1):
            tool = r.get("tool_name", "?")
            fpr_val = r.get("false_positive_rate")
            fpr_str = f"{fpr_val:<8.3f}" if fpr_val is not None else "N/A     "
            mcc_val = r.get("mcc")
            mcc_str = f"{mcc_val:<8.3f}" if mcc_val is not None else "N/A     "
            cov_f1_val = r.get("coverage_adjusted_f1")
            cov_f1_str = f"{cov_f1_val:<8.3f}" if cov_f1_val is not None else "N/A     "
            row = (
                f"  {i:<6}{tool:<25}"
                f"{r.get('f1_hallucination', 0):<8.3f}"
                f"{r.get('detection_rate', 0):<8.3f}"
                f"{fpr_str}"
                f"{r.get('tier_weighted_f1', 0):<8.3f}"
                f"{mcc_str}"
                f"{cov_f1_str}"
            )
            if clusters:
                row += f"{clusters.get(tool, ''):<8}"
            print(row)

        print(f"{'=' * width}")
        if clusters:
            print(
                "  * Tools sharing a cluster letter are not statistically distinguishable"
                " (paired bootstrap, p > 0.05, Holm-Bonferroni correction)."
            )
        print()
    return 0


def _cmd_list_baselines() -> int:
    """List all registered baselines and their availability."""
    from hallmark.baselines.registry import check_available, get_registry

    registry = get_registry()
    print(f"\n{'=' * 85}")
    print("  HALLMARK Registered Baselines")
    print(f"{'=' * 85}")
    print(f"  {'Name':<22}{'Available':<12}{'Free':<8}{'Confidence':<16}{'Description'}")
    print(f"  {'─' * 81}")

    for name, info in sorted(registry.items()):
        avail, msg = check_available(name)
        avail_str = "yes" if avail else "NO"
        free_str = "yes" if info.is_free else "no"
        print(
            f"  {name:<22}{avail_str:<12}{free_str:<8}{info.confidence_type:<16}{info.description}"
        )
        if not avail:
            print(f"    -> {msg}")

    print(f"{'=' * 85}\n")
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
            "mcc": data.get("mcc"),
            "ece": data.get("ece"),
            "auroc": data.get("auroc"),
            "auprc": data.get("auprc"),
            "coverage": data.get("coverage"),
            "coverage_adjusted_f1": data.get("coverage_adjusted_f1"),
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


def _cmd_validate_predictions(args: argparse.Namespace) -> int:
    """Validate a predictions JSONL file."""
    pred_file = Path(args.file)
    if not pred_file.exists():
        logging.error(f"File not found: {pred_file}")
        return 1

    errors: list[str] = []
    valid_preds: list[Prediction] = []

    with open(pred_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {lineno}: JSON parse error: {e}")
                continue
            try:
                pred = Prediction.from_dict(data)
                valid_preds.append(pred)
            except Exception as e:
                errors.append(f"Line {lineno}: Schema error: {e}")

    # Coverage check against split if requested
    coverage_info: str = ""
    if args.split is not None:
        try:
            entries = load_split(
                split=args.split,
                version=args.version,
                data_dir=args.data_dir,
            )
            entry_keys = {e.bibtex_key for e in entries}
            pred_keys = {p.bibtex_key for p in valid_preds}
            covered = pred_keys & entry_keys
            missing = entry_keys - pred_keys
            extra = pred_keys - entry_keys
            coverage_info = (
                f"\n  Split coverage ({args.split}):"
                f"\n    Entries in split:  {len(entry_keys)}"
                f"\n    Predictions matched: {len(covered)}"
                f"\n    Missing predictions: {len(missing)}"
                f"\n    Extra keys (not in split): {len(extra)}"
            )
            if missing:
                coverage_info += f"\n    First 10 missing: {sorted(missing)[:10]}"
        except FileNotFoundError as e:
            logging.warning(f"Could not load split for coverage check: {e}")

    total = len(valid_preds) + len(errors)
    print(f"\n{'=' * 60}")
    print(f"  HALLMARK Validate Predictions: {pred_file.name}")
    print(f"{'=' * 60}")
    print(f"  Total lines processed: {total}")
    print(f"  Valid predictions:     {len(valid_preds)}")
    print(f"  Invalid:               {len(errors)}")
    if coverage_info:
        print(coverage_info)
    if errors:
        print(f"\n{'─' * 60}")
        print("  Errors:")
        for err in errors:
            print(f"    {err}")
    print(f"{'=' * 60}\n")

    return 1 if errors else 0


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
        reason_trunc = (
            reason if args.full else (reason[:80] + "..." if len(reason) > 80 else reason)
        )
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

    if args.gate and wrong > 0:
        return 1
    return 0


def _run_baseline(
    name: str, entries: list[BenchmarkEntry], split: str | None = None
) -> list[Prediction]:
    """Run a baseline via the registry."""
    from hallmark.baselines.registry import run_baseline

    return run_baseline(name, entries, split=split)


if __name__ == "__main__":
    sys.exit(main())
