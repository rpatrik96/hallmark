#!/usr/bin/env python3
"""End-to-end orchestrator for rolling dataset updates.

Chains the full pipeline: scrape -> generate hallucinations -> create splits.
Optionally runs baselines on the resulting rolling split.

Usage:
    python scripts/update_rolling.py -v
    python scripts/update_rolling.py --years 2025 2026 --min-entries 10 -v
    python scripts/update_rolling.py --dry-run -v
    python scripts/update_rolling.py --run-baselines -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from hallmark.dataset.generator import (
    generate_tier1_batch,
    generate_tier2_batch,
    generate_tier3_batch,
)
from hallmark.dataset.schema import BenchmarkEntry, save_entries
from hallmark.dataset.scraper import ScraperConfig, scrape_proceedings

logger = logging.getLogger(__name__)

# Rolling split: single evaluation split matching v1.0 test_public dimensions
ROLLING_SPLIT_CONFIG = {
    "rolling_test": {"valid": 270, "hallucinated": 30},
}

# Tier ratios within hallucinated entries (same as create_splits.py)
TIER_RATIOS = {1: 0.40, 2: 0.35, 3: 0.25}

PIPELINE_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    current_year = date.today().year
    parser = argparse.ArgumentParser(
        description="Run the rolling dataset update pipeline",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--venues",
        nargs="+",
        default=["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR"],
        help="Venues to scrape",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[current_year - 1, current_year],
        help="Years to scrape (default: last 2 years)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: YYYYMMDD as int from today)",
    )
    parser.add_argument(
        "--min-entries",
        type=int,
        default=50,
        help="Minimum valid entries required (default: 50)",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run free baselines on the rolling split after creation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making network calls",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def compute_seed(seed: int | None) -> int:
    """Compute seed from today's date if not provided."""
    if seed is not None:
        return seed
    today = date.today()
    return today.year * 10000 + today.month * 100 + today.day


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the rolling update pipeline. Returns exit code."""
    today_str = date.today().isoformat()
    seed = compute_seed(args.seed)
    rolling_dir = Path(args.data_dir) / "rolling" / today_str

    logger.info("=== Rolling Dataset Update Pipeline ===")
    logger.info(f"Date: {today_str}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output: {rolling_dir}")
    logger.info(f"Venues: {args.venues}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Min entries: {args.min_entries}")

    if args.dry_run:
        logger.info(
            "[DRY RUN] Would scrape %d venues x %d years", len(args.venues), len(args.years)
        )
        logger.info("[DRY RUN] Would generate hallucinated entries with seed %d", seed)
        logger.info("[DRY RUN] Would create rolling_test split (270 valid + 30 hallucinated)")
        logger.info("[DRY RUN] Would write to %s", rolling_dir)
        if args.run_baselines:
            logger.info("[DRY RUN] Would run free baselines on rolling_test")
        return 0

    # Step 1: Scrape valid entries
    logger.info("Step 1/4: Scraping valid entries...")
    config = ScraperConfig(
        venues=args.venues,
        years=args.years,
    )
    valid_entries = scrape_proceedings(config)
    logger.info(f"  Scraped {len(valid_entries)} valid entries")

    # Health check: minimum entries
    if len(valid_entries) < args.min_entries:
        logger.error(
            f"Health check failed: only {len(valid_entries)} entries scraped, "
            f"minimum is {args.min_entries}"
        )
        return 1

    # Step 2: Generate hallucinated entries
    logger.info("Step 2/4: Generating hallucinated entries...")
    n_hall = ROLLING_SPLIT_CONFIG["rolling_test"]["hallucinated"]
    n_t1 = round(n_hall * TIER_RATIOS[1])
    n_t2 = round(n_hall * TIER_RATIOS[2])
    n_t3 = n_hall - n_t1 - n_t2  # remainder to tier 3

    tier1 = generate_tier1_batch(valid_entries, n_t1, seed=seed)
    tier2 = generate_tier2_batch(valid_entries, n_t2, seed=seed + 1)
    tier3 = generate_tier3_batch(valid_entries, n_t3, seed=seed + 2)
    hallucinated_entries = tier1 + tier2 + tier3

    logger.info(
        f"  Generated {len(hallucinated_entries)} hallucinated entries "
        f"(T1:{len(tier1)}, T2:{len(tier2)}, T3:{len(tier3)})"
    )

    # Health check: all tiers produced entries
    if not tier1 or not tier2 or not tier3:
        logger.error("Health check failed: one or more tiers produced zero entries")
        return 1

    # Step 3: Create rolling split
    logger.info("Step 3/4: Creating rolling split...")
    rolling_dir.mkdir(parents=True, exist_ok=True)

    from scripts.archive.create_splits import ROLLING_SPLIT_CONFIG as SPLIT_CFG
    from scripts.archive.create_splits import create_splits, update_metadata

    splits = create_splits(
        valid_entries,
        hallucinated_entries,
        seed=seed,
        split_config=SPLIT_CFG,
    )

    for split_name, entries in splits.items():
        path = rolling_dir / f"{split_name}.jsonl"
        save_entries(entries, path)
        logger.info(f"  Saved {split_name}: {len(entries)} entries -> {path}")

    # Write metadata with provenance
    update_metadata(rolling_dir, splits, rolling=True, seed=seed)

    # Step 4: Optionally run baselines
    if args.run_baselines:
        logger.info("Step 4/4: Running free baselines...")
        _run_baselines(rolling_dir, splits)
    else:
        logger.info("Step 4/4: Skipping baselines (use --run-baselines to enable)")

    # Print summary
    _print_summary(splits, rolling_dir, today_str, seed)
    return 0


def _run_baselines(rolling_dir: Path, splits: dict[str, list[BenchmarkEntry]]) -> None:
    """Run available free baselines on the rolling split."""
    try:
        from hallmark.baselines.registry import check_available, list_baselines, run_baseline
        from hallmark.evaluation.metrics import evaluate
    except ImportError:
        logger.warning("Baseline dependencies not available, skipping evaluation")
        return

    entries = splits.get("rolling_test", [])
    if not entries:
        logger.warning("No rolling_test split found, skipping baselines")
        return

    results_dir = rolling_dir / "results"
    results_dir.mkdir(exist_ok=True)

    for name in list_baselines():
        if not check_available(name):
            logger.info(f"  Skipping {name} (not available)")
            continue
        try:
            predictions = run_baseline(name, entries)
            result = evaluate(entries, predictions)
            result_path = results_dir / f"{name}.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"  {name}: F1={result.f1_hallucination:.3f}")
        except Exception:
            logger.exception(f"  {name} failed")


def _print_summary(
    splits: dict[str, list[BenchmarkEntry]],
    rolling_dir: Path,
    today_str: str,
    seed: int,
) -> None:
    """Print a summary table of the pipeline results."""
    print("\n" + "=" * 60)
    print(f"Rolling Dataset Update Summary ({today_str})")
    print("=" * 60)
    print(f"Output: {rolling_dir}")
    print(f"Seed:   {seed}")
    print()
    print(f"{'Split':<15} {'Total':<8} {'Valid':<8} {'Hall.':<8} {'T1':<5} {'T2':<5} {'T3':<5}")
    print("-" * 54)
    for split_name, entries in splits.items():
        n_valid = sum(1 for e in entries if e.label == "VALID")
        n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
        t1 = sum(1 for e in entries if e.difficulty_tier == 1)
        t2 = sum(1 for e in entries if e.difficulty_tier == 2)
        t3 = sum(1 for e in entries if e.difficulty_tier == 3)
        print(
            f"{split_name:<15} {len(entries):<8} {n_valid:<8} {n_hall:<8} {t1:<5} {t2:<5} {t3:<5}"
        )
    print()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
