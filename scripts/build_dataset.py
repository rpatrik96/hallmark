#!/usr/bin/env python3
"""HALLMARK Dataset Build Orchestrator.

Rebuilds the entire HALLMARK benchmark dataset from scratch in 9 deterministic stages.
All stages operate on in-memory data; files are only written at the end (Stage 9).

Usage:
    python scripts/build_dataset.py --skip-scrape --skip-llm --seed 42 -v
    python scripts/build_dataset.py --skip-scrape --skip-llm --dry-run
    python scripts/build_dataset.py --help

Stages:
    1. Scrape/load valid entries
    2. Generate base hallucinations (perturbation, 14 types)
    3. Create initial splits (dev/test/hidden, tier-stratified)
    4. Scale up hallucinations (>=30 per type per public split)
    5. Collect external sources (real-world, LLM, GPTZero, journal)
    6. Integrate external sources (stratified into splits)
    7. Expand hidden test set
    8. Sanitize (absorbs all fix scripts + P0)
    9. Finalize (anonymize, resplit, write, validate)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger("build_dataset")

# Seed offsets per stage (deterministic, non-overlapping)
STAGE_OFFSETS = {
    1: 0,
    2: 1000,
    3: 2000,
    4: 3000,
    5: 4000,
    6: 5000,
    7: 6000,
    8: 7000,
    9: 8000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HALLMARK benchmark dataset from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument(
        "--build-date",
        type=str,
        default="2026-02-13",
        help="Fixed build date for reproducibility (default: 2026-02-13)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v1.0",
        help="Output directory for public splits (default: data/v1.0)",
    )
    parser.add_argument(
        "--hidden-dir",
        type=str,
        default="data/hidden",
        help="Output directory for hidden split (default: data/hidden)",
    )
    parser.add_argument(
        "--valid-entries",
        type=str,
        default="data/raw/valid_entries.jsonl",
        help="Path to cached valid entries JSONL",
    )
    parser.add_argument(
        "--journal-articles",
        type=str,
        default="data/raw/journal_articles.jsonl",
        help="Path to journal articles JSONL",
    )
    parser.add_argument(
        "--llm-entries",
        type=str,
        default="data/v1.0/llm_generated.jsonl",
        help="Path to LLM-generated entries JSONL",
    )
    parser.add_argument(
        "--gptzero-entries",
        type=str,
        default="",
        help="Path to GPTZero seed entries JSONL (optional)",
    )
    parser.add_argument(
        "--min-per-type",
        type=int,
        default=30,
        help="Minimum hallucinated entries per type in public splits (default: 30)",
    )
    parser.add_argument(
        "--hidden-per-type",
        type=int,
        default=15,
        help="Minimum hallucinated entries per type in hidden split (default: 15)",
    )
    parser.add_argument(
        "--hidden-valid",
        type=int,
        default=200,
        help="Target valid entries in hidden split (default: 200)",
    )
    parser.add_argument(
        "--total-hallucinated",
        type=int,
        default=100,
        help="Total base hallucinated entries to generate in Stage 2 (default: 100)",
    )
    parser.add_argument(
        "--skip-scrape", action="store_true", help="Load cached valid entries instead of scraping"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Load cached LLM entries instead of generating"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="", help="Directory for stage checkpoints (optional)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run pipeline without writing output files"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def _save_checkpoint(
    splits: dict,
    stage: int,
    checkpoint_dir: Path,
) -> None:
    """Save a checkpoint after a stage completes."""
    import json

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"stage_{stage}.json"

    data = {}
    for split_name, entries in splits.items():
        data[split_name] = [e.to_dict() for e in entries]

    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.debug("Checkpoint saved: %s", path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = PROJECT_ROOT / args.output_dir
    hidden_dir = PROJECT_ROOT / args.hidden_dir
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    logger.info("=" * 70)
    logger.info("HALLMARK Dataset Build Pipeline")
    logger.info("=" * 70)
    logger.info("Seed: %d | Build date: %s", args.seed, args.build_date)
    logger.info("Output: %s | Hidden: %s", output_dir, hidden_dir)
    logger.info("Min per type: public=%d, hidden=%d", args.min_per_type, args.hidden_per_type)

    t_start = time.time()
    timings: dict[int, float] = {}

    # ── Stage 1: Scrape / Load valid entries ────────────────────────────────
    logger.info("\n── Stage 1: Load valid entries ──")
    t0 = time.time()

    from stages.scrape import (
        stage_load_cached_valid,
        stage_load_journal_articles,
        stage_scrape_valid,
    )

    if args.skip_scrape:
        valid_path = PROJECT_ROOT / args.valid_entries
        valid_entries = stage_load_cached_valid(valid_path)
    else:
        valid_entries = stage_scrape_valid(
            rng=None,  # uses default
        )

    # Load journal articles as additional valid entries
    journal_path = PROJECT_ROOT / args.journal_articles
    journal_entries = stage_load_journal_articles(journal_path)

    timings[1] = time.time() - t0
    logger.info("Stage 1 complete: %d valid entries (%.1fs)", len(valid_entries), timings[1])

    # ── Stage 2: Generate base hallucinations ───────────────────────────────
    logger.info("\n── Stage 2: Generate base hallucinations ──")
    t0 = time.time()

    from stages.generate import stage_generate_hallucinations

    seed_2 = args.seed + STAGE_OFFSETS[2]
    hallucinated_entries = stage_generate_hallucinations(
        valid_entries,
        args.total_hallucinated,
        seed_2,
    )

    timings[2] = time.time() - t0
    logger.info(
        "Stage 2 complete: %d hallucinated entries (%.1fs)", len(hallucinated_entries), timings[2]
    )

    # ── Stage 3: Create initial splits ──────────────────────────────────────
    logger.info("\n── Stage 3: Create initial splits ──")
    t0 = time.time()

    from stages.split import stage_create_splits

    seed_3 = args.seed + STAGE_OFFSETS[3]
    splits = stage_create_splits(valid_entries, hallucinated_entries, seed_3)

    timings[3] = time.time() - t0
    for name, entries in splits.items():
        logger.info("  %s: %d entries", name, len(entries))
    logger.info("Stage 3 complete (%.1fs)", timings[3])

    if checkpoint_dir:
        _save_checkpoint(splits, 3, checkpoint_dir)

    # ── Stage 4: Scale up hallucinations ────────────────────────────────────
    logger.info("\n── Stage 4: Scale up hallucinations ──")
    t0 = time.time()

    from stages.scale_up import stage_scale_up

    seed_4 = args.seed + STAGE_OFFSETS[4]
    splits = stage_scale_up(splits, args.min_per_type, seed_4, args.build_date)

    timings[4] = time.time() - t0
    for name, entries in splits.items():
        logger.info("  %s: %d entries", name, len(entries))
    logger.info("Stage 4 complete (%.1fs)", timings[4])

    if checkpoint_dir:
        _save_checkpoint(splits, 4, checkpoint_dir)

    # ── Stage 5: Collect external sources ───────────────────────────────────
    logger.info("\n── Stage 5: Collect external sources ──")
    t0 = time.time()

    from stages.external import (
        stage_collect_real_world,
        stage_load_cached_llm,
        stage_load_gptzero_seed,
    )

    real_world = stage_collect_real_world(args.build_date)

    llm_path = PROJECT_ROOT / args.llm_entries
    llm_entries = stage_load_cached_llm(llm_path) if args.skip_llm else []

    gptzero_entries = []
    if args.gptzero_entries:
        gptzero_path = PROJECT_ROOT / args.gptzero_entries
        gptzero_entries = stage_load_gptzero_seed(gptzero_path)

    timings[5] = time.time() - t0
    logger.info(
        "Stage 5 complete: %d real-world, %d LLM, %d GPTZero (%.1fs)",
        len(real_world),
        len(llm_entries),
        len(gptzero_entries),
        timings[5],
    )

    # ── Stage 6: Integrate external sources ─────────────────────────────────
    logger.info("\n── Stage 6: Integrate external sources ──")
    t0 = time.time()

    from stages.integrate import stage_integrate_external

    seed_6 = args.seed + STAGE_OFFSETS[6]
    splits = stage_integrate_external(
        splits,
        real_world,
        llm_entries,
        gptzero_entries,
        journal_entries,
        seed_6,
        args.build_date,
    )

    timings[6] = time.time() - t0
    for name, entries in splits.items():
        logger.info("  %s: %d entries", name, len(entries))
    logger.info("Stage 6 complete (%.1fs)", timings[6])

    if checkpoint_dir:
        _save_checkpoint(splits, 6, checkpoint_dir)

    # ── Stage 7: Expand hidden test ─────────────────────────────────────────
    logger.info("\n── Stage 7: Expand hidden test ──")
    t0 = time.time()

    from stages.expand_hidden import stage_expand_hidden

    seed_7 = args.seed + STAGE_OFFSETS[7]
    splits = stage_expand_hidden(
        splits,
        args.hidden_per_type,
        args.hidden_valid,
        seed_7,
        args.build_date,
    )

    timings[7] = time.time() - t0
    logger.info("  test_hidden: %d entries", len(splits.get("test_hidden", [])))
    logger.info("Stage 7 complete (%.1fs)", timings[7])

    if checkpoint_dir:
        _save_checkpoint(splits, 7, checkpoint_dir)

    # ── Stage 8: Sanitize ───────────────────────────────────────────────────
    logger.info("\n── Stage 8: Sanitize ──")
    t0 = time.time()

    from stages.sanitize import stage_sanitize

    seed_8 = args.seed + STAGE_OFFSETS[8]
    splits = stage_sanitize(splits, seed_8)

    timings[8] = time.time() - t0
    logger.info("Stage 8 complete (%.1fs)", timings[8])

    # ── Stage 9: Finalize ───────────────────────────────────────────────────
    logger.info("\n── Stage 9: Finalize ──")
    t0 = time.time()

    from stages.finalize import stage_finalize

    seed_9 = args.seed + STAGE_OFFSETS[9]
    splits = stage_finalize(
        splits,
        output_dir,
        hidden_dir,
        seed_9,
        args.build_date,
        min_per_type_public=args.min_per_type,
        min_per_type_hidden=args.hidden_per_type,
        dry_run=args.dry_run,
    )

    timings[9] = time.time() - t0
    logger.info("Stage 9 complete (%.1fs)", timings[9])

    # ── Summary ─────────────────────────────────────────────────────────────
    t_total = time.time() - t_start

    logger.info("\n" + "=" * 70)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 70)

    # Summary table
    header = f"{'Split':<15} {'Total':>6} {'Valid':>6} {'Hall.':>6} {'Types':>6}"
    logger.info(header)
    logger.info("-" * len(header))

    for split_name, entries in splits.items():
        n_valid = sum(1 for e in entries if e.label == "VALID")
        n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
        n_types = len({e.hallucination_type for e in entries if e.hallucination_type})
        logger.info(
            "%s  %6d %6d %6d %6d",
            f"{split_name:<15}",
            len(entries),
            n_valid,
            n_hall,
            n_types,
        )

    grand_total = sum(len(entries) for entries in splits.values())
    logger.info("-" * len(header))
    logger.info("Grand total: %d entries", grand_total)

    # Timing summary
    logger.info("\nStage timings:")
    for stage, elapsed in sorted(timings.items()):
        logger.info("  Stage %d: %.1fs", stage, elapsed)
    logger.info("  Total: %.1fs", t_total)

    if args.dry_run:
        logger.info("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
