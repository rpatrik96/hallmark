#!/usr/bin/env python3
"""Create benchmark splits from valid and hallucinated entries.

Combines valid + hallucinated entries and splits into:
- dev_public: 500 entries (450 valid, 50 hallucinated)
- test_public: 300 entries (270 valid, 30 hallucinated)
- test_hidden: 200 entries (180 valid, 20 hallucinated)

Maintains tier ratios: ~40% Tier 1, ~35% Tier 2, ~25% Tier 3

IMPORTANT: After running this script, you MUST run scripts/sanitize_dataset.py
to ensure consistency with P0 fixes:
- Anonymize bibtex keys
- Normalize bibtex_type to inproceedings
- Strip hallucination-only fields
- Constrain years to 2021-2023 (except future_date)
- Eliminate title overlaps across splits

Usage:
    python scripts/create_splits.py \
        --valid-entries data/raw/valid_entries.jsonl \
        --hallucinated-entries data/raw/hallucinated_entries.jsonl \
        --output-dir data/v1.0 \
        --hidden-dir data/hidden
    python scripts/sanitize_dataset.py --data-dir data/v1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, load_entries, save_entries

logger = logging.getLogger(__name__)

# Target split sizes
SPLIT_CONFIG = {
    "dev_public": {"valid": 450, "hallucinated": 50},
    "test_public": {"valid": 270, "hallucinated": 30},
    "test_hidden": {"valid": 180, "hallucinated": 20},
}

# Rolling split config: single evaluation split matching test_public dimensions
ROLLING_SPLIT_CONFIG = {
    "rolling_test": {"valid": 270, "hallucinated": 30},
}

# Target tier ratios within hallucinated entries
TIER_RATIOS = {1: 0.40, 2: 0.35, 3: 0.25}


def allocate_by_tier(entries_by_tier: dict[int, list], count: int, rng: random.Random) -> list:
    """Allocate entries from tiers proportionally."""
    allocated = []
    for tier, ratio in TIER_RATIOS.items():
        n = round(count * ratio)
        available = entries_by_tier.get(tier, [])
        if len(available) < n:
            logger.warning(f"Tier {tier}: requested {n} but only {len(available)} available")
            n = len(available)
        sampled = rng.sample(available, n)
        allocated.extend(sampled)
        # Remove allocated entries from pool
        for e in sampled:
            available.remove(e)
    # If we're short due to rounding, fill from any tier
    while len(allocated) < count:
        for tier in [1, 2, 3]:
            available = entries_by_tier.get(tier, [])
            if available:
                allocated.append(available.pop())
                break
        else:
            break
    return allocated[:count]


def create_splits(
    valid_entries: list[BenchmarkEntry],
    hallucinated_entries: list[BenchmarkEntry],
    seed: int = 42,
    split_config: dict[str, dict[str, int]] | None = None,
) -> dict[str, list[BenchmarkEntry]]:
    """Create benchmark splits maintaining tier ratios."""
    split_config = split_config or SPLIT_CONFIG
    rng = random.Random(seed)

    # Shuffle
    valid_pool = list(valid_entries)
    rng.shuffle(valid_pool)

    # Group hallucinated by tier
    hall_by_tier: dict[int, list[BenchmarkEntry]] = defaultdict(list)
    for e in hallucinated_entries:
        hall_by_tier[e.difficulty_tier or 1].append(e)
    for tier in hall_by_tier:
        rng.shuffle(hall_by_tier[tier])

    splits: dict[str, list[BenchmarkEntry]] = {}
    valid_offset = 0

    for split_name, config in split_config.items():
        n_valid = config["valid"]
        n_hall = config["hallucinated"]

        # Take valid entries
        split_valid = valid_pool[valid_offset : valid_offset + n_valid]
        valid_offset += n_valid

        if len(split_valid) < n_valid:
            logger.warning(
                f"{split_name}: requested {n_valid} valid but only {len(split_valid)} available"
            )

        # Allocate hallucinated by tier
        split_hall = allocate_by_tier(hall_by_tier, n_hall, rng)

        if len(split_hall) < n_hall:
            logger.warning(
                f"{split_name}: requested {n_hall} hallucinated but only "
                f"{len(split_hall)} allocated"
            )

        # Combine and shuffle
        split_entries = split_valid + split_hall
        rng.shuffle(split_entries)
        splits[split_name] = split_entries

        logger.info(
            f"{split_name}: {len(split_valid)} valid + {len(split_hall)} hallucinated "
            f"= {len(split_entries)} total"
        )

    return splits


def update_metadata(
    output_dir: Path,
    splits: dict[str, list[BenchmarkEntry]],
    rolling: bool = False,
    seed: int | None = None,
) -> None:
    """Update metadata.json with actual split statistics."""
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    for split_name, entries in splits.items():
        valid_count = sum(1 for e in entries if e.label == "VALID")
        hall_count = sum(1 for e in entries if e.label == "HALLUCINATED")
        tier_counts: dict[str, int] = defaultdict(int)
        type_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            if e.difficulty_tier:
                tier_counts[str(e.difficulty_tier)] += 1
            if e.hallucination_type:
                type_counts[e.hallucination_type] += 1

        if "splits" not in metadata:
            metadata["splits"] = {}
        metadata["splits"][split_name] = {
            "file": f"{split_name}.jsonl",
            "total": len(entries),
            "valid": valid_count,
            "hallucinated": hall_count,
            "tier_distribution": dict(tier_counts),
            "type_distribution": dict(type_counts),
        }

    if rolling:
        from datetime import date as date_cls

        today = date_cls.today().isoformat()
        metadata["rolling_metadata"] = {
            "created": today,
            "scrape_date": today,
            "seed": seed,
            "pipeline_version": "1.0.0",
        }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Updated metadata at {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark splits")
    parser.add_argument("--valid-entries", type=str, required=True)
    parser.add_argument("--hallucinated-entries", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/v1.0")
    parser.add_argument("--hidden-dir", type=str, default="data/hidden")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Create a rolling split instead of standard v1.0 splits",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Determine split config based on mode
    if args.rolling:
        from datetime import date

        split_config = ROLLING_SPLIT_CONFIG
        args.output_dir = f"data/rolling/{date.today().isoformat()}"
        logger.info("Rolling mode: writing to %s", args.output_dir)
    else:
        split_config = SPLIT_CONFIG

    valid = load_entries(args.valid_entries)
    hallucinated = load_entries(args.hallucinated_entries)
    logger.info(f"Loaded {len(valid)} valid + {len(hallucinated)} hallucinated entries")

    total_needed_valid = sum(c["valid"] for c in split_config.values())
    total_needed_hall = sum(c["hallucinated"] for c in split_config.values())
    logger.info(f"Need {total_needed_valid} valid + {total_needed_hall} hallucinated")

    if len(valid) < total_needed_valid:
        logger.warning(
            f"Only {len(valid)} valid entries available, need {total_needed_valid}. "
            "Splits will be smaller."
        )
    if len(hallucinated) < total_needed_hall:
        logger.warning(
            f"Only {len(hallucinated)} hallucinated entries available, "
            f"need {total_needed_hall}. Splits will be smaller."
        )

    splits = create_splits(valid, hallucinated, args.seed, split_config=split_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.rolling:
        hidden_dir = Path(args.hidden_dir)
        hidden_dir.mkdir(parents=True, exist_ok=True)

    for split_name, entries in splits.items():
        if split_name == "test_hidden" and not args.rolling:
            path = Path(args.hidden_dir) / "test_hidden.jsonl"
        else:
            path = output_dir / f"{split_name}.jsonl"
        save_entries(entries, path)
        logger.info(f"Saved {split_name}: {len(entries)} entries -> {path}")

    update_metadata(output_dir, splits, rolling=args.rolling, seed=args.seed)

    # Print summary
    print("\nSplit Summary:")
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


if __name__ == "__main__":
    main()
