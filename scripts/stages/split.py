"""Stage 3: Create initial train/test/hidden splits with tier stratification.

Reuses logic from scripts/create_splits.py but operates purely in-memory.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from hallmark.dataset.schema import BenchmarkEntry

logger = logging.getLogger(__name__)

# Target split sizes
DEFAULT_SPLIT_CONFIG = {
    "dev_public": {"valid": 450, "hallucinated": 50},
    "test_public": {"valid": 270, "hallucinated": 30},
    "test_hidden": {"valid": 180, "hallucinated": 20},
}

# Target tier ratios within hallucinated entries
TIER_RATIOS = {1: 0.40, 2: 0.35, 3: 0.25}


def _allocate_by_tier(
    entries_by_tier: dict[int, list[BenchmarkEntry]],
    count: int,
    rng: random.Random,
) -> list[BenchmarkEntry]:
    """Allocate entries from tiers proportionally, removing from pools."""
    allocated: list[BenchmarkEntry] = []
    for tier, ratio in TIER_RATIOS.items():
        n = round(count * ratio)
        available = entries_by_tier.get(tier, [])
        if len(available) < n:
            logger.warning("Tier %d: requested %d but only %d available", tier, n, len(available))
            n = len(available)
        sampled = rng.sample(available, n)
        allocated.extend(sampled)
        for e in sampled:
            available.remove(e)

    # Fill rounding shortfall from any tier
    while len(allocated) < count:
        for tier in [1, 2, 3]:
            available = entries_by_tier.get(tier, [])
            if available:
                allocated.append(available.pop())
                break
        else:
            break

    return allocated[:count]


def stage_create_splits(
    valid_entries: list[BenchmarkEntry],
    hallucinated_entries: list[BenchmarkEntry],
    seed: int,
    split_config: dict[str, dict[str, int]] | None = None,
) -> dict[str, list[BenchmarkEntry]]:
    """Create benchmark splits maintaining tier ratios.

    Args:
        valid_entries: Pool of valid entries.
        hallucinated_entries: Pool of hallucinated entries.
        seed: Random seed.
        split_config: Optional override for split sizes.

    Returns:
        Dict mapping split name to list of entries.
    """
    split_config = split_config or DEFAULT_SPLIT_CONFIG
    rng = random.Random(seed)

    # Shuffle pools (copies to avoid mutating caller's lists)
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

        split_valid = valid_pool[valid_offset : valid_offset + n_valid]
        valid_offset += n_valid

        if len(split_valid) < n_valid:
            logger.warning(
                "%s: requested %d valid but only %d available",
                split_name,
                n_valid,
                len(split_valid),
            )

        split_hall = _allocate_by_tier(hall_by_tier, n_hall, rng)

        if len(split_hall) < n_hall:
            logger.warning(
                "%s: requested %d hallucinated but only %d allocated",
                split_name,
                n_hall,
                len(split_hall),
            )

        combined = split_valid + split_hall
        rng.shuffle(combined)
        splits[split_name] = combined

        logger.info(
            "%s: %d valid + %d hallucinated = %d total",
            split_name,
            len(split_valid),
            len(split_hall),
            len(combined),
        )

    return splits
