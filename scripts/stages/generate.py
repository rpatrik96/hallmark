"""Stage 2: Generate base hallucinated entries via perturbation.

Wraps hallmark.dataset.generator batch functions with isolated RNG.
Produces hallucinated entries across all 14 types in 3 tiers.
"""

from __future__ import annotations

import logging
import random

from hallmark.dataset.schema import BenchmarkEntry

logger = logging.getLogger(__name__)


# Tier target ratios: ~40% T1, ~35% T2, ~25% T3
TIER_RATIOS = {1: 0.40, 2: 0.35, 3: 0.25}


def stage_generate_hallucinations(
    valid_entries: list[BenchmarkEntry],
    total_count: int,
    seed: int,
) -> list[BenchmarkEntry]:
    """Generate base hallucinated entries from valid entries using perturbation.

    Args:
        valid_entries: Pool of valid entries to perturb.
        total_count: Total number of hallucinated entries to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of hallucinated BenchmarkEntry objects across all 14 types.
    """
    from hallmark.dataset.generator import (
        generate_tier1_batch,
        generate_tier2_batch,
        generate_tier3_batch,
    )

    t1_count = round(total_count * TIER_RATIOS[1])
    t2_count = round(total_count * TIER_RATIOS[2])
    t3_count = total_count - t1_count - t2_count  # remainder to T3

    rng = random.Random(seed)
    # Use different sub-seeds for each tier to avoid RNG entanglement
    t1_seed = rng.randint(0, 2**31)
    t2_seed = rng.randint(0, 2**31)
    t3_seed = rng.randint(0, 2**31)

    logger.info(
        "Generating %d hallucinated entries (T1=%d, T2=%d, T3=%d)",
        total_count,
        t1_count,
        t2_count,
        t3_count,
    )

    tier1 = generate_tier1_batch(valid_entries, t1_count, seed=t1_seed)
    tier2 = generate_tier2_batch(valid_entries, t2_count, seed=t2_seed)
    tier3 = generate_tier3_batch(valid_entries, t3_count, seed=t3_seed)

    all_entries = tier1 + tier2 + tier3
    logger.info("Generated %d total hallucinated entries", len(all_entries))
    return all_entries
