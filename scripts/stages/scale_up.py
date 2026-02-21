"""Stage 4: Scale up hallucinated entries to meet minimum per-type thresholds.

Computes gaps per hallucination type and generates entries using the canonical
generators from hallmark.dataset.generator via the shared dispatch in _common.
"""

from __future__ import annotations

import logging
import random

from hallmark.dataset.schema import BenchmarkEntry

from ._common import ML_BUZZWORDS, compute_type_gaps, generate_for_type

logger = logging.getLogger(__name__)

# Re-export for backward compatibility (expand_hidden previously imported these)
__all__ = ["ML_BUZZWORDS", "stage_scale_up"]


def generate_entries_for_gaps(
    gaps: dict[str, int],
    valid_entries: list[BenchmarkEntry],
    split_name: str,
    rng: random.Random,
    existing_keys: set[str],
    build_date: str,
) -> list[BenchmarkEntry]:
    """Generate exactly the needed entries to fill gaps.

    Args:
        gaps: Dict mapping hallucination type to count needed.
        valid_entries: Pool of valid entries to perturb.
        split_name: Name of the split (for key generation).
        rng: Random number generator.
        existing_keys: Set of keys already in use (mutated in-place).
        build_date: ISO date string for added_to_benchmark.

    Returns:
        List of new hallucinated entries.
    """
    new_entries: list[BenchmarkEntry] = []

    # Prepare chimeric title pool
    available_buzzwords = list(ML_BUZZWORDS)
    rng.shuffle(available_buzzwords)
    chimeric_idx = [0]  # mutable counter for closure

    for type_val, count in sorted(gaps.items()):
        for i in range(count):
            source = rng.choice(valid_entries)
            entry = generate_for_type(
                type_val,
                source,
                valid_entries,
                rng,
                f"scaleup_{split_name}",
                i,
                available_buzzwords,
                chimeric_idx,
                build_date,
            )
            entry.source = "perturbation_scaleup"

            # Ensure unique key
            while entry.bibtex_key in existing_keys:
                i += 1
                entry.bibtex_key = f"scaleup_{split_name}_{type_val}_{i}"
            existing_keys.add(entry.bibtex_key)

            new_entries.append(entry)

    return new_entries


def stage_scale_up(
    splits: dict[str, list[BenchmarkEntry]],
    min_per_type: int,
    seed: int,
    build_date: str,
) -> dict[str, list[BenchmarkEntry]]:
    """Scale up hallucinated entries to >= min_per_type per type per public split.

    Args:
        splits: Current split data (modified in-place).
        min_per_type: Minimum entries per hallucination type.
        seed: Random seed.
        build_date: ISO date for new entries.

    Returns:
        Updated splits dict.
    """
    rng = random.Random(seed)
    all_keys: set[str] = set()
    for entries in splits.values():
        for e in entries:
            all_keys.add(e.bibtex_key)

    for split_name in ["dev_public", "test_public"]:
        entries = splits[split_name]
        valid = [e for e in entries if e.label == "VALID"]
        gaps = compute_type_gaps(entries, min_per_type)

        if not gaps:
            logger.info("%s: all types already >= %d", split_name, min_per_type)
            continue

        total_gap = sum(gaps.values())
        logger.info("%s: need %d entries across %d types", split_name, total_gap, len(gaps))

        new_entries = generate_entries_for_gaps(
            gaps,
            valid,
            split_name,
            rng,
            all_keys,
            build_date,
        )
        splits[split_name] = entries + new_entries
        logger.info(
            "%s: added %d entries (now %d total)",
            split_name,
            len(new_entries),
            len(splits[split_name]),
        )

    return splits
