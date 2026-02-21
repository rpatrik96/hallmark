"""Stage 7: Expand hidden test set to target coverage.

Uses canonical generators via the shared dispatch in _common.
Drops retracted_paper entries per user decision. Ensures all 14 hallucination
types have >= target_per_type entries in the hidden split.
"""

from __future__ import annotations

import copy
import logging
import random

from hallmark.dataset.schema import BenchmarkEntry

from ._common import ML_BUZZWORDS, compute_type_gaps, generate_for_type

logger = logging.getLogger(__name__)


def stage_expand_hidden(
    splits: dict[str, list[BenchmarkEntry]],
    target_per_type: int,
    target_valid: int,
    seed: int,
    build_date: str,
) -> dict[str, list[BenchmarkEntry]]:
    """Expand hidden test set to target coverage.

    Args:
        splits: Current split data (modified in-place for test_hidden).
        target_per_type: Minimum hallucinated entries per type in hidden.
        target_valid: Target number of valid entries in hidden.
        seed: Random seed.
        build_date: ISO date for new entries.

    Returns:
        Updated splits dict.
    """
    rng = random.Random(seed)
    hidden = splits["test_hidden"]

    # Drop retracted_paper entries (not in taxonomy)
    before = len(hidden)
    hidden = [e for e in hidden if e.hallucination_type != "retracted_paper"]
    dropped = before - len(hidden)
    if dropped:
        logger.info("Dropped %d retracted_paper entries from hidden", dropped)

    # Collect all keys across all splits for collision avoidance
    all_keys: set[str] = set()
    for entries in splits.values():
        for e in entries:
            all_keys.add(e.bibtex_key)

    # Get valid entries from all splits as source pool
    all_valid: list[BenchmarkEntry] = []
    for entries in splits.values():
        all_valid.extend(e for e in entries if e.label == "VALID")

    # Add valid entries if below target
    current_valid = sum(1 for e in hidden if e.label == "VALID")
    valid_gap = max(0, target_valid - current_valid)
    if valid_gap > 0:
        # Sample valid entries from dev/test that aren't already in hidden
        hidden_keys = {e.bibtex_key for e in hidden}
        available_valid = [e for e in all_valid if e.bibtex_key not in hidden_keys]
        rng.shuffle(available_valid)
        # Deep copy to avoid mutating entries in dev/test splits
        added_valid = [copy.deepcopy(e) for e in available_valid[:valid_gap]]

        # Assign new keys to avoid collisions
        for i, e in enumerate(added_valid):
            new_key = f"hidden_valid_{i}"
            while new_key in all_keys:
                new_key = f"hidden_valid_{i}_{rng.randint(0, 9999)}"
            e.bibtex_key = new_key
            all_keys.add(new_key)

        hidden.extend(added_valid)
        logger.info("Added %d valid entries to hidden (target=%d)", len(added_valid), target_valid)

    # Compute hallucination gaps
    gaps = compute_type_gaps(hidden, target_per_type)
    if not gaps:
        logger.info("Hidden split already meets all per-type targets")
        splits["test_hidden"] = hidden
        return splits

    total_gap = sum(gaps.values())
    logger.info("Hidden needs %d entries across %d types", total_gap, len(gaps))

    # Prepare chimeric title pool
    available_buzzwords = list(ML_BUZZWORDS)
    rng.shuffle(available_buzzwords)
    chimeric_idx = [0]

    # Generate entries
    new_entries: list[BenchmarkEntry] = []
    for type_val, count in sorted(gaps.items()):
        for _i in range(count):
            source = rng.choice(all_valid)
            entry = generate_for_type(
                type_val,
                source,
                all_valid,
                rng,
                "hidden",
                len(new_entries),
                available_buzzwords,
                chimeric_idx,
                build_date,
            )
            entry.source = "perturbation_hidden"

            # Ensure unique key
            while entry.bibtex_key in all_keys:
                entry.bibtex_key = f"hidden_{type_val}_{len(new_entries)}_{rng.randint(0, 9999)}"
            all_keys.add(entry.bibtex_key)

            new_entries.append(entry)

    hidden.extend(new_entries)
    splits["test_hidden"] = hidden

    n_valid = sum(1 for e in hidden if e.label == "VALID")
    n_hall = sum(1 for e in hidden if e.label == "HALLUCINATED")
    logger.info(
        "Hidden expanded: %d total (%d valid, %d hallucinated, +%d new)",
        len(hidden),
        n_valid,
        n_hall,
        len(new_entries),
    )

    return splits
