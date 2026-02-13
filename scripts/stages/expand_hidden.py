"""Stage 7: Expand hidden test set to target coverage.

Uses canonical generators from hallmark.dataset.generator (no duplicated logic).
Drops retracted_paper entries per user decision. Ensures all 14 hallucination
types have >= target_per_type entries in the hidden split.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import Counter

from hallmark.dataset.generator import (
    generate_chimeric_title,
    generate_fabricated_doi,
    generate_future_date,
    generate_hybrid_fabrication,
    generate_merged_citation,
    generate_near_miss_title,
    generate_nonexistent_venue,
    generate_partial_author_list,
    generate_placeholder_authors,
    generate_plausible_fabrication,
    generate_preprint_as_published,
    generate_swapped_authors,
    generate_version_confusion,
    generate_wrong_venue,
    is_preprint_source,
)
from hallmark.dataset.schema import BenchmarkEntry, HallucinationType

from .scale_up import ML_BUZZWORDS, VENUES

logger = logging.getLogger(__name__)


def _compute_hidden_gaps(
    entries: list[BenchmarkEntry],
    target_per_type: int,
) -> dict[str, int]:
    """Compute per-type gap for hidden split."""
    type_counts: Counter[str] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            type_counts[e.hallucination_type] += 1

    gaps: dict[str, int] = {}
    for ht in HallucinationType:
        current = type_counts.get(ht.value, 0)
        gap = max(0, target_per_type - current)
        if gap > 0:
            gaps[ht.value] = gap
    return gaps


def _generate_hidden_entry(
    type_val: str,
    source: BenchmarkEntry,
    valid_entries: list[BenchmarkEntry],
    rng: random.Random,
    idx: int,
    chimeric_titles: list[str],
    chimeric_idx: list[int],
    build_date: str,
) -> BenchmarkEntry:
    """Generate a single hidden test entry using canonical generators."""
    if type_val == HallucinationType.FABRICATED_DOI.value:
        entry = generate_fabricated_doi(source, rng)
    elif type_val == HallucinationType.NONEXISTENT_VENUE.value:
        entry = generate_nonexistent_venue(source, rng)
    elif type_val == HallucinationType.PLACEHOLDER_AUTHORS.value:
        entry = generate_placeholder_authors(source, rng)
    elif type_val == HallucinationType.FUTURE_DATE.value:
        entry = generate_future_date(source, rng)
    elif type_val == HallucinationType.CHIMERIC_TITLE.value:
        title = chimeric_titles[chimeric_idx[0] % len(chimeric_titles)]
        chimeric_idx[0] += 1
        entry = generate_chimeric_title(source, title, rng)
    elif type_val == HallucinationType.WRONG_VENUE.value:
        current = source.venue
        candidates = [v for v in VENUES if v != current]
        entry = generate_wrong_venue(source, rng.choice(candidates), rng=rng)
    elif type_val == HallucinationType.AUTHOR_MISMATCH.value:
        donor = rng.choice(valid_entries)
        while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
            donor = rng.choice(valid_entries)
        entry = generate_swapped_authors(source, donor, rng)
    elif type_val == HallucinationType.PREPRINT_AS_PUBLISHED.value:
        preprint_sources = [e for e in valid_entries if is_preprint_source(e)]
        if preprint_sources:
            source = rng.choice(preprint_sources)
        entry = generate_preprint_as_published(source, rng.choice(VENUES), rng)
    elif type_val == HallucinationType.HYBRID_FABRICATION.value:
        entry = generate_hybrid_fabrication(source, rng)
    elif type_val == HallucinationType.MERGED_CITATION.value:
        donor_b = rng.choice(valid_entries)
        while donor_b.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
            donor_b = rng.choice(valid_entries)
        donor_c = rng.choice(valid_entries) if rng.random() < 0.5 else None
        entry = generate_merged_citation(source, donor_b, donor_c, rng)
    elif type_val == HallucinationType.PARTIAL_AUTHOR_LIST.value:
        entry = generate_partial_author_list(source, rng)
    elif type_val == HallucinationType.NEAR_MISS_TITLE.value:
        entry = generate_near_miss_title(source, rng)
    elif type_val == HallucinationType.PLAUSIBLE_FABRICATION.value:
        entry = generate_plausible_fabrication(source, rng)
    elif type_val == HallucinationType.VERSION_CONFUSION.value:
        current = source.venue
        candidates = [v for v in VENUES if v != current]
        entry = generate_version_confusion(source, rng.choice(candidates), rng)
    else:
        raise ValueError(f"Unknown hallucination type: {type_val}")

    entry.added_to_benchmark = build_date
    entry.bibtex_key = f"hidden_{type_val}_{idx}"
    return entry


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
    gaps = _compute_hidden_gaps(hidden, target_per_type)
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
            entry = _generate_hidden_entry(
                type_val,
                source,
                all_valid,
                rng,
                len(new_entries),
                available_buzzwords,
                chimeric_idx,
                build_date,
            )

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
