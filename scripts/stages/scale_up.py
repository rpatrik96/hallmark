"""Stage 4: Scale up hallucinated entries to meet minimum per-type thresholds.

Computes gaps per hallucination type and generates entries using the canonical
generators from hallmark.dataset.generator via the shared dispatch in _common.
Also fills year gaps where a year appears in only one label (leakage vector).
"""

from __future__ import annotations

import logging
import random

from hallmark.dataset.schema import BenchmarkEntry, HallucinationType

from ._common import ML_BUZZWORDS, compute_type_gaps, generate_for_type

logger = logging.getLogger(__name__)

# Re-export for backward compatibility (expand_hidden previously imported these)
__all__ = ["ML_BUZZWORDS", "stage_scale_up"]

# Hallucination types that preserve the source entry's year field.
# Types that modify year (future_date, arxiv_version_mismatch) are excluded.
YEAR_PRESERVING_TYPES = [
    HallucinationType.NEAR_MISS_TITLE.value,
    HallucinationType.FABRICATED_DOI.value,
    HallucinationType.NONEXISTENT_VENUE.value,
    HallucinationType.PLACEHOLDER_AUTHORS.value,
    HallucinationType.WRONG_VENUE.value,
    HallucinationType.CHIMERIC_TITLE.value,
]


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


def fill_year_gaps(
    entries: list[BenchmarkEntry],
    split_name: str,
    rng: random.Random,
    existing_keys: set[str],
    build_date: str,
    entries_per_source: int = 3,
) -> list[BenchmarkEntry]:
    """Generate hallucinated entries for years that only appear in one label.

    Prevents year from being a label-leaking feature. For example, if year=2026
    only appears in VALID entries, generates hallucinated entries from those
    sources so year alone cannot predict the label.

    Only fills VALID-only years (generates HALLUCINATED). Years that only appear
    in HALLUCINATED (e.g., future_date years 2030+) are intentional by design.

    Args:
        entries: Current entries in the split.
        split_name: Name of the split (for key generation).
        rng: Random number generator.
        existing_keys: Set of keys already in use (mutated in-place).
        build_date: ISO date string for added_to_benchmark.
        entries_per_source: Max entries to generate per source entry.

    Returns:
        List of new hallucinated entries to append.
    """
    # Compute year → label distribution
    year_labels: dict[str, set[str]] = {}
    for e in entries:
        year = e.fields.get("year", "")
        if year and e.label in ("VALID", "HALLUCINATED"):
            year_labels.setdefault(year, set()).add(e.label)

    # Find years that only appear in VALID (HALLUCINATED-only years are by design)
    valid_only_years = {y for y, labels in year_labels.items() if labels == {"VALID"}}

    if not valid_only_years:
        return []

    logger.info(
        "%s: year gap detected — years %s only in VALID",
        split_name,
        sorted(valid_only_years),
    )

    # Collect source entries for gap years
    sources = [
        e for e in entries if e.label == "VALID" and e.fields.get("year", "") in valid_only_years
    ]

    if not sources:
        return []

    # Prepare chimeric title pool for generate_for_type
    available_buzzwords = list(ML_BUZZWORDS)
    rng.shuffle(available_buzzwords)
    chimeric_idx = [0]

    new_entries: list[BenchmarkEntry] = []
    all_valid = [e for e in entries if e.label == "VALID"]

    for src_i, source in enumerate(sources):
        # Pick 2-3 year-preserving types per source for diversity
        n_gens = min(len(YEAR_PRESERVING_TYPES), rng.randint(2, entries_per_source))
        selected_types = rng.sample(YEAR_PRESERVING_TYPES, n_gens)

        for gen_i, type_val in enumerate(selected_types):
            idx = src_i * 10 + gen_i
            entry = generate_for_type(
                type_val,
                source,
                all_valid,
                rng,
                f"yeargap_{split_name}",
                idx,
                available_buzzwords,
                chimeric_idx,
                build_date,
            )

            # Ensure year is preserved (should be by construction)
            source_year = source.fields.get("year", "")
            if entry.fields.get("year") != source_year:
                entry.fields["year"] = source_year

            # Ensure unique key
            while entry.bibtex_key in existing_keys:
                idx += 1
                entry.bibtex_key = f"yeargap_{split_name}_{type_val}_{idx}"
            existing_keys.add(entry.bibtex_key)

            new_entries.append(entry)

    logger.info(
        "%s: generated %d entries to fill year gap (years: %s)",
        split_name,
        len(new_entries),
        sorted(valid_only_years),
    )
    return new_entries


def stage_scale_up(
    splits: dict[str, list[BenchmarkEntry]],
    min_per_type: int,
    seed: int,
    build_date: str,
) -> dict[str, list[BenchmarkEntry]]:
    """Scale up hallucinated entries to >= min_per_type per type per public split.

    Also fills year gaps to prevent year from leaking labels.

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

        # Step 1: Fill type gaps
        gaps = compute_type_gaps(entries, min_per_type)

        if gaps:
            total_gap = sum(gaps.values())
            logger.info(
                "%s: need %d entries across %d types",
                split_name,
                total_gap,
                len(gaps),
            )

            new_entries = generate_entries_for_gaps(
                gaps,
                valid,
                split_name,
                rng,
                all_keys,
                build_date,
            )
            entries = entries + new_entries
            logger.info(
                "%s: added %d entries for type gaps (now %d total)",
                split_name,
                len(new_entries),
                len(entries),
            )
        else:
            logger.info("%s: all types already >= %d", split_name, min_per_type)

        # Step 2: Fill year gaps (prevent year from leaking labels)
        year_entries = fill_year_gaps(
            entries,
            split_name,
            rng,
            all_keys,
            build_date,
        )
        if year_entries:
            entries = entries + year_entries

        splits[split_name] = entries

    return splits
