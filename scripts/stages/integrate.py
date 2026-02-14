"""Stage 6: Integrate external sources into splits with stratified allocation.

Merges real-world, LLM-generated, GPTZero, and journal entries into the
existing dev/test/hidden splits using type-stratified proportional allocation.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from hallmark.dataset.schema import BenchmarkEntry

logger = logging.getLogger(__name__)

# Split ratios for new entries
SPLIT_RATIOS = {"dev_public": 0.40, "test_public": 0.33, "test_hidden": 0.27}

# Keys of fabricated entries that should be excluded from real-world data
FAKE_REALWORLD_KEYS = {
    "realworld_future_date_pattern",
    "realworld_nonexistent_venue",
    "realworld_fabricated_doi",
    "realworld_hybrid_fabrication",
}

# NOTE: cross_db_agreement is now set correctly by generators and
# classify_hallucination(). No post-hoc fixup needed.


def _split_entries_stratified(
    entries: list[BenchmarkEntry],
    prefix: str,
    rng: random.Random,
) -> dict[str, list[BenchmarkEntry]]:
    """Split entries across dev/test/hidden, stratified by hallucination_type.

    Assigns new bibtex_keys with split-specific prefixes.
    """
    by_type: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        ht = entry.hallucination_type or "valid"
        by_type[ht].append(entry)

    result: dict[str, list[BenchmarkEntry]] = {
        "dev_public": [],
        "test_public": [],
        "test_hidden": [],
    }

    for ht, type_entries in by_type.items():
        rng.shuffle(type_entries)
        n = len(type_entries)

        n_dev = max(1, round(n * SPLIT_RATIOS["dev_public"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test_public"]))
        n_hidden = max(0, n - n_dev - n_test)

        # Clamp to available
        if n_dev + n_test + n_hidden > n:
            n_hidden = max(0, n - n_dev - n_test)
            if n_dev + n_test > n:
                n_test = max(0, n - n_dev)
                if n_dev > n:
                    n_dev = n

        dev_chunk = type_entries[:n_dev]
        test_chunk = type_entries[n_dev : n_dev + n_test]
        hidden_chunk = type_entries[n_dev + n_test : n_dev + n_test + n_hidden]

        # Assign split-specific keys
        for i, e in enumerate(dev_chunk):
            e.bibtex_key = f"{prefix}_{ht}_dev_{i}"
        for i, e in enumerate(test_chunk):
            e.bibtex_key = f"{prefix}_{ht}_test_{i}"
        for i, e in enumerate(hidden_chunk):
            e.bibtex_key = f"{prefix}_{ht}_hidden_{i}"

        result["dev_public"].extend(dev_chunk)
        result["test_public"].extend(test_chunk)
        result["test_hidden"].extend(hidden_chunk)

    return result


def _filter_fake_realworld(entries: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
    """Remove fabricated entries from real-world collection."""
    filtered = [e for e in entries if e.bibtex_key not in FAKE_REALWORLD_KEYS]
    removed = len(entries) - len(filtered)
    if removed:
        logger.info("Removed %d fabricated real-world entries", removed)
    return filtered


def _fix_cross_db_agreement(entries: list[BenchmarkEntry]) -> int:
    """No-op: cross_db_agreement is now set correctly by generators."""
    return 0


def stage_integrate_external(
    splits: dict[str, list[BenchmarkEntry]],
    real_world: list[BenchmarkEntry],
    llm_entries: list[BenchmarkEntry],
    gptzero_entries: list[BenchmarkEntry],
    journal_entries: list[BenchmarkEntry],
    seed: int,
    build_date: str,
) -> dict[str, list[BenchmarkEntry]]:
    """Integrate external sources into splits with stratified allocation.

    Args:
        splits: Current split data (modified in-place).
        real_world: Real-world hallucination entries.
        llm_entries: LLM-generated hallucination entries.
        gptzero_entries: GPTZero hand-curated entries.
        journal_entries: Journal article valid entries.
        seed: Random seed.
        build_date: ISO date for new entries.

    Returns:
        Updated splits dict.
    """
    rng = random.Random(seed)

    # Collect all existing keys for collision detection
    all_keys: set[str] = set()
    for entries in splits.values():
        for e in entries:
            all_keys.add(e.bibtex_key)

    # Filter out fake real-world entries
    real_world = _filter_fake_realworld(real_world)

    # Fix cross_db_agreement on LLM entries before integration
    fixed = _fix_cross_db_agreement(llm_entries)
    if fixed:
        logger.info("Fixed cross_db_agreement on %d LLM entries", fixed)

    # Set source field and build date on all new entries
    for e in real_world:
        e.source = "real_world"
        e.added_to_benchmark = build_date
    for e in llm_entries:
        e.source = "llm_generated"
        e.added_to_benchmark = build_date
    for e in gptzero_entries:
        e.source = "gptzero_neurips2025"
        e.added_to_benchmark = build_date

    # Stratify and integrate each source
    sources = [
        ("rw", real_world),
        ("llm", llm_entries),
        ("gptzero", gptzero_entries),
    ]

    for prefix, source_entries in sources:
        if not source_entries:
            continue

        stratified = _split_entries_stratified(source_entries, prefix, rng)

        for split_name in ["dev_public", "test_public", "test_hidden"]:
            new_entries = stratified.get(split_name, [])
            # Ensure no key collisions
            for e in new_entries:
                while e.bibtex_key in all_keys:
                    e.bibtex_key = f"{e.bibtex_key}_{rng.randint(0, 9999)}"
                all_keys.add(e.bibtex_key)

            splits[split_name].extend(new_entries)

        total = sum(len(v) for v in stratified.values())
        logger.info("Integrated %d %s entries across splits", total, prefix)

    # Integrate journal articles as valid entries (stratified across splits)
    if journal_entries:
        rng.shuffle(journal_entries)
        n = len(journal_entries)
        n_dev = round(n * 0.45)
        n_test = round(n * 0.30)

        for e in journal_entries:
            e.source = "dblp"
            e.added_to_benchmark = build_date

        dev_journal = journal_entries[:n_dev]
        test_journal = journal_entries[n_dev : n_dev + n_test]
        hidden_journal = journal_entries[n_dev + n_test :]

        for i, e in enumerate(dev_journal):
            e.bibtex_key = f"journal_dev_{i}"
            while e.bibtex_key in all_keys:
                e.bibtex_key = f"journal_dev_{i}_{rng.randint(0, 9999)}"
            all_keys.add(e.bibtex_key)
        for i, e in enumerate(test_journal):
            e.bibtex_key = f"journal_test_{i}"
            while e.bibtex_key in all_keys:
                e.bibtex_key = f"journal_test_{i}_{rng.randint(0, 9999)}"
            all_keys.add(e.bibtex_key)
        for i, e in enumerate(hidden_journal):
            e.bibtex_key = f"journal_hidden_{i}"
            while e.bibtex_key in all_keys:
                e.bibtex_key = f"journal_hidden_{i}_{rng.randint(0, 9999)}"
            all_keys.add(e.bibtex_key)

        splits["dev_public"].extend(dev_journal)
        splits["test_public"].extend(test_journal)
        splits["test_hidden"].extend(hidden_journal)

        logger.info(
            "Integrated %d journal articles (dev=%d, test=%d, hidden=%d)",
            n,
            len(dev_journal),
            len(test_journal),
            len(hidden_journal),
        )

    return splits
