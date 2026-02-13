"""Stage 9: Finalize splits — anonymize keys, resplit for source separation, write output.

Applied to ALL 3 splits (fixing current gap where hidden was skipped).
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, HallucinationType, save_entries

logger = logging.getLogger(__name__)


def _resplit_with_source_separation(
    entries: list[BenchmarkEntry],
    split_name: str,
    seed: int,
) -> list[BenchmarkEntry]:
    """Remove hallucinated entries whose modified title accidentally matches a valid entry.

    Only applies to types that MODIFY the title (chimeric_title, near_miss_title,
    plausible_fabrication). Types that preserve the source title (wrong_venue,
    placeholder_authors, etc.) are expected to share titles with valid entries —
    that's what the benchmark tests.

    Returns filtered entries with accidental title collisions removed.
    """
    # Types that modify the title — only these can have "accidental" overlaps
    TITLE_MODIFIED_TYPES = {"chimeric_title", "near_miss_title", "plausible_fabrication"}

    valid_titles: set[str] = set()
    for e in entries:
        if e.label == "VALID":
            t = e.fields.get("title", "").strip().lower()
            if t:
                valid_titles.add(t)

    keep: list[BenchmarkEntry] = []
    removed = 0
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type in TITLE_MODIFIED_TYPES:
            t = e.fields.get("title", "").strip().lower()
            if t in valid_titles:
                removed += 1
                continue
        keep.append(e)

    if removed:
        logger.info(
            "%s: removed %d title-modified hallucinated entries with accidental title overlap",
            split_name,
            removed,
        )

    return keep


def _anonymize_and_shuffle(
    entries: list[BenchmarkEntry],
    split_name: str,
    seed: int,
) -> list[BenchmarkEntry]:
    """Anonymize bibtex_keys and shuffle entries.

    Keys follow pattern: hallmark_{split}_{NNNN}
    """
    rng = random.Random(seed)
    rng.shuffle(entries)

    # Map split names to short prefixes
    prefix_map = {
        "dev_public": "dev",
        "test_public": "test",
        "test_hidden": "hidden",
    }
    prefix = prefix_map.get(split_name, split_name)

    for i, entry in enumerate(entries):
        entry.bibtex_key = f"hallmark_{prefix}_{i:04d}"

    return entries


def _verify_no_title_overlap(splits: dict[str, list[BenchmarkEntry]]) -> int:
    """Verify no valid-hallucinated title overlap within any split."""
    violations = 0
    for split_name, entries in splits.items():
        valid_titles: set[str] = set()
        hall_titles: set[str] = set()
        for e in entries:
            t = e.fields.get("title", "").strip().lower()
            if not t:
                continue
            if e.label == "VALID":
                valid_titles.add(t)
            else:
                hall_titles.add(t)
        overlap = valid_titles & hall_titles
        if overlap:
            logger.warning("%s: %d title overlaps remain!", split_name, len(overlap))
            violations += len(overlap)
    return violations


def _verify_key_uniqueness(splits: dict[str, list[BenchmarkEntry]]) -> list[str]:
    """Verify all bibtex_keys are unique across all splits."""
    all_keys: list[str] = []
    for entries in splits.values():
        all_keys.extend(e.bibtex_key for e in entries)
    dupes = [k for k, c in Counter(all_keys).items() if c > 1]
    return dupes


def _verify_min_per_type(
    entries: list[BenchmarkEntry],
    split_name: str,
    min_count: int,
) -> list[str]:
    """Verify minimum entries per hallucination type."""
    type_counts: Counter[str] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            type_counts[e.hallucination_type] += 1

    failures: list[str] = []
    for ht in HallucinationType:
        count = type_counts.get(ht.value, 0)
        if count < min_count:
            failures.append(f"{split_name}/{ht.value}: {count} < {min_count}")
    return failures


def _compute_metadata(
    splits: dict[str, list[BenchmarkEntry]],
    seed: int,
    build_date: str,
) -> dict:
    """Compute metadata.json content from final splits."""
    metadata: dict = {
        "version": "1.0",
        "build_date": build_date,
        "seed": seed,
        "total_entries": sum(len(entries) for entries in splits.values()),
        "splits": {},
    }

    for split_name, entries in splits.items():
        valid_count = sum(1 for e in entries if e.label == "VALID")
        hall_count = sum(1 for e in entries if e.label == "HALLUCINATED")

        tier_counts: dict[str, int] = defaultdict(int)
        type_counts: dict[str, int] = defaultdict(int)
        method_counts: dict[str, int] = defaultdict(int)

        for e in entries:
            if e.difficulty_tier:
                tier_counts[str(e.difficulty_tier)] += 1
            if e.hallucination_type:
                type_counts[e.hallucination_type] += 1
            method_counts[e.generation_method] += 1

        metadata["splits"][split_name] = {
            "file": f"{split_name}.jsonl",
            "total": len(entries),
            "valid": valid_count,
            "hallucinated": hall_count,
            "tier_distribution": dict(sorted(tier_counts.items())),
            "type_distribution": dict(sorted(type_counts.items())),
            "generation_methods": dict(sorted(method_counts.items())),
        }

    return metadata


def _print_split_stats(entries: list[BenchmarkEntry], name: str) -> None:
    """Print summary statistics for a split."""
    valid = [e for e in entries if e.label == "VALID"]
    hall = [e for e in entries if e.label == "HALLUCINATED"]
    types = Counter(e.hallucination_type for e in hall)
    tiers = Counter(e.difficulty_tier for e in hall)

    logger.info(
        "%s: %d entries (%d valid, %d hallucinated)",
        name,
        len(entries),
        len(valid),
        len(hall),
    )
    logger.info("  Tiers: %s", dict(sorted(tiers.items())))
    logger.info("  Types: %s", dict(sorted(types.items())))


def stage_finalize(
    splits: dict[str, list[BenchmarkEntry]],
    output_dir: Path,
    hidden_dir: Path,
    seed: int,
    build_date: str,
    min_per_type_public: int = 30,
    min_per_type_hidden: int = 15,
    dry_run: bool = False,
) -> dict[str, list[BenchmarkEntry]]:
    """Finalize: anonymize, resplit, write output, validate.

    Applied to ALL 3 splits (fixing current gap where hidden was skipped).

    Args:
        splits: Dict mapping split name to list of entries.
        output_dir: Directory for dev_public.jsonl and test_public.jsonl.
        hidden_dir: Directory for test_hidden.jsonl.
        seed: Random seed.
        build_date: ISO date string.
        min_per_type_public: Minimum hallucinated entries per type in public splits.
        min_per_type_hidden: Minimum hallucinated entries per type in hidden split.
        dry_run: If True, skip writing files.

    Returns:
        Finalized splits dict.
    """
    logger.info("Starting finalization...")

    # 1. Resplit with source separation — apply to ALL 3 splits
    for split_name in list(splits.keys()):
        split_seed = seed + hash(split_name) % 10000
        splits[split_name] = _resplit_with_source_separation(
            splits[split_name],
            split_name,
            split_seed,
        )

    # 2. Anonymize and shuffle — apply to ALL 3 splits
    for i, split_name in enumerate(["dev_public", "test_public", "test_hidden"]):
        if split_name in splits:
            splits[split_name] = _anonymize_and_shuffle(
                splits[split_name],
                split_name,
                seed + i,
            )

    # 3. Validation checks
    errors: list[str] = []

    # Key uniqueness
    dupes = _verify_key_uniqueness(splits)
    if dupes:
        errors.append(f"Duplicate keys: {dupes[:5]}")

    # No title overlaps
    violations = _verify_no_title_overlap(splits)
    if violations:
        errors.append(f"{violations} title overlaps remain")

    # Min per type
    for split_name, entries in splits.items():
        min_count = min_per_type_hidden if "hidden" in split_name else min_per_type_public
        failures = _verify_min_per_type(entries, split_name, min_count)
        errors.extend(failures)

    if errors:
        for err in errors:
            logger.warning("VALIDATION: %s", err)
    else:
        logger.info("All validation checks passed")

    # 4. Print statistics
    for split_name, entries in splits.items():
        _print_split_stats(entries, split_name)

    # 5. Write output files
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        hidden_dir.mkdir(parents=True, exist_ok=True)

        save_entries(splits["dev_public"], output_dir / "dev_public.jsonl")
        save_entries(splits["test_public"], output_dir / "test_public.jsonl")
        save_entries(splits["test_hidden"], hidden_dir / "test_hidden.jsonl")

        logger.info("Written: %s/dev_public.jsonl", output_dir)
        logger.info("Written: %s/test_public.jsonl", output_dir)
        logger.info("Written: %s/test_hidden.jsonl", hidden_dir)

        # 6. Write metadata.json
        metadata = _compute_metadata(splits, seed, build_date)
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            f.write("\n")
        logger.info("Written: %s", metadata_path)
    else:
        logger.info("[DRY RUN] No files written")

    return splits
