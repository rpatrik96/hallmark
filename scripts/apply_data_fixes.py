#!/usr/bin/env python3
"""Apply data fixes to dev_public.jsonl and test_public.jsonl.

This script performs:
1. Fix DOI + subtests for chimeric_title, near_miss_title, version_confusion entries
2. Top up test_public types below 30
3. Populate source field for all entries
4. Update metadata.json
"""

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Import generator functions
from hallmark.dataset.generator import (
    generate_chimeric_title,
    generate_near_miss_title,
    generate_preprint_as_published,
    generate_version_confusion,
    generate_wrong_venue,
)
from hallmark.dataset.schema import BenchmarkEntry

# Paths
DATA_DIR = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark/data/v1.0")
DEV_FILE = DATA_DIR / "dev_public.jsonl"
TEST_FILE = DATA_DIR / "test_public.jsonl"
SOURCE_MAPPING_FILE = DATA_DIR / "source_mapping.json"
METADATA_FILE = DATA_DIR / "metadata.json"

# Seed for determinism
SEED = 2026
rng = random.Random(SEED)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    """Write JSONL file."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def get_doi_pool(entries: list[dict[str, Any]]) -> list[str]:
    """Extract DOIs from valid entries."""
    dois = []
    for entry in entries:
        if entry.get("label") == "VALID" and "doi" in entry.get("fields", {}):
            dois.append(entry["fields"]["doi"])
    return dois


def update_fields_complete(entry: dict[str, Any]) -> None:
    """Update fields_complete based on DOI/URL presence."""
    fields = entry.get("fields", {})
    has_doi = "doi" in fields and fields["doi"]
    has_url = "url" in fields and fields["url"]
    if "subtests" in entry:
        entry["subtests"]["fields_complete"] = has_doi or has_url


def fix_doi_for_type(
    entries: list[dict[str, Any]],
    hallucination_type: str,
    doi_pool: list[str],
    target_pct: float = 0.6,
    subtest_updates: dict[str, Any] | None = None,
) -> int:
    """Fix DOI presence and subtests for a specific hallucination type.

    Returns number of entries updated.
    """
    count = 0
    type_entries = [e for e in entries if e.get("hallucination_type") == hallucination_type]

    # Determine which entries need DOIs
    entries_without_doi = [e for e in type_entries if "doi" not in e.get("fields", {})]
    num_to_add = int(len(type_entries) * target_pct) - (
        len(type_entries) - len(entries_without_doi)
    )

    if num_to_add > 0 and entries_without_doi:
        # Randomly select entries to add DOIs to
        entries_to_update = rng.sample(
            entries_without_doi, min(num_to_add, len(entries_without_doi))
        )

        for entry in entries_to_update:
            # Add a DOI from the pool
            entry["fields"]["doi"] = rng.choice(doi_pool)
            count += 1

    # Update subtests for all entries of this type
    if subtest_updates:
        for entry in type_entries:
            if "subtests" not in entry:
                entry["subtests"] = {}

            # Update specified subtests
            for key, value in subtest_updates.items():
                # Special handling for doi_resolves - only set True if DOI present
                if key == "doi_resolves":
                    entry["subtests"][key] = value if "doi" in entry.get("fields", {}) else False
                else:
                    entry["subtests"][key] = value

            # Update fields_complete based on DOI/URL
            update_fields_complete(entry)

    return count


def step1_fix_dois(dev_entries: list[dict[str, Any]], test_entries: list[dict[str, Any]]) -> None:
    """Step 1: Fix DOI + subtests for chimeric_title, near_miss_title, version_confusion."""
    print("\n=== Step 1: Fix DOIs and subtests ===")

    # Get DOI pools from valid entries
    dev_doi_pool = get_doi_pool(dev_entries)
    test_doi_pool = get_doi_pool(test_entries)

    print(f"Dev DOI pool: {len(dev_doi_pool)} DOIs")
    print(f"Test DOI pool: {len(test_doi_pool)} DOIs")

    # Fix chimeric_title
    dev_count = fix_doi_for_type(
        dev_entries,
        "chimeric_title",
        dev_doi_pool,
        target_pct=0.6,
        subtest_updates={"doi_resolves": True, "title_exists": False},
    )
    test_count = fix_doi_for_type(
        test_entries,
        "chimeric_title",
        test_doi_pool,
        target_pct=0.6,
        subtest_updates={"doi_resolves": True, "title_exists": False},
    )
    print(f"chimeric_title: added {dev_count} DOIs to dev, {test_count} to test")

    # Fix near_miss_title
    dev_count = fix_doi_for_type(
        dev_entries,
        "near_miss_title",
        dev_doi_pool,
        target_pct=0.6,
        subtest_updates={"doi_resolves": True, "title_exists": False},
    )
    test_count = fix_doi_for_type(
        test_entries,
        "near_miss_title",
        test_doi_pool,
        target_pct=0.6,
        subtest_updates={"doi_resolves": True, "title_exists": False},
    )
    print(f"near_miss_title: added {dev_count} DOIs to dev, {test_count} to test")

    # Fix version_confusion
    dev_count = fix_doi_for_type(
        dev_entries,
        "version_confusion",
        dev_doi_pool,
        target_pct=0.6,
        subtest_updates={"doi_resolves": True, "cross_db_agreement": False},
    )
    test_count = fix_doi_for_type(
        test_entries,
        "version_confusion",
        test_doi_pool,
        target_pct=0.6,
        subtest_updates={"doi_resolves": True, "cross_db_agreement": False},
    )
    print(f"version_confusion: added {dev_count} DOIs to dev, {test_count} to test")


def step2_topup_test(test_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Step 2: Top up test_public types below 30."""
    print("\n=== Step 2: Top up test_public types ===")

    # Count current entries per type
    type_counts = defaultdict(int)
    for entry in test_entries:
        if entry.get("label") == "HALLUCINATED":
            type_counts[entry["hallucination_type"]] += 1

    # Fake venues and titles for generators
    fake_venues = [
        "International Conference on Advanced AI Systems",
        "Journal of Computational Intelligence and Applications",
        "Workshop on Emerging Methods in Deep Learning",
        "Transactions on Neural Computing Paradigms",
        "Symposium on Frontier Artificial Intelligence Research",
        "Annual Conference on Machine Learning Innovations",
        "IEEE International Conference on Cognitive Computing",
        "Pacific Rim Symposium on Neural Information Systems",
        "European Workshop on Probabilistic Machine Learning",
        "ACM Conference on Automated Reasoning and Learning",
    ]

    fake_titles = [
        "Deep Learning Approaches for Advanced Pattern Recognition",
        "Neural Network Optimization via Gradient Approximation",
        "Transformer Architectures for Sequential Data Processing",
        "Reinforcement Learning in Complex Decision Spaces",
        "Attention Mechanisms for Multi-Modal Learning",
        "Graph Neural Networks for Relational Reasoning",
        "Adversarial Training for Robust Classification",
        "Meta-Learning Strategies for Few-Shot Recognition",
        "Generative Models for Data Augmentation",
        "Causal Inference in Deep Learning Systems",
    ]

    # Define wrapper functions that provide required arguments
    def gen_chimeric(entry: BenchmarkEntry) -> BenchmarkEntry:
        return generate_chimeric_title(entry, rng.choice(fake_titles), rng=rng)

    def gen_near_miss(entry: BenchmarkEntry) -> BenchmarkEntry:
        return generate_near_miss_title(entry, rng=rng)

    def gen_wrong_venue(entry: BenchmarkEntry) -> BenchmarkEntry:
        return generate_wrong_venue(entry, rng.choice(fake_venues), rng=rng)

    def gen_preprint(entry: BenchmarkEntry) -> BenchmarkEntry:
        return generate_preprint_as_published(entry, rng.choice(fake_venues), rng=rng)

    def gen_version(entry: BenchmarkEntry) -> BenchmarkEntry:
        return generate_version_confusion(entry, rng.choice(fake_venues), rng=rng)

    # Define types that need top-up and their generators
    topup_needed = {
        "chimeric_title": (8, gen_chimeric),
        "near_miss_title": (6, gen_near_miss),
        "wrong_venue": (5, gen_wrong_venue),
        "preprint_as_published": (2, gen_preprint),
        "version_confusion": (2, gen_version),
    }

    # Get valid entries that haven't been used as perturbation sources
    valid_entries = [e for e in test_entries if e.get("label") == "VALID"]
    used_keys = {e.get("source_bibtex_key") for e in test_entries if e.get("source_bibtex_key")}
    available_sources = [e for e in valid_entries if e["bibtex_key"] not in used_keys]

    print(f"Available source entries: {len(available_sources)}")

    new_entries = []
    for halluc_type, (needed, generator_func) in topup_needed.items():
        current = type_counts[halluc_type]
        print(f"{halluc_type}: current={current}, need +{needed}")

        # Sample source entries
        sources = rng.sample(available_sources, min(needed, len(available_sources)))

        for i, source_dict in enumerate(sources):
            # Convert to BenchmarkEntry
            source_entry = BenchmarkEntry(**source_dict)

            # Generate hallucination
            halluc_entry = generator_func(source_entry)

            # Customize for top-up
            halluc_entry.bibtex_key = f"topup_{halluc_type}_{i + 1}"
            halluc_entry.source = "perturbation_scaleup"
            halluc_entry.generation_method = "perturbation"

            # Convert back to dict
            new_entries.append(asdict(halluc_entry))

        # Mark sources as used
        for source in sources:
            used_keys.add(source["bibtex_key"])

    print(f"Generated {len(new_entries)} new entries")
    return new_entries


def step3_populate_source(
    dev_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
    new_test_entries: list[dict[str, Any]],
) -> None:
    """Step 3: Populate source field for all entries."""
    print("\n=== Step 3: Populate source field ===")

    # Load source mapping
    with open(SOURCE_MAPPING_FILE) as f:
        source_mapping = json.load(f)

    # Apply to dev
    for entry in dev_entries:
        key = entry["bibtex_key"]
        if key in source_mapping:
            entry["source"] = source_mapping[key]
        else:
            print(f"Warning: {key} not in source_mapping")

    # Apply to test
    for entry in test_entries:
        key = entry["bibtex_key"]
        if key in source_mapping:
            entry["source"] = source_mapping[key]
        else:
            print(f"Warning: {key} not in source_mapping")

    # Apply to new entries
    for entry in new_test_entries:
        entry["source"] = "perturbation_scaleup"

    print(
        f"Applied source to {len(dev_entries)} dev + {len(test_entries)} test + {len(new_test_entries)} new entries"
    )


def step4_update_metadata(
    dev_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
) -> None:
    """Step 4: Update metadata.json with new statistics."""
    print("\n=== Step 4: Update metadata.json ===")

    # Load existing metadata
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    # Calculate statistics
    def calc_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
        type_counts = defaultdict(int)
        type_doi_counts = defaultdict(int)
        source_counts = defaultdict(int)

        total = len(entries)
        hallucinated = 0
        valid = 0

        for entry in entries:
            label = entry.get("label")
            if label == "HALLUCINATED":
                hallucinated += 1
                halluc_type = entry.get("hallucination_type")
                type_counts[halluc_type] += 1
                if "doi" in entry.get("fields", {}):
                    type_doi_counts[halluc_type] += 1
            elif label == "VALID":
                valid += 1

            source = entry.get("source")
            if source:
                source_counts[source] += 1

        return {
            "total": total,
            "valid": valid,
            "hallucinated": hallucinated,
            "type_counts": dict(type_counts),
            "type_doi_percentages": {
                t: round(type_doi_counts[t] / type_counts[t] * 100, 1) if type_counts[t] > 0 else 0
                for t in type_counts
            },
            "source_distribution": dict(source_counts),
        }

    dev_stats = calc_stats(dev_entries)
    test_stats = calc_stats(test_entries)

    # Update metadata
    metadata["splits"]["dev_public"]["total"] = dev_stats["total"]
    metadata["splits"]["dev_public"]["valid"] = dev_stats["valid"]
    metadata["splits"]["dev_public"]["hallucinated"] = dev_stats["hallucinated"]
    metadata["splits"]["dev_public"]["type_distribution"] = dev_stats["type_counts"]
    metadata["splits"]["dev_public"]["generation_method_distribution"] = dev_stats[
        "source_distribution"
    ]

    metadata["splits"]["test_public"]["total"] = test_stats["total"]
    metadata["splits"]["test_public"]["valid"] = test_stats["valid"]
    metadata["splits"]["test_public"]["hallucinated"] = test_stats["hallucinated"]
    metadata["splits"]["test_public"]["type_distribution"] = test_stats["type_counts"]
    metadata["splits"]["test_public"]["generation_method_distribution"] = test_stats[
        "source_distribution"
    ]

    # Add new statistics sections
    metadata["type_counts_per_split"] = {
        "dev_public": dev_stats["type_counts"],
        "test_public": test_stats["type_counts"],
    }

    metadata["doi_percentages_per_type"] = {
        "dev_public": dev_stats["type_doi_percentages"],
        "test_public": test_stats["type_doi_percentages"],
    }

    metadata["source_distribution"] = {
        "dev_public": dev_stats["source_distribution"],
        "test_public": test_stats["source_distribution"],
    }

    # Write updated metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata updated")


def print_summary(
    dev_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
) -> None:
    """Print verification summary."""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    def print_split_summary(name: str, entries: list[dict[str, Any]]) -> None:
        print(f"\n{name.upper()} SPLIT:")
        print(f"Total entries: {len(entries)}")

        # Entry counts per type
        type_counts = defaultdict(int)
        type_doi_counts = defaultdict(int)

        for entry in entries:
            if entry.get("label") == "HALLUCINATED":
                halluc_type = entry.get("hallucination_type")
                type_counts[halluc_type] += 1
                if "doi" in entry.get("fields", {}):
                    type_doi_counts[halluc_type] += 1

        print("\nEntry counts per type:")
        for halluc_type in sorted(type_counts.keys()):
            count = type_counts[halluc_type]
            doi_count = type_doi_counts[halluc_type]
            doi_pct = (doi_count / count * 100) if count > 0 else 0
            print(f"  {halluc_type}: {count} ({doi_count} with DOI, {doi_pct:.1f}%)")

        # Source field coverage
        source_counts = defaultdict(int)
        missing_source = 0
        for entry in entries:
            source = entry.get("source")
            if source:
                source_counts[source] += 1
            else:
                missing_source += 1

        print("\nSource field coverage:")
        for source in sorted(source_counts.keys()):
            print(f"  {source}: {source_counts[source]}")
        if missing_source > 0:
            print(f"  MISSING: {missing_source}")
        else:
            print("  Coverage: 100%")

    print_split_summary("dev", dev_entries)
    print_split_summary("test", test_entries)

    print("\n" + "=" * 60)


def main() -> None:
    """Main execution."""
    print("Loading data...")
    dev_entries = load_jsonl(DEV_FILE)
    test_entries = load_jsonl(TEST_FILE)

    print(f"Loaded {len(dev_entries)} dev entries, {len(test_entries)} test entries")

    # Step 1: Fix DOIs and subtests
    step1_fix_dois(dev_entries, test_entries)

    # Step 2: Top up test types
    new_test_entries = step2_topup_test(test_entries)

    # Step 3: Populate source field
    step3_populate_source(dev_entries, test_entries, new_test_entries)

    # Merge new entries into test
    test_entries.extend(new_test_entries)

    # Step 4: Update metadata
    step4_update_metadata(dev_entries, test_entries)

    # Write updated files
    print("\n=== Writing updated files ===")
    write_jsonl(DEV_FILE, dev_entries)
    print(f"Wrote {len(dev_entries)} entries to {DEV_FILE}")
    write_jsonl(TEST_FILE, test_entries)
    print(f"Wrote {len(test_entries)} entries to {TEST_FILE}")

    # Print summary
    print_summary(dev_entries, test_entries)


if __name__ == "__main__":
    main()
