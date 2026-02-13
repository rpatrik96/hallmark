"""Generate new hallucinated instances for HALLMARK benchmark.

Creates instances for missing hallucination types:
- chimeric_title
- plausible_fabrication
- arxiv_version_mismatch
- hybrid_fabrication
"""

from __future__ import annotations

import random
from pathlib import Path

from hallmark.dataset.generator import (
    generate_arxiv_version_mismatch,
    generate_chimeric_title,
    generate_hybrid_fabrication,
    generate_plausible_fabrication,
)
from hallmark.dataset.schema import BenchmarkEntry, load_entries, save_entries

# Real papers with arXiv versions for arxiv_version_mismatch
VERSION_CONFUSED_PAPERS = [
    {"arxiv_id": "1706.03762", "conference_venue": "NeurIPS", "conference_year": "2017"},
    {"arxiv_id": "1810.04805", "conference_venue": "NAACL", "conference_year": "2019"},
    {"arxiv_id": "2005.14165", "conference_venue": "NeurIPS", "conference_year": "2020"},
    {"arxiv_id": "2302.13971", "conference_venue": "ICLR", "conference_year": "2024"},
    {"arxiv_id": "2203.15556", "conference_venue": "NeurIPS", "conference_year": "2022"},
]

# ML buzzwords for chimeric titles
ML_BUZZWORDS = [
    "Self-Attention Mechanisms",
    "Cross-Modal Representation Learning",
    "Contrastive Self-Supervised Methods",
    "Few-Shot Meta-Learning Frameworks",
    "Diffusion-Based Generative Models",
    "Neural Architecture Search Strategies",
    "Multi-Task Transfer Learning",
    "Graph Neural Network Architectures",
]

SEED = 2026
ADDED_DATE = "2026-02-12"


def generate_instances_for_split(
    split_name: str,
    valid_entries: list[BenchmarkEntry],
    num_chimeric: int,
    num_plausible: int,
    num_version: int,
    num_hybrid: int,
    rng: random.Random,
) -> list[BenchmarkEntry]:
    """Generate new hallucinated instances for a split."""
    new_entries = []

    # Generate chimeric_title instances
    print(f"  Generating {num_chimeric} chimeric_title instances...")
    for i in range(num_chimeric):
        source = rng.choice(valid_entries)
        fake_title = rng.choice(ML_BUZZWORDS)
        entry = generate_chimeric_title(source, fake_title, rng)
        entry.added_to_benchmark = ADDED_DATE
        entry.bibtex_key = f"chimeric_{split_name}_{i + 1}"
        new_entries.append(entry)

    # Generate plausible_fabrication instances
    print(f"  Generating {num_plausible} plausible_fabrication instances...")
    for i in range(num_plausible):
        source = rng.choice(valid_entries)
        entry = generate_plausible_fabrication(source, rng)
        entry.added_to_benchmark = ADDED_DATE
        entry.bibtex_key = f"plausible_{split_name}_{i + 1}"
        new_entries.append(entry)

    # Generate arxiv_version_mismatch instances
    print(f"  Generating {num_version} arxiv_version_mismatch instances...")
    version_pool = VERSION_CONFUSED_PAPERS[:num_version]
    for i, version_data in enumerate(version_pool):
        source = rng.choice(valid_entries)
        entry = generate_arxiv_version_mismatch(
            source,
            version_data["arxiv_id"],
            version_data["conference_venue"],
            version_data["conference_year"],
            rng,
        )
        entry.added_to_benchmark = ADDED_DATE
        entry.bibtex_key = f"version_{split_name}_{i + 1}"
        new_entries.append(entry)

    # Generate hybrid_fabrication instances
    print(f"  Generating {num_hybrid} hybrid_fabrication instances...")
    for i in range(num_hybrid):
        source = rng.choice(valid_entries)
        entry = generate_hybrid_fabrication(source, rng)
        entry.added_to_benchmark = ADDED_DATE
        entry.bibtex_key = f"hybrid_{split_name}_{i + 1}"
        new_entries.append(entry)

    return new_entries


def main() -> None:
    """Generate new instances and append to dataset files."""
    data_dir = Path("data/v1.0")
    dev_path = data_dir / "dev_public.jsonl"
    test_path = data_dir / "test_public.jsonl"

    rng = random.Random(SEED)

    # Load existing entries
    print("Loading existing entries...")
    dev_entries = load_entries(dev_path)
    test_entries = load_entries(test_path)

    # Filter to get only valid entries for source material
    dev_valid = [e for e in dev_entries if e.label == "VALID"]
    test_valid = [e for e in test_entries if e.label == "VALID"]

    print(f"Dev split: {len(dev_entries)} total ({len(dev_valid)} valid)")
    print(f"Test split: {len(test_entries)} total ({len(test_valid)} valid)")

    # Generate new instances for dev
    print("\nGenerating instances for dev_public.jsonl...")
    dev_new = generate_instances_for_split(
        "dev",
        dev_valid,
        num_chimeric=5,
        num_plausible=5,
        num_version=3,
        num_hybrid=5,
        rng=rng,
    )

    # Generate new instances for test
    print("\nGenerating instances for test_public.jsonl...")
    test_new = generate_instances_for_split(
        "test",
        test_valid,
        num_chimeric=3,
        num_plausible=3,
        num_version=2,
        num_hybrid=3,
        rng=rng,
    )

    # Append to existing entries
    dev_entries.extend(dev_new)
    test_entries.extend(test_new)

    # Save updated files
    print("\nSaving updated dataset files...")
    save_entries(dev_entries, dev_path)
    save_entries(test_entries, test_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nDev split (dev_public.jsonl):")
    print(f"  Total entries: {len(dev_entries)} (+{len(dev_new)})")
    print(f"  New chimeric_title: {5}")
    print(f"  New plausible_fabrication: {5}")
    print(f"  New arxiv_version_mismatch: {3}")
    print(f"  New hybrid_fabrication: {5}")

    print("\nTest split (test_public.jsonl):")
    print(f"  Total entries: {len(test_entries)} (+{len(test_new)})")
    print(f"  New chimeric_title: {3}")
    print(f"  New plausible_fabrication: {3}")
    print(f"  New arxiv_version_mismatch: {2}")
    print(f"  New hybrid_fabrication: {3}")

    print("\nVerifying dataset integrity...")
    dev_hallucinated = [e for e in dev_entries if e.label == "HALLUCINATED"]
    test_hallucinated = [e for e in test_entries if e.label == "HALLUCINATED"]

    print(f"  Dev: {len(dev_entries)} total ({len(dev_hallucinated)} hallucinated)")
    print(f"  Test: {len(test_entries)} total ({len(test_hallucinated)} hallucinated)")

    # Count by type
    print("\nType distribution in dev_public.jsonl:")
    dev_type_counts: dict[str, int] = {}
    for e in dev_hallucinated:
        if e.hallucination_type:
            dev_type_counts[e.hallucination_type] = dev_type_counts.get(e.hallucination_type, 0) + 1
    for htype in sorted(dev_type_counts.keys()):
        print(f"  {htype}: {dev_type_counts[htype]}")

    print("\nType distribution in test_public.jsonl:")
    test_type_counts: dict[str, int] = {}
    for e in test_hallucinated:
        if e.hallucination_type:
            test_type_counts[e.hallucination_type] = (
                test_type_counts.get(e.hallucination_type, 0) + 1
            )
    for htype in sorted(test_type_counts.keys()):
        print(f"  {htype}: {test_type_counts[htype]}")

    print("\nDone! New instances added to dataset files.")


if __name__ == "__main__":
    main()
