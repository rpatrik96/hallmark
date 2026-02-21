"""P2.1: Test dataset generation seed sensitivity.  [analysis]

Generates entries with different random seeds and measures diversity.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from hallmark.dataset.generator import (
    generate_fabricated_doi,
    generate_future_date,
    generate_near_miss_title,
    generate_nonexistent_venue,
    generate_placeholder_authors,
    generate_plausible_fabrication,
)
from hallmark.dataset.schema import BenchmarkEntry


def load_valid_entries(path: Path, limit: int = 20) -> list[BenchmarkEntry]:
    """Load valid entries from dev_public.jsonl."""
    entries = []
    with path.open("r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("label") == "VALID":
                entry = BenchmarkEntry(
                    bibtex_key=data["bibtex_key"],
                    bibtex_type=data["bibtex_type"],
                    fields=data["fields"],
                    label=data["label"],
                    hallucination_type=data.get("hallucination_type"),
                    difficulty_tier=data.get("difficulty_tier"),
                    explanation=data.get("explanation", ""),
                    generation_method=data.get("generation_method", ""),
                    source_conference=data.get("source_conference", ""),
                    publication_date=data.get("publication_date", ""),
                    added_to_benchmark=data.get("added_to_benchmark", ""),
                    subtests=data.get("subtests", {}),
                    raw_bibtex=data.get("raw_bibtex"),
                )
                entries.append(entry)
                if len(entries) >= limit:
                    break
    return entries


def generate_with_seed(
    valid_entries: list[BenchmarkEntry], seed: int, samples_per_type: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Generate hallucinated entries with a specific seed."""
    rng = random.Random(seed)
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    generators = [
        ("fabricated_doi", generate_fabricated_doi),
        ("nonexistent_venue", generate_nonexistent_venue),
        ("placeholder_authors", generate_placeholder_authors),
        ("future_date", generate_future_date),
        ("near_miss_title", generate_near_miss_title),
        ("plausible_fabrication", generate_plausible_fabrication),
    ]

    for gen_type, gen_func in generators:
        for _ in range(samples_per_type):
            source = rng.choice(valid_entries)
            generated = gen_func(source, rng=rng)
            # Store relevant fields for comparison
            results[gen_type].append(
                {
                    "title": generated.fields.get("title", ""),
                    "author": generated.fields.get("author", ""),
                    "venue": generated.fields.get("booktitle")
                    or generated.fields.get("journal", ""),
                    "year": generated.fields.get("year", ""),
                    "doi": generated.fields.get("doi"),
                }
            )

    return results


def compute_diversity_metrics(all_results: dict[int, dict[str, list[dict[str, Any]]]]) -> None:
    """Compare results across seeds and compute diversity metrics."""
    seeds = sorted(all_results.keys())
    print(f"Analyzing {len(seeds)} seeds: {seeds}")

    for gen_type in [
        "fabricated_doi",
        "nonexistent_venue",
        "placeholder_authors",
        "future_date",
        "near_miss_title",
        "plausible_fabrication",
    ]:
        print(f"\n{'=' * 80}")
        print(f"Generator: {gen_type}")
        print(f"{'=' * 80}")

        # Collect all entries for this generator across seeds
        all_entries = []
        entries_by_seed = {}
        for seed in seeds:
            entries = all_results[seed].get(gen_type, [])
            all_entries.extend(entries)
            entries_by_seed[seed] = entries

        if not all_entries:
            print("No entries generated")
            continue

        # Count unique values per field
        unique_titles = set(e["title"] for e in all_entries)
        unique_authors = set(e["author"] for e in all_entries)
        unique_venues = set(e["venue"] for e in all_entries if e["venue"])
        unique_dois = set(e["doi"] for e in all_entries if e["doi"])

        print(f"Total entries: {len(all_entries)}")
        print(f"Unique titles: {len(unique_titles)}")
        print(f"Unique authors: {len(unique_authors)}")
        print(f"Unique venues: {len(unique_venues)}")
        print(f"Unique DOIs: {len(unique_dois)}")

        # Compare across seeds: how many entries are identical?
        identical_count = 0
        total_comparisons = 0

        for i, seed1 in enumerate(seeds):
            for seed2 in seeds[i + 1 :]:
                entries1 = entries_by_seed[seed1]
                entries2 = entries_by_seed[seed2]

                # Check how many entries match between these two seeds
                for e1 in entries1:
                    for e2 in entries2:
                        total_comparisons += 1
                        if (
                            e1["title"] == e2["title"]
                            and e1["author"] == e2["author"]
                            and e1["venue"] == e2["venue"]
                        ):
                            identical_count += 1

        if total_comparisons > 0:
            identical_pct = (identical_count / total_comparisons) * 100
            varying_pct = 100 - identical_pct
            print(
                f"\nCross-seed comparison: {identical_pct:.1f}% identical, {varying_pct:.1f}% vary"
            )

        # Feature diversity: DOI prefixes, venue diversity
        if gen_type == "fabricated_doi":
            doi_prefixes = set()
            for e in all_entries:
                if e["doi"]:
                    prefix = e["doi"].split("/")[0]
                    doi_prefixes.add(prefix)
            print(f"Unique DOI prefixes: {len(doi_prefixes)}")

        if gen_type == "nonexistent_venue":
            print(f"Unique fabricated venues: {len(unique_venues)}")

        if gen_type == "placeholder_authors":
            print(f"Unique placeholder author sets: {len(unique_authors)}")


def main() -> None:
    """Run seed sensitivity analysis."""
    data_dir = Path(__file__).parent.parent / "data" / "v1.0"
    dev_path = data_dir / "dev_public.jsonl"

    print("Loading valid entries from dev_public.jsonl...")
    valid_entries = load_valid_entries(dev_path, limit=20)
    print(f"Loaded {len(valid_entries)} valid entries as source material")

    seeds = [42, 123, 456, 789, 1024]
    all_results = {}

    print(f"\nGenerating entries with {len(seeds)} different seeds...")
    for seed in seeds:
        print(f"  Seed {seed}...")
        all_results[seed] = generate_with_seed(valid_entries, seed, samples_per_type=5)

    print("\n" + "=" * 80)
    print("SEED SENSITIVITY ANALYSIS")
    print("=" * 80)

    compute_diversity_metrics(all_results)

    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Analysis complete: {len(seeds)} seeds, ~{5 * 6 * len(seeds)} total entries generated")
    print("\nConclusion: High variation across seeds indicates proper randomization.")
    print("Low variation suggests deterministic patterns or insufficient randomness pool.")


if __name__ == "__main__":
    main()
