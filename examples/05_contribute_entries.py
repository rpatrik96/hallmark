#!/usr/bin/env python3
"""Example 5: Contributing new entries to the HALLMARK pool.

This example shows how to:
1. Create properly formatted benchmark entries
2. Validate entries before submission
3. Submit a contribution to the pool
"""

from hallmark.contribution.validate_entry import validate_batch, validate_entry
from hallmark.dataset.schema import BenchmarkEntry, save_entries


def main():
    # 1. Create a valid entry
    valid_entry = BenchmarkEntry(
        bibtex_key="hinton2006reducing",
        bibtex_type="article",
        fields={
            "title": "Reducing the Dimensionality of Data with Neural Networks",
            "author": "Geoffrey E. Hinton and Ruslan R. Salakhutdinov",
            "year": "2006",
            "journal": "Science",
            "doi": "10.1126/science.1127647",
        },
        label="VALID",
        explanation="Classic paper on autoencoders, verified via CrossRef",
        generation_method="scraped",
        source_conference="Science",
        publication_date="2006-07-28",
        added_to_benchmark="2026-02-09",
        subtests={
            "doi_resolves": True,
            "title_exists": True,
            "authors_match": True,
            "venue_correct": True,
            "fields_complete": True,
            "cross_db_agreement": True,
        },
    )

    # 2. Create a hallucinated entry
    hallucinated_entry = BenchmarkEntry(
        bibtex_key="hinton2006deep_fake",
        bibtex_type="article",
        fields={
            "title": "Deep Learning for Dimensionality Enhancement",
            "author": "Geoffrey E. Hinton and Ruslan R. Salakhutdinov",
            "year": "2006",
            "journal": "Science",
        },
        label="HALLUCINATED",
        hallucination_type="chimeric_title",
        difficulty_tier=2,
        explanation="Title is fabricated; authors and venue are real",
        generation_method="perturbation",
        source_conference="Science",
        publication_date="2006-07-28",
        added_to_benchmark="2026-02-09",
        subtests={
            "doi_resolves": False,
            "title_exists": False,
            "authors_match": True,
            "venue_correct": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        },
    )

    # 3. Validate entries
    print("=== Validating Entries ===\n")
    for entry in [valid_entry, hallucinated_entry]:
        result = validate_entry(entry)
        status = "VALID" if result.valid else "INVALID"
        print(f"  {entry.bibtex_key}: {status}")
        if result.errors:
            for err in result.errors:
                print(f"    ERROR: {err}")
        if result.warnings:
            for warn in result.warnings:
                print(f"    WARNING: {warn}")

    # 4. Batch validation
    entries = [valid_entry, hallucinated_entry]
    batch_results = validate_batch(entries)
    n_valid = sum(1 for r in batch_results if r.valid)
    print(f"\nBatch: {n_valid}/{len(batch_results)} valid")

    # 5. Save to JSONL for submission
    output_path = "/tmp/my_contribution.jsonl"
    save_entries(entries, output_path)
    print(f"\nSaved {len(entries)} entries to {output_path}")
    print(f"Submit with: hallmark contribute --file {output_path} --contributor 'Your Name'")


if __name__ == "__main__":
    main()
