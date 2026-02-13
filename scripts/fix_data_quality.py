#!/usr/bin/env python3
"""Fix data quality issues in HALLMARK benchmark data files.

This script:
1. Strips DBLP numeric suffixes from author names (e.g., "Author 0001" -> "Author")
2. Deduplicates bibtex_keys across dev_public and test_public splits
"""

import re
from pathlib import Path

from hallmark.dataset.schema import load_entries, save_entries


def strip_dblp_suffixes(entries):
    """Strip DBLP numeric suffixes from author names.

    Args:
        entries: List of BenchmarkEntry objects

    Returns:
        Number of entries modified
    """
    modified_count = 0
    suffix_pattern = re.compile(r" \d{4}")

    for entry in entries:
        if entry.fields.get("author"):
            original_author = entry.fields["author"]
            # Split by " and ", strip suffixes from each author, rejoin
            authors = original_author.split(" and ")
            cleaned_authors = [suffix_pattern.sub("", author.strip()) for author in authors]
            cleaned_author_string = " and ".join(cleaned_authors)

            if cleaned_author_string != original_author:
                entry.fields["author"] = cleaned_author_string
                modified_count += 1

    return modified_count


def deduplicate_bibtex_keys(dev_entries, test_entries):
    """Deduplicate bibtex_keys across splits by prefixing test collisions.

    Args:
        dev_entries: List of BenchmarkEntry objects from dev split
        test_entries: List of BenchmarkEntry objects from test split

    Returns:
        Number of keys deduplicated
    """
    dev_keys = {entry.bibtex_key for entry in dev_entries}

    collision_count = 0
    for entry in test_entries:
        if entry.bibtex_key in dev_keys:
            entry.bibtex_key = f"test_{entry.bibtex_key}"
            collision_count += 1

    return collision_count


def main():
    data_dir = Path("data/v1.0")
    dev_path = data_dir / "dev_public.jsonl"
    test_path = data_dir / "test_public.jsonl"

    print("Loading data files...")
    dev_entries = load_entries(dev_path)
    test_entries = load_entries(test_path)

    print(f"Loaded {len(dev_entries)} dev entries, {len(test_entries)} test entries")

    # Fix 1: Strip DBLP suffixes
    print("\n=== Stripping DBLP numeric suffixes ===")
    dev_suffix_count = strip_dblp_suffixes(dev_entries)
    test_suffix_count = strip_dblp_suffixes(test_entries)

    print(f"Dev split: {dev_suffix_count} entries had suffixes stripped")
    print(f"Test split: {test_suffix_count} entries had suffixes stripped")
    print(f"Total: {dev_suffix_count + test_suffix_count} entries modified")

    # Fix 2: Deduplicate bibtex_keys
    print("\n=== Deduplicating bibtex_keys ===")
    collision_count = deduplicate_bibtex_keys(dev_entries, test_entries)
    print(f"Fixed {collision_count} colliding keys in test_public (prefixed with 'test_')")

    # Save back to files
    print("\n=== Saving modified files ===")
    save_entries(dev_entries, dev_path)
    save_entries(test_entries, test_path)

    print(f"Saved {dev_path}")
    print(f"Saved {test_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
