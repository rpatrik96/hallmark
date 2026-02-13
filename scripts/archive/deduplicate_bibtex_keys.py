#!/usr/bin/env python3
"""Fix duplicate bibtex_keys in HALLMARK data files by appending suffixes."""

from collections import Counter
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, load_entries, save_entries


def deduplicate_keys(entries: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
    """Deduplicate bibtex_keys by appending _2, _3, etc. to duplicates."""
    key_counts: dict[str, int] = {}
    result = []

    for entry in entries:
        original_key = entry.bibtex_key
        if original_key not in key_counts:
            key_counts[original_key] = 1
            result.append(entry)
        else:
            key_counts[original_key] += 1
            suffix = key_counts[original_key]
            entry.bibtex_key = f"{original_key}_{suffix}"
            result.append(entry)

    return result


def main():
    data_dir = Path("data/v1.0")

    for split_file in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / split_file
        print(f"\nProcessing {path}...")

        # Load entries
        entries = load_entries(path)
        print(f"  Loaded {len(entries)} entries")

        # Check for duplicates
        keys = [e.bibtex_key for e in entries]
        dupes = {k: c for k, c in Counter(keys).items() if c > 1}
        if dupes:
            print(f"  Found {len(dupes)} duplicate keys:")
            for k, c in list(dupes.items())[:10]:
                print(f"    {k} appears {c} times")

            # Deduplicate
            deduped = deduplicate_keys(entries)

            # Verify no duplicates remain
            new_keys = [e.bibtex_key for e in deduped]
            new_dupes = {k: c for k, c in Counter(new_keys).items() if c > 1}
            if new_dupes:
                raise ValueError(f"Deduplication failed: {new_dupes}")

            # Save
            save_entries(deduped, path)
            print(f"  ✓ Deduplicated and saved {len(deduped)} entries")
        else:
            print("  No duplicates found")


if __name__ == "__main__":
    main()
