#!/usr/bin/env python3
"""Enrich valid entries with DOIs from CrossRef API.

This script adds DOIs to VALID entries that currently lack them, reducing
the DOI distributional artifact in the benchmark dataset.

Usage:
    python scripts/enrich_valid_dois.py
"""

import sys
import time
from pathlib import Path
from typing import Any

import requests
from rapidfuzz import fuzz

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hallmark.dataset.schema import BenchmarkEntry


def extract_last_names(author_string: str) -> set[str]:
    """Extract last names from BibTeX author string.

    Handles formats like:
    - "John Doe"
    - "John Doe and Jane Smith"
    - "Doe, John and Smith, Jane"
    """
    last_names = set()
    # Split by 'and' to get individual authors
    authors = author_string.split(" and ")
    for author in authors:
        author = author.strip()
        if not author:
            continue

        # Handle "Last, First" format
        if "," in author:
            last_name = author.split(",")[0].strip()
            last_names.add(last_name.lower())
        else:
            # Handle "First Last" format - take last word
            parts = author.split()
            if parts:
                last_names.add(parts[-1].lower())

    return last_names


def query_crossref(title: str) -> list[dict[str, Any]]:
    """Query CrossRef API for a title.

    Args:
        title: Paper title to search for

    Returns:
        List of matching works from CrossRef
    """
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": title,
        "rows": 3,
    }
    headers = {
        "User-Agent": "HALLMARK-Benchmark/1.0 (mailto:hallmark@example.com)",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("items", [])
    except Exception as e:
        print(f"  ⚠️  CrossRef API error: {e}", file=sys.stderr)
        return []


def find_doi_match(
    entry_title: str, entry_authors: str, crossref_results: list[dict[str, Any]]
) -> str | None:
    """Find a high-confidence DOI match from CrossRef results.

    Match criteria:
    - Title similarity >= 90 (using token_sort_ratio)
    - At least one author last name match

    Args:
        entry_title: Title from benchmark entry
        entry_authors: Author string from benchmark entry
        crossref_results: Results from CrossRef API

    Returns:
        DOI string if high-confidence match found, None otherwise
    """
    entry_last_names = extract_last_names(entry_authors)

    for result in crossref_results:
        # Check title similarity
        crossref_title = result.get("title", [""])[0]
        title_similarity = fuzz.token_sort_ratio(entry_title.lower(), crossref_title.lower())

        if title_similarity < 90:
            continue

        # Check author match
        crossref_authors = result.get("author", [])
        crossref_last_names = set()
        for author in crossref_authors:
            family = author.get("family", "").lower()
            if family:
                crossref_last_names.add(family)

        # At least one author last name must match
        if entry_last_names & crossref_last_names:
            doi = result.get("DOI")
            if doi:
                return doi

    return None


def enrich_file(input_path: Path, output_path: Path) -> dict[str, int]:
    """Enrich a single JSONL file with DOIs.

    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path

    Returns:
        Statistics dictionary
    """
    entries = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))

    stats = {
        "total_entries": len(entries),
        "valid_entries": 0,
        "valid_with_doi": 0,
        "valid_without_doi": 0,
        "enriched": 0,
        "skipped_no_match": 0,
    }

    enriched_entries = []

    for entry in entries:
        if entry.label != "VALID":
            enriched_entries.append(entry)
            continue

        stats["valid_entries"] += 1

        # Check if DOI already exists
        existing_doi = entry.fields.get("doi")
        if existing_doi:
            stats["valid_with_doi"] += 1
            enriched_entries.append(entry)
            continue

        stats["valid_without_doi"] += 1

        # Query CrossRef
        title = entry.fields.get("title", "")
        author = entry.fields.get("author", "")

        if not title or not author:
            print(
                f"  ⚠️  Skipping {entry.bibtex_key}: missing title or author",
                file=sys.stderr,
            )
            stats["skipped_no_match"] += 1
            enriched_entries.append(entry)
            continue

        print(f"  🔍 Querying: {entry.bibtex_key} - {title[:60]}...")

        crossref_results = query_crossref(title)
        doi = find_doi_match(title, author, crossref_results)

        if doi:
            print(f"    ✅ Found DOI: {doi}")
            # Add DOI to entry fields
            entry.fields["doi"] = doi
            stats["enriched"] += 1
        else:
            print("    ❌ No high-confidence match")
            stats["skipped_no_match"] += 1

        enriched_entries.append(entry)

        # Rate limit: 1 request per second
        time.sleep(1)

    # Write enriched entries to output file
    with open(output_path, "w") as f:
        for entry in enriched_entries:
            f.write(entry.to_json() + "\n")

    return stats


def main() -> None:
    """Main entry point."""
    # Define paths
    data_dir = Path(__file__).parent.parent / "data" / "v1.0"

    files_to_process = [
        ("dev_public.jsonl", "dev_public_enriched.jsonl"),
        ("test_public.jsonl", "test_public_enriched.jsonl"),
    ]

    print("=" * 80)
    print("DOI Enrichment Script for HALLMARK Benchmark")
    print("=" * 80)
    print()

    for input_name, output_name in files_to_process:
        input_path = data_dir / input_name
        output_path = data_dir / output_name

        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            continue

        print(f"📂 Processing: {input_name}")
        print(f"   Output to: {output_name}")
        print()

        stats = enrich_file(input_path, output_path)

        print()
        print(f"📊 Summary for {input_name}:")
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Valid entries: {stats['valid_entries']}")
        print(f"   Valid with DOI (before): {stats['valid_with_doi']}")
        print(f"   Valid without DOI (before): {stats['valid_without_doi']}")
        print(f"   ✅ Enriched with new DOI: {stats['enriched']}")
        print(f"   ❌ No match found: {stats['skipped_no_match']}")
        print(f"   Valid with DOI (after): {stats['valid_with_doi'] + stats['enriched']}")
        print()

    print("=" * 80)
    print("✅ Enrichment complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
