#!/usr/bin/env python3
"""Scrape valid journal articles from DBLP to diversify benchmark beyond conference papers.

This addresses P1.5: Add valid journal articles to eliminate article/misc type as shortcut.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def fetch_dblp_journal(
    journal_name: str, num_entries: int, start_year: int, end_year: int
) -> list[dict[str, Any]]:
    """Fetch journal articles from DBLP API.

    Args:
        journal_name: Journal name or abbreviation (e.g., 'JMLR', 'TMLR')
        num_entries: Target number of entries to fetch
        start_year: Start of year range
        end_year: End of year range (inclusive)

    Returns:
        List of DBLP publication records
    """
    base_url = "https://dblp.org/search/publ/api"
    entries = []

    # Query format: venue:JMLR year:2021..2023
    query = f"venue:{journal_name} year:{start_year}..{end_year}"
    params = {
        "q": query,
        "format": "json",
        "h": str(num_entries),  # Number of hits
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    print(
        f"Fetching {num_entries} entries from {journal_name} ({start_year}-{end_year})...",
        file=sys.stderr,
    )

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "HALLMARK-Benchmark/1.0 (mailto:research@example.com)")

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        if "result" in data and "hits" in data["result"]:
            hits = data["result"]["hits"].get("hit", [])
            for hit in hits:
                if "info" in hit:
                    entries.append(hit["info"])

        print(f"  → Fetched {len(entries)} entries", file=sys.stderr)

    except urllib.error.HTTPError as e:
        print(f"  ✗ HTTP error {e.code}: {e.reason}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"  ✗ URL error: {e.reason}", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}", file=sys.stderr)

    return entries


def convert_to_benchmark_entry(dblp_entry: dict[str, Any], seq_id: int) -> dict[str, Any]:
    """Convert DBLP entry to HALLMARK BenchmarkEntry format.

    Args:
        dblp_entry: DBLP publication record
        seq_id: Sequential ID for bibtex key

    Returns:
        HALLMARK BenchmarkEntry dict
    """
    # Extract authors (DBLP returns list of dicts with 'text' field or plain strings)
    authors_raw = dblp_entry.get("authors", {}).get("author", [])
    if isinstance(authors_raw, dict):
        authors_raw = [authors_raw]
    authors = []
    for author in authors_raw:
        if isinstance(author, dict):
            authors.append(author.get("text", ""))
        else:
            authors.append(str(author))
    author_str = " and ".join(authors) if authors else "Unknown Author"

    # Extract venue (journal name)
    venue = dblp_entry.get("venue", "Unknown Journal")

    # Extract DOI (may be missing)
    doi = dblp_entry.get("doi", "")

    # Extract year
    year_str = dblp_entry.get("year", "")

    # Extract title
    title = dblp_entry.get("title", "Untitled")

    # Extract volume, number, pages if available
    volume = dblp_entry.get("volume", "")
    number = dblp_entry.get("number", "")
    pages = dblp_entry.get("pages", "")

    # Construct BibTeX string
    bibtex_lines = [
        f"@article{{hallmark_journal_{seq_id:04d},",
        f"  title = {{{title}}},",
        f"  author = {{{author_str}}},",
        f"  journal = {{{venue}}},",
        f"  year = {{{year_str}}},",
    ]

    if volume:
        bibtex_lines.append(f"  volume = {{{volume}}},")
    if number:
        bibtex_lines.append(f"  number = {{{number}}},")
    if pages:
        bibtex_lines.append(f"  pages = {{{pages}}},")
    if doi:
        bibtex_lines.append(f"  doi = {{{doi}}},")

    # Add URL (DBLP provides 'url' or 'ee' fields)
    url = dblp_entry.get("ee", dblp_entry.get("url", ""))
    if url:
        # DBLP 'ee' can be list; take first if so
        if isinstance(url, list):
            url = url[0] if url else ""
        bibtex_lines.append(f"  url = {{{url}}},")

    bibtex_lines.append("}")
    bibtex_str = "\n".join(bibtex_lines)

    # Build subtests
    subtests = {
        "doi_resolves": bool(doi),
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": bool(authors and title and venue and year_str),
        "cross_db_agreement": bool(doi),  # If DOI exists, CrossRef can verify
    }

    return {
        "bibtex_key": f"hallmark_journal_{seq_id:04d}",
        "bibtex_type": "article",
        "bibtex": bibtex_str,
        "label": "VALID",
        "generation_method": "scraped",
        "source": f"DBLP:{venue}",
        "subtests": subtests,
        "metadata": {
            "dblp_key": dblp_entry.get("key", ""),
            "original_venue": venue,
            "has_doi": bool(doi),
        },
    }


def main() -> None:
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape valid journal articles from DBLP to diversify HALLMARK benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/v1.0/journal_articles.jsonl"),
        help="Output JSONL file path (default: data/v1.0/journal_articles.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview entries without saving",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between API requests (default: 1.0)",
    )
    parser.add_argument(
        "--num-per-journal",
        type=int,
        default=40,
        help="Number of entries to fetch per journal (default: 40)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year for publication range (default: 2021)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year for publication range (default: 2023)",
    )

    args = parser.parse_args()

    # Target journals
    journals = [
        "JMLR",  # Journal of Machine Learning Research
        "TMLR",  # Transactions on Machine Learning Research
        "Mach. Learn.",  # Machine Learning (Springer)
    ]

    all_entries = []
    seq_id = 0

    for journal in journals:
        dblp_entries = fetch_dblp_journal(
            journal_name=journal,
            num_entries=args.num_per_journal,
            start_year=args.start_year,
            end_year=args.end_year,
        )

        for dblp_entry in dblp_entries:
            benchmark_entry = convert_to_benchmark_entry(dblp_entry, seq_id)
            all_entries.append(benchmark_entry)
            seq_id += 1

        # Rate limiting between journals
        if journal != journals[-1]:  # Don't sleep after last journal
            time.sleep(args.rate_limit)

    print(f"\nTotal entries collected: {len(all_entries)}", file=sys.stderr)

    if args.dry_run:
        print("\n=== DRY RUN: Preview of first 3 entries ===\n", file=sys.stderr)
        for entry in all_entries[:3]:
            print(json.dumps(entry, indent=2))
            print()
        print(f"(showing 3/{len(all_entries)} entries)", file=sys.stderr)
    else:
        # Save to JSONL
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✓ Saved {len(all_entries)} entries to {args.output}", file=sys.stderr)

        # Print summary statistics
        with_doi = sum(1 for e in all_entries if e["metadata"]["has_doi"])
        pct = 100 * with_doi / len(all_entries)
        print("\nSummary:", file=sys.stderr)
        print(f"  Entries with DOI: {with_doi}/{len(all_entries)} ({pct:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
