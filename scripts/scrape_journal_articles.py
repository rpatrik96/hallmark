#!/usr/bin/env python3
"""CLI wrapper to scrape journal articles from DBLP.  [data-collection]

Delegates all logic to hallmark.dataset.scraper.scrape_journal_articles().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        help="Number of entries to fetch per journal per year (default: 40)",
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

    from hallmark.dataset.schema import save_entries
    from hallmark.dataset.scraper import ScraperConfig, scrape_journal_articles

    config = ScraperConfig(
        rate_limit_delay=args.rate_limit,
        verify_against_crossref=False,
        verify_against_s2=False,
    )

    years = list(range(args.start_year, args.end_year + 1))
    entries = scrape_journal_articles(
        journals=["JMLR", "MLJ", "TMLR"],
        years=years,
        max_per_journal_year=args.num_per_journal,
        config=config,
    )

    print(f"\nTotal entries collected: {len(entries)}", file=sys.stderr)

    if args.dry_run:
        import json

        print("\n=== DRY RUN: Preview of first 3 entries ===\n", file=sys.stderr)
        for entry in entries[:3]:
            print(json.dumps(entry.to_dict(), indent=2))
            print()
        print(f"(showing 3/{len(entries)} entries)", file=sys.stderr)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_entries(entries, args.output)
        print(f"Saved {len(entries)} entries to {args.output}", file=sys.stderr)

        if entries:
            with_doi = sum(1 for e in entries if e.fields.get("doi"))
            pct = 100 * with_doi / len(entries)
            print("\nSummary:", file=sys.stderr)
            print(
                f"  Entries with DOI: {with_doi}/{len(entries)} ({pct:.1f}%)",
                file=sys.stderr,
            )
        else:
            print("\nNo entries collected. Check network/SSL settings.", file=sys.stderr)


if __name__ == "__main__":
    main()
