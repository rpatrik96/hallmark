#!/usr/bin/env python3
"""Scrape valid BibTeX entries from conference proceedings via DBLP.

Usage:
    python scripts/scrape_proceedings.py --venues NeurIPS ICML ICLR \
        --years 2020 2021 2022 2023 2024 2025 \
        --max-per-venue-year 50 \
        --output data/v1.0/valid_entries.jsonl
"""

from __future__ import annotations

import argparse
import logging

from hallmark.dataset.schema import save_entries
from hallmark.dataset.scraper import ScraperConfig, scrape_proceedings


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape valid entries from proceedings")
    parser.add_argument(
        "--venues",
        nargs="+",
        default=["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR"],
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2020, 2026)),
    )
    parser.add_argument("--max-per-venue-year", type=int, default=100)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = ScraperConfig(
        venues=args.venues,
        years=args.years,
        max_per_venue_year=args.max_per_venue_year,
        rate_limit_delay=args.rate_limit,
    )

    entries = scrape_proceedings(config)
    save_entries(entries, args.output)
    logging.info(f"Saved {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
