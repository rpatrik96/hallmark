"""Stage 1: Scrape or load valid BibTeX entries.

Two modes:
- Live scrape via hallmark.dataset.scraper (requires network)
- Load cached valid entries from a JSONL file (default for reproducibility)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, load_entries

logger = logging.getLogger(__name__)


def stage_load_cached_valid(path: Path) -> list[BenchmarkEntry]:
    """Load pre-scraped valid entries from a JSONL file."""
    entries = load_entries(path)
    valid = [e for e in entries if e.label == "VALID"]
    logger.info("Loaded %d valid entries from %s", len(valid), path)
    return valid


def stage_scrape_valid(
    conferences: list[str] | None = None,
    years: list[int] | None = None,
    rng: random.Random | None = None,
    include_arxiv: bool = True,
) -> list[BenchmarkEntry]:
    """Scrape valid entries from DBLP/Semantic Scholar and optionally arXiv.

    Requires network access. Falls back to cached data if scraping fails.
    """
    from hallmark.dataset.scraper import ScraperConfig, scrape_arxiv_recent, scrape_proceedings

    conferences = conferences or ["NeurIPS", "ICML", "ICLR"]
    years = years or [2021, 2022, 2023]

    config = ScraperConfig(venues=conferences, years=years, include_arxiv=include_arxiv)
    entries = scrape_proceedings(config)

    # Set source field for scraped entries
    for entry in entries:
        entry.source = "dblp"

    logger.info("Scraped %d valid entries from %d conferences", len(entries), len(conferences))

    # Optionally scrape recent arXiv entries
    if include_arxiv:
        arxiv_entries = scrape_arxiv_recent(config=config)
        for entry in arxiv_entries:
            entry.source = "arxiv"
        entries.extend(arxiv_entries)
        logger.info("Added %d arXiv entries (total: %d)", len(arxiv_entries), len(entries))

    return entries


def _parse_bibtex_fields(raw: str) -> dict[str, str]:
    """Extract key=value fields from a raw BibTeX string."""
    import re

    fields: dict[str, str] = {}
    # Match field = {value} or field = "value"
    for m in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", raw):
        fields[m.group(1).lower()] = m.group(2).strip()
    return fields


def stage_load_journal_articles(path: Path) -> list[BenchmarkEntry]:
    """Load journal articles as additional valid entries.

    These come from scrape_journal_articles.py and may use a different schema
    (bibtex string instead of fields dict). Handles both formats.
    """
    import json

    if not path.exists():
        logger.warning("Journal articles file not found: %s", path)
        return []

    entries: list[BenchmarkEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Handle journal article format: has 'bibtex' key instead of 'fields'
            if "bibtex" in data and "fields" not in data:
                fields = _parse_bibtex_fields(data["bibtex"])
                # Convert journal -> booktitle for consistency
                if "journal" in fields and "booktitle" not in fields:
                    fields["booktitle"] = fields.pop("journal")

                entry = BenchmarkEntry(
                    bibtex_key=data["bibtex_key"],
                    bibtex_type="inproceedings",
                    fields=fields,
                    label=data.get("label", "VALID"),
                    generation_method=data.get("generation_method", "scraped"),
                    source_conference=data.get("source"),
                    source="dblp",
                    subtests=data.get("subtests", {}),
                    raw_bibtex=data.get("bibtex"),
                )
                if entry.label == "VALID":
                    entries.append(entry)
            else:
                # Standard BenchmarkEntry format
                try:
                    entry = BenchmarkEntry.from_dict(data)
                    if entry.label == "VALID":
                        entries.append(entry)
                except (TypeError, ValueError) as e:
                    logger.warning("Skipping malformed entry: %s", e)

    logger.info("Loaded %d journal article entries from %s", len(entries), path)
    return entries
