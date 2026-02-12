"""Scrape valid BibTeX entries from published proceedings via DBLP.

Targets major ML/AI conferences: NeurIPS, ICML, ICLR, AAAI, ACL, CVPR.
Each scraped entry is verified against at least 2 databases.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date

import httpx

from hallmark.dataset.schema import VALID_SUBTESTS, BenchmarkEntry, GenerationMethod

logger = logging.getLogger(__name__)


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs: object,
) -> httpx.Response | None:
    """Make an HTTP request with exponential backoff on transient failures.

    Returns the response on success, or None after exhausting retries.
    """
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.request(method, url, **kwargs)  # type: ignore[arg-type]
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if attempt == max_retries:
                logger.error(f"Request failed after {max_retries + 1} attempts: {url}: {e}")
                return None
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}, retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
    return None


# DBLP venue keys for major conferences
DBLP_VENUE_KEYS = {
    "NeurIPS": "conf/nips",
    "ICML": "conf/icml",
    "ICLR": "conf/iclr",
    "AAAI": "conf/aaai",
    "ACL": "conf/acl",
    "CVPR": "conf/cvpr",
    "ECCV": "conf/eccv",
    "EMNLP": "conf/emnlp",
    "AISTATS": "conf/aistats",
    "NAACL": "conf/naacl",
}

DBLP_API_BASE = "https://dblp.org/search/publ/api"
CROSSREF_API_BASE = "https://api.crossref.org/works"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


@dataclass
class ScraperConfig:
    """Configuration for the proceedings scraper."""

    venues: list[str] | None = None  # defaults to all
    years: list[int] | None = None  # defaults to 2020-2025
    max_per_venue_year: int = 100
    rate_limit_delay: float = 1.0  # seconds between requests
    timeout: float = 20.0
    verify_against_crossref: bool = True
    verify_against_s2: bool = True
    user_agent: str = "HALLMARK/0.1.0 (https://github.com/rpatrik96/hallmark)"


def scrape_dblp_venue(
    venue_key: str,
    year: int,
    max_results: int = 100,
    config: ScraperConfig | None = None,
) -> list[dict]:
    """Scrape entries from a DBLP venue for a given year.

    Returns raw DBLP hit records.
    """
    config = config or ScraperConfig()
    query = f"venue:{venue_key}/{year}"

    params: dict[str, str | int] = {
        "q": query,
        "format": "json",
        "h": min(max_results, 1000),
    }

    with httpx.Client(timeout=config.timeout) as client:
        resp = _request_with_retry(
            client,
            "GET",
            DBLP_API_BASE,
            params=params,
            headers={"User-Agent": config.user_agent},
        )
        if resp is None:
            logger.error(f"DBLP query failed for {venue_key}/{year} after retries")
            return []
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.error(f"DBLP JSON decode failed for {venue_key}/{year}: {e}")
            return []

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    return [h.get("info", {}) for h in hits if "info" in h]


def dblp_hit_to_entry(
    hit: dict,
    venue_name: str,
    today: str | None = None,
) -> BenchmarkEntry | None:
    """Convert a DBLP hit record to a BenchmarkEntry."""
    title = hit.get("title", "").rstrip(".")
    if not title:
        return None

    authors_info = hit.get("authors", {}).get("author", [])
    if isinstance(authors_info, dict):
        authors_info = [authors_info]
    author_names = [a.get("text", a) if isinstance(a, dict) else str(a) for a in authors_info]
    author_str = " and ".join(author_names)

    year = hit.get("year", "")
    doi = hit.get("doi", "")
    url = hit.get("url", "")
    venue = hit.get("venue", venue_name)

    entry_type = hit.get("type", "")
    bibtex_type = "inproceedings" if "Conference" in entry_type else "article"

    # Generate key: FirstAuthorYear
    first_author = author_names[0].split()[-1] if author_names else "unknown"
    first_word = title.split()[0].lower() if title else "untitled"
    bibtex_key = f"{first_author}{year}{first_word}"

    if today is None:
        today = date.today().isoformat()

    fields: dict[str, str] = {
        "title": title,
        "author": author_str,
        "year": year,
    }
    if doi:
        fields["doi"] = doi
    if url:
        fields["url"] = url
    if bibtex_type == "inproceedings":
        fields["booktitle"] = venue
    else:
        fields["journal"] = venue

    return BenchmarkEntry(
        bibtex_key=bibtex_key,
        bibtex_type=bibtex_type,
        fields=fields,
        label="VALID",
        explanation="Valid entry scraped from DBLP and verified",
        generation_method=GenerationMethod.SCRAPED.value,
        source_conference=venue_name,
        publication_date=f"{year}-01-01" if year else "",
        added_to_benchmark=today,
        subtests={**VALID_SUBTESTS, "doi_resolves": bool(doi)},
    )


def verify_entry_crossref(entry: BenchmarkEntry, config: ScraperConfig | None = None) -> bool:
    """Verify an entry against CrossRef."""
    config = config or ScraperConfig()
    doi = entry.fields.get("doi")

    with httpx.Client(timeout=config.timeout) as client:
        if doi:
            resp = _request_with_retry(
                client,
                "GET",
                f"{CROSSREF_API_BASE}/{doi}",
                headers={"User-Agent": config.user_agent},
            )
            if resp is not None:
                return True

        # Fallback to title search
        title = entry.fields.get("title", "")
        if not title:
            return False

        resp = _request_with_retry(
            client,
            "GET",
            CROSSREF_API_BASE,
            params={"query.title": title, "rows": 5},
            headers={"User-Agent": config.user_agent},
        )
        if resp is not None:
            try:
                items = resp.json().get("message", {}).get("items", [])
                return len(items) > 0
            except json.JSONDecodeError:
                pass

    return False


def verify_entry_s2(entry: BenchmarkEntry, config: ScraperConfig | None = None) -> bool:
    """Verify an entry against Semantic Scholar."""
    config = config or ScraperConfig()
    title = entry.fields.get("title", "")
    if not title:
        return False

    with httpx.Client(timeout=config.timeout) as client:
        resp = _request_with_retry(
            client,
            "GET",
            f"{S2_API_BASE}/paper/search",
            params={"query": title, "limit": 5},
            headers={"User-Agent": config.user_agent},
        )
        if resp is not None:
            try:
                papers = resp.json().get("data", [])
                return len(papers) > 0
            except json.JSONDecodeError:
                pass

    return False


def scrape_proceedings(
    config: ScraperConfig | None = None,
) -> list[BenchmarkEntry]:
    """Scrape valid entries from conference proceedings.

    Main entry point for dataset construction.
    """
    config = config or ScraperConfig()
    venues = config.venues or list(DBLP_VENUE_KEYS.keys())
    years = config.years or list(range(2020, 2026))
    today = date.today().isoformat()

    all_entries: list[BenchmarkEntry] = []
    total_hits = 0
    skipped = 0

    for venue in venues:
        venue_key = DBLP_VENUE_KEYS.get(venue)
        if not venue_key:
            logger.warning(f"Unknown venue: {venue}, skipping")
            continue

        for year in years:
            logger.info(f"Scraping {venue} {year}...")
            hits = scrape_dblp_venue(venue_key, year, config.max_per_venue_year, config)
            total_hits += len(hits)
            time.sleep(config.rate_limit_delay)

            for hit in hits:
                entry = dblp_hit_to_entry(hit, venue, today)
                if entry is None:
                    continue

                # Verify against secondary sources
                verified_count = 1  # Already from DBLP
                if config.verify_against_crossref:
                    time.sleep(config.rate_limit_delay)
                    if verify_entry_crossref(entry, config):
                        verified_count += 1

                if config.verify_against_s2 and verified_count < 2:
                    time.sleep(config.rate_limit_delay)
                    if verify_entry_s2(entry, config):
                        verified_count += 1

                if verified_count >= 2:
                    all_entries.append(entry)
                    logger.debug(f"  Verified: {entry.bibtex_key} ({verified_count} sources)")
                else:
                    skipped += 1
                    logger.debug(f"  Skipped (only {verified_count} source): {entry.bibtex_key}")

            logger.info(f"  {venue} {year}: {len(hits)} hits, {len(all_entries)} total verified")

    logger.info(
        f"Scraped {len(all_entries)} entries "
        f"({len(all_entries)}/{total_hits} verified, {skipped} skipped)"
    )
    return all_entries
