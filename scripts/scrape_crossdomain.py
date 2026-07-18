#!/usr/bin/env python3
"""Scrape valid bibliographic entries from non-ML domains for HALLMARK v1.1.

Three sub-corpora targeted by the v1.1 cross-domain test split:

1. CS-non-ML: DBLP venue keys for SIGIR, KDD, WWW, SIGMOD, SIGGRAPH, etc.
2. Bio/life sciences: bioRxiv API (recent preprints) + medRxiv.
3. Medical/clinical: PubMed E-utilities (peer-reviewed indexed papers).

Each scraped entry is tagged with ``source_conference`` describing the domain
sub-corpus it belongs to and ``source`` identifying the specific upstream
database. Output is the union of all three sub-corpora as a single JSONL file
of VALID entries; subsequent steps (perturbation, evaluation) operate on this
file the same way they do for the existing v1.0 valid pool.

Usage:

    python scripts/scrape_crossdomain.py \\
        --output data/v1.0/raw_crossdomain_valid.jsonl \\
        --target-bio 150 --target-med 150 --target-csml 200

The script never modifies any v1.0 split. Verification keys come from
``/tmp/.s2_env`` (Semantic Scholar) when present.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from datetime import date, timedelta

from hallmark.dataset.api_clients import (
    BioRxivClient,
    CrossRefClient,
    DBLPClient,
    PubMedClient,
)
from hallmark.dataset.schema import VALID_SUBTESTS, BenchmarkEntry, GenerationMethod, save_entries
from hallmark.dataset.scraper import (
    DBLP_VENUES_CS_NON_ML,
    dblp_hit_to_entry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_first_word(s: str) -> str:
    parts = s.split()
    return parts[0].lower() if parts else "untitled"


def _bibtex_key_from(author_names: list[str], year: str, title: str) -> str:
    last = author_names[0].split()[-1] if author_names and author_names[0] else "unknown"
    return f"{last}{year}{_safe_first_word(title)}"


_DBLP_DISAMBIG_RE = __import__("re").compile(r"\s+\d{4}\b")


def _strip_dblp_disambiguators(author_str: str) -> str:
    """Remove DBLP author homonym suffixes ("Zhang 0001 and ..." -> "Zhang and ...")."""
    return _DBLP_DISAMBIG_RE.sub("", author_str)


# ---------------------------------------------------------------------------
# bioRxiv → BenchmarkEntry
# ---------------------------------------------------------------------------


def biorxiv_record_to_entry(rec: dict, server: str = "biorxiv") -> BenchmarkEntry | None:
    """Convert a bioRxiv API ``collection`` record to a BenchmarkEntry."""
    title = (rec.get("title") or "").strip().rstrip(".")
    if not title:
        return None
    raw_authors = rec.get("authors") or ""
    # bioRxiv returns "Last1, F.; Last2, F2.;" — split on ";"
    parts = [p.strip() for p in raw_authors.split(";") if p.strip()]
    # Reorder "Last, First" -> "First Last" where possible
    cleaned: list[str] = []
    for p in parts:
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
            cleaned.append(f"{first} {last}".strip())
        else:
            cleaned.append(p)
    author_str = " and ".join(cleaned) if cleaned else ""
    doi = (rec.get("doi") or "").strip()
    # The bioRxiv API ``date`` is the LATEST version's posting date, but the DOI
    # (10.1101/YYYY.MM.DD.nnnnnn) is the immutable v1 DOI whose canonical record
    # carries the v1 year. Taking ``year`` from the API date leaves the entry
    # internally inconsistent (year != DOI), which a DOI-resolving verifier
    # correctly reads as a metadata mismatch. Derive the year from the DOI when it
    # embeds a date so the citation matches what a resolver returns.
    doi_date = re.match(r"10\.1101/(\d{4})\.(\d{2})\.(\d{2})\.", doi)
    if doi_date:
        year = doi_date.group(1)
        pub_date = f"{doi_date.group(1)}-{doi_date.group(2)}-{doi_date.group(3)}"
    else:
        pub_date = (rec.get("date") or "").strip()
        year = pub_date[:4] if pub_date else ""

    if not (author_str and year and doi):
        return None

    bibtex_key = _bibtex_key_from(cleaned, year, title)
    fields: dict[str, str] = {
        "title": title,
        "author": author_str,
        "year": year,
        "doi": doi,
        "url": f"https://doi.org/{doi}",
        "journal": server,  # bioRxiv/medRxiv as preprint server
    }
    return BenchmarkEntry(
        bibtex_key=bibtex_key,
        bibtex_type="article",
        fields=fields,
        label="VALID",
        explanation=f"Valid {server} preprint scraped via API.",
        generation_method=GenerationMethod.SCRAPED.value,
        source_conference=server,
        source=server,
        publication_date=pub_date if len(pub_date) == 10 else (f"{year}-01-01" if year else ""),
        added_to_benchmark=date.today().isoformat(),
        subtests={**VALID_SUBTESTS, "doi_resolves": True},
    )


def scrape_biorxiv(
    target: int,
    days_back: int = 60,
    server: str = "biorxiv",
    years: list[int] | None = None,
) -> list[BenchmarkEntry]:
    """Pull bioRxiv preprints and convert to BenchmarkEntry.

    When ``years`` is given, fetch a slice from each listed year (spreading the
    sample across the window so it is not clustered in one month) instead of the
    trailing ``days_back`` window. Used to build the recency-matched
    cross-domain split, where the valid pool must sit in a pre-cutoff year range.
    """
    client = BioRxivClient(server=server, rate_limit=0.3, timeout=30.0)

    if years:
        # Over-fetch per year, then random-sample to a per-year quota so the
        # pool spans the whole year rather than only early January.
        rng = random.Random(8042)
        per_year = max(1, target // len(years))
        out: list[BenchmarkEntry] = []
        seen: set[str] = set()
        for yr in years:
            from_date, to_date = f"{yr}-01-01", f"{yr}-12-31"
            logger.info(
                "[%s] fetching ~%d papers from %s..%s", server, per_year, from_date, to_date
            )
            raw = client.list_papers(from_date, to_date, max_results=per_year * 8)
            cand = []
            for rec in raw:
                e = biorxiv_record_to_entry(rec, server=server)
                if e is None:
                    continue
                key = e.fields.get("doi", "") or e.bibtex_key
                if key in seen:
                    continue
                seen.add(key)
                cand.append(e)
            rng.shuffle(cand)
            out.extend(cand[:per_year])
        logger.info("[%s] kept %d entries across years %s", server, len(out), years)
        return out

    today = date.today()
    from_date = (today - timedelta(days=days_back)).isoformat()
    to_date = today.isoformat()
    logger.info("[%s] fetching ~%d papers from %s..%s", server, target, from_date, to_date)
    raw = client.list_papers(from_date, to_date, max_results=target * 2)  # over-fetch then filter
    out = []
    seen = set()
    for rec in raw:
        e = biorxiv_record_to_entry(rec, server=server)
        if e is None:
            continue
        key = e.fields.get("doi", "") or e.bibtex_key
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= target:
            break
    logger.info("[%s] kept %d entries (from %d raw)", server, len(out), len(raw))
    return out


# ---------------------------------------------------------------------------
# PubMed → BenchmarkEntry
# ---------------------------------------------------------------------------


def pubmed_summary_to_entry(rec: dict) -> BenchmarkEntry | None:
    """Convert an esummary record to a BenchmarkEntry."""
    title = (rec.get("title") or "").strip().rstrip(".")
    if not title:
        return None
    pub_date = (rec.get("pubdate") or "").strip()
    year = pub_date[:4] if pub_date and pub_date[:4].isdigit() else ""
    if not year:
        return None

    authors_field = rec.get("authors") or []
    author_names = [
        a.get("name", "") for a in authors_field if a.get("authtype", "Author") == "Author"
    ]
    author_names = [n for n in author_names if n]
    if not author_names:
        return None
    # PubMed format is "Last FM" — flip to "FM Last" for downstream BibTeX style
    cleaned: list[str] = []
    for n in author_names:
        parts = n.split(" ", 1)
        if len(parts) == 2:
            last, initials = parts
            cleaned.append(f"{initials} {last}")
        else:
            cleaned.append(n)
    author_str = " and ".join(cleaned)

    journal = (rec.get("fulljournalname") or rec.get("source") or "").strip()
    pmid = str(rec.get("uid") or rec.get("pmid") or "")

    # Pull DOI from articleids
    doi = ""
    for aid in rec.get("articleids", []) or []:
        if aid.get("idtype") == "doi":
            doi = (aid.get("value") or "").strip()
            break

    bibtex_key = _bibtex_key_from(cleaned, year, title)
    fields: dict[str, str] = {
        "title": title,
        "author": author_str,
        "year": year,
    }
    if journal:
        fields["journal"] = journal
    if doi:
        fields["doi"] = doi
        fields["url"] = f"https://doi.org/{doi}"
    if pmid:
        fields["pmid"] = pmid

    return BenchmarkEntry(
        bibtex_key=bibtex_key,
        bibtex_type="article",
        fields=fields,
        label="VALID",
        explanation="Valid peer-reviewed paper indexed in PubMed.",
        generation_method=GenerationMethod.SCRAPED.value,
        source_conference="pubmed",
        source="pubmed",
        publication_date=f"{year}-01-01",
        added_to_benchmark=date.today().isoformat(),
        subtests={**VALID_SUBTESTS, "doi_resolves": bool(doi)},
    )


def scrape_pubmed(target: int, years: list[int] | None = None) -> list[BenchmarkEntry]:
    """Pull PubMed papers across mixed clinical/biomed terms.

    When ``years`` is given, restrict the search window to that (inclusive) year
    range instead of the trailing 180 days -- used for the recency-matched
    cross-domain split.
    """
    api_key = os.environ.get("NCBI_API_KEY")
    client = PubMedClient(api_key=api_key, rate_limit=0.4 if api_key else 0.5)
    if years:
        mindate = f"{min(years)}/01/01"
        maxdate = f"{max(years)}/12/31"
    else:
        today = date.today()
        mindate = (today - timedelta(days=180)).strftime("%Y/%m/%d")
        maxdate = today.strftime("%Y/%m/%d")

    # Diverse query terms across med/clinical/biomed sub-domains.
    queries = [
        "randomized controlled trial",
        "clinical outcome",
        "epidemiology cohort",
        "genome wide association",
        "meta-analysis treatment",
        "biomarker diagnosis",
    ]
    per_query = max(20, target // len(queries) + 5)

    entries: list[BenchmarkEntry] = []
    seen_pmids: set[str] = set()
    for q in queries:
        if len(entries) >= target:
            break
        pmids = client.esearch(q, retmax=per_query, mindate=mindate, maxdate=maxdate)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        if not new_pmids:
            continue
        # esummary in batches of 50
        for i in range(0, len(new_pmids), 50):
            batch = new_pmids[i : i + 50]
            for rec in client.esummary(batch):
                e = pubmed_summary_to_entry(rec)
                if e is None:
                    continue
                pmid = e.fields.get("pmid", "") or e.bibtex_key
                if pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                entries.append(e)
                if len(entries) >= target:
                    break
            if len(entries) >= target:
                break
    logger.info("[pubmed] kept %d entries across %d queries", len(entries), len(queries))
    return entries


# ---------------------------------------------------------------------------
# CS-non-ML via DBLP
# ---------------------------------------------------------------------------


def scrape_cs_non_ml(target: int, years: list[int]) -> list[BenchmarkEntry]:
    """Pull DBLP entries from CS-non-ML venues across recent years."""
    client = DBLPClient(rate_limit=0.5)
    venues = list(DBLP_VENUES_CS_NON_ML.items())
    per_venue_year = max(5, target // (len(venues) * len(years)) + 2)

    entries: list[BenchmarkEntry] = []
    seen_titles: set[str] = set()
    for venue_name, venue_key in venues:
        if len(entries) >= target:
            break
        for year in years:
            if len(entries) >= target:
                break
            try:
                hits = client.search_venue_year(venue_key, year, max_results=per_venue_year * 3)
            except Exception as e:
                logger.warning("DBLP %s/%d failed: %s", venue_name, year, e)
                continue
            kept = 0
            for hit in hits:
                if kept >= per_venue_year:
                    break
                e = dblp_hit_to_entry(hit, venue_name=venue_name)
                if e is None:
                    continue
                t = e.fields.get("title", "").lower()
                if not t or t in seen_titles:
                    continue
                seen_titles.add(t)
                # DBLP author records can contain "0001"-style disambiguator suffixes;
                # strip them to keep the entry looking like a normal BibTeX record.
                if a := e.fields.get("author"):
                    e.fields["author"] = _strip_dblp_disambiguators(a)
                # Recompute bibtex_key off the cleaned author surname.
                first = e.fields.get("author", "").split(" and ", 1)[0]
                if first:
                    last = first.split()[-1] if first.split() else "unknown"
                    e.bibtex_key = (
                        f"{last}{e.fields.get('year', '')}"
                        f"{_safe_first_word(e.fields.get('title', ''))}"
                    )
                # Tag the source so downstream filters can find it
                e.source = "dblp_cs_non_ml"
                entries.append(e)
                kept += 1
                if len(entries) >= target:
                    break
            time.sleep(0.5)
    logger.info(
        "[cs_non_ml] kept %d entries across %d venues x %d years",
        len(entries),
        len(venues),
        len(years),
    )
    return entries


# ---------------------------------------------------------------------------
# Verification (optional pass)
# ---------------------------------------------------------------------------


def verify_doi_resolution(entries: list[BenchmarkEntry], rate: float = 0.3) -> list[BenchmarkEntry]:
    """Update ``doi_resolves`` subtest by hitting CrossRef. Drops nothing."""
    cr = CrossRefClient(rate_limit=rate)
    for i, e in enumerate(entries):
        doi = e.fields.get("doi")
        if not doi:
            e.subtests["doi_resolves"] = False
            continue
        try:
            ok = cr.verify_doi(doi)
        except Exception:
            ok = False
        e.subtests["doi_resolves"] = ok
        if (i + 1) % 25 == 0:
            logger.info("verified %d/%d", i + 1, len(entries))
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="Output JSONL path for valid entries.")
    p.add_argument("--target-bio", type=int, default=150)
    p.add_argument("--target-med", type=int, default=150)
    p.add_argument("--target-csml", type=int, default=200)
    p.add_argument(
        "--bio-days-back",
        type=int,
        default=60,
        help="Days back from today to pull bioRxiv preprints.",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Years to scrape for DBLP CS-non-ML venues.",
    )
    p.add_argument(
        "--biomed-years",
        nargs="+",
        type=int,
        default=None,
        help="If set, restrict bioRxiv/medRxiv/PubMed to this (inclusive) year "
        "range instead of the trailing window. Use for the recency-matched "
        "cross-domain split (e.g. --biomed-years 2021 2022 2023).",
    )
    p.add_argument(
        "--include-medrxiv",
        action="store_true",
        help="Also scrape medRxiv (in addition to bioRxiv).",
    )
    p.add_argument(
        "--verify-dois",
        action="store_true",
        help="Verify each DOI against CrossRef (slow, but accurate).",
    )
    p.add_argument("--smoke", action="store_true", help="Smoke mode: target ~50 entries total.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.smoke:
        args.target_bio = 20
        args.target_med = 20
        args.target_csml = 20
        args.bio_days_back = 30

    all_entries: list[BenchmarkEntry] = []

    # Bio/life sciences
    bio_target = args.target_bio
    if args.include_medrxiv:
        bio_target = max(bio_target // 2, 1)
    bio_entries = scrape_biorxiv(
        bio_target, days_back=args.bio_days_back, server="biorxiv", years=args.biomed_years
    )
    all_entries.extend(bio_entries)
    if args.include_medrxiv:
        med_pre = scrape_biorxiv(
            args.target_bio - len(bio_entries),
            days_back=args.bio_days_back,
            server="medrxiv",
            years=args.biomed_years,
        )
        all_entries.extend(med_pre)

    # PubMed
    pm_entries = scrape_pubmed(args.target_med, years=args.biomed_years)
    all_entries.extend(pm_entries)

    # CS-non-ML
    cs_entries = scrape_cs_non_ml(args.target_csml, years=args.years)
    all_entries.extend(cs_entries)

    if not all_entries:
        logger.error("No entries scraped — aborting.")
        return 1

    # Hard year filter: API date filters (esp. PubMed pdat vs the pubdate we
    # extract the year from) are imperfect and leak out-of-window entries. For
    # the recency-matched split the year window is load-bearing (any post-cutoff
    # leak reintroduces the very confound the split removes), so enforce it here.
    if args.biomed_years or args.years:
        allowed = set(args.years or []) | set(args.biomed_years or [])
        before = len(all_entries)
        kept = []
        for e in all_entries:
            y = str(e.fields.get("year", ""))[:4]
            if y.isdigit() and int(y) in allowed:
                kept.append(e)
            else:
                logger.debug("dropping out-of-window entry: %s year=%s", e.bibtex_key, y)
        all_entries = kept
        logger.info(
            "year filter (allowed=%s): kept %d/%d entries",
            sorted(allowed),
            len(all_entries),
            before,
        )

    # Optional CrossRef verification
    if args.verify_dois:
        all_entries = verify_doi_resolution(all_entries)

    # De-dup by bibtex_key
    by_key: dict[str, BenchmarkEntry] = {}
    for e in all_entries:
        by_key[e.bibtex_key] = e
    final = list(by_key.values())

    save_entries(final, args.output)
    logger.info("Wrote %d entries to %s", len(final), args.output)

    # Per-source summary
    from collections import Counter

    by_src = Counter(e.source for e in final)
    for src, n in by_src.most_common():
        logger.info("  %s: %d", src, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
