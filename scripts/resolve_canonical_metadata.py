"""Rebuild a valid-entry pool with canonical CrossRef metadata.

The rapid cross-domain scrape produced valid biomedical entries whose author/year
metadata is unreliable: non-Western names get mangled (Portuguese suffixes,
hyphenated surnames, CJK given/family order), and preprint/epub years disagree
with the authoritative record. A strict metadata verifier (bibtex-updater) then
flags these as hallucinations, inflating FPR with what is really scrape noise
rather than a domain-transfer signal.

This step resolves each entry's DOI against CrossRef and REPLACES its fields with
the canonical record (full author names in canonical order, the issued year, the
container-title venue, the canonical title). Entries whose DOI does not resolve
are dropped (cannot confirm a real paper). Crucially, the pool is re-filtered on
the *canonical* year, which removes post-cutoff leaks the scraped year hid.

Output is a clean VALID pool; feed it to build_crossdomain_split.py exactly as
before to regenerate the split.

Usage:
    python scripts/resolve_canonical_metadata.py \
        --input  data/v1.1_crossdomain_matched/raw_crossdomain_matched_valid.jsonl \
        --output data/v1.1_crossdomain_matched/raw_crossdomain_matched_canonical_valid.jsonl \
        --years 2021 2022 2023
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from hallmark.dataset.api_clients import CrossRefClient
from hallmark.dataset.schema import VALID_SUBTESTS, BenchmarkEntry, GenerationMethod, save_entries

logger = logging.getLogger("resolve_canonical")


def canonical_year(msg: dict) -> str | None:
    for key in ("issued", "published-print", "published-online", "published", "posted", "created"):
        dp = (msg.get(key) or {}).get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            return str(dp[0][0])
    return None


def canonical_authors(msg: dict) -> str:
    names = []
    for a in msg.get("author", []) or []:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        full = " ".join(x for x in (given, family) if x) or (a.get("name") or "").strip()
        if full:
            names.append(full)
    return " and ".join(names)


def canonical_venue(msg: dict, fallback: str) -> str:
    ct = msg.get("container-title") or []
    if ct and ct[0].strip():
        return ct[0].strip()
    inst = msg.get("institution") or []
    if inst and isinstance(inst, list) and inst[0].get("name"):
        return inst[0]["name"]
    return fallback  # e.g. bioRxiv/medRxiv preprints have no container-title


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023])
    ap.add_argument("--rate", type=float, default=0.5, help="seconds between CrossRef calls")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    allowed = set(args.years)
    src_entries = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d scraped valid entries", len(src_entries))

    client = CrossRefClient()
    kept: list[BenchmarkEntry] = []
    dropped_unresolved = dropped_year = dropped_incomplete = 0

    for i, e in enumerate(src_entries):
        doi = (e.get("fields", {}).get("doi") or "").strip()
        if not doi:
            dropped_unresolved += 1
            continue
        time.sleep(args.rate)
        msg = client.query_by_doi(doi)
        if not msg:
            dropped_unresolved += 1
            continue
        year = canonical_year(msg)
        authors = canonical_authors(msg)
        title = (msg.get("title") or [""])[0].strip()
        if not (year and authors and title):
            dropped_incomplete += 1
            continue
        if not (year.isdigit() and int(year) in allowed):
            dropped_year += 1  # authoritative year is out of the pre-cutoff window
            continue
        src = e.get("source") or "crossref"
        venue = canonical_venue(msg, e.get("fields", {}).get("journal") or src)
        # deterministic canonical key
        first_family = ""
        if msg.get("author"):
            first_family = (msg["author"][0].get("family") or "").split()[-1:] or [""]
            first_family = first_family[0].lower()
        title_word = "".join(c for c in title.split()[0].lower() if c.isalnum()) if title else "x"
        kept.append(
            BenchmarkEntry(
                bibtex_key=f"{first_family}{year}{title_word}",
                bibtex_type="article",
                fields={
                    "title": title,
                    "author": authors,
                    "year": year,
                    "doi": doi,
                    "url": f"https://doi.org/{doi}",
                    "journal": venue,
                },
                label="VALID",
                hallucination_type=None,
                difficulty_tier=None,
                explanation="Valid entry with canonical CrossRef metadata.",
                generation_method=GenerationMethod.SCRAPED.value,
                source_conference=e.get("source_conference") or src,
                source=src,
                publication_date=f"{year}-01-01",
                subtests={**VALID_SUBTESTS, "doi_resolves": True},
            )
        )
        if (i + 1) % 25 == 0:
            logger.info("  resolved %d/%d (kept %d)", i + 1, len(src_entries), len(kept))

    save_entries(kept, args.output)
    logger.info(
        "Wrote %d canonical valid entries -> %s\n  dropped: %d unresolved, %d out-of-window-year, %d incomplete",
        len(kept),
        args.output,
        dropped_unresolved,
        dropped_year,
        dropped_incomplete,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
