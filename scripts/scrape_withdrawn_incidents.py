#!/usr/bin/env python3
"""Collect withdrawn bioRxiv/medRxiv preprints as real-world hallucination incidents.

A withdrawn preprint has a real DOI that resolves and a real title that
exists, but it has been formally retracted by the authors or the platform
and citing it in serious scientific work is a real-world failure mode.
This is the kind of "hallucination" that no perturbation generator
produces — the citation is genuinely valid metadata, but the underlying
paper has been pulled from circulation.

Output entries are tagged:
    label = "HALLUCINATED"
    hallucination_type = "plausible_fabrication"  (closest taxonomy fit)
    generation_method = "real_world"
    source = "biorxiv_withdrawn" | "medrxiv_withdrawn"
    explanation = "Withdrawn from {server}: {first sentence of statement}"

The output file ``data/v1.0/biorxiv_withdrawn_incidents.jsonl`` is meant
to be a reusable data source — analogous to ``real_world_incidents.jsonl``
and ``gptzero_neurips2025.jsonl`` — that downstream split builders can
sample from.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

from hallmark.dataset.api_clients import BioRxivClient
from hallmark.dataset.schema import (
    BenchmarkEntry,
    GenerationMethod,
    HallucinationType,
    save_entries,
)

logger = logging.getLogger(__name__)


_FIRST_SENTENCE_RE = re.compile(r"^([^\n.]{20,200}?[.!])", re.MULTILINE)


def _withdrawal_reason(abstract: str) -> str:
    """Extract a short human-readable withdrawal reason from the abstract."""
    head = (abstract or "")[:600].strip()
    # Most withdrawn papers start with "Withdrawal Statement" — strip that prefix
    head = re.sub(r"^Withdrawal Statement\s*[:\-]?\s*", "", head, flags=re.IGNORECASE)
    head = re.sub(r"^\[?Withdrawn\]?\s*[:\-]?\s*", "", head, flags=re.IGNORECASE)
    m = _FIRST_SENTENCE_RE.search(head)
    if m:
        return m.group(1).strip()
    return head[:200].strip()


def _record_to_entry(rec: dict, server: str) -> BenchmarkEntry | None:
    title = (rec.get("title") or "").strip().rstrip(".")
    if not title:
        return None
    raw_authors = rec.get("authors") or ""
    parts = [p.strip() for p in raw_authors.split(";") if p.strip()]
    cleaned: list[str] = []
    for p in parts:
        if "," in p:
            last, first = (x.strip() for x in p.split(",", 1))
            cleaned.append(f"{first} {last}".strip())
        else:
            cleaned.append(p)
    if not cleaned:
        return None

    pub_date = (rec.get("date") or "").strip()
    year = pub_date[:4] if pub_date else ""
    doi = (rec.get("doi") or "").strip()
    if not (year and doi):
        return None

    last = cleaned[0].split()[-1] if cleaned[0].split() else "withdrawn"
    first_word = title.split()[0].lower() if title else "untitled"
    import hashlib

    digest = hashlib.sha1(f"{title}|{doi}".encode()).hexdigest()[:6]
    bibtex_key = f"{last}{year}{first_word}_{digest}"

    fields: dict[str, str] = {
        "title": title,
        "author": " and ".join(cleaned),
        "year": year,
        "doi": doi,
        "url": f"https://doi.org/{doi}",
        "journal": server,
    }

    reason = _withdrawal_reason(rec.get("abstract") or "")
    explanation = f"Withdrawn from {server}: {reason}" if reason else f"Withdrawn from {server}."

    return BenchmarkEntry(
        bibtex_key=bibtex_key,
        bibtex_type="article",
        fields=fields,
        label="HALLUCINATED",
        hallucination_type=HallucinationType.PLAUSIBLE_FABRICATION.value,
        difficulty_tier=3,
        explanation=explanation,
        generation_method=GenerationMethod.REAL_WORLD.value,
        source_conference=server,
        source=f"{server}_withdrawn",
        publication_date=pub_date if len(pub_date) == 10 else f"{year}-01-01",
        added_to_benchmark=date.today().isoformat(),
        subtests={
            # The DOI does resolve and the title does exist — that's the trap.
            # The "hallucination signal" is venue/cross-DB disagreement.
            "doi_resolves": True,
            "title_exists": True,
            "authors_match": True,
            "venue_correct": True,
            "fields_complete": True,
            "cross_db_agreement": False,
        },
    )


def scrape_server(
    server: str,
    years: list[int],
    workers: int = 8,
    checkpoint_path: Path | None = None,
    seen_dois: set[str] | None = None,
) -> list[BenchmarkEntry]:
    """Pull withdrawn papers from a single server across years using parallel pagination.

    Logs an INFO line every ~5% of pages with running counts of pages
    processed and withdrawn hits found. Optional ``checkpoint_path`` writes
    every successfully-converted entry to a JSONL file as it arrives, so a
    long-running scan that gets killed still produces partial results.

    ``seen_dois`` (when provided) is the set of DOIs we already have on disk
    from a prior run — entries with those DOIs are skipped without being
    appended to the checkpoint a second time.
    """
    client = BioRxivClient(server=server, rate_limit=0.0, timeout=30.0)
    out: list[BenchmarkEntry] = []
    if seen_dois is None:
        seen_dois = set()

    for y in years:
        running_hits = 0
        last_logged_pct = -5

        def _on_batch(
            hits: list[dict],
            completed: int,
            total_pages: int,
            year: int = y,
        ) -> None:
            nonlocal running_hits, last_logged_pct
            running_hits += len(hits)
            pct = int(100 * completed / max(total_pages, 1))
            if pct >= last_logged_pct + 5 or completed == total_pages:
                logger.info(
                    "[%s %d] %d/%d pages (%d%%) — %d withdrawn so far",
                    server,
                    year,
                    completed,
                    total_pages,
                    pct,
                    running_hits,
                )
                last_logged_pct = pct

        logger.info("[%s %d] scanning with %d workers...", server, y, workers)
        recs = client.find_withdrawn_papers_parallel(
            from_date=f"{y}-01-01",
            to_date=f"{y}-12-31",
            workers=workers,
            on_batch=_on_batch,
        )
        logger.info("[%s %d] DONE — %d withdrawn records", server, y, len(recs))

        for rec in recs:
            e = _record_to_entry(rec, server)
            if e is None:
                continue
            doi = e.fields.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            out.append(e)
            if checkpoint_path is not None:
                with checkpoint_path.open("a") as fh:
                    fh.write(json.dumps(e.to_dict()) + "\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Path("data/v1.0/biorxiv_withdrawn_incidents.jsonl")
    )
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument(
        "--include-medrxiv",
        action="store_true",
        default=True,
        help="Also scan medRxiv (default: True; medRxiv has ~2x the withdrawn rate).",
    )
    p.add_argument("--no-medrxiv", action="store_false", dest="include_medrxiv")
    p.add_argument(
        "--workers", type=int, default=8, help="Parallel HTTP workers per (server, year) scan."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Append-only JSONL written incrementally as entries arrive. "
        "Defaults to <output>.partial.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --checkpoint exists, load its DOIs as the 'seen' set "
        "and skip them. New entries are appended.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    # Force unbuffered stdout/stderr so progress lines show up under nohup/pipes
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    checkpoint = args.checkpoint or args.output.with_suffix(args.output.suffix + ".partial")

    seen_dois: set[str] = set()
    resumed_entries: list[BenchmarkEntry] = []
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            doi = (d.get("fields") or {}).get("doi")
            if doi:
                seen_dois.add(doi)
            resumed_entries.append(BenchmarkEntry.from_dict(d))
        logger.info("Resuming with %d existing entries from %s", len(resumed_entries), checkpoint)
    elif checkpoint.exists():
        checkpoint.unlink()

    all_entries: list[BenchmarkEntry] = list(resumed_entries)
    all_entries.extend(
        scrape_server(
            "biorxiv",
            args.years,
            workers=args.workers,
            checkpoint_path=checkpoint,
            seen_dois=seen_dois,
        )
    )
    if args.include_medrxiv:
        all_entries.extend(
            scrape_server(
                "medrxiv",
                args.years,
                workers=args.workers,
                checkpoint_path=checkpoint,
                seen_dois=seen_dois,
            )
        )

    if not all_entries:
        logger.error("No withdrawn entries collected.")
        return 1

    # De-dup by DOI
    by_doi: dict[str, BenchmarkEntry] = {}
    for e in all_entries:
        doi = e.fields.get("doi", "")
        if doi and doi not in by_doi:
            by_doi[doi] = e
    final = list(by_doi.values())

    save_entries(final, args.output)
    logger.info("Wrote %d unique withdrawn-paper entries to %s", len(final), args.output)

    from collections import Counter

    by_src = Counter(e.source for e in final)
    by_year = Counter(e.fields.get("year") for e in final)
    for src, n in by_src.most_common():
        logger.info("  %s: %d", src, n)
    for y, n in sorted(by_year.items()):
        logger.info("  year %s: %d", y, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
