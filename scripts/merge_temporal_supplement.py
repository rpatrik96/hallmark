#!/usr/bin/env python3
"""Merge the new 2024-2025 ML scrape into the existing temporal_supplement.

The existing ``results/temporal_supplement/temporal_supplement_2024_2025.jsonl``
(April 22, 448 entries) has pre-computed baseline evaluations across 10+ LLMs,
keyed by ``bibtex_key``. We must NOT change keys for entries already present —
that would invalidate every eval JSON in ``results/temporal_supplement/``.

The May 2026 re-scrape (``data/v1.0/test_temporal_2024_2025.jsonl``, 450
entries) covers the same intent but has more even hallucination type coverage,
particularly on swapped_authors and merged_citation which the original is
sparse on.

Merge strategy:
  1. Load both files.
  2. Take all existing entries verbatim (keys preserved).
  3. From new entries, take only those whose DOI is not already in the
     existing file's DOI set.
  4. Re-key those new entries using the corrected DBLP scheme
     (real surname + year + first-word + 6-char SHA-1 hash) to avoid the
     "0001"-disambiguator bug that contaminated the original keys.
  5. Write the union back to the same file.

The new entries' baseline evals will need to be computed separately. Existing
evals on the existing-DOI subset remain valid because their keys are
preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger("merge_temporal_supplement")


_DISAMBIG_RE = re.compile(r"\s+\d{4}\b")


def _clean_author(name: str) -> str:
    return _DISAMBIG_RE.sub("", name).strip()


def _safe_first_word(s: str) -> str:
    return s.split()[0].lower() if s.split() else "untitled"


def _new_key(entry: dict) -> str:
    """Generate a corrected bibtex_key for a new entry."""
    fields = entry.get("fields", {}) or {}
    title = fields.get("title", "")
    year = fields.get("year", "")
    author = _clean_author((fields.get("author") or "").split(" and ")[0])
    last = author.split()[-1] if author.split() else "unknown"
    digest_input = (
        f"{title}|{fields.get('author', '')}|{year}|{fields.get('doi') or fields.get('url', '')}"
    )
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:6]
    return f"{last}{year}{_safe_first_word(title)}_{digest}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--existing",
        type=Path,
        default=Path("results/temporal_supplement/temporal_supplement_2024_2025.jsonl"),
    )
    p.add_argument(
        "--new",
        type=Path,
        default=Path("data/v1.0/test_temporal_2024_2025.jsonl"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/temporal_supplement/temporal_supplement_2024_2025.jsonl"),
        help="Where to write the merged result (defaults to overwriting --existing).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    existing = [json.loads(line) for line in args.existing.read_text().splitlines() if line.strip()]
    new = [json.loads(line) for line in args.new.read_text().splitlines() if line.strip()]
    logger.info("loaded existing=%d, new=%d", len(existing), len(new))

    existing_dois = {
        (e.get("fields") or {}).get("doi") for e in existing if (e.get("fields") or {}).get("doi")
    }
    existing_keys = {e["bibtex_key"] for e in existing}

    appended: list[dict] = []
    skipped_doi_overlap = 0
    rekey_collisions = 0

    for n in new:
        label = n.get("label")
        doi = (n.get("fields") or {}).get("doi")

        # Dedup VALID entries by DOI against the existing pool. HALLUCINATED
        # entries are kept regardless of DOI presence — most have intentionally
        # damaged DOIs (fabricated_doi, hybrid_fabrication, etc.) and they are
        # *content-unique* perturbations of distinct source entries.
        if label == "VALID" and doi and doi in existing_dois:
            skipped_doi_overlap += 1
            continue

        # Re-key with the corrected scheme
        candidate = _new_key(n)
        # Defensive: ensure no collision with existing or already-appended entries
        if candidate in existing_keys or candidate in {a["bibtex_key"] for a in appended}:
            rekey_collisions += 1
            # Mix DOI/title/source key in so it's globally unique
            mix = doi or n.get("source") or json.dumps(n.get("fields", {}), sort_keys=True)
            digest = hashlib.sha1((candidate + str(mix)).encode()).hexdigest()[:8]
            candidate = f"{candidate}_{digest}"
        n["bibtex_key"] = candidate
        appended.append(n)

    logger.info(
        "appending %d new entries (skipped %d valid for DOI overlap; %d rekey collisions)",
        len(appended),
        skipped_doi_overlap,
        rekey_collisions,
    )

    merged = existing + appended

    # Final dedup safety check
    keys = [e["bibtex_key"] for e in merged]
    if len(keys) != len(set(keys)):
        from collections import Counter

        dups = [k for k, n in Counter(keys).items() if n > 1]
        logger.error("duplicate keys after merge: %s", dups[:10])
        return 1

    if args.dry_run:
        logger.info("DRY RUN: would write %d entries to %s", len(merged), args.output)
    else:
        with args.output.open("w") as f:
            for e in merged:
                f.write(json.dumps(e) + "\n")
        logger.info("wrote %d merged entries to %s", len(merged), args.output)

    # Summary by label / type
    from collections import Counter

    labels = Counter(e["label"] for e in merged)
    types = Counter(e.get("hallucination_type") or "valid" for e in merged)
    logger.info("merged labels: %s", dict(labels))
    logger.info("merged types:")
    for k, v in types.most_common():
        logger.info("  %s: %d", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
