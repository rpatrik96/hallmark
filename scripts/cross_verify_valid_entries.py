#!/usr/bin/env python3
"""Cross-verify existing valid entries against CrossRef API.  [analysis]

This addresses P1.7: Verify that valid entries are truly valid by checking against CrossRef.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from hallmark.dataset.api_clients import CrossRefClient
from hallmark.dataset.text_utils import extract_last_names, parse_bibtex_fields, title_similarity


def verify_entry(entry: dict[str, Any], rate_limit: float) -> dict[str, Any]:
    """Verify a single valid entry against CrossRef.

    Args:
        entry: BenchmarkEntry dict
        rate_limit: Seconds to sleep between API calls

    Returns:
        Verification result dict
    """
    bibtex_key = entry.get("bibtex_key", "")
    fields = entry.get("fields", {})

    # Extract fields directly from entry (or fall back to BibTeX parsing)
    title = fields.get("title", "")
    author = fields.get("author", "")
    doi = fields.get("doi", "")

    # Fall back to BibTeX string parsing if fields are empty
    if not title:
        bibtex = entry.get("bibtex", entry.get("raw_bibtex", ""))
        parsed = parse_bibtex_fields(bibtex)
        title = parsed.get("title", "")
        author = parsed.get("author", "")
        doi = parsed.get("doi", "")

    result = {
        "bibtex_key": bibtex_key,
        "has_doi": bool(doi),
        "verified": False,
        "verification_method": None,
        "title_similarity": 0.0,
        "authors_match": False,
        "details": "",
    }

    client = CrossRefClient(rate_limit=rate_limit)

    if doi:
        # Verify by DOI
        crossref_data = client.query_by_doi(doi)
        if crossref_data is None:
            result["details"] = f"DOI {doi} not found in CrossRef"
            return result

        # Check title similarity
        crossref_title = crossref_data.get("title", [""])[0]
        sim = title_similarity(title, crossref_title)
        result["title_similarity"] = sim

        # Check author match
        our_last_names = extract_last_names(author)
        crossref_authors = crossref_data.get("author", [])
        crossref_last_names = {
            a.get("family", "").lower() for a in crossref_authors if "family" in a
        }

        authors_match = bool(our_last_names & crossref_last_names)
        result["authors_match"] = authors_match

        # Verify if both pass thresholds
        if sim >= 0.85 and authors_match:
            result["verified"] = True
            result["verification_method"] = "doi"
            result["details"] = f"Verified via DOI (title_sim={sim:.2f})"
        else:
            reasons = []
            if sim < 0.85:
                reasons.append(f"title_sim={sim:.2f}<0.85")
            if not authors_match:
                reasons.append("no_author_match")
            result["details"] = f"DOI found but failed: {', '.join(reasons)}"

    else:
        # Verify by title search
        items = client.query_by_title(title, rows=1)
        crossref_data = items[0] if items else None
        if crossref_data is None:
            result["details"] = "No CrossRef search results"
            return result

        # Check title similarity with top result
        crossref_title = crossref_data.get("title", [""])[0]
        sim = title_similarity(title, crossref_title)
        result["title_similarity"] = sim

        # Check author match
        our_last_names = extract_last_names(author)
        crossref_authors = crossref_data.get("author", [])
        crossref_last_names = {
            a.get("family", "").lower() for a in crossref_authors if "family" in a
        }

        authors_match = bool(our_last_names & crossref_last_names)
        result["authors_match"] = authors_match

        # Verify if both pass thresholds
        if sim >= 0.85 and authors_match:
            result["verified"] = True
            result["verification_method"] = "title_search"
            result["details"] = f"Verified via title search (title_sim={sim:.2f})"
        else:
            reasons = []
            if sim < 0.85:
                reasons.append(f"title_sim={sim:.2f}<0.85")
            if not authors_match:
                reasons.append("no_author_match")
            result["details"] = f"Title search found but failed: {', '.join(reasons)}"

    return result


def main() -> None:
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-verify valid HALLMARK entries against CrossRef API"
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("data/v1.0/dev_public.jsonl"),
        help="Dev set path (default: data/v1.0/dev_public.jsonl)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/v1.0/test_public.jsonl"),
        help="Test set path (default: data/v1.0/test_public.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/v1.0/valid_entry_verification.json"),
        help="Output JSON file path (default: data/v1.0/valid_entry_verification.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Verify only N random entries (for testing)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between API requests (default: 1.0)",
    )

    args = parser.parse_args()

    # Load valid entries
    valid_entries = []

    for path in [args.dev, args.test]:
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue

        with open(path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("label") == "VALID":
                    valid_entries.append(entry)

    print(f"Loaded {len(valid_entries)} valid entries", file=sys.stderr)

    # Sample if requested
    if args.sample is not None and args.sample < len(valid_entries):
        valid_entries = random.sample(valid_entries, args.sample)
        print(f"Sampled {len(valid_entries)} entries for verification", file=sys.stderr)

    # Verify entries
    results = []
    for i, entry in enumerate(valid_entries, 1):
        bibtex_key = entry.get("bibtex_key", f"entry_{i}")
        print(f"[{i}/{len(valid_entries)}] Verifying {bibtex_key}...", file=sys.stderr)

        result = verify_entry(entry, args.rate_limit)
        results.append(result)

        # Print immediate feedback
        if result["verified"]:
            print(f"  ✓ {result['details']}", file=sys.stderr)
        else:
            print(f"  ✗ {result['details']}", file=sys.stderr)

    # Compute summary statistics
    verified_count = sum(1 for r in results if r["verified"])
    failed_count = sum(1 for r in results if not r["verified"] and r["has_doi"])
    no_doi_count = sum(1 for r in results if not r["has_doi"])

    summary = {
        "total_entries": len(results),
        "verified_count": verified_count,
        "failed_count": failed_count,
        "no_doi_count": no_doi_count,
        "verification_rate": verified_count / len(results) if results else 0.0,
    }

    # Save results
    output_data = {
        "summary": summary,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Verification complete. Results saved to {args.output}", file=sys.stderr)
    print("\nSummary:", file=sys.stderr)
    print(f"  Total entries verified: {len(results)}", file=sys.stderr)
    vr_pct = 100 * summary["verification_rate"]
    print(f"  Successfully verified: {verified_count} ({vr_pct:.1f}%)", file=sys.stderr)
    print(f"  Failed verification: {failed_count}", file=sys.stderr)
    print(f"  No DOI (used title search): {no_doi_count}", file=sys.stderr)

    # Print sample failures for debugging
    failures = [r for r in results if not r["verified"]]
    if failures:
        print("\nSample failures (showing up to 5):", file=sys.stderr)
        for failure in failures[:5]:
            print(f"  - {failure['bibtex_key']}: {failure['details']}", file=sys.stderr)


if __name__ == "__main__":
    main()
