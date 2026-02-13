#!/usr/bin/env python3
"""Cross-verify existing valid entries against CrossRef API.

This addresses P1.7: Verify that valid entries are truly valid by checking against CrossRef.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# SSL context with certifi fallback for macOS
try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def normalize_title(title: str) -> str:
    """Normalize title for comparison.

    Args:
        title: Raw title string

    Returns:
        Normalized title (lowercase, stripped punctuation)
    """
    # Remove common punctuation and extra whitespace
    normalized = title.lower()
    for char in ".,;:!?\"'()[]{}":
        normalized = normalized.replace(char, " ")
    # Collapse whitespace
    return " ".join(normalized.split())


def title_similarity(title1: str, title2: str) -> float:
    """Compute similarity between two titles.

    Args:
        title1: First title
        title2: Second title

    Returns:
        Similarity score in [0, 1]
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()


def extract_last_names(author_str: str) -> set[str]:
    """Extract last names from BibTeX author field.

    Args:
        author_str: BibTeX author string (e.g., "Smith, John and Doe, Jane")

    Returns:
        Set of last names (lowercased)
    """
    last_names = set()

    # Split by "and"
    authors = author_str.split(" and ")

    for author in authors:
        author = author.strip()
        if not author:
            continue

        # Handle "Last, First" format
        if "," in author:
            last_name = author.split(",")[0].strip()
        else:
            # Handle "First Last" format (take last token)
            tokens = author.split()
            last_name = tokens[-1] if tokens else ""

        if last_name:
            last_names.add(last_name.lower())

    return last_names


def query_crossref_by_doi(doi: str) -> dict[str, Any] | None:
    """Query CrossRef API by DOI.

    Args:
        doi: DOI string

    Returns:
        CrossRef metadata dict or None on failure
    """
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "HALLMARK-Benchmark/1.0 (mailto:research@example.com)")

        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as response:
            data = json.loads(response.read().decode())
            message = data.get("message")
            return message if isinstance(message, dict) else None

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"    CrossRef HTTP error {e.code}: {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    CrossRef error: {e}", file=sys.stderr)
        return None


def query_crossref_by_title(title: str) -> dict[str, Any] | None:
    """Query CrossRef API by title search.

    Args:
        title: Paper title

    Returns:
        Top search result metadata or None on failure
    """
    params = {
        "query.bibliographic": title,
        "rows": "1",
    }
    url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "HALLMARK-Benchmark/1.0 (mailto:research@example.com)")

        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as response:
            data = json.loads(response.read().decode())
            items = data.get("message", {}).get("items", [])
            return items[0] if items else None

    except Exception as e:
        print(f"    CrossRef search error: {e}", file=sys.stderr)
        return None


def extract_bibtex_field(bibtex: str, field: str) -> str:
    """Extract field value from BibTeX string.

    Args:
        bibtex: BibTeX entry string
        field: Field name (e.g., "title", "author", "doi")

    Returns:
        Field value or empty string if not found
    """
    # Simple regex-free parsing: look for "field = {value},"
    lines = bibtex.split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith(f"{field.lower()} = "):
            # Extract value between { and }
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and end > start:
                return line[start + 1 : end]
    return ""


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
        title = extract_bibtex_field(bibtex, "title")
        author = extract_bibtex_field(bibtex, "author")
        doi = extract_bibtex_field(bibtex, "doi")

    result = {
        "bibtex_key": bibtex_key,
        "has_doi": bool(doi),
        "verified": False,
        "verification_method": None,
        "title_similarity": 0.0,
        "authors_match": False,
        "details": "",
    }

    # Rate limiting
    time.sleep(rate_limit)

    if doi:
        # Verify by DOI
        crossref_data = query_crossref_by_doi(doi)
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
        crossref_data = query_crossref_by_title(title)
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
