#!/usr/bin/env python3
"""Re-classify LLM-generated entries with proper verification thresholds.

Fixes three bugs in the original classification pipeline:
1. No title similarity threshold (CrossRef always returns results)
2. Exact string author comparison (format-sensitive)
3. No DOI-first verification

Re-queries CrossRef for each entry and applies:
- rapidfuzz token_sort_ratio >= 0.85 for title matching
- Jaccard similarity on last names >= 0.5 for author matching
- DOI-first verification when available
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent.parent))

from hallmark.dataset.schema import (
    HALLUCINATION_TIER_MAP,
    BenchmarkEntry,
    HallucinationType,
)

CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_HEADERS = {"User-Agent": "HALLMARK-Generator/1.0 (mailto:hallmark@example.com)"}
RATE_LIMIT_DELAY = 1.0

TITLE_SIMILARITY_THRESHOLD = 85.0  # rapidfuzz token_sort_ratio
AUTHOR_JACCARD_THRESHOLD = 0.5  # Jaccard on last names


def extract_last_names(author_string: str) -> set[str]:
    """Extract last names from author string (handles multiple formats)."""
    last_names: set[str] = set()
    authors = author_string.split(" and ")
    for author in authors:
        author = author.strip()
        if not author:
            continue
        # "et al." is not a real author
        if author.lower() in ("et al.", "et al", "others"):
            continue
        # Handle "Last, First" format
        if "," in author:
            last_name = author.split(",")[0].strip()
            last_names.add(last_name.lower())
        else:
            # Handle "First Last" format
            parts = author.split()
            if parts:
                last_names.add(parts[-1].lower())
    return last_names


def author_jaccard(authors_a: str, authors_b: str) -> float:
    """Compute Jaccard similarity on last names between two author strings."""
    names_a = extract_last_names(authors_a)
    names_b = extract_last_names(authors_b)
    if not names_a or not names_b:
        return 0.0
    intersection = names_a & names_b
    union = names_a | names_b
    return len(intersection) / len(union) if union else 0.0


def query_crossref_title(title: str) -> list[dict]:
    """Query CrossRef for a title, returning top 3 results."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        params = {"query.bibliographic": title, "rows": 3}
        response = requests.get(CROSSREF_API, params=params, headers=CROSSREF_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("items", [])
    except Exception as e:
        print(f"  CrossRef error: {e}")
        return []


def query_crossref_doi(doi: str) -> dict | None:
    """Resolve a DOI via CrossRef."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, headers=CROSSREF_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json().get("message")
    except Exception:
        return None


def extract_crossref_authors(cr_data: dict) -> str:
    """Extract author string from CrossRef result."""
    authors = []
    for author in cr_data.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        if given and family:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)
    return " and ".join(authors)


def extract_crossref_title(cr_data: dict) -> str:
    """Extract title from CrossRef result."""
    titles = cr_data.get("title", [])
    return titles[0] if titles else ""


def find_best_title_match(title: str, results: list[dict]) -> tuple[dict | None, float]:
    """Find the best title match above threshold from CrossRef results."""
    best_match = None
    best_score = 0.0
    for result in results:
        cr_title = extract_crossref_title(result)
        if not cr_title:
            continue
        score = fuzz.token_sort_ratio(title.lower(), cr_title.lower())
        if score > best_score:
            best_score = score
            best_match = result
    return best_match, best_score


def classify_entry(entry: BenchmarkEntry) -> tuple[str, int, str, dict]:
    """Re-classify a single entry with proper verification.

    Returns: (hallucination_type, difficulty_tier, explanation, subtests)
    """
    fields = entry.fields
    title = fields.get("title", "")
    authors = fields.get("author", "")
    doi = fields.get("doi", "")
    year = fields.get("year", "")
    venue = fields.get("booktitle") or fields.get("journal", "")

    subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": False,
        "venue_real": False,
        "fields_complete": bool(title and authors and year),
        "cross_db_agreement": False,
    }

    # Step 1: Try DOI-first verification
    doi_data = None
    if doi:
        doi_data = query_crossref_doi(doi)
        if doi_data:
            subtests["doi_resolves"] = True
            cr_title = extract_crossref_title(doi_data)
            cr_authors = extract_crossref_authors(doi_data)

            title_sim = fuzz.token_sort_ratio(title.lower(), cr_title.lower())
            auth_jacc = author_jaccard(authors, cr_authors)

            if title_sim >= TITLE_SIMILARITY_THRESHOLD:
                subtests["title_exists"] = True
                if auth_jacc >= AUTHOR_JACCARD_THRESHOLD:
                    # DOI resolves, title matches, authors match → real paper, not hallucinated
                    subtests["authors_match"] = True
                    subtests["venue_real"] = True
                    subtests["cross_db_agreement"] = True
                    ht = HallucinationType.PLAUSIBLE_FABRICATION
                    return (
                        ht.value,
                        HALLUCINATION_TIER_MAP[ht].value,
                        f"DOI-verified real paper (title_sim={title_sim:.0f}, auth_jacc={auth_jacc:.2f}). "
                        f"LLM accurately recalled this paper.",
                        subtests,
                    )
                else:
                    # DOI resolves, title matches, authors DON'T match → genuine swapped_authors
                    subtests["authors_match"] = False
                    subtests["venue_real"] = True
                    ht = HallucinationType.AUTHOR_MISMATCH
                    return (
                        ht.value,
                        HALLUCINATION_TIER_MAP[ht].value,
                        f"DOI-verified author mismatch (title_sim={title_sim:.0f}, "
                        f"auth_jacc={auth_jacc:.2f}): '{authors}' vs '{cr_authors}'",
                        subtests,
                    )
            else:
                # DOI resolves but title doesn't match → hybrid fabrication
                subtests["title_exists"] = False
                ht = HallucinationType.HYBRID_FABRICATION
                return (
                    ht.value,
                    HALLUCINATION_TIER_MAP[ht].value,
                    f"Real DOI {doi} but title mismatch (sim={title_sim:.0f}): "
                    f"'{title[:60]}' vs '{cr_title[:60]}'",
                    subtests,
                )

    # Step 2: Title-based search with similarity threshold
    results = query_crossref_title(title)
    best_match, title_score = find_best_title_match(title, results)

    if best_match and title_score >= TITLE_SIMILARITY_THRESHOLD:
        # We have a genuine title match
        subtests["title_exists"] = True
        cr_title = extract_crossref_title(best_match)
        cr_authors = extract_crossref_authors(best_match)
        cr_venue = ""
        if best_match.get("container-title"):
            cr_venue = best_match["container-title"][0]

        auth_jacc = author_jaccard(authors, cr_authors)

        if auth_jacc >= AUTHOR_JACCARD_THRESHOLD:
            subtests["authors_match"] = True
            # Title and authors match — check venue
            if venue and cr_venue:
                venue_sim = fuzz.token_sort_ratio(venue.lower(), cr_venue.lower())
                if venue_sim < 70:
                    subtests["venue_real"] = True  # venue exists, just wrong
                    ht = HallucinationType.WRONG_VENUE
                    return (
                        ht.value,
                        HALLUCINATION_TIER_MAP[ht].value,
                        f"Title matches (sim={title_score:.0f}), authors match "
                        f"(jacc={auth_jacc:.2f}), but venue differs: "
                        f"'{venue}' vs '{cr_venue}'",
                        subtests,
                    )
                else:
                    subtests["venue_real"] = True

            # Near-miss title check
            if title_score < 95:
                ht = HallucinationType.NEAR_MISS_TITLE
                return (
                    ht.value,
                    HALLUCINATION_TIER_MAP[ht].value,
                    f"Near-miss title (sim={title_score:.0f}): '{title[:60]}' vs '{cr_title[:60]}'",
                    subtests,
                )

            # Everything matches — LLM recalled a real paper
            subtests["cross_db_agreement"] = True
            subtests["venue_real"] = True
            ht = HallucinationType.PLAUSIBLE_FABRICATION
            return (
                ht.value,
                HALLUCINATION_TIER_MAP[ht].value,
                f"LLM accurately recalled real paper (title_sim={title_score:.0f}, "
                f"auth_jacc={auth_jacc:.2f}). May not be hallucinated.",
                subtests,
            )
        else:
            # Title matches but authors don't → genuine swapped_authors
            subtests["authors_match"] = False
            subtests["venue_real"] = True
            ht = HallucinationType.AUTHOR_MISMATCH
            return (
                ht.value,
                HALLUCINATION_TIER_MAP[ht].value,
                f"Title matches real paper (sim={title_score:.0f}) but authors differ "
                f"(jacc={auth_jacc:.2f}): '{authors}' vs '{cr_authors}'",
                subtests,
            )
    elif best_match and title_score >= 70:
        # Partial title match — near-miss or chimeric
        cr_authors = extract_crossref_authors(best_match)
        auth_jacc = author_jaccard(authors, cr_authors)

        if auth_jacc >= AUTHOR_JACCARD_THRESHOLD:
            # Authors match but title is different → chimeric title
            subtests["authors_match"] = True
            subtests["title_exists"] = False
            ht = HallucinationType.CHIMERIC_TITLE
            return (
                ht.value,
                HALLUCINATION_TIER_MAP[ht].value,
                f"Partial title match (sim={title_score:.0f}) with matching authors "
                f"(jacc={auth_jacc:.2f}). Real authors, fabricated title.",
                subtests,
            )
        else:
            # Neither title nor authors match well → plausible fabrication
            ht = HallucinationType.PLAUSIBLE_FABRICATION
            return (
                ht.value,
                HALLUCINATION_TIER_MAP[ht].value,
                f"Weak title match (sim={title_score:.0f}), weak author match "
                f"(jacc={auth_jacc:.2f}). Likely fabricated.",
                subtests,
            )

    # Step 3: No title match — check for other patterns
    # Check DOI that didn't resolve
    if doi and not doi_data:
        ht = HallucinationType.FABRICATED_DOI
        return (
            ht.value,
            HALLUCINATION_TIER_MAP[ht].value,
            f"DOI {doi} does not resolve and no title match in CrossRef",
            subtests,
        )

    # Check for future date
    try:
        from datetime import date as date_cls

        year_int = int(year)
        if year_int > date_cls.today().year:
            subtests["fields_complete"] = False
            ht = HallucinationType.FUTURE_DATE
            return (
                ht.value,
                HALLUCINATION_TIER_MAP[ht].value,
                f"Publication year {year} is in the future",
                subtests,
            )
    except (ValueError, TypeError):
        pass

    # Check for placeholder authors
    import re

    placeholder_patterns = [
        r"\b(john|jane)\s+(doe|smith)\b",
        r"\b(test|example|sample)\s+author\b",
        r"\bfirstname\s+lastname\b",
    ]
    authors_lower = authors.lower()
    for pattern in placeholder_patterns:
        if re.search(pattern, authors_lower):
            subtests["authors_match"] = False
            ht = HallucinationType.PLACEHOLDER_AUTHORS
            return (
                ht.value,
                HALLUCINATION_TIER_MAP[ht].value,
                f"Placeholder authors: {authors}",
                subtests,
            )

    # Default: plausible fabrication
    ht = HallucinationType.PLAUSIBLE_FABRICATION
    return (
        ht.value,
        HALLUCINATION_TIER_MAP[ht].value,
        f"No match in CrossRef (best title sim={title_score:.0f}). Completely fabricated paper.",
        subtests,
    )


def main() -> int:
    data_dir = Path(__file__).parent.parent / "data" / "v1.0"
    llm_path = data_dir / "llm_generated.jsonl"

    if not llm_path.exists():
        print(f"ERROR: {llm_path} not found")
        return 1

    # Load entries
    entries: list[BenchmarkEntry] = []
    with open(llm_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))

    print(f"Loaded {len(entries)} LLM-generated entries")
    print("Current distribution:")
    from collections import Counter

    old_dist = Counter(e.hallucination_type for e in entries)
    for ht, count in old_dist.most_common():
        print(f"  {ht}: {count}")

    # Re-classify each entry
    print("\nRe-classifying with proper thresholds...")
    print(f"  Title similarity >= {TITLE_SIMILARITY_THRESHOLD}")
    print(f"  Author Jaccard >= {AUTHOR_JACCARD_THRESHOLD}")
    print()

    reclassified = 0
    for i, entry in enumerate(entries):
        print(f"  [{i + 1}/{len(entries)}] {entry.bibtex_key}: ", end="", flush=True)

        old_type = entry.hallucination_type
        new_type, new_tier, new_explanation, new_subtests = classify_entry(entry)

        entry.hallucination_type = new_type
        entry.difficulty_tier = new_tier
        entry.explanation = (
            f"[Re-classified] {new_explanation} (Original prompt logged in generation script)"
        )
        entry.subtests = new_subtests

        if old_type != new_type:
            print(f"{old_type} → {new_type}")
            reclassified += 1
        else:
            print(f"{new_type} (unchanged)")

    # Save corrected staging file
    print(f"\nRe-classified {reclassified}/{len(entries)} entries")

    print("\nNew distribution:")
    new_dist = Counter(e.hallucination_type for e in entries)
    for ht, count in new_dist.most_common():
        change = count - old_dist.get(ht, 0)
        sign = "+" if change > 0 else ""
        print(f"  {ht}: {count} ({sign}{change})")

    # Save
    with open(llm_path, "w") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    print(f"\nSaved corrected entries to {llm_path}")

    # Now update the entries in the main split files
    print("\nUpdating entries in split files...")
    llm_entries_by_key: dict[str, BenchmarkEntry] = {}
    for entry in entries:
        llm_entries_by_key[entry.bibtex_key] = entry

    split_files = [
        data_dir / "dev_public.jsonl",
        data_dir / "test_public.jsonl",
        Path(__file__).parent.parent / "data" / "hidden" / "test_hidden.jsonl",
    ]

    for split_path in split_files:
        if not split_path.exists():
            continue
        split_entries: list[BenchmarkEntry] = []
        updated = 0
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = BenchmarkEntry.from_json(line)
                if e.generation_method == "llm_generated" and e.bibtex_key:
                    # Find corresponding re-classified entry by matching fields
                    for llm_e in entries:
                        title_a = e.fields.get("title", "")
                        title_b = llm_e.fields.get("title", "")
                        if title_a and title_b and title_a == title_b:
                            e.hallucination_type = llm_e.hallucination_type
                            e.difficulty_tier = llm_e.difficulty_tier
                            e.explanation = llm_e.explanation
                            e.subtests = llm_e.subtests
                            updated += 1
                            break
                split_entries.append(e)

        with open(split_path, "w") as f:
            for e in split_entries:
                f.write(e.to_json() + "\n")
        print(f"  {split_path.name}: updated {updated} entries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
