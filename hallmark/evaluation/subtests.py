"""Per-entry sub-test definitions for HALLMARK.

Each BibTeX entry is an atomic test unit with multiple verification criteria
(inspired by HumanEval's ~7.7 tests per problem). Sub-tests include:
1. DOI resolution
2. Title matching
3. Author consistency
4. Venue verification
5. Field completeness
6. Cross-database agreement
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from rapidfuzz import fuzz


@dataclass
class SubTestResult:
    """Result of a single sub-test on a BibTeX entry."""

    name: str
    passed: bool | None  # None = not evaluated / skipped
    detail: str = ""
    score: float = 0.0  # Continuous score [0, 1] for partial credit


# --- Sub-test implementations ---


def check_doi_resolves(doi: str | None, timeout: float = 10.0) -> SubTestResult:
    """Check if a DOI resolves via doi.org."""
    if not doi:
        return SubTestResult(name="doi_resolves", passed=None, detail="No DOI provided")

    doi = doi.strip()
    # Normalize DOI
    if doi.startswith("http"):
        doi = re.sub(r"https?://doi\.org/", "", doi)

    url = f"https://doi.org/{doi}"
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=timeout)
        resolved = resp.status_code == 200
        return SubTestResult(
            name="doi_resolves",
            passed=resolved,
            detail=f"HTTP {resp.status_code}" + (f" -> {resp.url}" if resolved else ""),
            score=1.0 if resolved else 0.0,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        return SubTestResult(
            name="doi_resolves",
            passed=None,
            detail=f"Connection error: {e}",
        )


def check_title_exists(
    title: str,
    api_results: list[dict] | None = None,
    threshold: float = 0.90,
) -> SubTestResult:
    """Check if a title exists in API search results using fuzzy matching."""
    if not title:
        return SubTestResult(name="title_exists", passed=None, detail="No title provided")

    if api_results is None:
        return SubTestResult(name="title_exists", passed=None, detail="No API results to check against")

    title_normalized = _normalize_title(title)
    best_score = 0.0
    best_match = ""

    for result in api_results:
        candidate = result.get("title", "")
        if not candidate:
            continue
        candidate_normalized = _normalize_title(candidate)
        score = fuzz.token_sort_ratio(title_normalized, candidate_normalized) / 100.0
        if score > best_score:
            best_score = score
            best_match = candidate

    passed = best_score >= threshold
    return SubTestResult(
        name="title_exists",
        passed=passed,
        detail=f"Best match ({best_score:.2f}): {best_match[:80]}" if best_match else "No matches found",
        score=best_score,
    )


def check_authors_match(
    authors_entry: str,
    authors_api: list[str] | None = None,
    threshold: float = 0.80,
) -> SubTestResult:
    """Check author consistency using Jaccard similarity on last names."""
    if not authors_entry:
        return SubTestResult(name="authors_match", passed=None, detail="No authors provided")

    if authors_api is None:
        return SubTestResult(name="authors_match", passed=None, detail="No API authors to check against")

    entry_names = _extract_last_names(authors_entry)
    api_names = {n.lower() for n in authors_api}

    if not entry_names or not api_names:
        return SubTestResult(name="authors_match", passed=None, detail="Could not parse author names")

    intersection = entry_names & api_names
    union = entry_names | api_names
    jaccard = len(intersection) / len(union) if union else 0.0

    passed = jaccard >= threshold
    return SubTestResult(
        name="authors_match",
        passed=passed,
        detail=f"Jaccard={jaccard:.2f}, entry={entry_names}, api={api_names}",
        score=jaccard,
    )


def check_venue_real(
    venue: str,
    known_venues: set[str] | None = None,
    api_venue: str | None = None,
    threshold: float = 0.70,
) -> SubTestResult:
    """Check if the venue is real and matches API records."""
    if not venue:
        return SubTestResult(name="venue_real", passed=None, detail="No venue provided")

    # Check against known venues list
    if known_venues:
        venue_lower = venue.lower().strip()
        best = max(
            (fuzz.token_sort_ratio(venue_lower, kv.lower()) / 100.0 for kv in known_venues),
            default=0.0,
        )
        if best >= threshold:
            return SubTestResult(
                name="venue_real", passed=True, detail="Venue found in known venues list", score=1.0
            )
        else:
            return SubTestResult(
                name="venue_real",
                passed=False,
                detail=f"Venue not in known venues (best match: {best:.2f})",
                score=best,
            )

    # Check against API result
    if api_venue:
        score = fuzz.token_sort_ratio(venue.lower(), api_venue.lower()) / 100.0
        passed = score >= threshold
        return SubTestResult(
            name="venue_real",
            passed=passed,
            detail=f"API venue match: {score:.2f} (entry='{venue}', api='{api_venue}')",
            score=score,
        )

    return SubTestResult(name="venue_real", passed=None, detail="No reference data to verify venue")


def check_fields_complete(
    bibtex_type: str,
    fields: dict[str, str],
) -> SubTestResult:
    """Check if required BibTeX fields are present and well-formed."""
    required_fields = _required_fields_for_type(bibtex_type)
    missing = [f for f in required_fields if not fields.get(f)]
    malformed = []

    # Check year is a valid number
    year = fields.get("year", "")
    if year and not re.match(r"^\d{4}$", year.strip()):
        malformed.append(f"year '{year}' is not a 4-digit number")

    # Check DOI format
    doi = fields.get("doi", "")
    if doi and not re.match(r"^10\.\d{4,}", doi.strip()):
        malformed.append(f"doi '{doi}' doesn't match expected format")

    total_checks = len(required_fields) + 2  # +2 for year and doi format
    issues = len(missing) + len(malformed)
    score = (total_checks - issues) / total_checks if total_checks > 0 else 1.0

    passed = len(missing) == 0 and len(malformed) == 0
    details = []
    if missing:
        details.append(f"Missing: {missing}")
    if malformed:
        details.append(f"Malformed: {malformed}")

    return SubTestResult(
        name="fields_complete",
        passed=passed,
        detail="; ".join(details) if details else "All required fields present",
        score=score,
    )


def check_cross_db_agreement(
    results_by_source: dict[str, dict],
    title: str = "",
    threshold: float = 0.85,
) -> SubTestResult:
    """Check if multiple databases agree on the entry's metadata."""
    if len(results_by_source) < 2:
        return SubTestResult(
            name="cross_db_agreement",
            passed=None,
            detail=f"Only {len(results_by_source)} source(s) available, need >=2",
        )

    sources = list(results_by_source.keys())
    agreements = 0
    comparisons = 0

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s1, s2 = results_by_source[sources[i]], results_by_source[sources[j]]
            t1 = _normalize_title(s1.get("title", ""))
            t2 = _normalize_title(s2.get("title", ""))
            if t1 and t2:
                score = fuzz.token_sort_ratio(t1, t2) / 100.0
                comparisons += 1
                if score >= threshold:
                    agreements += 1

    if comparisons == 0:
        return SubTestResult(name="cross_db_agreement", passed=None, detail="No comparable fields")

    agreement_rate = agreements / comparisons
    return SubTestResult(
        name="cross_db_agreement",
        passed=agreement_rate >= 0.5,
        detail=f"{agreements}/{comparisons} source pairs agree",
        score=agreement_rate,
    )


def run_all_subtests(
    bibtex_type: str,
    fields: dict[str, str],
    doi: str | None = None,
    api_results: list[dict] | None = None,
    api_authors: list[str] | None = None,
    api_venue: str | None = None,
    known_venues: set[str] | None = None,
    results_by_source: dict[str, dict] | None = None,
    skip_network: bool = False,
) -> list[SubTestResult]:
    """Run all sub-tests on a single entry."""
    results = []

    # DOI resolution (network call)
    if skip_network:
        results.append(SubTestResult(name="doi_resolves", passed=None, detail="Skipped (no network)"))
    else:
        results.append(check_doi_resolves(doi))

    # Title matching
    results.append(check_title_exists(fields.get("title", ""), api_results))

    # Author consistency
    results.append(check_authors_match(fields.get("author", ""), api_authors))

    # Venue verification
    venue = fields.get("booktitle", "") or fields.get("journal", "")
    results.append(check_venue_real(venue, known_venues, api_venue))

    # Field completeness
    results.append(check_fields_complete(bibtex_type, fields))

    # Cross-database agreement
    results.append(check_cross_db_agreement(results_by_source or {}))

    return results


# --- Helper functions ---


def _normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip LaTeX, strip punctuation."""
    title = title.lower()
    # Remove LaTeX commands
    title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
    title = re.sub(r"[{}\\$]", "", title)
    # Remove punctuation
    title = re.sub(r"[^\w\s]", " ", title)
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _extract_last_names(author_string: str) -> set[str]:
    """Extract last names from a BibTeX author string."""
    names = set()
    for author in author_string.split(" and "):
        author = author.strip()
        if not author:
            continue
        if "," in author:
            # "Family, Given" format
            last = author.split(",")[0].strip()
        else:
            # "Given Family" format
            parts = author.split()
            last = parts[-1] if parts else author
        # Remove LaTeX
        last = re.sub(r"[{}\\]", "", last)
        names.add(last.lower())
    return names


def _required_fields_for_type(bibtex_type: str) -> list[str]:
    """Return required fields for a given BibTeX entry type."""
    required = {
        "article": ["author", "title", "journal", "year"],
        "inproceedings": ["author", "title", "booktitle", "year"],
        "book": ["author", "title", "publisher", "year"],
        "misc": ["author", "title", "year"],
        "phdthesis": ["author", "title", "school", "year"],
        "mastersthesis": ["author", "title", "school", "year"],
        "techreport": ["author", "title", "institution", "year"],
    }
    return required.get(bibtex_type.lower(), ["author", "title", "year"])
