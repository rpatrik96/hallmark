"""Pre-screening checks for citation verification baselines.

Lightweight, local (no-API or minimal-API) checks that run BEFORE calling external
tools like bibtex-check or harcx. Catches obvious hallucinations that external tools
may miss.

Note: ``hybrid_fabrication`` entries have a valid DOI that resolves, but the metadata
(authors/title) doesn't match the DOI target. Detecting these requires cross-referencing
resolved metadata, which is the job of external tools — pre-screening alone will
return VALID for the DOI check on these entries.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


@dataclass
class PreScreenResult:
    """Result from a single pre-screening check."""

    label: Literal["HALLUCINATED", "VALID", "UNKNOWN"]
    confidence: float
    reason: str
    check_name: str


def check_doi_resolves(entry: BenchmarkEntry) -> PreScreenResult:
    """Check if DOI resolves via HTTP HEAD request.

    Returns:
        VALID (0.85) if DOI resolves
        HALLUCINATED (0.85) if DOI returns 404
        UNKNOWN if no DOI, network error, or other HTTP status
    """
    doi = entry.fields.get("doi")
    if not doi:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason="No DOI field present",
            check_name="check_doi_resolves",
        )

    # Normalize DOI (strip URLs like https://doi.org/...)
    doi_match = re.search(r"10\.\d+/[^\s]+", doi)
    if doi_match:
        normalized_doi = doi_match.group(0)
    else:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason=f"Malformed DOI: {doi}",
            check_name="check_doi_resolves",
        )

    url = f"https://doi.org/{normalized_doi}"

    try:
        response = httpx.head(url, timeout=10.0, follow_redirects=True)
        if response.status_code == 200:
            return PreScreenResult(
                label="VALID",
                confidence=0.85,
                reason=f"DOI {normalized_doi} resolves successfully",
                check_name="check_doi_resolves",
            )
        elif response.status_code == 404:
            return PreScreenResult(
                label="HALLUCINATED",
                confidence=0.85,
                reason=f"DOI {normalized_doi} returns 404",
                check_name="check_doi_resolves",
            )
        else:
            return PreScreenResult(
                label="UNKNOWN",
                confidence=0.0,
                reason=f"DOI {normalized_doi} returned HTTP {response.status_code}",
                check_name="check_doi_resolves",
            )
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.debug(f"DOI check failed for {normalized_doi}: {e}")
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason=f"Network error checking DOI: {type(e).__name__}",
            check_name="check_doi_resolves",
        )


def check_year_bounds(entry: BenchmarkEntry) -> PreScreenResult:
    """Check if publication year is within plausible bounds.

    Returns:
        HALLUCINATED (0.95) if year is in the future
        HALLUCINATED (0.70) if year < 1900 (implausibly old)
        VALID (0.60) if year is within plausible range
        UNKNOWN if year is missing or non-numeric
    """
    year_str = entry.fields.get("year")
    if not year_str:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason="No year field present",
            check_name="check_year_bounds",
        )

    try:
        year = int(year_str)
    except ValueError:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason=f"Non-numeric year: {year_str}",
            check_name="check_year_bounds",
        )

    current_year = datetime.datetime.now().year

    if year > current_year:
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.95,
            reason=f"Publication year {year} is in the future",
            check_name="check_year_bounds",
        )

    if year < 1900:
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.70,
            reason=f"Implausibly old publication year: {year}",
            check_name="check_year_bounds",
        )

    return PreScreenResult(
        label="VALID",
        confidence=0.60,
        reason=f"Year {year} is within plausible range",
        check_name="check_year_bounds",
    )


def check_author_heuristics(entry: BenchmarkEntry) -> PreScreenResult:
    """Check for placeholder or synthetic author patterns.

    Returns:
        HALLUCINATED (0.80) if placeholder patterns detected
        UNKNOWN otherwise (don't claim valid based on authors alone)
    """
    author_field = entry.fields.get("author", "")
    if not author_field:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason="No author field present",
            check_name="check_author_heuristics",
        )

    # Check for very short author field
    if len(author_field.strip()) < 3:
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.80,
            reason=f"Author field too short: '{author_field}'",
            check_name="check_author_heuristics",
        )

    # Check for "et al." as sole author
    if author_field.strip().lower() in ["et al.", "et al"]:
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.80,
            reason="Author field contains only 'et al.'",
            check_name="check_author_heuristics",
        )

    # Check for placeholder patterns: Author1, Author2, AuthorA, etc.
    if re.search(r"\bAuthor\d+\b|\bAuthor[A-Z]\b", author_field):
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.80,
            reason=f"Synthetic author pattern detected: {author_field}",
            check_name="check_author_heuristics",
        )

    # Check if all authors have single-letter last names
    # BibTeX format: "Lastname1, Firstname1 and Lastname2, Firstname2"
    # or "Firstname1 Lastname1 and Firstname2 Lastname2"
    authors = re.split(r"\s+and\s+", author_field)
    if len(authors) > 1:
        single_letter_count = 0
        for author in authors:
            # Extract last name (assume comma-separated format or last word)
            if "," in author:
                lastname = author.split(",")[0].strip()
            else:
                parts = author.strip().split()
                if parts:
                    lastname = parts[-1]
                else:
                    continue

            if len(lastname) == 1:
                single_letter_count += 1

        if single_letter_count == len(authors):
            return PreScreenResult(
                label="HALLUCINATED",
                confidence=0.80,
                reason="All authors have single-letter last names",
                check_name="check_author_heuristics",
            )

    return PreScreenResult(
        label="UNKNOWN",
        confidence=0.0,
        reason="No placeholder patterns detected",
        check_name="check_author_heuristics",
    )


# Registry of all checks
ALL_CHECKS: list[Callable[[BenchmarkEntry], PreScreenResult]] = [
    check_doi_resolves,
    check_year_bounds,
    check_author_heuristics,
]


def prescreen_entry(entry: BenchmarkEntry) -> list[PreScreenResult]:
    """Run all pre-screening checks on a single entry.

    Returns:
        List of results, one per check.
    """
    results = []
    for check_fn in ALL_CHECKS:
        try:
            result = check_fn(entry)
            results.append(result)
        except Exception as e:
            logger.error(f"Check {check_fn.__name__} failed for {entry.bibtex_key}: {e}")
            results.append(
                PreScreenResult(
                    label="UNKNOWN",
                    confidence=0.0,
                    reason=f"Check failed with error: {type(e).__name__}",
                    check_name=check_fn.__name__,
                )
            )
    return results


def prescreen_entries(entries: list[BenchmarkEntry]) -> dict[str, list[PreScreenResult]]:
    """Run pre-screening on all entries.

    Returns:
        Dictionary mapping bibtex_key to list of PreScreenResults.
    """
    results = {}
    for entry in entries:
        results[entry.bibtex_key] = prescreen_entry(entry)
    return results


def merge_with_predictions(
    entries: list[BenchmarkEntry],
    tool_predictions: list[Prediction],
    prescreen_results: dict[str, list[PreScreenResult]],
) -> list[Prediction]:
    """Merge pre-screening results with tool predictions.

    Logic:
    - If pre-screening found HALLUCINATED and tool said VALID → override to HALLUCINATED
    - If both say HALLUCINATED → keep higher confidence
    - If pre-screening says UNKNOWN → keep tool prediction unchanged
    - For entries with no tool prediction (timeout/missing) → use pre-screening if available

    Args:
        entries: Original benchmark entries
        tool_predictions: Predictions from external tool
        prescreen_results: Results from pre-screening checks

    Returns:
        Merged predictions (one per entry)
    """
    # Build lookup map
    predictions_by_key = {p.bibtex_key: p for p in tool_predictions}

    merged = []

    for entry in entries:
        key = entry.bibtex_key
        tool_pred = predictions_by_key.get(key)
        prescreens = prescreen_results.get(key, [])

        # Find strongest HALLUCINATED signal from pre-screening
        hallucinated_signals = [r for r in prescreens if r.label == "HALLUCINATED"]
        strongest_hallucinated = max(hallucinated_signals, key=lambda r: r.confidence, default=None)

        if tool_pred is None:
            # No tool prediction — use pre-screening if available
            if strongest_hallucinated:
                merged.append(
                    Prediction(
                        bibtex_key=key,
                        label="HALLUCINATED",
                        confidence=strongest_hallucinated.confidence,
                        reason=f"[Pre-screening] {strongest_hallucinated.reason}",
                        subtest_results={strongest_hallucinated.check_name: False},
                        api_sources_queried=[],
                        wall_clock_seconds=0.0,
                        api_calls=0,
                    )
                )
            else:
                # No tool prediction and no strong pre-screening signal — default to UNKNOWN
                merged.append(
                    Prediction(
                        bibtex_key=key,
                        label="VALID",
                        confidence=0.0,
                        reason="[Pre-screening] No tool prediction available, no hallucination detected",
                        subtest_results={},
                        api_sources_queried=[],
                        wall_clock_seconds=0.0,
                        api_calls=0,
                    )
                )
        else:
            # Tool prediction exists
            if strongest_hallucinated and tool_pred.label == "VALID":
                # Override: pre-screening found hallucination, tool said valid
                merged.append(
                    Prediction(
                        bibtex_key=key,
                        label="HALLUCINATED",
                        confidence=strongest_hallucinated.confidence,
                        reason=f"{tool_pred.reason} | [Pre-screening override] {strongest_hallucinated.reason}",
                        subtest_results={
                            **tool_pred.subtest_results,
                            strongest_hallucinated.check_name: False,
                        },
                        api_sources_queried=tool_pred.api_sources_queried,
                        wall_clock_seconds=tool_pred.wall_clock_seconds,
                        api_calls=tool_pred.api_calls,
                    )
                )
            elif strongest_hallucinated and tool_pred.label == "HALLUCINATED":
                # Both say hallucinated — keep higher confidence
                if strongest_hallucinated.confidence > tool_pred.confidence:
                    merged.append(
                        Prediction(
                            bibtex_key=key,
                            label="HALLUCINATED",
                            confidence=strongest_hallucinated.confidence,
                            reason=f"{tool_pred.reason} | [Pre-screening confirms] {strongest_hallucinated.reason}",
                            subtest_results={
                                **tool_pred.subtest_results,
                                strongest_hallucinated.check_name: False,
                            },
                            api_sources_queried=tool_pred.api_sources_queried,
                            wall_clock_seconds=tool_pred.wall_clock_seconds,
                            api_calls=tool_pred.api_calls,
                        )
                    )
                else:
                    # Tool confidence is higher — keep tool prediction, append pre-screening reason
                    merged.append(
                        Prediction(
                            bibtex_key=key,
                            label="HALLUCINATED",
                            confidence=tool_pred.confidence,
                            reason=f"{tool_pred.reason} | [Pre-screening confirms] {strongest_hallucinated.reason}",
                            subtest_results={
                                **tool_pred.subtest_results,
                                strongest_hallucinated.check_name: False,
                            },
                            api_sources_queried=tool_pred.api_sources_queried,
                            wall_clock_seconds=tool_pred.wall_clock_seconds,
                            api_calls=tool_pred.api_calls,
                        )
                    )
            else:
                # Pre-screening says UNKNOWN or VALID — keep tool prediction unchanged
                merged.append(tool_pred)

    return merged
