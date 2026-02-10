"""Baseline wrapper for HaRC (Hallucinated Reference Checker).

HaRC (https://pypi.org/project/harcx/) validates BibTeX citations against
Semantic Scholar, DBLP, Google Scholar, and Open Library.

Install: pip install harcx
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


def run_harc(
    entries: list[BenchmarkEntry],
    author_threshold: float = 0.6,
    year_tolerance: int = 1,
    api_key: str | None = None,
    check_urls: bool = False,
) -> list[Prediction]:
    """Run HaRC verification on benchmark entries.

    Requires: pip install harcx

    Args:
        entries: Benchmark entries to verify.
        author_threshold: Author match threshold (0.0-1.0, default: 0.6).
        year_tolerance: Year tolerance (default: 1).
        api_key: Optional Semantic Scholar API key.
        check_urls: Whether to verify URL reachability.

    Returns:
        List of Predictions.
    """
    try:
        from reference_checker import (  # type: ignore[import-untyped]
            check_citations,
            check_web_citations,
        )
    except ImportError:
        raise ImportError(
            "harcx is required for the HaRC baseline. Install with: pip install harcx\n"
            "(The importable module is 'reference_checker')"
        ) from None

    # Write entries to a temporary .bib file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
        for entry in entries:
            f.write(entry.to_bibtex() + "\n\n")
        bib_path = f.name

    start = time.time()
    try:
        issues = check_citations(
            bib_path,
            author_threshold=author_threshold,
            year_tolerance=year_tolerance,
            api_key=api_key,
            verbose=False,
        )
        url_issues = []
        if check_urls:
            url_issues = check_web_citations(bib_path, verbose=False)
    finally:
        Path(bib_path).unlink(missing_ok=True)

    total_time = time.time() - start
    per_entry_time = total_time / max(len(entries), 1)

    # Build a set of flagged entry keys from issues
    flagged_keys: dict[str, list[str]] = {}
    for issue in issues:
        key = getattr(issue, "key", None) or getattr(issue, "entry_key", "")
        msg = getattr(issue, "message", str(issue))
        if key:
            flagged_keys.setdefault(key, []).append(msg)

    url_flagged: dict[str, list[str]] = {}
    for issue in url_issues:
        key = getattr(issue, "key", None) or getattr(issue, "entry_key", "")
        msg = getattr(issue, "message", str(issue))
        if key:
            url_flagged.setdefault(key, []).append(msg)

    # Map to predictions
    predictions = []
    api_sources = ["semantic_scholar", "dblp", "google_scholar", "open_library"]

    for entry in entries:
        citation_issues = flagged_keys.get(entry.bibtex_key, [])
        url_problems = url_flagged.get(entry.bibtex_key, [])
        all_issues = citation_issues + url_problems

        if all_issues:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=min(0.5 + 0.15 * len(all_issues), 0.95),
                    reason=f"HaRC flagged: {'; '.join(all_issues)}",
                    api_sources_queried=api_sources,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(api_sources),
                )
            )
        else:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.85,
                    reason="HaRC: No issues found across databases",
                    api_sources_queried=api_sources,
                    wall_clock_seconds=per_entry_time,
                    api_calls=len(api_sources),
                )
            )

    return predictions
