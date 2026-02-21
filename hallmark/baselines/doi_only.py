"""DOI-only baseline: the simplest verification strategy.

Only checks whether DOIs resolve via doi.org. Entries without DOIs
are assumed valid (conservative).
"""

from __future__ import annotations

import logging
import time

import httpx

from hallmark.baselines.prescreening import merge_with_predictions, prescreen_entries
from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


def check_doi(doi: str, timeout: float = 10.0) -> tuple[bool | None, str]:
    """Check if a DOI resolves."""
    if not doi:
        return True, "No DOI to check"

    # Normalize
    doi = doi.strip()
    if doi.startswith("http"):
        # Extract DOI from URL
        for prefix in ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/"]:
            if doi.startswith(prefix):
                doi = doi[len(prefix) :]
                break

    url = f"https://doi.org/{doi}"
    try:
        from hallmark.baselines._cache import retry_with_backoff

        resp = retry_with_backoff(
            lambda: httpx.head(url, follow_redirects=True, timeout=timeout),
            max_retries=2,
            base_delay=1.0,
            exceptions=(httpx.TimeoutException, httpx.ConnectError),
        )
        if resp.status_code == 200:
            return True, f"DOI resolves -> {resp.url}"
        else:
            return False, f"DOI returned HTTP {resp.status_code}"
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        return None, f"Network error (unresolved): {e}"


def run_doi_only(
    entries: list[BenchmarkEntry],
    timeout_per_doi: float = 10.0,
    skip_prescreening: bool = False,
) -> list[Prediction]:
    """Run DOI-only verification on all entries.

    Pre-screening (DOI check, year bounds, author heuristics) runs before
    DOI resolution to catch obvious hallucinations early, then results are merged.

    Args:
        entries: Benchmark entries to verify.
        timeout_per_doi: Timeout per DOI resolution request (default: 10.0).
        skip_prescreening: Skip pre-screening checks (default: False).
    """
    # Run pre-screening before DOI checks to catch obvious hallucinations
    prescreen_results = prescreen_entries(entries) if not skip_prescreening else {}

    predictions = []

    for entry in entries:
        start = time.time()
        doi = entry.fields.get("doi")

        if not doi:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason="No DOI field present",
                    wall_clock_seconds=0.0,
                    api_calls=0,
                )
            )
            continue

        resolves, detail = check_doi(doi, timeout_per_doi)
        elapsed = time.time() - start

        if resolves is None:
            # Indeterminate: network error — treat as VALID with low confidence
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason=f"network_error | {detail}",
                    subtest_results={"doi_resolves": None},
                    api_sources_queried=["doi.org"],
                    wall_clock_seconds=elapsed,
                    api_calls=1,
                )
            )
        else:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID" if resolves else "HALLUCINATED",
                    confidence=0.85 if resolves else 0.75,
                    reason=detail,
                    subtest_results={"doi_resolves": resolves},
                    api_sources_queried=["doi.org"],
                    wall_clock_seconds=elapsed,
                    api_calls=1,
                )
            )

    # Merge pre-screening results with tool predictions (unless skipped)
    if not skip_prescreening:
        predictions = merge_with_predictions(entries, predictions, prescreen_results)

    return predictions
