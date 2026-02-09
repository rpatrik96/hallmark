"""DOI-only baseline: the simplest verification strategy.

Only checks whether DOIs resolve via doi.org. Entries without DOIs
are assumed valid (conservative).
"""

from __future__ import annotations

import logging
import time

import httpx

from hallmark.dataset.schema import BenchmarkEntry, Prediction

logger = logging.getLogger(__name__)


def check_doi(doi: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Check if a DOI resolves."""
    if not doi:
        return True, "No DOI to check"

    # Normalize
    doi = doi.strip()
    if doi.startswith("http"):
        # Extract DOI from URL
        for prefix in ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/"]:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break

    url = f"https://doi.org/{doi}"
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=timeout)
        if resp.status_code == 200:
            return True, f"DOI resolves -> {resp.url}"
        else:
            return False, f"DOI returned HTTP {resp.status_code}"
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        return True, f"Network error (assuming valid): {e}"


def run_doi_only(
    entries: list[BenchmarkEntry],
    timeout_per_doi: float = 10.0,
) -> list[Prediction]:
    """Run DOI-only verification on all entries."""
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

    return predictions
