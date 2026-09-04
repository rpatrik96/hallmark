"""DOI-only baseline: the simplest verification strategy.

Only checks whether DOIs resolve via doi.org. Entries without DOIs
are assumed valid (conservative).
"""

from __future__ import annotations

import logging
import time

import httpx

from hallmark.baselines.prescreening import merge_with_predictions, prescreen_entries
from hallmark.dataset.schema import BlindEntry, Prediction

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
            # RequestError is the base of ConnectError, RemoteProtocolError,
            # ReadError and the rest. Catching only two of them let a server
            # disconnect propagate and abort the whole run mid-split.
            exceptions=(httpx.TimeoutException, httpx.RequestError),
        )
        # Only doi.org answering 404/410 itself is evidence the DOI is not
        # registered. Everything else that is not a 200 is indeterminate, and
        # this used to return False for all of it -- scoring the citation as
        # fabricated at confidence 0.75.
        #
        # It is not a corner case. A sample of 150 VALID entries carrying a DOI
        # returned HTTP 202 for 56 of them and 403 for one, all IEEE and ACM
        # landing pages applying bot mitigation after doi.org redirected
        # successfully. So the measured false-positive rate was substantially a
        # measurement of a publisher's bot policy on the day the run happened.
        #
        # ``prescreening.check_doi_resolves`` already draws these lines; this is
        # the same logic, and the two agreeing is what makes a DOI check
        # comparable across the pipeline.
        if resp.status_code == 200:
            return True, f"DOI resolves -> {resp.url}"
        if resp.status_code in (404, 410):
            if resp.history:
                # The 404 came from the redirect target, not from doi.org: the
                # DOI is registered and the landing page is broken or blocks us.
                return None, (
                    f"DOI resolved at doi.org but redirect target returned "
                    f"HTTP {resp.status_code} (indeterminate)"
                )
            return False, f"DOI returns HTTP {resp.status_code} at doi.org"
        # 202/403/429/5xx and friends: bot blocks, rate limits and server
        # errors are transient, never evidence of fabrication.
        return None, f"DOI returned HTTP {resp.status_code} (indeterminate)"
    except (httpx.TimeoutException, httpx.RequestError) as e:
        return None, f"Network error (unresolved): {type(e).__name__}: {e}"


def run_doi_only(
    entries: list[BlindEntry],
    timeout_per_doi: float = 10.0,
    skip_prescreening: bool = False,
    reference_year: int | None = None,
    **_kw: object,
) -> list[Prediction]:
    """Run DOI-only verification on all entries.

    Pre-screening (DOI check, year bounds, author heuristics) runs before
    DOI resolution to catch obvious hallucinations early, then results are merged.

    Args:
        entries: Benchmark entries to verify.
        timeout_per_doi: Timeout per DOI resolution request (default: 10.0).
        skip_prescreening: Skip pre-screening checks (default: False).
        reference_year: Upper bound year for future-date detection. When None,
            defaults to the current calendar year. Pass an explicit value for
            reproducible evaluation runs.
    """
    # Run pre-screening before DOI checks to catch obvious hallucinations
    prescreen_results = (
        prescreen_entries(entries, reference_year=reference_year) if not skip_prescreening else {}
    )

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
            # Indeterminate: a network error, a transient HTTP status (202, 403,
            # 429, 5xx), or a 404 from the redirect target rather than from
            # doi.org. None of these is evidence about the citation, so the
            # baseline declines to flag. Conservative VALID at 0.5 keeps the
            # existing convention that this baseline never abstains outright.
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason=f"indeterminate | {detail}",
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
