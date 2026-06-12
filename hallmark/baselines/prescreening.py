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

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)

# The benchmark reference year is pinned to the dataset freeze date.
# Callers can override via the reference_year parameter.
_BENCHMARK_REFERENCE_YEAR: int = 2026


@dataclass
class PreScreenResult:
    """Result from a single pre-screening check."""

    label: Literal["HALLUCINATED", "VALID", "UNKNOWN"]
    confidence: float
    reason: str
    check_name: str


# --- DOI normalization (mirrors bibtex-updater's normalize_doi_for_resolution) ---

# Leading resolver-URL prefix: https://doi.org/..., http://dx.doi.org/...
_DOI_URL_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)

# Core DOI shape: "10.<registrant>/<suffix>".
_DOI_CORE_RE = re.compile(r"10\.\d+/[^\s]+")

#: arXiv DataCite DOI prefix. arXiv mints *versioned* DOIs
#: (``10.48550/arXiv.2010.11929v1``), but only the unversioned DOI
#: (``10.48550/arXiv.2010.11929``) resolves via doi.org — the versioned form
#: 404s. Stripping the version suffix is therefore safe ONLY for this prefix;
#: other DOIs may legitimately end in a letter+digit token.
_ARXIV_DOI_PREFIX = "10.48550/arxiv."
_ARXIV_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_doi_for_resolution(doi: str) -> str | None:
    """Normalize a raw DOI field value for resolution against doi.org.

    Replicates the normalization bibtex-updater applies before resolving a DOI
    (``normalize_doi_for_resolution`` in ``bibtex_updater.utils``):

    - strip surrounding whitespace and a leading ``https://doi.org/`` /
      ``http://dx.doi.org/`` URL prefix, then extract the ``10.<reg>/<suffix>``
      core;
    - for arXiv DataCite DOIs (prefix ``10.48550/arXiv.``, case-insensitive),
      strip a trailing version suffix (``v1``, ``v2``, ...) — the versioned
      form 404s at doi.org while the unversioned form resolves.

    Returns:
        The normalized DOI, or None when no DOI core can be extracted.
    """
    stripped = _DOI_URL_PREFIX_RE.sub("", doi.strip())
    core = _DOI_CORE_RE.search(stripped)
    if core is None:
        return None
    normalized = core.group(0)
    if normalized.lower().startswith(_ARXIV_DOI_PREFIX):
        normalized = _ARXIV_VERSION_SUFFIX_RE.sub("", normalized)
    return normalized


def check_doi_resolves(entry: BlindEntry) -> PreScreenResult:
    """Check if DOI resolves via HTTP HEAD request.

    The DOI is normalized first (see :func:`normalize_doi_for_resolution`):
    URL prefixes are stripped and arXiv DataCite version suffixes (``vN``)
    removed, because versioned arXiv DOIs 404 at doi.org even though the
    unversioned preprint DOI resolves.

    Only a definitive 404/410 served by doi.org itself flags HALLUCINATED.
    Anything ambiguous — network errors, timeouts, rate limits (429), bot
    blocks (403), server errors (5xx), or a 404/410 served by a redirect
    *target* after doi.org resolved the DOI — returns UNKNOWN so that
    pre-screening never overrides the tool on transient failures.

    Returns:
        VALID (0.85) if DOI resolves
        HALLUCINATED (0.85) if doi.org itself returns 404 or 410
        UNKNOWN if no DOI, malformed DOI, network error, or any other HTTP status
    """
    raw_doi = entry.fields.get("doi")
    if not raw_doi:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason="No DOI field present",
            check_name="check_doi_resolves",
        )

    normalized_doi = normalize_doi_for_resolution(raw_doi)
    if normalized_doi is None:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason=f"Malformed DOI: {raw_doi}",
            check_name="check_doi_resolves",
        )

    # Reason strings keep the raw DOI; mention the normalization when it changed it.
    display_doi = raw_doi.strip()
    norm_note = (
        f" (normalized to {normalized_doi} for resolution)" if normalized_doi != display_doi else ""
    )

    url = f"https://doi.org/{normalized_doi}"

    try:
        from hallmark.baselines._cache import retry_with_backoff

        response = retry_with_backoff(
            lambda: httpx.head(url, timeout=10.0, follow_redirects=True),
            max_retries=2,
            base_delay=1.0,
            exceptions=(httpx.RequestError, httpx.TimeoutException),
        )
        if response.status_code == 200:
            return PreScreenResult(
                label="VALID",
                confidence=0.85,
                reason=f"DOI {display_doi} resolves successfully{norm_note}",
                check_name="check_doi_resolves",
            )
        elif response.status_code in (404, 410):
            if response.history:
                # The 404/410 came from a redirect target, not from doi.org: the
                # DOI is registered (doi.org redirected) but the landing page is
                # broken or blocks HEAD requests. Not evidence of fabrication.
                return PreScreenResult(
                    label="UNKNOWN",
                    confidence=0.0,
                    reason=(
                        f"DOI {display_doi} resolved at doi.org but redirect target "
                        f"returned HTTP {response.status_code}{norm_note}"
                    ),
                    check_name="check_doi_resolves",
                )
            return PreScreenResult(
                label="HALLUCINATED",
                confidence=0.85,
                reason=f"DOI {display_doi} returns {response.status_code} at doi.org{norm_note}",
                check_name="check_doi_resolves",
            )
        else:
            # 403/429/5xx etc.: bot blocks, rate limits, or server errors are
            # transient — never treated as evidence of fabrication.
            return PreScreenResult(
                label="UNKNOWN",
                confidence=0.0,
                reason=f"DOI {display_doi} returned HTTP {response.status_code}{norm_note}",
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


def check_year_bounds(entry: BlindEntry, reference_year: int | None = None) -> PreScreenResult:
    """Check if publication year is within plausible bounds.

    Args:
        entry: Benchmark entry to check.
        reference_year: Year to use as the upper bound for "future" detection.
            When None, defaults to the benchmark reference year (2026). Pass an
            explicit value for reproducible evaluation runs.

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

    current_year = reference_year if reference_year is not None else _BENCHMARK_REFERENCE_YEAR

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


def check_author_heuristics(entry: BlindEntry) -> PreScreenResult:
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


# Patterns for capitalized-token "fake-author" heuristics.
# These complement check_author_heuristics: that function catches "Author1"/"AuthorA"
# and single-letter last names; this one targets capitalized-word-followed-by-digit
# tokens (e.g. "Smith2"), fields that reduce entirely to initials/placeholders (e.g.
# "A." or "A. B. and C. D."), and majority-uppercase fields.
_CAPITALIZED_DIGIT_TOKEN = re.compile(r"^[A-Z][a-z]*\d+$")
_INITIAL_ONLY_TOKEN = re.compile(r"^[A-Z]\.$")
# A bare single uppercase letter with no period ("A") — also counts as an initial when
# deciding whether a whole name reduces to initials (e.g. "A B" has no real surname).
_BARE_INITIAL_TOKEN = re.compile(r"^[A-Z]$")
_PURE_UPPERCASE_TOKEN = re.compile(r"^[A-Z]{2,}$")


def _author_chunks(author_field: str) -> list[list[str]]:
    """Split the author field into per-author token lists.

    Splits on ' and ' to separate distinct authors, then on whitespace (and stripping
    BibTeX "Last, First" commas) within each author. Returns one token list per author,
    preserving the author boundary that a flat token list discards — Case 2 needs this
    boundary to decide whether a *whole name* reduces to initials.
    """
    chunks: list[list[str]] = []
    for chunk in re.split(r"\s+and\s+", author_field):
        parts = [p.strip().strip(",") for p in chunk.split()]
        tokens = [p for p in parts if p]
        if tokens:
            chunks.append(tokens)
    return chunks


def _is_initials_only_name(tokens: list[str]) -> bool:
    """True when every token of a single author name is an initial/placeholder.

    A name reduces to initials when it carries no real surname — every token is either
    a lone "X." initial or a bare single uppercase letter "X". A mid-name initial inside
    a full name (e.g. "John A. Smith") has real-word tokens, so this returns False.
    """
    if not tokens:
        return False
    return all(_INITIAL_ONLY_TOKEN.match(t) or _BARE_INITIAL_TOKEN.match(t) for t in tokens)


def check_capitalized_unknown_authors(entry: BlindEntry) -> PreScreenResult:
    """Flag entries whose authors look like capitalized-token placeholders.

    Local approximation of CheckIfExist's fake-author detection (which cross-validates
    against retrieved candidates). Pre-screening runs before any API call, so this
    instead matches surface-level placeholder patterns that ``check_author_heuristics``
    misses:

    - ``Smith2``, ``Jones3``: capitalized word followed by digits
    - whole field reduces to initials/placeholders, i.e. *every* author name carries
      no real surname (e.g. ``A.`` alone, or ``A. B. and C. D.``). A mid-name initial
      inside a full name (``John A. Smith``) is left untouched — that is a real author.
    - Author fields where >50% of tokens are pure uppercase words (length > 1)

    Returns:
        HALLUCINATED (confidence 0.75) if any case above triggers; otherwise UNKNOWN.
    """
    author_field = entry.fields.get("author", "")
    if not author_field or not author_field.strip():
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason="No author field present",
            check_name="check_capitalized_unknown_authors",
        )

    chunks = _author_chunks(author_field)
    if not chunks:
        return PreScreenResult(
            label="UNKNOWN",
            confidence=0.0,
            reason="No tokens in author field",
            check_name="check_capitalized_unknown_authors",
        )
    tokens = [t for chunk in chunks for t in chunk]

    # Case 1: capitalized-word-followed-by-digit (Smith2, Jones3) or "AuthorN" (overlap-safe)
    for token in tokens:
        if _CAPITALIZED_DIGIT_TOKEN.match(token) and not re.fullmatch(r"Author\d+", token):
            # Skip "Author1" / "AuthorN" — those are caught by check_author_heuristics already.
            return PreScreenResult(
                label="HALLUCINATED",
                confidence=0.75,
                reason=f"Capitalized-word-with-digit author token: {token!r}",
                check_name="check_capitalized_unknown_authors",
            )

    # Case 2: the WHOLE field reduces to initials/placeholders — every author name is
    # initials-only with no real surname (e.g. "A.", "A. B. and C. D."). A mid-name
    # initial inside a full name ("John A. Smith") leaves real-word tokens, so that name
    # is NOT initials-only and the field is not flagged here.
    if all(_is_initials_only_name(chunk) for chunk in chunks):
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.75,
            reason=f"All authors reduce to initials/placeholders: {author_field!r}",
            check_name="check_capitalized_unknown_authors",
        )

    # Case 3: >50% pure-uppercase tokens of length > 1 (e.g. "ABC DEF and GHI JKL")
    pure_upper = sum(1 for t in tokens if _PURE_UPPERCASE_TOKEN.match(t))
    if len(tokens) > 1 and pure_upper / len(tokens) > 0.5:
        return PreScreenResult(
            label="HALLUCINATED",
            confidence=0.75,
            reason=f"Majority pure-uppercase tokens ({pure_upper}/{len(tokens)})",
            check_name="check_capitalized_unknown_authors",
        )

    return PreScreenResult(
        label="UNKNOWN",
        confidence=0.0,
        reason="No capitalized placeholder pattern detected",
        check_name="check_capitalized_unknown_authors",
    )


# Registry of checks that share the common (BlindEntry) -> PreScreenResult interface.
# check_year_bounds is NOT included here — it takes an additional reference_year parameter
# and is special-cased in prescreen_entry(). The type annotation reflects the common
# interface for all checks in this list.
ALL_CHECKS: list[Callable[[BlindEntry], PreScreenResult]] = [
    check_doi_resolves,
    check_author_heuristics,
    check_capitalized_unknown_authors,
]


def prescreen_entry(entry: BlindEntry, reference_year: int | None = None) -> list[PreScreenResult]:
    """Run all pre-screening checks on a single entry.

    Args:
        entry: Benchmark entry to check.
        reference_year: Optional year for reproducible year-bound checks.
            When None, defaults to the benchmark reference year (2026).

    Returns:
        List of results, one per check.
    """
    results = []

    # Run check_year_bounds separately — it takes an extra reference_year parameter.
    try:
        results.append(check_year_bounds(entry, reference_year=reference_year))
    except Exception as e:
        logger.error(f"Check check_year_bounds failed for {entry.bibtex_key}: {e}")
        results.append(
            PreScreenResult(
                label="UNKNOWN",
                confidence=0.0,
                reason=f"Check failed with error: {type(e).__name__}",
                check_name="check_year_bounds",
            )
        )

    # Run all standard checks (common BlindEntry -> PreScreenResult interface).
    for check_fn in ALL_CHECKS:
        try:
            results.append(check_fn(entry))
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


@dataclass
class PrescreeningBreakdown:
    """Summary of how many predictions were influenced by pre-screening overrides.

    An "override" is a prediction whose ``reason`` field contains the marker
    ``"[Pre-screening override]"``, indicating that pre-screening changed the
    outcome from the tool's raw prediction.
    """

    total: int
    """Total number of predictions examined."""

    override_count: int
    """Number of predictions where pre-screening overrode the tool."""

    override_correct: int
    """Override predictions that matched the true label."""

    tool_only_total: int
    """Predictions not affected by a pre-screening override."""

    tool_only_correct: int
    """Tool-only predictions that matched the true label."""

    @property
    def override_accuracy(self) -> float | None:
        """Fraction of overrides that were correct (None if no overrides)."""
        if self.override_count == 0:
            return None
        return self.override_correct / self.override_count

    @property
    def tool_only_accuracy(self) -> float | None:
        """Fraction of tool-only predictions that were correct (None if none)."""
        if self.tool_only_total == 0:
            return None
        return self.tool_only_correct / self.tool_only_total


_PRESCREENING_OVERRIDE_MARKER = "[Pre-screening override]"


def compute_prescreening_breakdown(
    predictions: list[Prediction],
    true_labels: Mapping[str, str],
) -> PrescreeningBreakdown:
    """Compute a breakdown of pre-screening overrides vs. tool-only predictions.

    Args:
        predictions: Merged predictions (one per entry) from a prescreening-enabled run.
        true_labels: Mapping of bibtex_key → true label string ("HALLUCINATED"/"VALID").

    Returns:
        PrescreeningBreakdown with counts and accuracy split by override/tool-only.
    """
    override_count = 0
    override_correct = 0
    tool_only_total = 0
    tool_only_correct = 0

    for pred in predictions:
        true_label = true_labels.get(pred.bibtex_key)
        is_correct = true_label is not None and pred.label == true_label
        reason = pred.reason or ""

        if _PRESCREENING_OVERRIDE_MARKER in reason:
            override_count += 1
            if is_correct:
                override_correct += 1
        else:
            tool_only_total += 1
            if is_correct:
                tool_only_correct += 1

    return PrescreeningBreakdown(
        total=len(predictions),
        override_count=override_count,
        override_correct=override_correct,
        tool_only_total=tool_only_total,
        tool_only_correct=tool_only_correct,
    )


def format_prescreening_breakdown(breakdown: PrescreeningBreakdown) -> str:
    """Return a human-readable multi-line summary of the breakdown."""
    lines = ["Pre-screening breakdown:"]
    pct = f"{breakdown.override_count / breakdown.total:.1%}" if breakdown.total > 0 else "N/A"
    lines.append(f"  Prescreening overrides: {breakdown.override_count}/{breakdown.total} ({pct})")

    if breakdown.override_accuracy is not None:
        oa_pct = f"{breakdown.override_accuracy:.1%}"
        lines.append(
            f"  Override accuracy:      {breakdown.override_correct}/{breakdown.override_count} ({oa_pct})"
        )
    else:
        lines.append("  Override accuracy:      N/A (no overrides)")

    if breakdown.tool_only_accuracy is not None:
        ta_pct = f"{breakdown.tool_only_accuracy:.1%}"
        lines.append(
            f"  Tool-only accuracy:     {breakdown.tool_only_correct}/{breakdown.tool_only_total} ({ta_pct})"
        )
    else:
        lines.append("  Tool-only accuracy:     N/A")

    return "\n".join(lines)


def prescreen_entries(
    entries: list[BlindEntry], reference_year: int | None = None
) -> dict[str, list[PreScreenResult]]:
    """Run pre-screening on all entries.

    Args:
        entries: Benchmark entries to check.
        reference_year: Optional year for reproducible year-bound checks.
            When None, defaults to the benchmark reference year (2026).

    Returns:
        Dictionary mapping bibtex_key to list of PreScreenResults.
    """
    results = {}
    for entry in entries:
        results[entry.bibtex_key] = prescreen_entry(entry, reference_year=reference_year)
    return results


def merge_with_predictions(
    entries: list[BlindEntry],
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
                        source="prescreening",
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
                        source="prescreening",
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
                        source="prescreening_override",
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
                            source="tool",
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
                            source="tool",
                        )
                    )
            else:
                # Pre-screening says UNKNOWN or VALID — keep tool prediction unchanged
                merged.append(replace(tool_pred, source="tool"))

    return merged
