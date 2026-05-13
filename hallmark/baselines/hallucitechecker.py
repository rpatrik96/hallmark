"""HalluCiteChecker baseline (port of Sakai et al., 2026).

Title-centric fuzzy matching against multiple academic search APIs (CrossRef,
arXiv, Semantic Scholar). Per the original paper, the title is the only
stable bibliographic field and is used as the primary matching key. We adapt
the original local-snapshot design to live API calls so the baseline can run
without a pre-downloaded corpus.

Algorithm
---------
1. For each entry, query each configured source with the title.
2. Compute character-level Levenshtein similarity (rapidfuzz) between the
   query title and each candidate title, after lowercasing + alphanumeric-strip
   normalisation.
3. If any candidate from any source has similarity >= ``title_threshold``
   (default 0.9), predict VALID; otherwise predict HALLUCINATED.
4. Confidence: 0.85 when matched; otherwise ``1 - max_similarity`` clamped to
   [0.6, 0.9].
5. If all configured sources raise (network errors), emit UNCERTAIN @0.5.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from rapidfuzz import fuzz

# Re-exported as module-level names so tests can patch them via
# ``patch("hallmark.baselines.hallucitechecker.search_crossref", ...)``.
from hallmark.baselines._agentic_tools import (
    search_arxiv,
    search_crossref,
    search_semantic_scholar,
)
from hallmark.dataset.schema import BlindEntry, Prediction

__all__ = [
    "run_hallucitechecker",
    "search_arxiv",
    "search_crossref",
    "search_semantic_scholar",
]

logger = logging.getLogger(__name__)


def _get_source_funcs() -> dict[str, Callable[[str, int], list[dict[str, str]]]]:
    """Return the source name → callable map by looking up module-level names.

    Resolved at call time so test code can ``patch`` the module attributes
    (e.g. ``hallmark.baselines.hallucitechecker.search_crossref``).
    """
    import sys

    mod = sys.modules[__name__]
    return {
        "crossref": mod.search_crossref,
        "arxiv": mod.search_arxiv,
        "semantic_scholar": mod.search_semantic_scholar,
    }


_VALID_SOURCES: tuple[str, ...] = ("crossref", "arxiv", "semantic_scholar")


def _normalize_title(s: str) -> str:
    """Lowercase + strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _title_similarity(a: str, b: str) -> float:
    """Character-level Levenshtein similarity in [0, 1] after normalisation."""
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return 0.0
    return float(fuzz.ratio(na, nb)) / 100.0


def _filter_by_year(
    candidates: list[dict[str, str]], cutoff_year: int | None
) -> list[dict[str, str]]:
    """Filter candidates to those with year <= cutoff (keep candidates with no year)."""
    if cutoff_year is None:
        return candidates
    out: list[dict[str, str]] = []
    for cand in candidates:
        year_str = (cand.get("year") or "").strip()
        if not year_str:
            out.append(cand)
            continue
        try:
            yr = int(year_str[:4])
        except ValueError:
            out.append(cand)
            continue
        if yr <= cutoff_year:
            out.append(cand)
    return out


def run_hallucitechecker(
    entries: list[BlindEntry],
    *,
    knowledge_cutoff_year: int | None = None,
    sources: tuple[str, ...] = ("crossref", "arxiv", "semantic_scholar"),
    title_threshold: float = 0.9,
    candidates_per_source: int = 5,
    **kw: Any,
) -> list[Prediction]:
    """Run the HalluCiteChecker baseline on a list of entries.

    Args:
        entries: Blind entries to verify.
        knowledge_cutoff_year: Drop candidates with ``year > cutoff``.
            Candidates with no year are kept (conservative).
        sources: Which sources to query. Order does not matter.
        title_threshold: Normalised Levenshtein similarity threshold for
            VALID. Default 0.9 per the paper.
        candidates_per_source: Max candidates fetched from each source.
        **kw: Forward-compatible kwargs (ignored).

    Returns:
        List of Predictions, one per entry.
    """
    del kw

    source_funcs = _get_source_funcs()
    unknown_sources = set(sources) - set(source_funcs.keys())
    if unknown_sources:
        raise ValueError(
            f"Unknown source(s): {sorted(unknown_sources)}. "
            f"Available: {sorted(source_funcs.keys())}"
        )

    predictions: list[Prediction] = []

    for entry in entries:
        start = time.time()
        title = (entry.fields.get("title") or "").strip()
        api_calls = 0
        sources_queried: list[str] = []
        sources_succeeded: list[str] = []
        max_similarity = 0.0
        best_source = ""
        best_candidate_title = ""

        if not title:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason="No title field present; cannot match",
                    wall_clock_seconds=time.time() - start,
                    api_calls=0,
                )
            )
            continue

        for source_name in sources:
            sources_queried.append(source_name)
            api_calls += 1
            fn = source_funcs[source_name]
            try:
                cands = fn(title, candidates_per_source)
            except Exception as exc:
                logger.debug("HalluCiteChecker source %s failed: %s", source_name, exc)
                continue
            sources_succeeded.append(source_name)
            cands = _filter_by_year(cands, knowledge_cutoff_year)
            for cand in cands:
                sim = _title_similarity(title, cand.get("title", ""))
                if sim > max_similarity:
                    max_similarity = sim
                    best_source = source_name
                    best_candidate_title = cand.get("title", "")
                    if sim >= title_threshold:
                        # Early-stop optimisation: still keep accumulated bookkeeping
                        break
            if max_similarity >= title_threshold:
                break

        elapsed = time.time() - start

        if not sources_succeeded:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason=(
                        f"All sources unavailable ({', '.join(sources_queried)}); cannot determine."
                    ),
                    api_sources_queried=sources_queried,
                    wall_clock_seconds=elapsed,
                    api_calls=api_calls,
                )
            )
            continue

        if max_similarity >= title_threshold:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.85,
                    reason=(
                        f"Title match in {best_source} "
                        f"(similarity={max_similarity:.3f}, threshold={title_threshold:.2f}): "
                        f"{best_candidate_title!r}"
                    ),
                    subtest_results={"title_exists": True},
                    api_sources_queried=sources_succeeded,
                    wall_clock_seconds=elapsed,
                    api_calls=api_calls,
                )
            )
        else:
            confidence = max(0.6, min(0.9, 1.0 - max_similarity))
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=confidence,
                    reason=(
                        f"No title match across {sources_succeeded} "
                        f"(max similarity={max_similarity:.3f} < {title_threshold:.2f})"
                    ),
                    subtest_results={"title_exists": False},
                    api_sources_queried=sources_succeeded,
                    wall_clock_seconds=elapsed,
                    api_calls=api_calls,
                    predicted_hallucination_type="plausible_fabrication",
                )
            )

    return predictions
