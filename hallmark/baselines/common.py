"""Shared utilities for HALLMARK baseline wrappers."""

from __future__ import annotations

from hallmark.dataset.schema import BenchmarkEntry, Prediction


def fallback_predictions(
    entries: list[BenchmarkEntry],
    reason: str = "Tool unavailable",
    api_sources: list[str] | None = None,
    api_calls: int = 0,
) -> list[Prediction]:
    """Conservative fallback predictions when a baseline tool fails.

    Returns VALID with confidence=0.5 for all entries.
    """
    return [
        Prediction(
            bibtex_key=e.bibtex_key,
            label="VALID",
            confidence=0.5,
            reason=reason,
            api_sources_queried=api_sources or [],
            api_calls=api_calls,
        )
        for e in entries
    ]


def entries_to_bib(entries: list[BenchmarkEntry]) -> str:
    """Convert benchmark entries to a BibTeX string."""
    return "\n\n".join(e.to_bibtex() for e in entries)
