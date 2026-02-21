"""Shared utilities for HALLMARK baseline wrappers."""

from __future__ import annotations

from collections.abc import Callable

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


def run_with_prescreening(
    entries: list[BenchmarkEntry],
    run_tool: Callable[[list[BenchmarkEntry]], list[Prediction]],
    skip_prescreening: bool = False,
    backfill_reason: str = "Entry not in tool output",
) -> list[Prediction]:
    """Wrap a tool runner with pre-screening, backfill, and merge.

    Standard wrapper for CLI-based baselines that:
    1. Runs pre-screening checks (unless skipped)
    2. Calls the tool runner
    3. Backfills missing predictions (entries not in tool output)
    4. Merges pre-screening results with tool predictions

    Args:
        entries: Benchmark entries to verify.
        run_tool: Function that runs the external tool and returns predictions.
            May return fewer predictions than entries (e.g. on timeout).
        skip_prescreening: Skip pre-screening checks.
        backfill_reason: Reason string for backfilled predictions.

    Returns:
        Complete list of predictions (one per entry).
    """
    from hallmark.baselines.prescreening import merge_with_predictions, prescreen_entries

    prescreen_results = prescreen_entries(entries) if not skip_prescreening else {}

    predictions = run_tool(entries)

    # Backfill missing predictions
    predicted_keys = {p.bibtex_key for p in predictions}
    for entry in entries:
        if entry.bibtex_key not in predicted_keys:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason=backfill_reason,
                )
            )

    # Merge pre-screening results
    if not skip_prescreening:
        predictions = merge_with_predictions(entries, predictions, prescreen_results)

    return predictions
