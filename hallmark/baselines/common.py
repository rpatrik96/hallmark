"""Shared utilities for HALLMARK baseline wrappers."""

from __future__ import annotations

from collections.abc import Callable

from hallmark.dataset.schema import BlindEntry, Prediction

BothVariantsResult = dict[str, list[Prediction]]


def fallback_predictions(
    entries: list[BlindEntry],
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


def entries_to_bib(entries: list[BlindEntry]) -> str:
    """Convert benchmark entries to a BibTeX string."""
    return "\n\n".join(e.to_bibtex() for e in entries)


def run_with_prescreening(
    entries: list[BlindEntry],
    run_tool: Callable[[list[BlindEntry]], list[Prediction]],
    skip_prescreening: bool = False,
    backfill_reason: str = "Entry not in tool output",
    reference_year: int | None = None,
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
        reference_year: Upper bound year for future-date detection. When None,
            defaults to the current calendar year. Pass an explicit value for
            reproducible evaluation runs.

    Returns:
        Complete list of predictions (one per entry).
    """
    from hallmark.baselines.prescreening import merge_with_predictions, prescreen_entries

    prescreen_results = (
        prescreen_entries(entries, reference_year=reference_year) if not skip_prescreening else {}
    )

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


def run_baseline_both_variants(
    entries: list[BlindEntry],
    baseline_runner: Callable[[list[BlindEntry]], list[Prediction]],
) -> BothVariantsResult:
    """Run a baseline with and without pre-screening for fair comparison reporting.

    Runs the baseline twice — once without pre-screening (tool only) and once
    with pre-screening applied — and returns both result sets. Predictions are
    tagged via their ``source`` field: ``"tool"`` for pure-tool results and
    ``"prescreening"`` / ``"prescreening_override"`` where pre-screening
    changed the outcome.

    Args:
        entries: Benchmark entries to verify.
        baseline_runner: Callable that accepts a list of entries and returns
            predictions **without** any pre-screening. Typically the raw
            ``run_<baseline>`` function called with ``skip_prescreening=True``.

    Returns:
        Dict with keys ``"without_prescreening"`` and ``"with_prescreening"``,
        each mapping to a list of ``Prediction`` objects (one per entry).
    """
    from hallmark.baselines.prescreening import merge_with_predictions, prescreen_entries

    # Run tool without pre-screening
    without = baseline_runner(entries)

    # Tag all tool-only predictions explicitly
    from dataclasses import replace as _replace

    without_tagged = [_replace(p, source="tool") for p in without]

    # Backfill missing entries for the without-prescreening variant
    predicted_keys = {p.bibtex_key for p in without_tagged}
    for entry in entries:
        if entry.bibtex_key not in predicted_keys:
            without_tagged.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason="Entry not in tool output",
                    source="tool",
                )
            )

    # Run pre-screening and merge with the tool predictions
    prescreen_results = prescreen_entries(entries)
    with_prescreening = merge_with_predictions(entries, list(without_tagged), prescreen_results)

    return {
        "without_prescreening": without_tagged,
        "with_prescreening": with_prescreening,
    }
