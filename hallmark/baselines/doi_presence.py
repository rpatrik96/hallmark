"""DOI-presence heuristic baseline.

Predicts HALLUCINATED when the DOI field is absent or empty.
This is intentionally a trivial shortcut baseline to measure
how much DOI-absence correlates with hallucination labels.
"""

from __future__ import annotations

from hallmark.dataset.schema import BenchmarkEntry, Prediction


def run_doi_presence_heuristic(entries: list[BenchmarkEntry]) -> list[Prediction]:
    """Run DOI-presence heuristic on all entries.

    This baseline uses a simple rule:
    - No DOI → predict HALLUCINATED (confidence 0.8)
    - Has DOI → predict VALID (confidence 0.6)

    Args:
        entries: Benchmark entries to evaluate.

    Returns:
        List of predictions, one per entry.
    """
    predictions = []

    for entry in entries:
        doi = entry.fields.get("doi", "").strip()

        if not doi:
            # No DOI → likely hallucinated (based on dataset distribution)
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=0.8,
                    reason="No DOI field present (heuristic shortcut)",
                    wall_clock_seconds=0.0,
                    api_calls=0,
                )
            )
        else:
            # Has DOI → assume valid (conservative)
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="VALID",
                    confidence=0.6,
                    reason="DOI field present (heuristic shortcut)",
                    wall_clock_seconds=0.0,
                    api_calls=0,
                )
            )

    return predictions
