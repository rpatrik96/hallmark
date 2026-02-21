"""Degenerate baselines providing statistical reference points."""

from __future__ import annotations

import random as _random

from hallmark.dataset.schema import BlindEntry, Prediction


def random_baseline(
    entries: list[BlindEntry],
    seed: int = 42,
    prevalence: float = 0.5,
) -> list[Prediction]:
    """Random baseline: predict HALLUCINATED with probability=prevalence.

    Confidence is prevalence for HALLUCINATED predictions,
    1-prevalence for VALID predictions.
    """
    rng = _random.Random(seed)
    predictions = []
    for entry in entries:
        is_hall = rng.random() < prevalence
        confidence = prevalence if is_hall else (1.0 - prevalence)
        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label="HALLUCINATED" if is_hall else "VALID",  # type: ignore[arg-type]
                confidence=confidence,
            )
        )
    return predictions


def always_hallucinated_baseline(
    entries: list[BlindEntry],
) -> list[Prediction]:
    """Constant baseline: predict HALLUCINATED for every entry."""
    return [
        Prediction(
            bibtex_key=entry.bibtex_key,
            label="HALLUCINATED",
            confidence=1.0,
        )
        for entry in entries
    ]


def always_valid_baseline(
    entries: list[BlindEntry],
) -> list[Prediction]:
    """Constant baseline: predict VALID for every entry."""
    return [
        Prediction(
            bibtex_key=entry.bibtex_key,
            label="VALID",
            confidence=1.0,
        )
        for entry in entries
    ]
