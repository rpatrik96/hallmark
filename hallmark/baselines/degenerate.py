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


# The 8 venues that appear in VALID entries in the v1.0 dataset.
# A venue oracle flags anything outside this set as HALLUCINATED.
VALID_VENUE_SET: frozenset[str] = frozenset(
    {
        # Conferences
        "NeurIPS",
        "ICML",
        "ICLR",
        "AAAI",
        "CVPR",
        # Journals
        "J. Mach. Learn. Res.",
        "Mach. Learn.",
        "Trans. Mach. Learn. Res.",
    }
)


def venue_oracle_baseline(
    entries: list[BlindEntry],
    valid_venues: frozenset[str] | None = None,
) -> list[Prediction]:
    """Venue-oracle baseline: predict HALLUCINATED if venue is not in the known valid set.

    This is a diagnostic baseline that quantifies venue distribution bias in the
    benchmark. Entries whose booktitle/journal appears in the valid venue set are
    predicted VALID; all others are predicted HALLUCINATED with confidence=1.0.
    Entries with no venue field are predicted VALID (conservative default).

    NOT a legitimate detector — exploits dataset construction artifacts.
    """
    venues = valid_venues if valid_venues is not None else VALID_VENUE_SET
    predictions = []
    for entry in entries:
        # Check booktitle first (inproceedings), then journal (article)
        venue = entry.fields.get("booktitle") or entry.fields.get("journal") or ""
        # No venue field → conservative default: predict VALID
        is_known = (not venue) or (venue in venues)
        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label="VALID" if is_known else "HALLUCINATED",
                confidence=1.0,
            )
        )
    return predictions
