from __future__ import annotations

import datetime
import random
import string

from hallmark.dataset.schema import (
    EXPECTED_SUBTESTS,
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)

from ._helpers import _clone_entry
from ._pools import FAKE_AUTHORS, FAKE_DOI_PREFIXES, FAKE_VENUES
from ._registry import register_generator


def _current_reference_year() -> int:
    """Return the current year as the default reference for future-date generation.

    New dataset versions should use the current year so that generated "future"
    entries remain clearly in the future. For exact reproduction of a frozen
    dataset (e.g., v1.0 built in 2025), pass ``reference_year=2025`` explicitly.
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).year


@register_generator(HallucinationType.FABRICATED_DOI, description="Replace DOI with fabricated one")
def generate_fabricated_doi(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 1: Replace DOI with a non-resolving fabricated DOI."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_prefix = rng.choice(FAKE_DOI_PREFIXES)
    # Vary suffix patterns to avoid a single learnable format
    suffix_style = rng.choice(["path", "id", "year_id", "conf_id"])
    if suffix_style == "path":
        seg1 = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))
        seg2 = "".join(rng.choices(string.digits, k=rng.randint(4, 7)))
        fake_suffix = f"{seg1}.{seg2}"
    elif suffix_style == "id":
        fake_suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
    elif suffix_style == "year_id":
        year = rng.choice(["2019", "2020", "2021", "2022", "2023", "2024"])
        seq = "".join(rng.choices(string.digits, k=5))
        fake_suffix = f"{year}.{seq}"
    else:  # conf_id
        conf = rng.choice(["conf", "proc", "jour", "art", "pub"])
        seq = "".join(rng.choices(string.digits, k=7))
        fake_suffix = f"{conf}/{seq}"
    new_entry.fields["doi"] = f"{fake_prefix}/{fake_suffix}"
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.FABRICATED_DOI.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"DOI fabricated: {new_entry.fields['doi']} does not resolve"
    new_entry.subtests = dict(EXPECTED_SUBTESTS[HallucinationType.FABRICATED_DOI])
    new_entry.bibtex_key = f"fabricated_doi_{new_entry.bibtex_key}"
    return new_entry


@register_generator(
    HallucinationType.NONEXISTENT_VENUE, description="Replace venue with invented name"
)
def generate_nonexistent_venue(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 1: Replace venue with an invented conference/journal name.

    Note: Always uses booktitle (normalized to inproceedings per P0.2).
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_venue = rng.choice(FAKE_VENUES)
    # Always use booktitle (normalized to inproceedings per P0.2)
    new_entry.fields["booktitle"] = fake_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.NONEXISTENT_VENUE.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Venue fabricated: '{fake_venue}' does not exist"
    has_doi = bool(new_entry.fields.get("doi"))
    subtests = dict(EXPECTED_SUBTESTS[HallucinationType.NONEXISTENT_VENUE])
    subtests["doi_resolves"] = True if has_doi else None
    new_entry.subtests = subtests
    new_entry.bibtex_key = f"fake_venue_{new_entry.bibtex_key}"
    return new_entry


@register_generator(
    HallucinationType.PLACEHOLDER_AUTHORS, description="Replace authors with generic/fake names"
)
def generate_placeholder_authors(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 1: Replace authors with generic/fake names."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    new_entry.fields["author"] = rng.choice(FAKE_AUTHORS)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PLACEHOLDER_AUTHORS.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Authors are placeholders: {new_entry.fields['author']}"
    has_doi = bool(new_entry.fields.get("doi"))
    subtests = dict(EXPECTED_SUBTESTS[HallucinationType.PLACEHOLDER_AUTHORS])
    subtests["doi_resolves"] = True if has_doi else None
    new_entry.subtests = subtests
    new_entry.bibtex_key = f"fake_authors_{new_entry.bibtex_key}"
    return new_entry


@register_generator(
    HallucinationType.FUTURE_DATE,
    extra_args=("reference_year",),
    description="Set publication year to the future",
)
def generate_future_date(
    entry: BenchmarkEntry,
    rng: random.Random | None = None,
    reference_year: int | None = None,
) -> BenchmarkEntry:
    """Tier 1: Set publication year to the future.

    Args:
        reference_year: Base year for computing future dates. Defaults to the
            current year. Pass explicitly (e.g., 2025) to reproduce a frozen dataset.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    base_year = reference_year if reference_year is not None else _current_reference_year()
    future_year = str(base_year + rng.randint(4, 10))
    new_entry.fields["year"] = future_year
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.FUTURE_DATE.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Publication year {future_year} is in the future"
    has_doi = bool(new_entry.fields.get("doi"))
    subtests = dict(EXPECTED_SUBTESTS[HallucinationType.FUTURE_DATE])
    subtests["doi_resolves"] = True if has_doi else None
    new_entry.subtests = subtests
    new_entry.bibtex_key = f"future_{new_entry.bibtex_key}"
    return new_entry
