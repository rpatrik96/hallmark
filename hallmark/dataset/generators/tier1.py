from __future__ import annotations

import random
import string
from datetime import date

from hallmark.dataset.schema import (
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)

from ._helpers import _clone_entry
from ._pools import FAKE_AUTHORS, FAKE_DOI_PREFIXES, FAKE_VENUES


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
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fabricated_doi_{new_entry.bibtex_key}"
    return new_entry


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
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fake_venue_{new_entry.bibtex_key}"
    return new_entry


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
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fake_authors_{new_entry.bibtex_key}"
    return new_entry


def generate_future_date(entry: BenchmarkEntry, rng: random.Random | None = None) -> BenchmarkEntry:
    """Tier 1: Set publication year to the future."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    future_year = str(date.today().year + rng.randint(4, 10))
    new_entry.fields["year"] = future_year
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.FUTURE_DATE.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Publication year {future_year} is in the future"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": False,  # future year = malformed metadata
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"future_{new_entry.bibtex_key}"
    return new_entry
