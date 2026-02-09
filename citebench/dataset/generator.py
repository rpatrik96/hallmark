"""Generate hallucinated BibTeX entries for CiteBench.

Supports four generation methods:
1. Systematic perturbation of valid entries
2. LLM-generated hallucinations (via API)
3. Real-world hallucinations (manual curation)
4. Adversarial crafting
"""

from __future__ import annotations

import copy
import random
import string
from datetime import date

from citebench.dataset.schema import (
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)


def generate_fabricated_doi(entry: BenchmarkEntry, rng: random.Random | None = None) -> BenchmarkEntry:
    """Tier 1: Replace DOI with a non-resolving fabricated DOI."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_prefix = rng.choice(["10.9999", "10.8888", "10.7777"])
    fake_suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=8))
    new_entry.fields["doi"] = f"{fake_prefix}/fake.{fake_suffix}"
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


def generate_nonexistent_venue(entry: BenchmarkEntry, rng: random.Random | None = None) -> BenchmarkEntry:
    """Tier 1: Replace venue with an invented conference/journal name."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_venues = [
        "International Conference on Advanced AI Systems",
        "Journal of Computational Intelligence and Applications",
        "Workshop on Emerging Methods in Deep Learning",
        "Transactions on Neural Computing Paradigms",
        "Symposium on Frontier Artificial Intelligence Research",
        "Annual Conference on Machine Learning Innovations",
    ]
    fake_venue = rng.choice(fake_venues)
    venue_field = "booktitle" if "booktitle" in new_entry.fields else "journal"
    new_entry.fields[venue_field] = fake_venue
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.NONEXISTENT_VENUE.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Venue fabricated: '{fake_venue}' does not exist"
    new_entry.subtests = {
        "doi_resolves": True,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"fake_venue_{new_entry.bibtex_key}"
    return new_entry


def generate_placeholder_authors(entry: BenchmarkEntry, rng: random.Random | None = None) -> BenchmarkEntry:
    """Tier 1: Replace authors with generic/fake names."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    fake_author_sets = [
        "John Doe and Jane Smith",
        "Alice Johnson and Bob Williams",
        "Test Author and Another Author",
        "A. Researcher and B. Scientist",
        "First Last and Second Person",
    ]
    new_entry.fields["author"] = rng.choice(fake_author_sets)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PLACEHOLDER_AUTHORS.value
    new_entry.difficulty_tier = DifficultyTier.EASY.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Authors are placeholders: {new_entry.fields['author']}"
    new_entry.subtests = {
        "doi_resolves": True,
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
    new_entry.subtests = {
        "doi_resolves": True,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": False,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"future_{new_entry.bibtex_key}"
    return new_entry


def generate_chimeric_title(
    entry: BenchmarkEntry,
    fake_title: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Keep real author but replace title with a fabricated one."""
    new_entry = _clone_entry(entry)
    new_entry.fields["title"] = fake_title
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.CHIMERIC_TITLE.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Title '{fake_title}' is fabricated; authors are real"
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"chimeric_{new_entry.bibtex_key}"
    # Remove DOI since the title changed
    new_entry.fields.pop("doi", None)
    return new_entry


def generate_wrong_venue(
    entry: BenchmarkEntry,
    wrong_venue: str,
    wrong_year: str | None = None,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Correct title/authors but wrong venue or year."""
    new_entry = _clone_entry(entry)
    venue_field = "booktitle" if "booktitle" in new_entry.fields else "journal"
    new_entry.fields[venue_field] = wrong_venue
    if wrong_year:
        new_entry.fields["year"] = wrong_year
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.WRONG_VENUE.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Venue changed to '{wrong_venue}' (original was different)"
    new_entry.subtests = {
        "doi_resolves": True,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"wrong_venue_{new_entry.bibtex_key}"
    return new_entry


def generate_swapped_authors(
    entry: BenchmarkEntry,
    donor_entry: BenchmarkEntry,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Correct title but authors from a different paper."""
    new_entry = _clone_entry(entry)
    new_entry.fields["author"] = donor_entry.fields.get("author", "Unknown Author")
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.SWAPPED_AUTHORS.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Authors swapped from '{donor_entry.bibtex_key}'"
    new_entry.subtests = {
        "doi_resolves": True,
        "title_exists": True,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"swapped_{new_entry.bibtex_key}"
    return new_entry


def generate_near_miss_title(
    entry: BenchmarkEntry,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 3: Title off by 1-2 words from the real paper."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)
    title = new_entry.fields.get("title", "")
    words = title.split()

    if len(words) >= 3:
        idx = rng.randint(0, len(words) - 1)
        replacements = {
            "need": "want",
            "all": "everything",
            "learning": "training",
            "deep": "hierarchical",
            "neural": "cognitive",
            "network": "architecture",
            "model": "framework",
            "attention": "focus",
            "optimal": "efficient",
            "robust": "resilient",
            "generative": "creative",
        }
        word_lower = words[idx].lower().rstrip(".,;:!?")
        if word_lower in replacements:
            words[idx] = replacements[word_lower]
        else:
            # Replace with a random synonym-ish word
            words[idx] = rng.choice(["Improved", "Enhanced", "Novel", "Adaptive", "Scalable"])

    new_title = " ".join(words)
    new_entry.fields["title"] = new_title
    new_entry.fields.pop("doi", None)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.NEAR_MISS_TITLE.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Title slightly modified: '{new_title}' (original: '{title}')"
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"near_miss_{new_entry.bibtex_key}"
    return new_entry


def generate_preprint_as_published(
    entry: BenchmarkEntry,
    fake_venue: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: arXiv preprint cited as if published at a venue."""
    new_entry = _clone_entry(entry)
    # Add a fake venue
    new_entry.fields["booktitle"] = fake_venue
    new_entry.bibtex_type = "inproceedings"
    # Remove arXiv indicators
    new_entry.fields.pop("eprint", None)
    new_entry.fields.pop("archiveprefix", None)
    new_entry.fields.pop("primaryclass", None)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PREPRINT_AS_PUBLISHED.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Preprint falsely cited as published at '{fake_venue}'"
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": True,
        "authors_match": True,
        "venue_real": False,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"preprint_pub_{new_entry.bibtex_key}"
    return new_entry


# --- Batch generation ---


TIER1_GENERATORS = [
    generate_fabricated_doi,
    generate_nonexistent_venue,
    generate_placeholder_authors,
    generate_future_date,
]


def generate_tier1_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 1 hallucinated entries from valid entries."""
    rng = random.Random(seed)
    results = []
    for i in range(count):
        source = rng.choice(valid_entries)
        generator = rng.choice(TIER1_GENERATORS)
        results.append(generator(source, rng))
    return results


def generate_tier2_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 2 hallucinated entries."""
    rng = random.Random(seed)
    results = []
    venues = [
        "NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR",
        "ECCV", "EMNLP", "AISTATS", "UAI",
    ]

    for i in range(count):
        source = rng.choice(valid_entries)
        method = rng.choice(["wrong_venue", "swapped_authors", "preprint_as_published"])

        if method == "wrong_venue":
            wrong_v = rng.choice(venues)
            results.append(generate_wrong_venue(source, wrong_v, rng=rng))
        elif method == "swapped_authors":
            donor = rng.choice(valid_entries)
            while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
                donor = rng.choice(valid_entries)
            results.append(generate_swapped_authors(source, donor, rng))
        elif method == "preprint_as_published":
            fake_v = rng.choice(venues)
            results.append(generate_preprint_as_published(source, fake_v, rng))

    return results


def generate_tier3_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 3 hallucinated entries (near-miss titles)."""
    rng = random.Random(seed)
    results = []
    for i in range(count):
        source = rng.choice(valid_entries)
        results.append(generate_near_miss_title(source, rng))
    return results


# --- Helpers ---


def _clone_entry(entry: BenchmarkEntry) -> BenchmarkEntry:
    """Deep clone a BenchmarkEntry."""
    return BenchmarkEntry(
        bibtex_key=entry.bibtex_key,
        bibtex_type=entry.bibtex_type,
        fields=copy.deepcopy(entry.fields),
        label=entry.label,
        hallucination_type=entry.hallucination_type,
        difficulty_tier=entry.difficulty_tier,
        explanation=entry.explanation,
        generation_method=entry.generation_method,
        source_conference=entry.source_conference,
        publication_date=entry.publication_date,
        added_to_benchmark=entry.added_to_benchmark,
        subtests=copy.deepcopy(entry.subtests),
        raw_bibtex=None,
    )
