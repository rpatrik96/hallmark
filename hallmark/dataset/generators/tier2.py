from __future__ import annotations

import random

from hallmark.dataset.schema import (
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)

from ._helpers import _clone_entry
from ._pools import HYBRID_FAKE_AUTHORS, HYBRID_SWAP_WORDS


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
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,  # DOI still resolves to original paper
        "title_exists": False,
        "authors_match": True,
        "venue_correct": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"chimeric_{new_entry.bibtex_key}"
    return new_entry


def generate_wrong_venue(
    entry: BenchmarkEntry,
    wrong_venue: str,
    wrong_year: str | None = None,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Correct title/authors but wrong venue or year.

    Note: Always uses booktitle (normalized to inproceedings per P0.2).
    """
    new_entry = _clone_entry(entry)
    # Always use booktitle (normalized to inproceedings per P0.2)
    new_entry.fields["booktitle"] = wrong_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"
    if wrong_year:
        new_entry.fields["year"] = wrong_year
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.WRONG_VENUE.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Venue changed to '{wrong_venue}' (original was different)"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": True,
        "venue_correct": False,
        "fields_complete": True,
        "cross_db_agreement": False,  # venue mismatch causes cross-DB disagreement
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
    new_entry.hallucination_type = HallucinationType.AUTHOR_MISMATCH.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Authors swapped from '{donor_entry.bibtex_key}'"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": True,
        "authors_match": False,
        "venue_correct": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"swapped_{new_entry.bibtex_key}"
    return new_entry


def generate_preprint_as_published(
    entry: BenchmarkEntry,
    fake_venue: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: arXiv preprint cited as if published at a venue.

    Note: source entry should ideally be a genuine preprint (use is_preprint_source()
    to filter). If the source is already a conference paper, the result is effectively
    a wrong_venue entry.

    Per P0.3: Does NOT add eprint/archiveprefix fields (stripped as hallucination-only).
    """
    new_entry = _clone_entry(entry)
    # Add a fake venue
    new_entry.fields["booktitle"] = fake_venue
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"
    # Remove arXiv indicators (P0.3: these are hallucination-only fields)
    new_entry.fields.pop("eprint", None)
    new_entry.fields.pop("archiveprefix", None)
    new_entry.fields.pop("primaryclass", None)
    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PREPRINT_AS_PUBLISHED.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = f"Preprint falsely cited as published at '{fake_venue}'"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,  # real paper DOI resolves; N/A if no DOI
        "title_exists": True,
        "authors_match": True,
        "venue_correct": False,
        "fields_complete": True,
        "cross_db_agreement": False,  # fabricated venue causes cross-DB disagreement
    }
    new_entry.bibtex_key = f"preprint_pub_{new_entry.bibtex_key}"
    return new_entry


def generate_merged_citation(
    entry_a: BenchmarkEntry,
    entry_b: BenchmarkEntry,
    entry_c: BenchmarkEntry | None = None,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Merge metadata from 2-3 real papers into one BibTeX entry.

    Takes authors from entry_a, title from entry_b, venue from entry_c (or entry_a).
    The resulting entry looks plausible but no single real paper matches all fields.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry_b)  # start from entry_b (title source)

    # Authors from entry_a
    new_entry.fields["author"] = entry_a.fields.get("author", "")

    # Title from entry_b (already cloned)

    # Venue from entry_c if provided, else entry_a
    venue_source = entry_c if entry_c is not None else entry_a
    # Always use booktitle (normalized to inproceedings per P0.2)
    for vf in ("booktitle", "journal"):
        if vf in venue_source.fields:
            new_entry.fields["booktitle"] = venue_source.fields[vf]
            break
    new_entry.fields.pop("journal", None)  # Remove journal if present
    new_entry.bibtex_type = "inproceedings"

    # Year from venue source
    if "year" in venue_source.fields:
        new_entry.fields["year"] = venue_source.fields["year"]

    # Use entry_b's DOI if available (title matches but authors won't)
    if entry_b.doi:
        new_entry.fields["doi"] = entry_b.doi
    else:
        new_entry.fields.pop("doi", None)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.MERGED_CITATION.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = (
        "Metadata merged from multiple real papers: "
        f"authors from '{entry_a.bibtex_key}', title from '{entry_b.bibtex_key}'"
    )
    new_entry.subtests = {
        "doi_resolves": entry_b.doi is not None,
        "title_exists": True,
        "authors_match": False,
        # Venue comes from entry_c (or entry_a), not from entry_b whose title is used.
        # The merged entry attributes entry_b's title to a different paper's venue, so
        # the venue does not match what entry_b was actually published at.
        "venue_correct": False,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"merged_{entry_b.bibtex_key}"
    return new_entry


def generate_partial_author_list(
    entry: BenchmarkEntry,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 2: Cite a real paper with a subset of its authors.

    Keeps first and last author, drops middle co-authors. Common in LLM outputs
    when the model remembers only prominent authors.

    Raises:
        ValueError: If the entry has fewer than 2 authors and cannot produce a
            genuinely different partial list. Callers should skip such entries.
    """
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    authors_str = entry.fields.get("author", "")
    # BibTeX uses " and " as separator
    authors = [a.strip() for a in authors_str.split(" and ") if a.strip()]

    if len(authors) >= 3:
        # Keep first + last, drop random middle authors
        first = authors[0]
        last = authors[-1]
        middle = authors[1:-1]
        # Keep at most 1 middle author (randomly selected) for some variation
        kept_middle = [rng.choice(middle)] if middle and rng.random() < 0.3 else []
        new_authors = [first, *kept_middle, last]
        new_entry.fields["author"] = " and ".join(new_authors)
    elif len(authors) == 2:
        # Drop one author randomly
        new_entry.fields["author"] = authors[rng.randint(0, 1)]
    else:
        # Single author (or empty) — cannot produce a meaningfully different partial
        # list without fabricating data. Skip these entries at the call site.
        raise ValueError(
            f"Cannot generate partial_author_list for entry '{entry.bibtex_key}': "
            f"found {len(authors)} author(s); need at least 2 to drop one."
        )

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PARTIAL_AUTHOR_LIST.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = (
        f"Partial author list: {len(authors)} authors reduced to "
        f"{len(new_entry.fields.get('author', '').split(' and '))}"
    )
    new_entry.subtests = {
        "doi_resolves": entry.doi is not None,
        "title_exists": True,
        "authors_match": False,  # partial list doesn't fully match
        "venue_correct": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"partial_authors_{new_entry.bibtex_key}"
    return new_entry


def generate_hybrid_fabrication(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 2: Keep real DOI but fabricate authors and slightly modify title."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    # Keep the DOI from the original entry (it will still resolve)
    # Replace authors with fabricated names
    new_entry.fields["author"] = rng.choice(HYBRID_FAKE_AUTHORS)

    # Slightly modify the title (swap 2-3 words)
    title = new_entry.fields.get("title", "")
    words = title.split()
    if len(words) >= 4:
        # Swap 2-3 words with synonyms or related terms
        num_swaps = min(rng.randint(2, 3), len(words))
        for _ in range(num_swaps):
            idx = rng.randint(0, len(words) - 1)
            word_lower = words[idx].lower().rstrip(".,;:!?")
            if word_lower in HYBRID_SWAP_WORDS:
                words[idx] = HYBRID_SWAP_WORDS[word_lower]
            elif rng.random() < 0.5:
                words[idx] = rng.choice(["Enhanced", "Improved", "Advanced", "Novel", "Efficient"])
        modified_title = " ".join(words)
        # Guarantee the title actually changed; the swap dict is small and the
        # per-attempt fallback fires with p=0.5, so (0.5)^num_swaps ≈ 12.5% chance
        # of zero modification without this guard.
        if modified_title == title:
            modified_title = "Improved " + title
        new_entry.fields["title"] = modified_title
    else:
        # Short titles: prepend modifier to guarantee title change
        new_entry.fields["title"] = "Improved " + title

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.HYBRID_FABRICATION.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.ADVERSARIAL.value
    new_entry.explanation = "Real DOI with fabricated authors and modified title"
    has_doi = bool(new_entry.fields.get("doi"))
    new_entry.subtests = {
        "doi_resolves": True if has_doi else None,
        "title_exists": False,
        "authors_match": False,
        "venue_correct": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"hybrid_{new_entry.bibtex_key}"
    return new_entry
