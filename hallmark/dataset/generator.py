"""Generate hallucinated BibTeX entries for HALLMARK.

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

from hallmark.dataset.schema import (
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
)


def generate_fabricated_doi(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
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


def generate_nonexistent_venue(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
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


def generate_placeholder_authors(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
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
    new_entry.hallucination_type = HallucinationType.AUTHOR_MISMATCH.value
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


def generate_plausible_fabrication(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 3: Fabricate a realistic but non-existent paper at a real prestigious venue."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    # ML buzzwords for plausible-sounding titles
    buzzwords = [
        "Attention",
        "Transformer",
        "Self-Supervised",
        "Contrastive",
        "Meta-Learning",
        "Few-Shot",
        "Multi-Modal",
        "Reinforcement",
        "Diffusion",
        "Retrieval-Augmented",
        "Neural",
        "Deep",
        "Efficient",
        "Scalable",
        "Adaptive",
    ]
    tasks = [
        "Classification",
        "Generation",
        "Reasoning",
        "Understanding",
        "Translation",
        "Segmentation",
        "Detection",
        "Retrieval",
        "Synthesis",
        "Alignment",
    ]

    # Generate plausible title combining buzzwords
    title_template = rng.choice(
        [
            f"{rng.choice(buzzwords)} {rng.choice(buzzwords)} for {rng.choice(tasks)}",
            f"{rng.choice(buzzwords)}-Based {rng.choice(tasks)}",
            f"Learning {rng.choice(tasks)} via {rng.choice(buzzwords)} Methods",
            f"{rng.choice(buzzwords)} Approaches to {rng.choice(tasks)}",
        ]
    )
    new_entry.fields["title"] = title_template

    # Fabricated but realistic author names
    author_sets = [
        "Wei Zhang and Sarah Chen and Marcus Johnson",
        "Yuki Tanaka and Elena Rodriguez and James Kim",
        "Sofia Andersson and Ravi Patel and Maria Santos",
        "Alex Wu and Emma Thompson and David Lee",
        "Nina Kowalski and Omar Hassan and Lisa Wang",
    ]
    new_entry.fields["author"] = rng.choice(author_sets)

    # Keep a real prestigious venue
    real_venues = ["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR"]
    venue_field = "booktitle" if new_entry.bibtex_type == "inproceedings" else "journal"
    new_entry.fields[venue_field] = rng.choice(real_venues)

    # Remove DOI (fabricated paper won't have one)
    new_entry.fields.pop("doi", None)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.PLAUSIBLE_FABRICATION.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.ADVERSARIAL.value
    new_entry.explanation = "Completely fabricated paper with plausible metadata at real venue"
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": False,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"plausible_{new_entry.bibtex_key}"
    return new_entry


def generate_retracted_paper(
    entry: BenchmarkEntry,
    retracted_doi: str,
    retracted_title: str,
    retracted_authors: str,
    retracted_venue: str,
    retracted_year: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 3: Create entry citing a real but retracted paper."""
    new_entry = _clone_entry(entry)

    # Replace all fields with the retracted paper's metadata
    new_entry.fields["doi"] = retracted_doi
    new_entry.fields["title"] = retracted_title
    new_entry.fields["author"] = retracted_authors
    new_entry.fields["year"] = retracted_year

    venue_field = "booktitle" if new_entry.bibtex_type == "inproceedings" else "journal"
    new_entry.fields[venue_field] = retracted_venue

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.RETRACTED_PAPER.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.REAL_WORLD.value
    new_entry.explanation = f"Paper '{retracted_title}' was retracted after publication"
    new_entry.subtests = {
        "doi_resolves": True,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": True,
    }
    new_entry.bibtex_key = f"retracted_{new_entry.bibtex_key}"
    return new_entry


def generate_version_confusion(
    entry: BenchmarkEntry,
    arxiv_id: str,
    conference_venue: str,
    conference_year: str,
    rng: random.Random | None = None,
) -> BenchmarkEntry:
    """Tier 3: Mix arXiv preprint metadata with conference publication metadata."""
    new_entry = _clone_entry(entry)

    # Keep the title from the original entry
    # Set eprint field to arxiv_id (arXiv metadata)
    new_entry.fields["eprint"] = arxiv_id
    new_entry.fields["archiveprefix"] = "arXiv"

    # But claim it was published at a conference (venue metadata)
    new_entry.fields["booktitle"] = conference_venue
    new_entry.fields["year"] = conference_year
    new_entry.bibtex_type = "inproceedings"

    # Remove DOI since this creates version confusion
    new_entry.fields.pop("doi", None)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.VERSION_CONFUSION.value
    new_entry.difficulty_tier = DifficultyTier.HARD.value
    new_entry.generation_method = GenerationMethod.PERTURBATION.value
    new_entry.explanation = (
        f"arXiv preprint {arxiv_id} cited with conference venue {conference_venue}"
    )
    new_entry.subtests = {
        "doi_resolves": False,
        "title_exists": True,
        "authors_match": True,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"version_{new_entry.bibtex_key}"
    return new_entry


def generate_hybrid_fabrication(
    entry: BenchmarkEntry, rng: random.Random | None = None
) -> BenchmarkEntry:
    """Tier 2: Keep real DOI but fabricate authors and slightly modify title."""
    rng = rng or random.Random()
    new_entry = _clone_entry(entry)

    # Keep the DOI from the original entry (it will still resolve)
    # Replace authors with fabricated names
    fake_author_sets = [
        "Michael Zhang and Jennifer Liu and Robert Chen",
        "Anna Petrov and Carlos Martinez and Yuki Nakamura",
        "Thomas Anderson and Sophia Kumar and Daniel Park",
        "Isabella Rossi and Ahmed Ali and Emma Williams",
        "Lucas Brown and Maria Garcia and Kevin Nguyen",
    ]
    new_entry.fields["author"] = rng.choice(fake_author_sets)

    # Slightly modify the title (swap 2-3 words)
    title = new_entry.fields.get("title", "")
    words = title.split()
    if len(words) >= 4:
        # Swap 2-3 words with synonyms or related terms
        num_swaps = min(rng.randint(2, 3), len(words))
        swap_words = {
            "learning": "training",
            "neural": "deep",
            "network": "model",
            "efficient": "optimized",
            "robust": "stable",
            "novel": "new",
            "approach": "method",
            "framework": "system",
            "attention": "focus",
            "transformer": "architecture",
        }
        for _ in range(num_swaps):
            idx = rng.randint(0, len(words) - 1)
            word_lower = words[idx].lower().rstrip(".,;:!?")
            if word_lower in swap_words:
                words[idx] = swap_words[word_lower]
            elif rng.random() < 0.5:
                words[idx] = rng.choice(["Enhanced", "Improved", "Advanced", "Novel", "Efficient"])
        new_entry.fields["title"] = " ".join(words)

    new_entry.label = "HALLUCINATED"
    new_entry.hallucination_type = HallucinationType.HYBRID_FABRICATION.value
    new_entry.difficulty_tier = DifficultyTier.MEDIUM.value
    new_entry.generation_method = GenerationMethod.ADVERSARIAL.value
    new_entry.explanation = "Real DOI with fabricated authors and modified title"
    new_entry.subtests = {
        "doi_resolves": True,
        "title_exists": False,
        "authors_match": False,
        "venue_real": True,
        "fields_complete": True,
        "cross_db_agreement": False,
    }
    new_entry.bibtex_key = f"hybrid_{new_entry.bibtex_key}"
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
    for _i in range(count):
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
        "NeurIPS",
        "ICML",
        "ICLR",
        "AAAI",
        "ACL",
        "CVPR",
        "ECCV",
        "EMNLP",
        "AISTATS",
        "UAI",
    ]

    # ML buzzwords for chimeric_title
    ml_buzzwords = [
        "Attention",
        "Transformer",
        "Self-Supervised",
        "Contrastive",
        "Few-Shot",
        "Multi-Modal",
        "Reinforcement",
        "Diffusion",
        "Neural Architecture Search",
        "Meta-Learning",
    ]

    for _i in range(count):
        source = rng.choice(valid_entries)
        method = rng.choice(
            [
                "wrong_venue",
                "swapped_authors",
                "preprint_as_published",
                "chimeric_title",
                "hybrid_fabrication",
            ]
        )

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
        elif method == "chimeric_title":
            # Generate a fake title using ML buzzwords
            fake_title = f"{rng.choice(ml_buzzwords)} for {rng.choice(['Classification', 'Generation', 'Reasoning'])}"
            results.append(generate_chimeric_title(source, fake_title, rng))
        elif method == "hybrid_fabrication":
            results.append(generate_hybrid_fabrication(source, rng))

    return results


def generate_tier3_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 3 hallucinated entries."""
    rng = random.Random(seed)
    results = []

    # Conference venues and years for version_confusion
    conferences = [
        ("NeurIPS", "2023"),
        ("ICML", "2023"),
        ("ICLR", "2024"),
        ("AAAI", "2024"),
        ("CVPR", "2023"),
    ]

    for _i in range(count):
        source = rng.choice(valid_entries)
        # Randomly choose between near_miss_title, plausible_fabrication, and version_confusion
        # Skip retracted_paper since it needs external data
        method = rng.choice(["near_miss_title", "plausible_fabrication", "version_confusion"])

        if method == "near_miss_title":
            results.append(generate_near_miss_title(source, rng))
        elif method == "plausible_fabrication":
            results.append(generate_plausible_fabrication(source, rng))
        elif method == "version_confusion":
            # Generate a plausible arXiv ID
            arxiv_id = f"{rng.randint(2001, 2312)}.{rng.randint(10000, 99999)}"
            conf_venue, conf_year = rng.choice(conferences)
            results.append(generate_version_confusion(source, arxiv_id, conf_venue, conf_year, rng))

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
