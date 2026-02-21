from __future__ import annotations

import random
from collections.abc import Callable

from hallmark.dataset.schema import BenchmarkEntry

from ._pools import ML_BUZZWORDS, REAL_VENUES
from .tier1 import (
    generate_fabricated_doi,
    generate_future_date,
    generate_nonexistent_venue,
    generate_placeholder_authors,
)
from .tier2 import (
    generate_chimeric_title,
    generate_hybrid_fabrication,
    generate_merged_citation,
    generate_partial_author_list,
    generate_preprint_as_published,
    generate_swapped_authors,
    generate_wrong_venue,
)
from .tier3 import (
    generate_arxiv_version_mismatch,
    generate_near_miss_title,
    generate_plausible_fabrication,
)

TIER1_GENERATORS: list[Callable[..., BenchmarkEntry]] = [
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

    for _i in range(count):
        source = rng.choice(valid_entries)
        method = rng.choice(
            [
                "wrong_venue",
                "swapped_authors",
                "preprint_as_published",
                "chimeric_title",
                "hybrid_fabrication",
                "merged_citation",
                "partial_author_list",
            ]
        )

        if method == "wrong_venue":
            wrong_v = rng.choice(REAL_VENUES)
            results.append(generate_wrong_venue(source, wrong_v, rng=rng))
        elif method == "swapped_authors":
            donor = rng.choice(valid_entries)
            while donor.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
                donor = rng.choice(valid_entries)
            results.append(generate_swapped_authors(source, donor, rng))
        elif method == "preprint_as_published":
            fake_v = rng.choice(REAL_VENUES)
            results.append(generate_preprint_as_published(source, fake_v, rng))
        elif method == "chimeric_title":
            buzzword = rng.choice(ML_BUZZWORDS)
            fake_title = (
                f"{buzzword} for {rng.choice(['Classification', 'Generation', 'Reasoning'])}"
            )
            results.append(generate_chimeric_title(source, fake_title, rng))
        elif method == "hybrid_fabrication":
            results.append(generate_hybrid_fabrication(source, rng))
        elif method == "merged_citation":
            donor_b = rng.choice(valid_entries)
            while donor_b.bibtex_key == source.bibtex_key and len(valid_entries) > 1:
                donor_b = rng.choice(valid_entries)
            donor_c = rng.choice(valid_entries) if rng.random() < 0.5 else None
            results.append(generate_merged_citation(source, donor_b, donor_c, rng))
        elif method == "partial_author_list":
            results.append(generate_partial_author_list(source, rng))

    return results


def generate_tier3_batch(
    valid_entries: list[BenchmarkEntry],
    count: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Generate a batch of Tier 3 hallucinated entries."""
    rng = random.Random(seed)
    results = []

    # Conference venues and years for arxiv_version_mismatch
    conferences = [
        ("NeurIPS", "2023"),
        ("ICML", "2023"),
        ("ICLR", "2024"),
        ("AAAI", "2024"),
        ("CVPR", "2023"),
    ]

    for _i in range(count):
        source = rng.choice(valid_entries)
        # Randomly choose between near_miss_title, plausible_fabrication, and arxiv_version_mismatch
        method = rng.choice(["near_miss_title", "plausible_fabrication", "arxiv_version_mismatch"])

        if method == "near_miss_title":
            results.append(generate_near_miss_title(source, rng))
        elif method == "plausible_fabrication":
            results.append(generate_plausible_fabrication(source, rng))
        elif method == "arxiv_version_mismatch":
            conf_venue, _conf_year = rng.choice(conferences)
            results.append(generate_arxiv_version_mismatch(source, conf_venue, rng))

    return results
