"""Dataset loading and filtering for CiteBench benchmark splits."""

from __future__ import annotations

import json
from pathlib import Path

from citebench.dataset.schema import BenchmarkEntry, DifficultyTier, load_entries

# Default data directory (relative to package root)
_PACKAGE_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = _PACKAGE_DIR / "data"


def load_split(
    split: str = "dev_public",
    version: str = "v1.0",
    data_dir: str | Path | None = None,
) -> list[BenchmarkEntry]:
    """Load a benchmark split.

    Args:
        split: One of "dev_public", "test_public", "test_hidden".
        version: Dataset version (e.g., "v1.0").
        data_dir: Override data directory. Defaults to data/ in package root.

    Returns:
        List of BenchmarkEntry objects.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)

    if split == "test_hidden":
        path = data_dir / "hidden" / "test_hidden.jsonl"
    else:
        path = data_dir / version / f"{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    return load_entries(path)


def load_metadata(
    version: str = "v1.0",
    data_dir: str | Path | None = None,
) -> dict:
    """Load metadata for a benchmark version."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    path = data_dir / version / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path) as f:
        return json.load(f)


def filter_by_tier(
    entries: list[BenchmarkEntry],
    tier: int | DifficultyTier,
) -> list[BenchmarkEntry]:
    """Filter entries to a specific difficulty tier."""
    tier_val = tier.value if isinstance(tier, DifficultyTier) else tier
    return [e for e in entries if e.difficulty_tier == tier_val or e.label == "VALID"]


def filter_by_type(
    entries: list[BenchmarkEntry],
    hallucination_type: str,
) -> list[BenchmarkEntry]:
    """Filter entries to a specific hallucination type (plus all valid entries)."""
    return [
        e
        for e in entries
        if e.hallucination_type == hallucination_type or e.label == "VALID"
    ]


def filter_by_date_range(
    entries: list[BenchmarkEntry],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[BenchmarkEntry]:
    """Filter entries by publication date range (ISO format YYYY-MM-DD)."""
    result = []
    for e in entries:
        if not e.publication_date:
            continue
        if start_date and e.publication_date < start_date:
            continue
        if end_date and e.publication_date > end_date:
            continue
        result.append(e)
    return result


def filter_by_venue(
    entries: list[BenchmarkEntry],
    venue: str,
) -> list[BenchmarkEntry]:
    """Filter entries by source conference/venue."""
    venue_lower = venue.lower()
    return [
        e
        for e in entries
        if e.source_conference and venue_lower in e.source_conference.lower()
    ]


def filter_by_bibtex_type(
    entries: list[BenchmarkEntry],
    bibtex_type: str,
) -> list[BenchmarkEntry]:
    """Filter entries by BibTeX entry type."""
    return [e for e in entries if e.bibtex_type.lower() == bibtex_type.lower()]


def get_statistics(entries: list[BenchmarkEntry]) -> dict:
    """Compute summary statistics for a set of entries."""
    total = len(entries)
    valid = sum(1 for e in entries if e.label == "VALID")
    hallucinated = total - valid

    tier_counts = {1: 0, 2: 0, 3: 0}
    type_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    venue_counts: dict[str, int] = {}

    for e in entries:
        if e.difficulty_tier:
            tier_counts[e.difficulty_tier] = tier_counts.get(e.difficulty_tier, 0) + 1
        if e.hallucination_type:
            type_counts[e.hallucination_type] = type_counts.get(e.hallucination_type, 0) + 1
        method_counts[e.generation_method] = method_counts.get(e.generation_method, 0) + 1
        if e.source_conference:
            venue_counts[e.source_conference] = venue_counts.get(e.source_conference, 0) + 1

    return {
        "total": total,
        "valid": valid,
        "hallucinated": hallucinated,
        "hallucination_rate": hallucinated / total if total > 0 else 0.0,
        "tier_distribution": tier_counts,
        "type_distribution": type_counts,
        "method_distribution": method_counts,
        "venue_distribution": venue_counts,
    }
