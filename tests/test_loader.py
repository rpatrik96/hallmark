"""Tests for hallmark.dataset.loader."""

from __future__ import annotations

import warnings

import pytest

from hallmark.dataset.loader import (
    SPLIT_PATHS,
    filter_by_date_range,
    filter_by_tier,
    filter_by_type,
    get_statistics,
    load_split,
)
from hallmark.dataset.schema import BenchmarkEntry


@pytest.fixture(scope="module")
def dev_entries() -> list[BenchmarkEntry]:
    return load_split("dev_public")


def test_split_paths_keys() -> None:
    assert set(SPLIT_PATHS.keys()) == {"dev_public", "test_public", "stress_test", "test_hidden"}


def test_load_split_returns_nonempty_list(dev_entries: list[BenchmarkEntry]) -> None:
    assert len(dev_entries) > 0
    assert all(isinstance(e, BenchmarkEntry) for e in dev_entries)


def test_filter_by_tier_returns_only_tier1(dev_entries: list[BenchmarkEntry]) -> None:
    tier1 = filter_by_tier(dev_entries, tier=1)
    # All hallucinated entries must be tier 1; valid entries pass through
    for e in tier1:
        if e.label == "HALLUCINATED":
            assert e.difficulty_tier == 1


def test_filter_by_type_fabricated_doi(dev_entries: list[BenchmarkEntry]) -> None:
    filtered = filter_by_type(dev_entries, "fabricated_doi")
    for e in filtered:
        if e.label == "HALLUCINATED":
            assert e.hallucination_type == "fabricated_doi"


def test_filter_by_date_range_no_bounds_returns_all(dev_entries: list[BenchmarkEntry]) -> None:
    result = filter_by_date_range(dev_entries)
    assert len(result) == len(dev_entries)


def test_filter_by_date_range_with_start_date(dev_entries: list[BenchmarkEntry]) -> None:
    result = filter_by_date_range(dev_entries, start_date="2024")
    # Every returned entry must have a publication date >= 2024
    for e in result:
        assert e.publication_date is not None
        assert e.publication_date >= "2024"
    # Result should be a strict subset (dev split contains pre-2024 entries)
    assert len(result) < len(dev_entries)


def test_get_statistics_keys(dev_entries: list[BenchmarkEntry]) -> None:
    stats = get_statistics(dev_entries)
    expected_keys = {
        "total",
        "valid",
        "hallucinated",
        "hallucination_rate",
        "tier_distribution",
        "type_distribution",
        "method_distribution",
        "venue_distribution",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["total"] == len(dev_entries)
    assert stats["valid"] + stats["hallucinated"] == stats["total"]


def test_filter_by_type_nonexistent_emits_warning(dev_entries: list[BenchmarkEntry]) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = filter_by_type(dev_entries, "nonexistent_type_xyz")
    # Only VALID entries returned — all hallucinated entries are filtered out
    hallucinated_returned = [e for e in result if e.label == "HALLUCINATED"]
    assert hallucinated_returned == []
    # A warning must have been emitted
    assert any("nonexistent_type_xyz" in str(w.message) for w in caught)
