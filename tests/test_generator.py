"""Tests for hallmark.dataset.generator."""

from hallmark.dataset.generator import (
    generate_hybrid_fabrication,
    generate_plausible_fabrication,
    generate_version_confusion,
)
from hallmark.dataset.schema import BenchmarkEntry


def _make_base_entry() -> BenchmarkEntry:
    """Create a base valid entry for testing generators."""
    return BenchmarkEntry(
        bibtex_key="test2024paper",
        bibtex_type="inproceedings",
        fields={
            "title": "Deep Learning for Computer Vision",
            "author": "John Smith and Jane Doe",
            "year": "2024",
            "booktitle": "NeurIPS",
            "doi": "10.1234/test.doi.2024",
        },
        label="VALID",
        explanation="Valid test entry",
        generation_method="scraped",
        publication_date="2024-01-15",
        added_to_benchmark="2026-01-15",
        subtests={
            "doi_resolves": True,
            "title_exists": True,
            "authors_match": True,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": True,
        },
    )


class TestGeneratePlausibleFabrication:
    def test_creates_hallucinated_entry(self):
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert result.label == "HALLUCINATED"

    def test_hallucination_type(self):
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert result.hallucination_type == "plausible_fabrication"

    def test_difficulty_tier_is_hard(self):
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert result.difficulty_tier == 3

    def test_doi_removed(self):
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert "doi" not in result.fields

    def test_venue_real_subtest_true(self):
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert result.subtests["venue_real"] is True
        assert result.subtests["doi_resolves"] is False
        assert result.subtests["title_exists"] is False


class TestGenerateHybridFabrication:
    def test_creates_hallucinated_entry(self):
        entry = _make_base_entry()
        result = generate_hybrid_fabrication(entry)
        assert result.label == "HALLUCINATED"

    def test_hallucination_type(self):
        entry = _make_base_entry()
        result = generate_hybrid_fabrication(entry)
        assert result.hallucination_type == "hybrid_fabrication"

    def test_doi_preserved(self):
        entry = _make_base_entry()
        result = generate_hybrid_fabrication(entry)
        assert "doi" in result.fields
        assert result.fields["doi"] == entry.fields["doi"]

    def test_doi_resolves_subtest_true(self):
        entry = _make_base_entry()
        result = generate_hybrid_fabrication(entry)
        assert result.subtests["doi_resolves"] is True

    def test_authors_match_subtest_false(self):
        entry = _make_base_entry()
        result = generate_hybrid_fabrication(entry)
        assert result.subtests["authors_match"] is False

    def test_difficulty_tier_is_medium(self):
        entry = _make_base_entry()
        result = generate_hybrid_fabrication(entry)
        assert result.difficulty_tier == 2


class TestGenerateVersionConfusion:
    def test_creates_hallucinated_entry(self):
        entry = _make_base_entry()
        result = generate_version_confusion(entry, "ICML")
        assert result.label == "HALLUCINATED"

    def test_hallucination_type(self):
        entry = _make_base_entry()
        result = generate_version_confusion(entry, "ICML")
        assert result.hallucination_type == "version_confusion"

    def test_eprint_field_not_set(self):
        """Per P0.3: eprint/archiveprefix are hallucination-only fields and should not be generated."""
        entry = _make_base_entry()
        result = generate_version_confusion(entry, "ICML")
        assert result.fields.get("eprint") is None
        assert result.fields.get("archiveprefix") is None

    def test_booktitle_field_set(self):
        entry = _make_base_entry()
        result = generate_version_confusion(entry, "ICML")
        assert result.fields.get("booktitle") == "ICML"

    def test_year_shifted(self):
        entry = _make_base_entry()
        result = generate_version_confusion(entry, "ICML")
        original_year = int(entry.fields["year"])
        result_year = int(result.fields["year"])
        assert abs(result_year - original_year) == 1

    def test_difficulty_tier_is_hard(self):
        entry = _make_base_entry()
        result = generate_version_confusion(entry, "ICML")
        assert result.difficulty_tier == 3


class TestBackwardCompatibility:
    def test_swapped_authors_value_compatibility(self):
        """Test that entries with hallucination_type='swapped_authors' still work."""
        entry = BenchmarkEntry(
            bibtex_key="test",
            bibtex_type="article",
            fields={"title": "Test", "author": "A", "year": "2024"},
            label="HALLUCINATED",
            hallucination_type="swapped_authors",
            difficulty_tier=2,
        )
        assert entry.hallucination_type == "swapped_authors"
        assert entry.label == "HALLUCINATED"
