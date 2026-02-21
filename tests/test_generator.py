"""Tests for hallmark.dataset.generator."""

import random

import pytest

from hallmark.dataset.generator import (
    FAKE_DOI_PREFIXES,
    generate_arxiv_version_mismatch,
    generate_chimeric_title,
    generate_hybrid_fabrication,
    generate_near_miss_title,
    generate_plausible_fabrication,
)
from hallmark.dataset.schema import BenchmarkEntry

# Known real DOI registrant prefixes — none of these should appear in FAKE_DOI_PREFIXES.
_KNOWN_REAL_PREFIXES = {
    "10.48550",  # arXiv
    "10.1145",  # ACM
    "10.1109",  # IEEE
    "10.1007",  # Springer
    "10.1038",  # Nature
    "10.1016",  # Elsevier
    "10.32614",  # CRAN / R
    "10.15439",  # ACSIS
    "10.21033",  # various
    "10.60715",  # various
    "10.47281",  # various
    "10.93105",  # various
    "10.82004",  # various
    "10.55910",  # various
    "10.71336",  # various
}


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
            "venue_correct": True,
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

    def test_venue_correct_subtest_true(self):
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert result.subtests["venue_correct"] is True
        assert result.subtests["doi_resolves"] is None  # DOI removed; not applicable
        assert result.subtests["title_exists"] is False

    def test_fields_complete_is_false(self):
        """Issue 11: has_identifier is always False (DOI removed, URL stripped by _clone_entry).

        fields_complete reflects has_identifier, so it must be False.
        """
        entry = _make_base_entry()
        result = generate_plausible_fabrication(entry)
        assert result.subtests["fields_complete"] is False


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

    def test_title_always_modified(self):
        """Issue 3: title must always differ from the original (100 seeds)."""
        base_title = "Deep Learning for Computer Vision"
        for seed in range(100):
            entry = _make_base_entry()
            rng = random.Random(seed)
            result = generate_hybrid_fabrication(entry, rng=rng)
            assert result.fields["title"] != base_title, (
                f"Title unchanged at seed {seed}: '{result.fields['title']}'"
            )


class TestGenerateVersionConfusion:
    def test_creates_hallucinated_entry(self):
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.label == "HALLUCINATED"

    def test_hallucination_type(self):
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.hallucination_type == "arxiv_version_mismatch"

    def test_eprint_field_not_set(self):
        """Per P0.3: eprint/archiveprefix are hallucination-only fields and should not be generated."""
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.fields.get("eprint") is None
        assert result.fields.get("archiveprefix") is None

    def test_booktitle_field_set(self):
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.fields.get("booktitle") == "ICML"

    def test_year_shifted(self):
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        original_year = int(entry.fields["year"])
        result_year = int(result.fields["year"])
        assert abs(result_year - original_year) == 1

    def test_difficulty_tier_is_hard(self):
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.difficulty_tier == 3

    def test_doi_preserved(self):
        """DOI should be kept since it points to the real paper."""
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.fields.get("doi") == entry.fields["doi"]

    def test_subtests_correct(self):
        """Verify subtests reflect DOI resolving but metadata mismatch."""
        entry = _make_base_entry()
        result = generate_arxiv_version_mismatch(entry, "ICML")
        assert result.subtests["doi_resolves"] is True
        assert result.subtests["cross_db_agreement"] is False


class TestGenerateChimericTitle:
    def test_creates_hallucinated_entry(self):
        entry = _make_base_entry()
        result = generate_chimeric_title(entry, "Fake Title for Testing")
        assert result.label == "HALLUCINATED"

    def test_hallucination_type(self):
        entry = _make_base_entry()
        result = generate_chimeric_title(entry, "Fake Title for Testing")
        assert result.hallucination_type == "chimeric_title"

    def test_doi_preserved(self):
        """DOI should be kept since it points to the real paper with real authors."""
        entry = _make_base_entry()
        result = generate_chimeric_title(entry, "Fake Title for Testing")
        assert result.fields.get("doi") == entry.fields["doi"]

    def test_subtests_correct(self):
        """DOI resolves but title doesn't match."""
        entry = _make_base_entry()
        result = generate_chimeric_title(entry, "Fake Title for Testing")
        assert result.subtests["doi_resolves"] is True
        assert result.subtests["title_exists"] is False
        assert result.subtests["authors_match"] is True

    def test_fields_complete_with_doi(self):
        """fields_complete should be True when DOI is present."""
        entry = _make_base_entry()
        result = generate_chimeric_title(entry, "Fake Title for Testing")
        assert result.subtests["fields_complete"] is True

    def test_fields_complete_without_doi(self):
        """fields_complete is True per taxonomy (F=checkmark) regardless of DOI presence."""
        entry = _make_base_entry()
        entry.fields.pop("doi")
        entry.fields["url"] = "https://example.com"
        result = generate_chimeric_title(entry, "Fake Title for Testing")
        # Taxonomy: chimeric_title always has F=checkmark (fields present and well-formed)
        assert result.subtests["fields_complete"] is True


class TestGenerateNearMissTitle:
    def test_creates_hallucinated_entry(self):
        entry = _make_base_entry()
        result = generate_near_miss_title(entry)
        assert result.label == "HALLUCINATED"

    def test_hallucination_type(self):
        entry = _make_base_entry()
        result = generate_near_miss_title(entry)
        assert result.hallucination_type == "near_miss_title"

    def test_doi_preserved(self):
        """DOI should be kept since it resolves to the original paper."""
        entry = _make_base_entry()
        result = generate_near_miss_title(entry)
        assert result.fields.get("doi") == entry.fields["doi"]

    def test_subtests_correct(self):
        """DOI resolves to original paper but title is slightly off."""
        entry = _make_base_entry()
        result = generate_near_miss_title(entry)
        assert result.subtests["doi_resolves"] is True
        assert result.subtests["title_exists"] is False
        assert result.subtests["authors_match"] is True

    def test_title_differs(self):
        """Title should be modified from original."""
        entry = _make_base_entry()
        result = generate_near_miss_title(entry)
        assert result.fields["title"] != entry.fields["title"]

    def test_difficulty_tier_is_hard(self):
        entry = _make_base_entry()
        result = generate_near_miss_title(entry)
        assert result.difficulty_tier == 3

    def test_short_title_one_word_always_changes(self):
        """Issue 2: 1-word title must produce a different title."""
        for seed in range(20):
            entry = _make_base_entry()
            entry.fields["title"] = "Attention"
            rng = random.Random(seed)
            result = generate_near_miss_title(entry, rng=rng)
            assert result.fields["title"] != "Attention", f"1-word title unchanged at seed {seed}"

    def test_short_title_two_word_always_changes(self):
        """Issue 2: 2-word title must produce a different title."""
        for seed in range(20):
            entry = _make_base_entry()
            entry.fields["title"] = "Deep Networks"
            rng = random.Random(seed)
            result = generate_near_miss_title(entry, rng=rng)
            assert result.fields["title"] != "Deep Networks", (
                f"2-word title unchanged at seed {seed}"
            )


class TestFakeDoiPrefixes:
    def test_no_real_registrant_prefixes(self):
        """Issue 13: FAKE_DOI_PREFIXES must not contain any real registrant prefix."""
        for prefix in FAKE_DOI_PREFIXES:
            assert prefix not in _KNOWN_REAL_PREFIXES, (
                f"FAKE_DOI_PREFIXES contains real registrant prefix: {prefix}"
            )

    def test_prefix_count(self):
        """Ensure we still have 20 prefixes (same count as before the fix)."""
        assert len(FAKE_DOI_PREFIXES) == 20


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


class TestGeneratePlausibleFabricationVariation:
    def test_bibtex_type_variation(self):
        """generate_plausible_fabrication produces both article and inproceedings over 200 seeds."""
        types_seen: set[str] = set()
        for seed in range(200):
            entry = _make_base_entry()
            rng = random.Random(seed)
            result = generate_plausible_fabrication(entry, rng=rng)
            types_seen.add(result.bibtex_type)
            if len(types_seen) == 2:
                break
        assert "article" in types_seen, "Never produced bibtex_type='article'"
        assert "inproceedings" in types_seen, "Never produced bibtex_type='inproceedings'"

    def test_author_count_varies_2_to_7(self):
        """Author count spans at least 3 to 5 across 100 seeds (true range 2-7)."""
        counts: list[int] = []
        for seed in range(100):
            entry = _make_base_entry()
            rng = random.Random(seed)
            result = generate_plausible_fabrication(entry, rng=rng)
            author_field = result.fields.get("author", "")
            count = len(author_field.split(" and ")) if author_field else 0
            counts.append(count)
        assert min(counts) <= 3, f"Minimum author count {min(counts)} never reached <=3"
        assert max(counts) >= 5, f"Maximum author count {max(counts)} never reached >=5"


class TestGenerateNonexistentVenuePreservesBibtexType:
    def test_article_source_preserved(self):
        """generate_nonexistent_venue with bibtex_type='article' source keeps article type."""
        from hallmark.dataset.generators.tier1 import generate_nonexistent_venue

        entry = _make_base_entry()
        entry.bibtex_type = "article"
        entry.fields.pop("booktitle", None)
        entry.fields["journal"] = "Nature Machine Intelligence"
        result = generate_nonexistent_venue(entry)
        assert result.bibtex_type == "article"
        assert "journal" in result.fields
        assert "booktitle" not in result.fields

    def test_inproceedings_source_stays_inproceedings(self):
        """generate_nonexistent_venue with inproceedings source uses inproceedings."""
        from hallmark.dataset.generators.tier1 import generate_nonexistent_venue

        entry = _make_base_entry()
        assert entry.bibtex_type == "inproceedings"
        result = generate_nonexistent_venue(entry)
        assert result.bibtex_type == "inproceedings"
        assert "booktitle" in result.fields


class TestGeneratorRegistry:
    """Tests for the generator registry pattern."""

    def test_all_types_registered(self):
        """Every HallucinationType has a registered generator."""
        import hallmark.dataset.generators  # noqa: F401 — triggers all @register_generator decorators
        from hallmark.dataset.generators._registry import all_generators
        from hallmark.dataset.schema import HallucinationType

        registered = all_generators()
        for ht in HallucinationType:
            assert ht in registered, f"{ht} not registered"

    def test_get_generator_func_callable(self):
        """get_generator_func returns a callable for every registered type."""
        import hallmark.dataset.generators  # noqa: F401
        from hallmark.dataset.generators._registry import get_generator_func
        from hallmark.dataset.schema import HallucinationType

        for ht in HallucinationType:
            func = get_generator_func(ht)
            assert callable(func), f"{ht} generator not callable"

    def test_get_generator_invalid_raises(self):
        """get_generator with an unregistered value raises ValueError."""
        import hallmark.dataset.generators  # noqa: F401
        from hallmark.dataset.generators._registry import get_generator

        with pytest.raises(ValueError):
            get_generator("nonexistent_type")  # type: ignore[arg-type]

    def test_registry_spec_has_extra_args(self):
        """Generators requiring extra args declare them in their spec."""
        import hallmark.dataset.generators  # noqa: F401
        from hallmark.dataset.generators._registry import get_generator
        from hallmark.dataset.schema import HallucinationType

        spec_chimeric = get_generator(HallucinationType.CHIMERIC_TITLE)
        assert "fake_title" in spec_chimeric.extra_args

        spec_author = get_generator(HallucinationType.AUTHOR_MISMATCH)
        assert "donor_entry" in spec_author.extra_args
