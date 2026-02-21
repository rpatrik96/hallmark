"""Tests for scripts.stages.sanitize helper functions."""

from hallmark.dataset.schema import BenchmarkEntry
from scripts.stages.sanitize import _drop_unknown_authors, _fix_mislabeled_llm_entries


def _make_entry(
    key: str,
    label: str = "HALLUCINATED",
    hallucination_type: str = "fabricated_doi",
    generation_method: str = "perturbation",
    difficulty_tier: int = 1,
    author: str = "Alice Smith",
) -> BenchmarkEntry:
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "inproceedings",
        "fields": {
            "title": f"Paper {key}",
            "author": author,
            "year": "2022",
            "booktitle": "NeurIPS",
        },
        "label": label,
        "explanation": "test",
        "generation_method": generation_method,
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = hallucination_type
        kwargs["difficulty_tier"] = difficulty_tier
    return BenchmarkEntry(**kwargs)


class TestFixMislabeledLlmEntries:
    def test_relabels_fabricated_doi_llm_to_plausible_fabrication(self):
        """LLM-generated fabricated_doi entry is relabeled to plausible_fabrication."""
        entry = _make_entry(
            "e1",
            label="HALLUCINATED",
            hallucination_type="fabricated_doi",
            generation_method="llm_generated",
            difficulty_tier=1,
        )
        count = _fix_mislabeled_llm_entries([entry])
        assert count == 1
        assert entry.hallucination_type == "plausible_fabrication"
        assert entry.difficulty_tier == 3

    def test_does_not_relabel_perturbation_fabricated_doi(self):
        """Perturbation-generated fabricated_doi entries are not relabeled."""
        entry = _make_entry(
            "e2",
            label="HALLUCINATED",
            hallucination_type="fabricated_doi",
            generation_method="perturbation",
            difficulty_tier=1,
        )
        count = _fix_mislabeled_llm_entries([entry])
        assert count == 0
        assert entry.hallucination_type == "fabricated_doi"
        assert entry.difficulty_tier == 1

    def test_does_not_relabel_llm_other_types(self):
        """LLM-generated entries with a non-fabricated_doi type are not touched."""
        entry = _make_entry(
            "e3",
            label="HALLUCINATED",
            hallucination_type="plausible_fabrication",
            generation_method="llm_generated",
            difficulty_tier=3,
        )
        count = _fix_mislabeled_llm_entries([entry])
        assert count == 0
        assert entry.hallucination_type == "plausible_fabrication"
        assert entry.difficulty_tier == 3

    def test_returns_count_of_fixed_entries(self):
        """Returns correct count when multiple entries are relabeled."""
        entries = [
            _make_entry(
                f"e{i}",
                hallucination_type="fabricated_doi",
                generation_method="llm_generated",
            )
            for i in range(4)
        ]
        # Add one that should not be relabeled
        entries.append(
            _make_entry(
                "e_skip", hallucination_type="fabricated_doi", generation_method="perturbation"
            )
        )
        count = _fix_mislabeled_llm_entries(entries)
        assert count == 4

    def test_relabeled_entry_difficulty_tier_is_hard(self):
        """Relabeled entry gets difficulty_tier == 3 (hard)."""
        entry = _make_entry(
            "e_hard",
            hallucination_type="fabricated_doi",
            generation_method="llm_generated",
            difficulty_tier=1,
        )
        _fix_mislabeled_llm_entries([entry])
        assert entry.difficulty_tier == 3


class TestDropUnknownAuthors:
    def test_drops_unknown_author_entries(self):
        """Entries with author='Unknown' are removed."""
        entries = [
            _make_entry("good1", author="Alice Smith"),
            _make_entry("bad", author="Unknown"),
            _make_entry("good2", author="Bob Jones"),
        ]
        filtered, dropped = _drop_unknown_authors(entries)
        assert dropped == 1
        assert len(filtered) == 2
        keys = [e.bibtex_key for e in filtered]
        assert "bad" not in keys
        assert "good1" in keys
        assert "good2" in keys

    def test_case_insensitive_unknown(self):
        """'unknown' (any case) is dropped."""
        entries = [
            _make_entry("e1", author="UNKNOWN"),
            _make_entry("e2", author="unknown"),
            _make_entry("e3", author="Unknown"),
        ]
        filtered, dropped = _drop_unknown_authors(entries)
        assert dropped == 3
        assert len(filtered) == 0

    def test_keeps_all_known_authors(self):
        """Entries with real author names are all preserved."""
        entries = [
            _make_entry("e1", author="Alice Smith"),
            _make_entry("e2", author="Bob Jones and Carol Lee"),
            _make_entry("e3", label="VALID", author="Dan Park"),
        ]
        filtered, dropped = _drop_unknown_authors(entries)
        assert dropped == 0
        assert len(filtered) == 3

    def test_empty_list(self):
        """Empty input returns empty output with zero dropped."""
        filtered, dropped = _drop_unknown_authors([])
        assert filtered == []
        assert dropped == 0
