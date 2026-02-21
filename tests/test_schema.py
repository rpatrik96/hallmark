"""Tests for hallmark.dataset.schema."""

import tempfile
from pathlib import Path

import pytest

from hallmark.dataset.schema import (
    BenchmarkEntry,
    DifficultyTier,
    GenerationMethod,
    HallucinationType,
    Prediction,
    load_entries,
    load_predictions,
    save_entries,
    save_predictions,
)

# --- Fixtures ---


def make_valid_entry(**kwargs) -> BenchmarkEntry:
    defaults = {
        "bibtex_key": "vaswani2017attention",
        "bibtex_type": "inproceedings",
        "fields": {
            "title": "Attention Is All You Need",
            "author": "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki",
            "year": "2017",
            "booktitle": "NeurIPS",
            "doi": "10.5555/3295222.3295349",
        },
        "label": "VALID",
        "explanation": "Valid entry from NeurIPS 2017",
        "generation_method": "scraped",
        "source_conference": "NeurIPS",
        "publication_date": "2017-12-04",
        "added_to_benchmark": "2026-01-15",
        "subtests": {
            "doi_resolves": True,
            "title_exists": True,
            "authors_match": True,
            "venue_real": True,
            "fields_complete": True,
            "cross_db_agreement": True,
        },
    }
    defaults.update(kwargs)
    return BenchmarkEntry(**defaults)


def make_hallucinated_entry(**kwargs) -> BenchmarkEntry:
    defaults = {
        "bibtex_key": "doe2024fake",
        "bibtex_type": "article",
        "fields": {
            "title": "A Novel Approach to Fake Research",
            "author": "John Doe and Jane Smith",
            "year": "2024",
            "journal": "Journal of Fake Science",
            "doi": "10.9999/fake.2024.001",
        },
        "label": "HALLUCINATED",
        "hallucination_type": "fabricated_doi",
        "difficulty_tier": 1,
        "explanation": "DOI is fabricated and does not resolve",
        "generation_method": "perturbation",
        "publication_date": "2024-06-15",
        "added_to_benchmark": "2026-01-15",
        "subtests": {
            "doi_resolves": False,
            "title_exists": False,
            "authors_match": False,
            "venue_real": False,
            "fields_complete": True,
            "cross_db_agreement": False,
        },
    }
    defaults.update(kwargs)
    return BenchmarkEntry(**defaults)


# --- Tests ---


class TestBenchmarkEntry:
    def test_create_valid_entry(self):
        entry = make_valid_entry()
        assert entry.label == "VALID"
        assert entry.hallucination_type is None
        assert entry.title == "Attention Is All You Need"
        assert entry.author.startswith("Vaswani")

    def test_create_hallucinated_entry(self):
        entry = make_hallucinated_entry()
        assert entry.label == "HALLUCINATED"
        assert entry.hallucination_type == "fabricated_doi"
        assert entry.difficulty_tier == 1

    def test_valid_entry_rejects_hallucination_type(self):
        with pytest.raises(ValueError, match="must not have"):
            make_valid_entry(hallucination_type="fabricated_doi")

    def test_hallucinated_entry_requires_type(self):
        with pytest.raises(ValueError, match="must have a hallucination_type"):
            BenchmarkEntry(
                bibtex_key="test",
                bibtex_type="article",
                fields={"title": "Test", "author": "A", "year": "2024"},
                label="HALLUCINATED",
                difficulty_tier=1,
            )

    def test_hallucinated_entry_requires_tier(self):
        with pytest.raises(ValueError, match="must have a difficulty_tier"):
            BenchmarkEntry(
                bibtex_key="test",
                bibtex_type="article",
                fields={"title": "Test", "author": "A", "year": "2024"},
                label="HALLUCINATED",
                hallucination_type="fabricated_doi",
            )

    def test_properties(self):
        entry = make_valid_entry()
        assert entry.title == "Attention Is All You Need"
        assert "Vaswani" in entry.author
        assert entry.year == "2017"
        assert entry.doi == "10.5555/3295222.3295349"
        assert entry.venue == "NeurIPS"

    def test_to_bibtex(self):
        entry = make_valid_entry()
        bib = entry.to_bibtex()
        assert "@inproceedings{vaswani2017attention," in bib
        assert "Attention Is All You Need" in bib

    def test_serialization_roundtrip(self):
        entry = make_hallucinated_entry()
        json_str = entry.to_json()
        restored = BenchmarkEntry.from_json(json_str)
        assert restored.bibtex_key == entry.bibtex_key
        assert restored.label == entry.label
        assert restored.hallucination_type == entry.hallucination_type
        assert restored.difficulty_tier == entry.difficulty_tier
        assert restored.fields == entry.fields

    def test_to_dict(self):
        entry = make_valid_entry()
        d = entry.to_dict()
        assert d["bibtex_key"] == "vaswani2017attention"
        assert d["label"] == "VALID"
        assert isinstance(d["fields"], dict)


class TestPrediction:
    def test_create_prediction(self):
        pred = Prediction(
            bibtex_key="test",
            label="HALLUCINATED",
            confidence=0.85,
            reason="DOI does not resolve",
        )
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(ValueError, match="Confidence"):
            Prediction(bibtex_key="test", label="VALID", confidence=1.5)
        with pytest.raises(ValueError, match="Confidence"):
            Prediction(bibtex_key="test", label="VALID", confidence=-0.1)

    def test_serialization_roundtrip(self):
        pred = Prediction(
            bibtex_key="test",
            label="VALID",
            confidence=0.95,
            reason="Verified",
            api_sources_queried=["crossref", "dblp"],
            wall_clock_seconds=1.5,
            api_calls=2,
        )
        restored = Prediction.from_json(pred.to_json())
        assert restored.bibtex_key == pred.bibtex_key
        assert restored.confidence == pred.confidence
        assert restored.api_sources_queried == pred.api_sources_queried


class TestFileIO:
    def test_save_load_entries(self):
        entries = [make_valid_entry(), make_hallucinated_entry()]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        save_entries(entries, path)
        loaded = load_entries(path)

        assert len(loaded) == 2
        assert loaded[0].bibtex_key == "vaswani2017attention"
        assert loaded[1].label == "HALLUCINATED"

        path.unlink()

    def test_save_load_predictions(self):
        preds = [
            Prediction(bibtex_key="a", label="VALID", confidence=0.9),
            Prediction(bibtex_key="b", label="HALLUCINATED", confidence=0.8),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        save_predictions(preds, path)
        loaded = load_predictions(path)

        assert len(loaded) == 2
        assert loaded[0].label == "VALID"
        assert loaded[1].label == "HALLUCINATED"

        path.unlink()


class TestEnums:
    def test_hallucination_types(self):
        assert HallucinationType.FABRICATED_DOI.value == "fabricated_doi"
        assert HallucinationType.NEAR_MISS_TITLE.value == "near_miss_title"

    def test_hybrid_fabrication_type(self):
        assert HallucinationType.HYBRID_FABRICATION.value == "hybrid_fabrication"

    def test_author_mismatch_backward_compat(self):
        # Backward compatibility: AUTHOR_MISMATCH enum uses "swapped_authors" value
        assert HallucinationType.AUTHOR_MISMATCH.value == "swapped_authors"

    def test_difficulty_tiers(self):
        assert DifficultyTier.EASY.value == 1
        assert DifficultyTier.MEDIUM.value == 2
        assert DifficultyTier.HARD.value == 3

    def test_generation_methods(self):
        assert GenerationMethod.SCRAPED.value == "scraped"
        assert GenerationMethod.LLM_GENERATED.value == "llm_generated"

    def test_hallucination_tier_map_hybrid_fabrication(self):
        from hallmark.dataset.schema import HALLUCINATION_TIER_MAP

        assert HALLUCINATION_TIER_MAP[HallucinationType.HYBRID_FABRICATION] == DifficultyTier.MEDIUM


class TestEvaluationResult:
    def test_ece_field_defaults_to_none(self):
        from hallmark.dataset.schema import EvaluationResult

        result = EvaluationResult(
            tool_name="test",
            split_name="dev",
            num_entries=10,
            num_hallucinated=5,
            num_valid=5,
            detection_rate=0.8,
            false_positive_rate=0.1,
            f1_hallucination=0.85,
            tier_weighted_f1=0.9,
        )
        assert result.ece is None


class TestLabelValidation:
    def test_invalid_entry_label_raises(self):
        with pytest.raises(ValueError, match="Invalid BenchmarkEntry label"):
            BenchmarkEntry(
                bibtex_key="test_invalid",
                bibtex_type="article",
                fields={"title": "Test", "author": "A", "year": "2024"},
                label="INVALID",
            )

    def test_invalid_prediction_label_raises(self):
        with pytest.raises(ValueError, match="Invalid Prediction label"):
            Prediction(bibtex_key="test_invalid", label="INVALID", confidence=0.5)

    def test_valid_entry_labels_accepted(self):
        for label in ("VALID", "HALLUCINATED"):
            kwargs: dict = {
                "bibtex_key": f"test_{label}",
                "bibtex_type": "article",
                "fields": {"title": "T", "author": "A", "year": "2024"},
                "label": label,
            }
            if label == "HALLUCINATED":
                kwargs["hallucination_type"] = "fabricated_doi"
                kwargs["difficulty_tier"] = 1
            entry = BenchmarkEntry(**kwargs)
            assert entry.label == label

    def test_valid_prediction_labels_accepted(self):
        for label in ("VALID", "HALLUCINATED", "UNCERTAIN"):
            pred = Prediction(bibtex_key="test", label=label, confidence=0.5)
            assert pred.label == label


class TestBackwardCompatibility:
    def test_swapped_authors_value_still_works(self):
        # Ensure old entries with hallucination_type="swapped_authors" still work
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
