"""Tests for the title-match oracle baseline."""

from __future__ import annotations

from hallmark.baselines.title_oracle import (
    HALLUCINATED_CONFIDENCE,
    VALID_CONFIDENCE,
    build_valid_title_set,
    run_title_oracle,
)
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, Prediction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_entry(key: str, title: str) -> BenchmarkEntry:
    return BenchmarkEntry(
        bibtex_key=key,
        bibtex_type="inproceedings",
        fields={"title": title, "author": "A. Author", "year": "2023"},
        label="VALID",
        explanation="valid entry",
    )


def _hallucinated_entry(key: str, title: str) -> BenchmarkEntry:
    return BenchmarkEntry(
        bibtex_key=key,
        bibtex_type="inproceedings",
        fields={"title": title, "author": "A. Author", "year": "2023"},
        label="HALLUCINATED",
        hallucination_type="fabricated_doi",
        difficulty_tier=1,
        explanation="hallucinated entry",
    )


def _blind(key: str, title: str) -> BlindEntry:
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="inproceedings",
        fields={"title": title, "author": "A. Author", "year": "2023"},
    )


# ---------------------------------------------------------------------------
# build_valid_title_set
# ---------------------------------------------------------------------------


class TestBuildValidTitleSet:
    def test_includes_valid_titles(self):
        pool = [_valid_entry("v1", "Attention Is All You Need")]
        titles = build_valid_title_set(pool)
        assert "attention is all you need" in titles

    def test_excludes_hallucinated_titles(self):
        pool = [
            _valid_entry("v1", "Real Paper Title"),
            _hallucinated_entry("h1", "Fake Paper Title"),
        ]
        titles = build_valid_title_set(pool)
        assert "real paper title" in titles
        assert "fake paper title" not in titles

    def test_normalises_case_and_whitespace(self):
        pool = [_valid_entry("v1", "  BERT: Pre-training of Deep Bidirectional  ")]
        titles = build_valid_title_set(pool)
        assert "bert: pre-training of deep bidirectional" in titles

    def test_empty_pool_returns_empty_set(self):
        assert build_valid_title_set([]) == set()

    def test_skips_entries_with_empty_title(self):
        entry = BenchmarkEntry(
            bibtex_key="notitle",
            bibtex_type="article",
            fields={"author": "Nobody", "year": "2020"},
            label="VALID",
            explanation="no title",
        )
        titles = build_valid_title_set([entry])
        assert len(titles) == 0

    def test_multiple_valid_entries(self):
        pool = [
            _valid_entry("v1", "Paper Alpha"),
            _valid_entry("v2", "Paper Beta"),
            _valid_entry("v3", "Paper Gamma"),
        ]
        titles = build_valid_title_set(pool)
        assert titles == {"paper alpha", "paper beta", "paper gamma"}


# ---------------------------------------------------------------------------
# run_title_oracle
# ---------------------------------------------------------------------------


class TestRunTitleOracle:
    def test_title_in_pool_predicts_hallucinated(self):
        pool = [_valid_entry("v1", "Attention Is All You Need")]
        entries = [_blind("e1", "Attention Is All You Need")]

        preds = run_title_oracle(entries, pool)

        assert len(preds) == 1
        assert preds[0].label == "HALLUCINATED"
        assert preds[0].confidence == HALLUCINATED_CONFIDENCE
        assert preds[0].bibtex_key == "e1"

    def test_title_absent_predicts_valid(self):
        pool = [_valid_entry("v1", "Attention Is All You Need")]
        entries = [_blind("e2", "A Brand New Paper Not In Pool")]

        preds = run_title_oracle(entries, pool)

        assert len(preds) == 1
        assert preds[0].label == "VALID"
        assert preds[0].confidence == VALID_CONFIDENCE
        assert preds[0].bibtex_key == "e2"

    def test_case_insensitive_match(self):
        pool = [_valid_entry("v1", "BERT: Pre-training")]
        entries = [_blind("e1", "bert: pre-training")]

        preds = run_title_oracle(entries, pool)

        assert preds[0].label == "HALLUCINATED"

    def test_whitespace_normalised_match(self):
        pool = [_valid_entry("v1", "  GPT-4 Technical Report  ")]
        entries = [_blind("e1", "GPT-4 Technical Report")]

        preds = run_title_oracle(entries, pool)

        assert preds[0].label == "HALLUCINATED"

    def test_empty_reference_pool_all_valid(self):
        entries = [_blind("e1", "Some Title"), _blind("e2", "Another Title")]
        preds = run_title_oracle(entries, [])

        assert all(p.label == "VALID" for p in preds)
        assert all(p.confidence == VALID_CONFIDENCE for p in preds)

    def test_empty_entries_returns_empty(self):
        pool = [_valid_entry("v1", "Some Title")]
        preds = run_title_oracle([], pool)
        assert preds == []

    def test_preserves_order_and_keys(self):
        pool = [_valid_entry("v1", "Known Title")]
        entries = [
            _blind("a", "Unknown Title"),
            _blind("b", "Known Title"),
            _blind("c", "Another Unknown"),
        ]
        preds = run_title_oracle(entries, pool)

        assert [p.bibtex_key for p in preds] == ["a", "b", "c"]
        assert preds[0].label == "VALID"
        assert preds[1].label == "HALLUCINATED"
        assert preds[2].label == "VALID"

    def test_returns_prediction_objects(self):
        pool = [_valid_entry("v1", "Title A")]
        entries = [_blind("e1", "Title A"), _blind("e2", "Title B")]
        preds = run_title_oracle(entries, pool)

        assert all(isinstance(p, Prediction) for p in preds)

    def test_reason_strings_are_informative(self):
        pool = [_valid_entry("v1", "Known Title")]
        entries = [_blind("match", "Known Title"), _blind("miss", "Unknown Title")]
        preds = run_title_oracle(entries, pool)

        assert "oracle" in preds[0].reason.lower()
        assert "oracle" in preds[1].reason.lower()

    def test_entry_with_no_title_predicts_valid(self):
        pool = [_valid_entry("v1", "Some Title")]
        entry = BlindEntry(
            bibtex_key="notitle",
            bibtex_type="article",
            fields={"author": "Nobody", "year": "2020"},
        )
        preds = run_title_oracle([entry], pool)

        assert preds[0].label == "VALID"

    def test_hallucinated_in_pool_are_ignored(self):
        """HALLUCINATED entries in the reference pool must not influence predictions."""
        pool = [
            _valid_entry("v1", "Real Paper"),
            _hallucinated_entry("h1", "Fake Paper"),
        ]
        entries = [_blind("e1", "Fake Paper")]

        preds = run_title_oracle(entries, pool)

        # "Fake Paper" is HALLUCINATED in the pool, not VALID — should predict VALID
        assert preds[0].label == "VALID"

    def test_mixed_batch(self):
        pool = [
            _valid_entry("v1", "Alpha"),
            _valid_entry("v2", "Beta"),
        ]
        entries = [
            _blind("e1", "Alpha"),  # in pool → HALLUCINATED
            _blind("e2", "Gamma"),  # not in pool → VALID
            _blind("e3", "Beta"),  # in pool → HALLUCINATED
            _blind("e4", "Delta"),  # not in pool → VALID
        ]
        preds = run_title_oracle(entries, pool)

        labels = [p.label for p in preds]
        assert labels == ["HALLUCINATED", "VALID", "HALLUCINATED", "VALID"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestTitleOracleRegistry:
    def test_title_oracle_registered(self):
        from hallmark.baselines.registry import list_baselines

        assert "title_oracle" in list_baselines()

    def test_title_oracle_is_free(self):
        from hallmark.baselines.registry import get_registry

        info = get_registry()["title_oracle"]
        assert info.is_free is True
        assert info.requires_api_key is False

    def test_title_oracle_confidence_type(self):
        from hallmark.baselines.registry import get_registry

        info = get_registry()["title_oracle"]
        assert info.confidence_type == "binary"

    def test_title_oracle_description_flags_diagnostic(self):
        from hallmark.baselines.registry import get_registry

        info = get_registry()["title_oracle"]
        desc_lower = info.description.lower()
        assert "diagnostic" in desc_lower or "oracle" in desc_lower

    def test_title_oracle_check_available(self):
        from hallmark.baselines.registry import check_available

        avail, msg = check_available("title_oracle")
        assert avail is True
        assert msg == "OK"
