"""Tests for hallmark.evaluation.factor_analysis."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.factor_analysis import (
    PCAResult,
    compute_score_matrix,
    pca_analysis,
    tier_stratified_pca,
)

# --- Helpers ---


def _entry(
    key: str,
    label: str,
    tier: int | None = None,
    h_type: str | None = None,
):
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2024"},
        "label": label,
        "explanation": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type or "fabricated_doi"
        kwargs["difficulty_tier"] = tier or 1
    return BenchmarkEntry(**kwargs)


def _pred(key: str, label: str, confidence: float = 0.9):
    return Prediction(bibtex_key=key, label=label, confidence=confidence)


# --- Tests ---


class TestComputeScoreMatrix:
    def test_basic(self):
        """2 tools, 2 types: verify matrix values match expected detection rates."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h3", "HALLUCINATED", tier=2, h_type="wrong_venue"),
            _entry("h4", "HALLUCINATED", tier=2, h_type="wrong_venue"),
        ]
        # tool_a detects both fabricated_doi, misses both wrong_venue
        # tool_b misses both fabricated_doi, detects both wrong_venue
        tool_predictions = {
            "tool_a": [
                _pred("h1", "HALLUCINATED"),
                _pred("h2", "HALLUCINATED"),
                _pred("h3", "VALID"),
                _pred("h4", "VALID"),
            ],
            "tool_b": [
                _pred("h1", "VALID"),
                _pred("h2", "VALID"),
                _pred("h3", "HALLUCINATED"),
                _pred("h4", "HALLUCINATED"),
            ],
        }
        tool_names, type_names, matrix = compute_score_matrix(entries, tool_predictions)

        assert sorted(tool_names) == ["tool_a", "tool_b"]
        assert sorted(type_names) == ["fabricated_doi", "wrong_venue"]

        # Build lookup by name
        idx_tool_a = tool_names.index("tool_a")
        idx_tool_b = tool_names.index("tool_b")
        idx_fab = type_names.index("fabricated_doi")
        idx_venue = type_names.index("wrong_venue")

        assert matrix[idx_tool_a][idx_fab] == pytest.approx(1.0)
        assert matrix[idx_tool_a][idx_venue] == pytest.approx(0.0)
        assert matrix[idx_tool_b][idx_fab] == pytest.approx(0.0)
        assert matrix[idx_tool_b][idx_venue] == pytest.approx(1.0)

    def test_missing_predictions_yields_zero(self):
        """Tool with no predictions for a type should get 0.0, not error."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=2, h_type="wrong_venue"),
        ]
        # tool_a only has prediction for h1, no prediction for h2
        tool_predictions = {
            "tool_a": [_pred("h1", "HALLUCINATED")],
            "tool_b": [_pred("h1", "HALLUCINATED"), _pred("h2", "HALLUCINATED")],
        }
        tool_names, type_names, matrix = compute_score_matrix(entries, tool_predictions)

        idx_tool_a = tool_names.index("tool_a")
        idx_venue = type_names.index("wrong_venue")
        # tool_a has no prediction for h2 (wrong_venue) -> 0/1 = 0.0
        assert matrix[idx_tool_a][idx_venue] == pytest.approx(0.0)

    def test_excludes_valid_entries(self):
        """VALID entries must not appear in the score matrix."""
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        ]
        tool_predictions = {
            "tool_a": [
                _pred("v1", "HALLUCINATED"),  # FP, but irrelevant to DR matrix
                _pred("v2", "HALLUCINATED"),  # FP
                _pred("h1", "HALLUCINATED"),  # TP
            ],
        }
        _tool_names, type_names, matrix = compute_score_matrix(entries, tool_predictions)

        # Only one hallucination type
        assert type_names == ["fabricated_doi"]
        assert len(matrix) == 1
        # DR for fabricated_doi: 1/1 = 1.0
        assert matrix[0][0] == pytest.approx(1.0)

    def test_partial_detection_rate(self):
        """DR should be fractional when only some entries are detected."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h3", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h4", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        ]
        tool_predictions = {
            "tool_a": [
                _pred("h1", "HALLUCINATED"),
                _pred("h2", "HALLUCINATED"),
                _pred("h3", "VALID"),
                _pred("h4", "VALID"),
            ],
        }
        _tool_names, _type_names, matrix = compute_score_matrix(entries, tool_predictions)
        assert matrix[0][0] == pytest.approx(0.5)


class TestPCAAnalysis:
    def test_identical_tools_zero_variance(self):
        """Two identical tools → zero variance after centering → degenerate PCA."""
        pytest.importorskip("numpy")

        tool_names = ["tool_a", "tool_b"]
        type_names = ["fabricated_doi", "wrong_venue", "future_date"]
        # Identical rows: centering zeros everything → no variance
        score_matrix = [
            [0.9, 0.5, 0.7],
            [0.9, 0.5, 0.7],
        ]
        result = pca_analysis(tool_names, type_names, score_matrix)

        assert isinstance(result, PCAResult)
        assert result.total_variance_explained_by_pc1 == pytest.approx(0.0)

    def test_near_identical_tools_low_rank(self):
        """Nearly identical tools → rank 1 (one dominant PC)."""
        pytest.importorskip("numpy")

        tool_names = ["tool_a", "tool_b", "tool_c"]
        type_names = ["fabricated_doi", "wrong_venue"]
        # All vary along same direction
        score_matrix = [
            [0.9, 0.3],
            [0.8, 0.2],
            [0.7, 0.1],
        ]
        result = pca_analysis(tool_names, type_names, score_matrix)
        assert result.effective_rank == 1

    def test_orthogonal_tools_rank_two(self):
        """Three tools with independent variation → effective_rank 2."""
        pytest.importorskip("numpy")

        # 3 tools, 3 types: two independent axes of variation
        tool_names = ["tool_a", "tool_b", "tool_c"]
        type_names = ["fabricated_doi", "wrong_venue", "future_date"]
        score_matrix = [
            [1.0, 0.0, 0.5],  # good at type 0
            [0.0, 1.0, 0.5],  # good at type 1
            [0.5, 0.5, 0.0],  # good at neither of above, bad at type 2
        ]
        result = pca_analysis(tool_names, type_names, score_matrix, variance_threshold=0.90)

        assert result.effective_rank == 2
        # PC1 alone doesn't explain everything
        assert result.total_variance_explained_by_pc1 < 0.99

    def test_requires_min_two_tools(self):
        """Fewer than 2 tools raises ValueError."""
        pytest.importorskip("numpy")

        with pytest.raises(ValueError, match="at least 2 tools"):
            pca_analysis(["tool_a"], ["fabricated_doi", "wrong_venue"], [[0.5, 0.5]])

    def test_requires_min_two_types(self):
        """Fewer than 2 types raises ValueError."""
        pytest.importorskip("numpy")

        with pytest.raises(ValueError, match="at least 2 hallucination types"):
            pca_analysis(["tool_a", "tool_b"], ["fabricated_doi"], [[0.5], [0.5]])

    def test_result_fields(self):
        """PCAResult has correct keys and lengths."""
        pytest.importorskip("numpy")

        tool_names = ["tool_a", "tool_b", "tool_c"]
        type_names = ["fabricated_doi", "wrong_venue", "future_date"]
        score_matrix = [
            [0.9, 0.3, 0.8],
            [0.6, 0.7, 0.4],
            [0.2, 0.9, 0.6],
        ]
        result = pca_analysis(tool_names, type_names, score_matrix)

        assert result.tool_names == tool_names
        assert result.type_names == type_names
        assert set(result.pc1_loadings.keys()) == set(type_names)
        assert set(result.pc1_tool_scores.keys()) == set(tool_names)
        assert len(result.explained_variance_ratio) <= min(len(tool_names), len(type_names))
        assert result.effective_rank >= 1
        # Explained variance ratios sum to ~1

        assert sum(result.explained_variance_ratio) == pytest.approx(1.0, abs=1e-6)

    def test_numpy_not_installed_raises_import_error(self):
        """When numpy import fails, pca_analysis raises ImportError with message."""
        tool_names = ["tool_a", "tool_b"]
        type_names = ["fabricated_doi", "wrong_venue"]
        score_matrix = [[0.9, 0.3], [0.6, 0.7]]

        with (
            patch.dict("sys.modules", {"numpy": None}),
            pytest.raises(ImportError, match="numpy is required"),
        ):
            pca_analysis(tool_names, type_names, score_matrix)


class TestTierStratifiedPCA:
    def test_basic_tier_stratification(self):
        """Returns PCAResult for each tier that has enough entries."""
        pytest.importorskip("numpy")

        # Tier 1: fabricated_doi, future_date (easy)
        # Tier 2: wrong_venue (medium)
        entries = [
            # Tier 1 entries
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h3", "HALLUCINATED", tier=1, h_type="future_date"),
            _entry("h4", "HALLUCINATED", tier=1, h_type="future_date"),
            # Tier 2 entries
            _entry("h5", "HALLUCINATED", tier=2, h_type="wrong_venue"),
            _entry("h6", "HALLUCINATED", tier=2, h_type="wrong_venue"),
            _entry("h7", "HALLUCINATED", tier=2, h_type="chimeric_title"),
            _entry("h8", "HALLUCINATED", tier=2, h_type="chimeric_title"),
        ]
        tool_predictions = {
            "tool_a": [
                _pred("h1", "HALLUCINATED"),
                _pred("h2", "HALLUCINATED"),
                _pred("h3", "VALID"),
                _pred("h4", "VALID"),
                _pred("h5", "HALLUCINATED"),
                _pred("h6", "HALLUCINATED"),
                _pred("h7", "VALID"),
                _pred("h8", "VALID"),
            ],
            "tool_b": [
                _pred("h1", "VALID"),
                _pred("h2", "VALID"),
                _pred("h3", "HALLUCINATED"),
                _pred("h4", "HALLUCINATED"),
                _pred("h5", "VALID"),
                _pred("h6", "VALID"),
                _pred("h7", "HALLUCINATED"),
                _pred("h8", "HALLUCINATED"),
            ],
        }
        results = tier_stratified_pca(entries, tool_predictions)

        # Tiers 1 and 2 have 2 tools and 2 types -> PCA possible
        assert 1 in results
        assert 2 in results
        assert isinstance(results[1], PCAResult)
        assert isinstance(results[2], PCAResult)

        # Tier 3 has no entries -> not in results
        assert 3 not in results

    def test_tier_with_insufficient_types_is_skipped(self):
        """Tier with only one hallucination type is skipped (needs >= 2)."""
        pytest.importorskip("numpy")

        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        ]
        tool_predictions = {
            "tool_a": [_pred("h1", "HALLUCINATED"), _pred("h2", "VALID")],
            "tool_b": [_pred("h1", "VALID"), _pred("h2", "HALLUCINATED")],
        }
        # Only one type in tier 1 -> can't do PCA
        results = tier_stratified_pca(entries, tool_predictions)
        assert 1 not in results
