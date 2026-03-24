"""Tests for hallmark.evaluation.ranking_stability."""

from __future__ import annotations

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.ranking_stability import (
    SubtypeRankingResult,
    _kendall_tau,
    iia_violation_check,
    per_subtype_ranking_stability,
)


def _entry(key: str, label: str, tier: int | None = None, h_type: str | None = None):
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


class TestKendallTau:
    def test_identical_rankings(self):
        assert _kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_reversed_rankings(self):
        assert _kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)

    def test_partial_swap(self):
        # One swap: a,b,c vs a,c,b → 1 concordant, 1 discordant among (b,c) pair only
        tau = _kendall_tau(["a", "b", "c"], ["a", "c", "b"])
        assert -1.0 <= tau <= 1.0
        assert tau < 1.0  # not perfect

    def test_single_item(self):
        assert _kendall_tau(["a"], ["a"]) == pytest.approx(0.0)

    def test_no_common_items(self):
        assert _kendall_tau(["a", "b"], ["c", "d"]) == pytest.approx(0.0)


class TestPerSubtypeRankingStability:
    def test_dominant_tool_stable(self):
        """One tool detects everything, others nothing → stable."""
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=1, h_type="fabricated_doi") for i in range(20)
        ]
        tool_predictions = {
            "perfect": [_pred(f"h{i}", "HALLUCINATED") for i in range(20)],
            "blind": [_pred(f"h{i}", "VALID") for i in range(20)],
        }
        results = per_subtype_ranking_stability(entries, tool_predictions, n_bootstrap=200)
        assert "fabricated_doi" in results
        r = results["fabricated_doi"]
        assert r.is_stable is True
        assert r.tool_rankings[0][0] == "perfect"

    def test_close_tools_not_distinguishable(self):
        """Two tools with nearly equal performance → pairwise_distinguishable=False."""
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=1, h_type="fabricated_doi") for i in range(20)
        ]
        # tool_a detects 10/20, tool_b detects 11/20 — very close
        tool_predictions = {
            "tool_a": [_pred(f"h{i}", "HALLUCINATED" if i < 10 else "VALID") for i in range(20)],
            "tool_b": [_pred(f"h{i}", "HALLUCINATED" if i < 11 else "VALID") for i in range(20)],
        }
        results = per_subtype_ranking_stability(entries, tool_predictions, n_bootstrap=200)
        r = results["fabricated_doi"]
        # Close tools should not be distinguishable
        assert r.pairwise_distinguishable["tool_a_vs_tool_b"] is False

    def test_result_structure(self):
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=1, h_type="fabricated_doi") for i in range(10)
        ]
        tool_predictions = {
            "tool_a": [_pred(f"h{i}", "HALLUCINATED") for i in range(10)],
            "tool_b": [_pred(f"h{i}", "VALID") for i in range(10)],
        }
        results = per_subtype_ranking_stability(entries, tool_predictions, n_bootstrap=50)
        r = results["fabricated_doi"]
        assert isinstance(r, SubtypeRankingResult)
        assert r.subtype == "fabricated_doi"
        assert r.n_entries == 10
        assert "tool_a" in r.rank_ci_per_tool
        assert "tool_b" in r.rank_ci_per_tool
        assert len(r.tool_rankings) == 2


class TestRankingSensitivity:
    def test_requires_numpy(self):
        from unittest.mock import patch

        entries = [_entry("h1", "HALLUCINATED", tier=1)]
        tool_predictions = {"t": [_pred("h1", "HALLUCINATED")]}
        with patch.dict("sys.modules", {"numpy": None}):
            from hallmark.evaluation.ranking_stability import ranking_sensitivity_analysis

            with pytest.raises(ImportError, match="numpy"):
                ranking_sensitivity_analysis(entries, tool_predictions, n_samples=10)


class TestIIACheck:
    def test_score_based_satisfies_iia(self):
        """Score-based ranking trivially satisfies IIA."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=2, h_type="wrong_venue"),
            _entry("v1", "VALID"),
        ]
        tool_predictions = {
            "tool_a": [
                _pred("h1", "HALLUCINATED"),
                _pred("h2", "HALLUCINATED"),
                _pred("v1", "VALID"),
            ],
            "tool_b": [
                _pred("h1", "VALID"),
                _pred("h2", "VALID"),
                _pred("v1", "VALID"),
            ],
        }
        result = iia_violation_check(entries, tool_predictions)
        assert result["iia_satisfied"] is True
        assert result["n_tools"] == 2
        assert len(result["violations"]) == 0
