"""Tests for metrics.py extensions (per-tier rankings, concordance, hard subset)."""

from __future__ import annotations

import pytest

from hallmark.dataset.schema import BenchmarkEntry, EvaluationResult, Prediction
from hallmark.evaluation.metrics import (
    evaluate,
    hard_subset_report,
    per_tier_rankings,
    per_type_rankings,
    ranking_concordance,
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


def _make_mixed_entries():
    """Entries spanning all 3 tiers plus valid entries."""
    return [
        _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        _entry("h2", "HALLUCINATED", tier=1, h_type="future_date"),
        _entry("h3", "HALLUCINATED", tier=2, h_type="wrong_venue"),
        _entry("h4", "HALLUCINATED", tier=2, h_type="chimeric_title"),
        _entry("h5", "HALLUCINATED", tier=3, h_type="near_miss_title"),
        _entry("h6", "HALLUCINATED", tier=3, h_type="plausible_fabrication"),
        _entry("v1", "VALID"),
        _entry("v2", "VALID"),
    ]


class TestPerTierRankings:
    def test_returns_all_tiers(self):
        entries = _make_mixed_entries()
        tool_preds = {
            "tool_a": [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]]
            + [_pred("v1", "VALID"), _pred("v2", "VALID")],
            "tool_b": [_pred(k, "VALID") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]]
            + [_pred("v1", "VALID"), _pred("v2", "VALID")],
        }
        result = per_tier_rankings(entries, tool_preds)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_sorted_descending(self):
        entries = _make_mixed_entries()
        tool_preds = {
            "good": [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]]
            + [_pred("v1", "VALID"), _pred("v2", "VALID")],
            "bad": [_pred(k, "VALID") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]]
            + [_pred("v1", "VALID"), _pred("v2", "VALID")],
        }
        result = per_tier_rankings(entries, tool_preds)
        for tier in (1, 2, 3):
            ranking = result[tier]
            assert ranking[0][0] == "good"
            assert ranking[0][1] >= ranking[1][1]


class TestPerTypeRankings:
    def test_includes_all_types(self):
        entries = _make_mixed_entries()
        tool_preds = {
            "tool_a": [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]]
            + [_pred("v1", "VALID"), _pred("v2", "VALID")],
        }
        result = per_type_rankings(entries, tool_preds)
        expected_types = {
            "fabricated_doi",
            "future_date",
            "wrong_venue",
            "chimeric_title",
            "near_miss_title",
            "plausible_fabrication",
        }
        assert expected_types.issubset(set(result.keys()))


class TestRankingConcordance:
    def test_identical_ordering(self):
        per_tier = {
            1: [("a", 0.9), ("b", 0.5), ("c", 0.1)],
            2: [("a", 0.8), ("b", 0.4), ("c", 0.2)],
            3: [("a", 0.7), ("b", 0.3), ("c", 0.1)],
        }
        result = ranking_concordance(per_tier)
        assert result["tau_t1_t2"] == pytest.approx(1.0)
        assert result["tau_t1_t3"] == pytest.approx(1.0)
        assert result["tau_t2_t3"] == pytest.approx(1.0)
        assert result["all_concordant"] is True

    def test_reversed_ordering(self):
        per_tier = {
            1: [("a", 0.9), ("b", 0.5), ("c", 0.1)],
            3: [("c", 0.9), ("b", 0.5), ("a", 0.1)],
        }
        result = ranking_concordance(per_tier)
        assert result["tau_t1_t3"] == pytest.approx(-1.0)
        assert result["all_concordant"] is False


class TestHardSubsetReport:
    def test_only_tier3_types(self):
        entries = _make_mixed_entries()
        preds = [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]] + [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
        ]
        result = hard_subset_report(entries, preds)
        # Should contain tier 3 types and aggregate
        assert "near_miss_title" in result
        assert "plausible_fabrication" in result
        assert "aggregate_tier3" in result
        # Should NOT contain tier 1/2 types
        assert "fabricated_doi" not in result
        assert "wrong_venue" not in result

    def test_aggregate_tier3_present(self):
        entries = _make_mixed_entries()
        preds = [_pred(k, "HALLUCINATED") for k in ["h5", "h6"]] + [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
        ]
        result = hard_subset_report(entries, preds)
        agg = result["aggregate_tier3"]
        assert "detection_rate" in agg
        assert "f1" in agg
        assert agg["detection_rate"] == pytest.approx(1.0)


class TestEvaluateTier3F1:
    def test_populates_tier3_f1(self):
        entries = _make_mixed_entries()
        preds = [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]] + [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert result.tier3_f1 > 0

    def test_tier3_f1_zero_when_no_tier3(self):
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("v1", "VALID"),
        ]
        preds = [_pred("h1", "HALLUCINATED"), _pred("v1", "VALID")]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert result.tier3_f1 == 0.0


class TestEvaluationResultSerialization:
    def test_tier3_f1_round_trip(self):
        entries = _make_mixed_entries()
        preds = [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]] + [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        d = result.to_dict()
        restored = EvaluationResult.from_dict(d)
        assert restored.tier3_f1 == pytest.approx(result.tier3_f1)

    def test_tier3_f1_in_summary(self):
        entries = _make_mixed_entries()
        preds = [_pred(k, "HALLUCINATED") for k in ["h1", "h2", "h3", "h4", "h5", "h6"]] + [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert "tier3_f1" in result.summary()
