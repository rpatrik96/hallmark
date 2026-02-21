"""Tests for hallmark.evaluation.ranking module."""

from __future__ import annotations

import importlib.util

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.ranking import (
    build_results_matrix,
    load_results_matrix,
    rank_tools,
    rank_tools_mean_score,
    rank_tools_plackett_luce,
    save_results_matrix,
)

HAS_CHOIX = importlib.util.find_spec("choix") is not None


def _entry(key, label="VALID", tier=0, h_type=None):
    kwargs = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2024"},
        "label": label,
        "explanation": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type or "fabricated_doi"
        kwargs["difficulty_tier"] = tier or 1
    return kwargs  # Use BenchmarkEntry(**kwargs)


def _pred(key, label, confidence=0.9):
    return Prediction(bibtex_key=key, label=label, confidence=confidence)


class TestBuildResultsMatrix:
    def test_basic_matrix(self):
        entries = [
            BenchmarkEntry(**_entry("a", "HALLUCINATED", 1, "fabricated_doi")),
            BenchmarkEntry(**_entry("b", "VALID")),
        ]
        tool_preds = {
            "tool1": [_pred("a", "HALLUCINATED", 0.9), _pred("b", "VALID", 0.8)],
            "tool2": [_pred("a", "VALID", 0.6), _pred("b", "VALID", 0.9)],
        }
        keys, names, matrix = build_results_matrix(entries, tool_preds)
        assert keys == ["a", "b"]
        assert sorted(names) == ["tool1", "tool2"]
        assert len(matrix) == 2
        assert len(matrix[0]) == 2

    def test_missing_prediction_is_none(self):
        entries = [
            BenchmarkEntry(**_entry("a", "VALID")),
            BenchmarkEntry(**_entry("b", "VALID")),
        ]
        tool_preds = {
            "tool1": [_pred("a", "VALID", 0.9)],  # missing b
        }
        _keys, names, matrix = build_results_matrix(entries, tool_preds)
        # tool1's score on b should be None
        t1_idx = names.index("tool1")
        assert matrix[1][t1_idx] is None

    def test_correct_valid_prediction_scores_confidence(self):
        entries = [BenchmarkEntry(**_entry("a", "VALID"))]
        tool_preds = {"t": [_pred("a", "VALID", 0.9)]}
        _, _, matrix = build_results_matrix(entries, tool_preds)
        # Symmetric scoring: correct → confidence
        assert matrix[0][0] == pytest.approx(0.9)

    def test_incorrect_valid_prediction_scores_1_minus_confidence(self):
        entries = [BenchmarkEntry(**_entry("a", "VALID"))]
        tool_preds = {"t": [_pred("a", "HALLUCINATED", 0.9)]}
        _, _, matrix = build_results_matrix(entries, tool_preds)
        # Symmetric scoring: incorrect → 1 - confidence
        assert matrix[0][0] == pytest.approx(0.1)

    def test_correct_hallucinated_weighted_by_confidence(self):
        entries = [BenchmarkEntry(**_entry("a", "HALLUCINATED", 1, "fabricated_doi"))]
        tool_preds = {"t": [_pred("a", "HALLUCINATED", 0.8)]}
        _, _, matrix = build_results_matrix(entries, tool_preds)
        assert matrix[0][0] == pytest.approx(0.8)  # 1.0 * 0.8

    def test_empty_predictions(self):
        entries = [BenchmarkEntry(**_entry("a", "VALID"))]
        tool_preds = {"t": []}
        _keys, _names, matrix = build_results_matrix(entries, tool_preds)
        assert matrix[0][0] is None


class TestRankToolsMeanScore:
    def test_perfect_tool_ranks_first(self):
        keys = ["a", "b"]
        names = ["good", "bad"]
        matrix = [[1.0, 0.0], [1.0, 0.0]]
        ranking = rank_tools_mean_score(keys, names, matrix)
        assert ranking[0][0] == "good"
        assert ranking[0][1] == 1.0
        assert ranking[1][0] == "bad"
        assert ranking[1][1] == 0.0

    def test_handles_none_values(self):
        keys = ["a", "b"]
        names = ["t1"]
        matrix = [[1.0], [None]]
        ranking = rank_tools_mean_score(keys, names, matrix)
        assert ranking[0][1] == 1.0  # Only one non-None score

    def test_all_none_scores_zero(self):
        keys = ["a"]
        names = ["t1"]
        matrix = [[None]]
        ranking = rank_tools_mean_score(keys, names, matrix)
        assert ranking[0][1] == 0.0


class TestRankTools:
    def test_method_mean(self):
        entries = [
            BenchmarkEntry(**_entry("a", "VALID")),
            BenchmarkEntry(**_entry("b", "HALLUCINATED", 1, "fabricated_doi")),
        ]
        tool_preds = {
            "good": [_pred("a", "VALID", 0.9), _pred("b", "HALLUCINATED", 0.8)],
            "bad": [_pred("a", "HALLUCINATED", 0.9), _pred("b", "VALID", 0.5)],
        }
        ranking = rank_tools(entries, tool_preds, method="mean")
        assert ranking[0][0] == "good"

    def test_invalid_method_raises(self):
        entries = [BenchmarkEntry(**_entry("a", "VALID"))]
        with pytest.raises(ValueError, match="Unknown ranking method"):
            rank_tools(entries, {}, method="invalid")


class TestSaveLoadMatrix:
    def test_roundtrip(self, tmp_path):
        keys = ["a", "b"]
        names = ["t1", "t2"]
        matrix = [[1.0, 0.5], [None, 0.8]]
        path = tmp_path / "matrix.csv"
        save_results_matrix(keys, names, matrix, str(path))
        loaded_keys, loaded_names, loaded_matrix = load_results_matrix(str(path))
        assert loaded_keys == keys
        assert loaded_names == names
        assert loaded_matrix[0][0] == pytest.approx(1.0)
        assert loaded_matrix[0][1] == pytest.approx(0.5)
        assert loaded_matrix[1][0] is None
        assert loaded_matrix[1][1] == pytest.approx(0.8)


def _make_entries_and_preds():
    """Return 4 entries + two tool prediction sets (perfect and random)."""
    entries = [
        BenchmarkEntry(**_entry("a", "VALID")),
        BenchmarkEntry(**_entry("b", "VALID")),
        BenchmarkEntry(**_entry("c", "HALLUCINATED", 1, "fabricated_doi")),
        BenchmarkEntry(**_entry("d", "HALLUCINATED", 2, "near_miss_title")),
    ]
    perfect_preds = [
        _pred("a", "VALID", 0.95),
        _pred("b", "VALID", 0.95),
        _pred("c", "HALLUCINATED", 0.95),
        _pred("d", "HALLUCINATED", 0.95),
    ]
    random_preds = [
        _pred("a", "HALLUCINATED", 0.5),  # wrong
        _pred("b", "VALID", 0.5),
        _pred("c", "VALID", 0.5),  # wrong
        _pred("d", "HALLUCINATED", 0.5),
    ]
    return entries, perfect_preds, random_preds


class TestPerfectToolRanksHigher:
    """F-19a: A perfect tool should rank above a random/bad tool."""

    def test_perfect_tool_ranks_higher_mean(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"perfect": perfect_preds, "random": random_preds}
        ranking = rank_tools(entries, tool_preds, method="mean")
        names = [r[0] for r in ranking]
        assert names[0] == "perfect", f"Expected 'perfect' first, got {names}"

    def test_scores_ordered_descending(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"perfect": perfect_preds, "random": random_preds}
        ranking = rank_tools(entries, tool_preds, method="mean")
        scores = [r[1] for r in ranking]
        assert scores[0] >= scores[1]


class TestRankToolsMeanMethod:
    """F-19b: rank_tools(..., method='mean') — correct ordering."""

    def test_rank_tools_mean_method(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"alpha": perfect_preds, "beta": random_preds}
        ranking = rank_tools(entries, tool_preds, method="mean")
        # Result must be list of (name, score) tuples
        assert isinstance(ranking, list)
        assert len(ranking) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranking)
        # Perfect tool ranks first
        assert ranking[0][0] == "alpha"
        assert ranking[0][1] > ranking[1][1]

    def test_rank_tools_mean_three_tools(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        # Add a third "zero" tool that gets everything wrong
        zero_preds = [
            _pred("a", "HALLUCINATED", 0.99),
            _pred("b", "HALLUCINATED", 0.99),
            _pred("c", "VALID", 0.99),
            _pred("d", "VALID", 0.99),
        ]
        tool_preds = {
            "perfect": perfect_preds,
            "random": random_preds,
            "zero": zero_preds,
        }
        ranking = rank_tools(entries, tool_preds, method="mean")
        names = [r[0] for r in ranking]
        assert names[0] == "perfect"
        assert names[-1] == "zero"


@pytest.mark.skipif(not HAS_CHOIX, reason="choix not installed")
class TestPlackettLuceRanking:
    """F-19c: Plackett-Luce ranking when choix is available."""

    def test_plackett_luce_returns_ranking(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"perfect": perfect_preds, "random": random_preds}
        ranking = rank_tools(entries, tool_preds, method="plackett_luce")
        assert isinstance(ranking, list)
        assert len(ranking) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranking)

    def test_plackett_luce_perfect_ranks_first(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"perfect": perfect_preds, "random": random_preds}
        ranking = rank_tools(entries, tool_preds, method="plackett_luce")
        assert ranking[0][0] == "perfect"

    def test_plackett_luce_scores_are_positive(self):
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"perfect": perfect_preds, "random": random_preds}
        ranking = rank_tools(entries, tool_preds, method="plackett_luce")
        for _name, score in ranking:
            assert score > 0.0

    def test_rank_tools_plackett_luce_with_ci(self):
        """rank_tools_plackett_luce with compute_ci=True returns (ranked, ci_dict)."""
        entries, perfect_preds, random_preds = _make_entries_and_preds()
        tool_preds = {"perfect": perfect_preds, "random": random_preds}
        entry_keys, tool_names, matrix = build_results_matrix(entries, tool_preds)

        result = rank_tools_plackett_luce(
            entry_keys, tool_names, matrix, compute_ci=True, n_bootstrap=50, seed=42
        )
        ranked, cis = result
        assert isinstance(ranked, list)
        assert isinstance(cis, dict)
        # CI tuples must be (lower, upper) with lower <= upper
        for tool_name, (lo, hi) in cis.items():
            assert lo <= hi, f"CI for {tool_name} has lo > hi: ({lo}, {hi})"
