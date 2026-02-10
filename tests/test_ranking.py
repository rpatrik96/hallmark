"""Tests for hallmark.evaluation.ranking module."""

from __future__ import annotations

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.ranking import (
    build_results_matrix,
    load_results_matrix,
    rank_tools,
    rank_tools_mean_score,
    save_results_matrix,
)


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

    def test_correct_valid_prediction_scores_1(self):
        entries = [BenchmarkEntry(**_entry("a", "VALID"))]
        tool_preds = {"t": [_pred("a", "VALID", 0.9)]}
        _, _, matrix = build_results_matrix(entries, tool_preds)
        assert matrix[0][0] == 1.0

    def test_incorrect_valid_prediction_scores_0(self):
        entries = [BenchmarkEntry(**_entry("a", "VALID"))]
        tool_preds = {"t": [_pred("a", "HALLUCINATED", 0.9)]}
        _, _, matrix = build_results_matrix(entries, tool_preds)
        assert matrix[0][0] == 0.0

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
