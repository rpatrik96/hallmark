"""Tests for hallmark.evaluation.aggregator."""

import importlib.util

import pytest

from hallmark.evaluation.aggregator import SparseEvaluation, aggregate_scores

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sparse(
    tool_entry_subtest_score: list[tuple[str, str, str, float | None]],
) -> SparseEvaluation:
    """Build a SparseEvaluation from (tool, entry, subtest, score) tuples."""
    se = SparseEvaluation()
    for tool, entry, subtest, score in tool_entry_subtest_score:
        se.add_score(tool, entry, subtest, score)
    return se


# ---------------------------------------------------------------------------
# SparseEvaluation basic behaviour
# ---------------------------------------------------------------------------


class TestSparseEvaluation:
    def test_add_and_get_score(self):
        se = SparseEvaluation()
        se.add_score("tool_a", "entry_1", "doi_resolves", 1.0)
        assert se.get_score("tool_a", "entry_1", "doi_resolves") == 1.0

    def test_missing_score_returns_none(self):
        se = SparseEvaluation()
        assert se.get_score("tool_a", "missing_entry", "doi_resolves") is None

    def test_sets_are_updated(self):
        se = SparseEvaluation()
        se.add_score("tool_a", "entry_1", "doi_resolves", 1.0)
        assert "tool_a" in se.tool_names
        assert "entry_1" in se.entry_keys
        assert "doi_resolves" in se.subtest_names

    def test_coverage_full(self):
        se = SparseEvaluation()
        se.add_score("tool_a", "entry_1", "sub1", 1.0)
        se.add_score("tool_a", "entry_1", "sub2", 0.0)
        assert se.coverage("tool_a") == pytest.approx(1.0)

    def test_coverage_partial(self):
        se = SparseEvaluation()
        # 2 entries x 2 subtests = 4 cells total; tool_a covers 2
        se.add_score("tool_a", "e1", "s1", 1.0)
        se.add_score("tool_a", "e1", "s2", 0.0)
        se.add_score("tool_b", "e2", "s1", 1.0)
        se.add_score("tool_b", "e2", "s2", 0.0)
        # tool_a: 2 of 4 total (e1/s1, e1/s2)
        assert se.coverage("tool_a") == pytest.approx(0.5)

    def test_coverage_empty(self):
        se = SparseEvaluation()
        assert se.coverage("nonexistent") == pytest.approx(0.0)

    def test_none_score_stored(self):
        se = SparseEvaluation()
        se.add_score("tool_a", "e1", "s1", None)
        assert se.get_score("tool_a", "e1", "s1") is None


# ---------------------------------------------------------------------------
# aggregate_scores — mean_of_means
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
class TestAggregateMeanOfMeans:
    def test_single_tool_returns_one_result(self):
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 0.8),
                ("tool_a", "e1", "s2", 0.6),
            ]
        )
        results = aggregate_scores(se, method="mean_of_means")
        assert len(results) == 1
        assert results[0].tool_name == "tool_a"

    def test_single_tool_correct_score(self):
        # 2 subtests: s1 mean=0.8, s2 mean=0.6 → overall=(0.8+0.6)/2=0.7
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 0.8),
                ("tool_a", "e2", "s1", 0.8),
                ("tool_a", "e1", "s2", 0.6),
                ("tool_a", "e2", "s2", 0.6),
            ]
        )
        results = aggregate_scores(se, method="mean_of_means")
        assert results[0].overall_score == pytest.approx(0.7)

    def test_two_tools_identical_scores_tie(self):
        rows = [
            ("tool_a", "e1", "s1", 0.8),
            ("tool_b", "e1", "s1", 0.8),
        ]
        se = _build_sparse(rows)
        results = aggregate_scores(se, method="mean_of_means")
        assert len(results) == 2
        assert results[0].overall_score == pytest.approx(results[1].overall_score)

    def test_sorted_descending_by_score(self):
        se = _build_sparse(
            [
                ("tool_low", "e1", "s1", 0.2),
                ("tool_high", "e1", "s1", 0.9),
            ]
        )
        results = aggregate_scores(se, method="mean_of_means")
        assert results[0].overall_score >= results[1].overall_score

    def test_missing_data_handled(self):
        # tool_a covers all, tool_b only covers e1
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 1.0),
                ("tool_a", "e2", "s1", 1.0),
                ("tool_b", "e1", "s1", 0.5),
                # tool_b has no score for e2
            ]
        )
        se.add_score("tool_b", "e2", "s1", None)
        results = aggregate_scores(se, method="mean_of_means")
        # tool_a should score higher than tool_b
        tool_scores = {r.tool_name: r.overall_score for r in results}
        assert tool_scores["tool_a"] > tool_scores["tool_b"]


# ---------------------------------------------------------------------------
# aggregate_scores — entry_mean
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
class TestAggregateEntryMean:
    def test_single_tool_returns_one_result(self):
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 1.0),
            ]
        )
        results = aggregate_scores(se, method="entry_mean")
        assert len(results) == 1

    def test_correct_score(self):
        # e1: mean(1.0, 0.0) = 0.5; e2: mean(1.0, 1.0) = 1.0 → overall = 0.75
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 1.0),
                ("tool_a", "e1", "s2", 0.0),
                ("tool_a", "e2", "s1", 1.0),
                ("tool_a", "e2", "s2", 1.0),
            ]
        )
        results = aggregate_scores(se, method="entry_mean")
        assert results[0].overall_score == pytest.approx(0.75)

    def test_two_tools_identical_tie(self):
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 0.5),
                ("tool_b", "e1", "s1", 0.5),
            ]
        )
        results = aggregate_scores(se, method="entry_mean")
        assert results[0].overall_score == pytest.approx(results[1].overall_score)


# ---------------------------------------------------------------------------
# aggregate_scores — pairwise
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
class TestAggregatePairwise:
    def test_better_tool_wins(self):
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 0.9),
                ("tool_b", "e1", "s1", 0.1),
            ]
        )
        results = aggregate_scores(se, method="pairwise")
        assert len(results) == 2
        tool_scores = {r.tool_name: r.overall_score for r in results}
        assert tool_scores["tool_a"] > tool_scores["tool_b"]

    def test_tie_produces_equal_scores(self):
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 0.5),
                ("tool_b", "e1", "s1", 0.5),
            ]
        )
        results = aggregate_scores(se, method="pairwise")
        assert results[0].overall_score == pytest.approx(results[1].overall_score)

    def test_coverage_tracked(self):
        se = _build_sparse(
            [
                ("tool_a", "e1", "s1", 1.0),
                ("tool_b", "e1", "s1", 0.0),
            ]
        )
        results = aggregate_scores(se, method="pairwise")
        for r in results:
            assert 0.0 <= r.coverage <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
class TestAggregateEdgeCases:
    def test_empty_sparse_evaluation_returns_empty(self):
        se = SparseEvaluation()
        for method in ("mean_of_means", "entry_mean", "pairwise"):
            results = aggregate_scores(se, method=method)
            assert results == [], f"Expected empty for method {method}"

    def test_all_none_scores(self):
        se = SparseEvaluation()
        se.add_score("tool_a", "e1", "s1", None)
        se.add_score("tool_a", "e1", "s2", None)
        results = aggregate_scores(se, method="mean_of_means")
        # No non-None scores → overall_score should be 0.0
        assert len(results) == 1
        assert results[0].overall_score == pytest.approx(0.0)

    def test_single_entry_mean_of_means(self):
        se = _build_sparse([("only_tool", "only_entry", "only_subtest", 0.75)])
        results = aggregate_scores(se, method="mean_of_means")
        assert len(results) == 1
        assert results[0].overall_score == pytest.approx(0.75)

    def test_invalid_method_raises(self):
        se = _build_sparse([("tool_a", "e1", "s1", 1.0)])
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_scores(se, method="invalid_method")


@pytest.mark.skipif(HAS_NUMPY, reason="only runs when numpy is absent")
class TestAggregateRequiresNumpy:
    def test_raises_import_error_without_numpy(self):
        se = SparseEvaluation()
        with pytest.raises(ImportError, match="numpy"):
            aggregate_scores(se, method="mean_of_means")
