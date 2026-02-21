"""Tests for the prescreening module."""

from unittest.mock import Mock, patch

import httpx
import pytest

from hallmark.baselines.prescreening import (
    PrescreeningBreakdown,
    PreScreenResult,
    check_author_heuristics,
    check_doi_resolves,
    check_year_bounds,
    compute_prescreening_breakdown,
    format_prescreening_breakdown,
    merge_with_predictions,
    prescreen_entry,
)
from hallmark.dataset.schema import BenchmarkEntry, Prediction


class TestCheckYearBounds:
    """Tests for check_year_bounds function."""

    def test_future_year(self):
        """Year 2099 should be flagged as HALLUCINATED with high confidence."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "2099"},
            label="VALID",
        )
        result = check_year_bounds(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.95
        assert "future" in result.reason.lower()
        assert result.check_name == "check_year_bounds"

    def test_normal_year(self):
        """Year 2023 should be flagged as VALID."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "2023"},
            label="VALID",
        )
        result = check_year_bounds(entry)
        assert result.label == "VALID"
        assert result.confidence == 0.60
        assert "plausible" in result.reason.lower()
        assert result.check_name == "check_year_bounds"

    def test_old_year(self):
        """Year 1800 should be flagged as HALLUCINATED with medium confidence."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "1800"},
            label="VALID",
        )
        result = check_year_bounds(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.70
        assert "implausibly old" in result.reason.lower()
        assert result.check_name == "check_year_bounds"

    def test_missing_year(self):
        """No year field should return UNKNOWN."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test"},
            label="VALID",
        )
        result = check_year_bounds(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "no year" in result.reason.lower()
        assert result.check_name == "check_year_bounds"

    def test_non_numeric_year(self):
        """Non-numeric year should return UNKNOWN."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "TBD"},
            label="VALID",
        )
        result = check_year_bounds(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "non-numeric" in result.reason.lower()
        assert result.check_name == "check_year_bounds"


class TestCheckAuthorHeuristics:
    """Tests for check_author_heuristics function."""

    def test_placeholder_author1(self):
        """Author1 and Author2 pattern should be flagged as HALLUCINATED."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "author": "Author1 and Author2"},
            label="VALID",
        )
        result = check_author_heuristics(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.80
        assert "synthetic" in result.reason.lower()
        assert result.check_name == "check_author_heuristics"

    def test_et_al_sole_author(self):
        """'et al.' as sole author should be flagged as HALLUCINATED."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "author": "et al."},
            label="VALID",
        )
        result = check_author_heuristics(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.80
        assert "et al" in result.reason.lower()
        assert result.check_name == "check_author_heuristics"

    def test_normal_authors(self):
        """Normal authors should return UNKNOWN (not VALID)."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "author": "Smith, John and Doe, Jane"},
            label="VALID",
        )
        result = check_author_heuristics(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "no placeholder" in result.reason.lower()
        assert result.check_name == "check_author_heuristics"

    def test_short_author(self):
        """Single letter author should be flagged as HALLUCINATED."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "author": "A"},
            label="VALID",
        )
        result = check_author_heuristics(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.80
        assert "too short" in result.reason.lower()
        assert result.check_name == "check_author_heuristics"

    def test_single_letter_lastnames(self):
        """All authors with single-letter last names should be flagged as HALLUCINATED."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "author": "A, John and B, Jane"},
            label="VALID",
        )
        result = check_author_heuristics(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.80
        assert "single-letter last names" in result.reason.lower()
        assert result.check_name == "check_author_heuristics"

    def test_missing_author(self):
        """No author field should return UNKNOWN."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test"},
            label="VALID",
        )
        result = check_author_heuristics(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "no author" in result.reason.lower()
        assert result.check_name == "check_author_heuristics"


class TestCheckDoiResolves:
    """Tests for check_doi_resolves function."""

    def test_no_doi(self):
        """Entry without DOI should return UNKNOWN."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test"},
            label="VALID",
        )
        result = check_doi_resolves(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "no doi" in result.reason.lower()
        assert result.check_name == "check_doi_resolves"

    def test_malformed_doi(self):
        """Malformed DOI should return UNKNOWN."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "doi": "not-a-doi"},
            label="VALID",
        )
        result = check_doi_resolves(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "malformed" in result.reason.lower()
        assert result.check_name == "check_doi_resolves"

    @patch("hallmark.baselines.prescreening.httpx.head")
    def test_doi_resolves(self, mock_head):
        """DOI that resolves (200 response) should return VALID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "doi": "10.1234/test"},
            label="VALID",
        )
        result = check_doi_resolves(entry)
        assert result.label == "VALID"
        assert result.confidence == 0.85
        assert "resolves successfully" in result.reason.lower()
        assert result.check_name == "check_doi_resolves"
        mock_head.assert_called_once_with(
            "https://doi.org/10.1234/test", timeout=10.0, follow_redirects=True
        )

    @patch("hallmark.baselines.prescreening.httpx.head")
    def test_doi_not_found(self, mock_head):
        """DOI that returns 404 should be flagged as HALLUCINATED."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response

        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "doi": "10.1234/nonexistent"},
            label="VALID",
        )
        result = check_doi_resolves(entry)
        assert result.label == "HALLUCINATED"
        assert result.confidence == 0.85
        assert "404" in result.reason
        assert result.check_name == "check_doi_resolves"

    @patch("hallmark.baselines.prescreening.httpx.head")
    def test_doi_network_error(self, mock_head):
        """Network error should return UNKNOWN."""
        mock_head.side_effect = httpx.RequestError("Connection failed")

        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "doi": "10.1234/test"},
            label="VALID",
        )
        result = check_doi_resolves(entry)
        assert result.label == "UNKNOWN"
        assert result.confidence == 0.0
        assert "network error" in result.reason.lower()
        assert result.check_name == "check_doi_resolves"


class TestPrescreenEntry:
    """Tests for prescreen_entry function."""

    @patch("hallmark.baselines.prescreening.httpx.head")
    def test_runs_all_checks(self, mock_head):
        """Should run all three checks and return three results."""
        # Mock DOI check to avoid network call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={
                "title": "Test",
                "author": "Smith, John",
                "year": "2023",
                "doi": "10.1234/test",
            },
            label="VALID",
        )
        results = prescreen_entry(entry)

        # Should have 3 results (one per check)
        assert len(results) == 3

        # Verify check names
        check_names = {r.check_name for r in results}
        assert check_names == {
            "check_doi_resolves",
            "check_year_bounds",
            "check_author_heuristics",
        }

        # All should be PreScreenResult instances
        assert all(isinstance(r, PreScreenResult) for r in results)


class TestMergeWithPredictions:
    """Tests for merge_with_predictions function."""

    def test_prescreen_overrides_tool_valid(self):
        """Pre-screening HALLUCINATED + tool VALID should result in HALLUCINATED."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "2099"},
            label="VALID",
        )

        tool_pred = Prediction(
            bibtex_key="test_key",
            label="VALID",
            confidence=0.80,
            reason="Tool says valid",
            subtest_results={},
            api_sources_queried=["CrossRef"],
            wall_clock_seconds=1.0,
            api_calls=1,
        )

        prescreen_results = {
            "test_key": [
                PreScreenResult(
                    label="HALLUCINATED",
                    confidence=0.95,
                    reason="Year is in future",
                    check_name="check_year_bounds",
                )
            ]
        }

        merged = merge_with_predictions([entry], [tool_pred], prescreen_results)

        assert len(merged) == 1
        assert merged[0].label == "HALLUCINATED"
        assert merged[0].confidence == 0.95
        assert "override" in merged[0].reason.lower()
        assert "check_year_bounds" in merged[0].subtest_results

    def test_prescreen_unknown_keeps_tool(self):
        """Pre-screening UNKNOWN + tool VALID should result in VALID."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test"},
            label="VALID",
        )

        tool_pred = Prediction(
            bibtex_key="test_key",
            label="VALID",
            confidence=0.80,
            reason="Tool says valid",
            subtest_results={},
            api_sources_queried=["CrossRef"],
            wall_clock_seconds=1.0,
            api_calls=1,
        )

        prescreen_results = {
            "test_key": [
                PreScreenResult(
                    label="UNKNOWN",
                    confidence=0.0,
                    reason="No year field",
                    check_name="check_year_bounds",
                )
            ]
        }

        merged = merge_with_predictions([entry], [tool_pred], prescreen_results)

        assert len(merged) == 1
        assert merged[0].label == "VALID"
        assert merged[0].confidence == 0.80
        assert merged[0].reason == "Tool says valid"

    def test_both_hallucinated_keeps_higher_confidence(self):
        """Both HALLUCINATED should keep higher confidence."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "2099"},
            label="VALID",
        )

        tool_pred = Prediction(
            bibtex_key="test_key",
            label="HALLUCINATED",
            confidence=0.70,
            reason="Tool detected issue",
            subtest_results={"tool_check": False},
            api_sources_queried=["CrossRef"],
            wall_clock_seconds=1.0,
            api_calls=1,
        )

        prescreen_results = {
            "test_key": [
                PreScreenResult(
                    label="HALLUCINATED",
                    confidence=0.95,
                    reason="Year is in future",
                    check_name="check_year_bounds",
                )
            ]
        }

        merged = merge_with_predictions([entry], [tool_pred], prescreen_results)

        assert len(merged) == 1
        assert merged[0].label == "HALLUCINATED"
        assert merged[0].confidence == 0.95
        assert "confirms" in merged[0].reason.lower()

    def test_no_tool_prediction_uses_prescreen(self):
        """No tool prediction + pre-screening HALLUCINATED should result in HALLUCINATED."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test", "year": "2099"},
            label="VALID",
        )

        prescreen_results = {
            "test_key": [
                PreScreenResult(
                    label="HALLUCINATED",
                    confidence=0.95,
                    reason="Year is in future",
                    check_name="check_year_bounds",
                )
            ]
        }

        merged = merge_with_predictions([entry], [], prescreen_results)

        assert len(merged) == 1
        assert merged[0].label == "HALLUCINATED"
        assert merged[0].confidence == 0.95
        assert "pre-screening" in merged[0].reason.lower()

    def test_no_tool_no_prescreen(self):
        """No tool prediction + no strong pre-screening should result in VALID with low confidence."""
        entry = BenchmarkEntry(
            bibtex_key="test_key",
            bibtex_type="article",
            fields={"title": "Test"},
            label="VALID",
        )

        prescreen_results = {
            "test_key": [
                PreScreenResult(
                    label="UNKNOWN",
                    confidence=0.0,
                    reason="No year field",
                    check_name="check_year_bounds",
                )
            ]
        }

        merged = merge_with_predictions([entry], [], prescreen_results)

        assert len(merged) == 1
        assert merged[0].label == "VALID"
        assert merged[0].confidence == 0.0
        assert "no tool prediction" in merged[0].reason.lower()


class TestPrescreeningBreakdown:
    """Tests for compute_prescreening_breakdown and format_prescreening_breakdown."""

    def _make_pred(self, key: str, label: str, reason: str = "") -> Prediction:
        return Prediction(bibtex_key=key, label=label, confidence=0.8, reason=reason)

    def test_no_overrides(self):
        """All tool-only predictions — override count is zero."""
        preds = [
            self._make_pred("a", "HALLUCINATED", "Tool detected issue"),
            self._make_pred("b", "VALID", "Looks fine"),
        ]
        true_labels = {"a": "HALLUCINATED", "b": "VALID"}

        bd = compute_prescreening_breakdown(preds, true_labels)

        assert bd.override_count == 0
        assert bd.tool_only_total == 2
        assert bd.tool_only_correct == 2
        assert bd.override_accuracy is None
        assert bd.tool_only_accuracy == 1.0

    def test_all_overrides_correct(self):
        """All predictions are overrides and all are correct."""
        preds = [
            self._make_pred(
                "a", "HALLUCINATED", "Tool ok | [Pre-screening override] Year in future"
            ),
            self._make_pred(
                "b", "HALLUCINATED", "Tool ok | [Pre-screening override] Author placeholder"
            ),
        ]
        true_labels = {"a": "HALLUCINATED", "b": "HALLUCINATED"}

        bd = compute_prescreening_breakdown(preds, true_labels)

        assert bd.override_count == 2
        assert bd.override_correct == 2
        assert bd.tool_only_total == 0
        assert bd.override_accuracy == 1.0
        assert bd.tool_only_accuracy is None

    def test_mixed_overrides(self):
        """One correct override, one wrong override, one correct tool-only prediction."""
        preds = [
            self._make_pred("a", "HALLUCINATED", "Tool ok | [Pre-screening override] DOI 404"),
            self._make_pred("b", "HALLUCINATED", "Tool ok | [Pre-screening override] Year future"),
            self._make_pred("c", "VALID", "Tool says valid"),
        ]
        true_labels = {"a": "HALLUCINATED", "b": "VALID", "c": "VALID"}

        bd = compute_prescreening_breakdown(preds, true_labels)

        assert bd.total == 3
        assert bd.override_count == 2
        assert bd.override_correct == 1
        assert bd.tool_only_total == 1
        assert bd.tool_only_correct == 1
        assert bd.override_accuracy == pytest.approx(0.5)
        assert bd.tool_only_accuracy == pytest.approx(1.0)

    def test_format_output_contains_expected_lines(self):
        """format_prescreening_breakdown produces the expected summary lines."""
        bd = PrescreeningBreakdown(
            total=100,
            override_count=10,
            override_correct=9,
            tool_only_total=90,
            tool_only_correct=81,
        )
        text = format_prescreening_breakdown(bd)

        assert "Pre-screening breakdown" in text
        assert "10/100" in text
        assert "9/10" in text
        assert "81/90" in text

    def test_format_no_overrides(self):
        """format_prescreening_breakdown says N/A when there are no overrides."""
        bd = PrescreeningBreakdown(
            total=50,
            override_count=0,
            override_correct=0,
            tool_only_total=50,
            tool_only_correct=45,
        )
        text = format_prescreening_breakdown(bd)

        assert "N/A" in text
        assert "45/50" in text
