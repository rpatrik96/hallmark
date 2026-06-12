"""Tests for the bibtex-check (bibtex-updater) baseline wrapper.

Covers ``_parse_jsonl_output`` across the tool's output-contract versions:
pre-1.2.0 records, 1.2.0 realness records (``confidence_score`` /
``abstained``), and post-1.2.0 records carrying ``coverage_incomplete`` and
``p_valid``.  No subprocess or network — JSONL output is faked on disk,
following the pattern of ``TestParseJsonlToRaw`` in
``test_llm_tool_augmented.py``.

These tests live in their own module (not ``test_baselines.py``) because that
module is skipped entirely when the optional ``openai`` dependency is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hallmark.baselines.bibtexupdater import (
    STATUS_TO_CONFIDENCE,
    STATUS_TO_LABEL,
    _parse_jsonl_output,
)
from hallmark.dataset.schema import Prediction


def _parse(tmp_path: Path, records: list[dict[str, Any]]) -> list[Prediction]:
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return _parse_jsonl_output(jsonl_path, 1.0, len(records))


class TestStatusMaps:
    def test_new_problem_statuses_mapped(self) -> None:
        assert STATUS_TO_LABEL["nonexistent_venue"] == "HALLUCINATED"
        assert STATUS_TO_LABEL["unpublished_at_claimed_venue"] == "HALLUCINATED"
        assert STATUS_TO_CONFIDENCE["nonexistent_venue"] == 0.85
        assert STATUS_TO_CONFIDENCE["unpublished_at_claimed_venue"] == 0.75

    def test_default_mode_statuses_already_mapped(self) -> None:
        """Statuses the upgraded tool now emits in default mode were already
        mapped — keep them HALLUCINATED."""
        assert STATUS_TO_LABEL["author_truncated"] == "HALLUCINATED"
        assert STATUS_TO_LABEL["preprint_only"] == "HALLUCINATED"

    def test_every_label_status_has_a_confidence(self) -> None:
        assert set(STATUS_TO_LABEL) == set(STATUS_TO_CONFIDENCE)


class TestParseJsonlOutput:
    """``_parse_jsonl_output`` across bibtex-check output-contract versions.

    The post-1.2.0 tool adds ``coverage_incomplete`` and ``p_valid`` fields
    plus new problem statuses; old-format records (including the precomputed
    reference results) must keep parsing with identical label and confidence.
    """

    def test_old_format_records_parse_identically(self, tmp_path: Path) -> None:
        """Regression pin: pre-1.2.0-shaped records (no confidence_score /
        abstained / p_valid / coverage_incomplete) keep label AND confidence
        exactly as before."""
        records: list[dict[str, Any]] = [
            {
                "key": "a",
                "status": "not_found",
                "confidence": 0.8,
                "mismatched_fields": [],
                "api_sources": ["crossref"],
                "errors": [],
            },
            {
                "key": "b",
                "status": "verified",
                "confidence": 0.95,
                "mismatched_fields": [],
                "api_sources": ["dblp"],
                "errors": [],
            },
            # No confidence field at all → STATUS_TO_CONFIDENCE fallback.
            {"key": "c", "status": "venue_mismatch"},
        ]
        preds = {p.bibtex_key: p for p in _parse(tmp_path, records)}
        assert preds["a"].label == "HALLUCINATED"
        assert preds["a"].confidence == 0.8
        assert preds["a"].reason == "Status: not_found"
        assert preds["b"].label == "VALID"
        assert preds["b"].confidence == 0.95
        assert preds["c"].label == "HALLUCINATED"
        assert preds["c"].confidence == 0.80

    def test_v12_format_inversion_heuristic_still_applies(self, tmp_path: Path) -> None:
        """1.2.0-shaped records (confidence_score/abstained but no p_valid)
        keep the realness-inversion heuristic for HALLUCINATED labels."""
        records: list[dict[str, Any]] = [
            {
                "key": "m",
                "status": "title_mismatch",
                "confidence": 0.1,
                "confidence_score": 10.0,
                "abstained": False,
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == pytest.approx(0.9)  # 1 - 0.1

    def test_new_format_p_valid_on_valid_status(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "v",
                "status": "verified",
                "confidence": 0.88,
                "abstained": False,
                "coverage_incomplete": False,
                "p_valid": 0.94,
                "confidence_score": 88.0,
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "VALID"
        assert pred.confidence == pytest.approx(0.94)

    def test_new_format_p_valid_on_hallucinated_status(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "h",
                "status": "nonexistent_venue",
                "confidence": 0.78,
                "abstained": False,
                "coverage_incomplete": False,
                "p_valid": 0.11,
                "confidence_score": 78.0,
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == pytest.approx(0.89)  # 1 - p_valid

    def test_not_found_coverage_incomplete_is_abstention(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "x",
                "status": "not_found",
                "confidence": 0.45,
                "abstained": True,
                "coverage_incomplete": True,
                "p_valid": 0.5,
                "confidence_score": 45.0,
                "errors": ["semanticscholar: 429 Too Many Requests"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "VALID"
        assert pred.confidence == pytest.approx(0.45)
        # Reason explains the abstention while keeping the leading raw-status
        # segment that run_bibtex_check_with_status parses for the cascade.
        assert pred.reason.startswith("Status: not_found")
        assert "incomplete" in pred.reason
        assert "throttling" in pred.reason

    def test_not_found_clean_miss_still_hallucinated(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "y",
                "status": "not_found",
                "confidence": 0.45,
                "abstained": True,
                "coverage_incomplete": False,
                "p_valid": 0.35,
                "confidence_score": 45.0,
                "errors": [],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "HALLUCINATED"
        assert pred.confidence == pytest.approx(0.65)  # 1 - p_valid

    def test_coverage_incomplete_informational_for_other_statuses(self, tmp_path: Path) -> None:
        """coverage_incomplete only rewrites not_found; api_error keeps its
        conservative-VALID mapping with p_valid-derived confidence."""
        records: list[dict[str, Any]] = [
            {
                "key": "e",
                "status": "api_error",
                "confidence": 0.0,
                "abstained": True,
                "coverage_incomplete": True,
                "p_valid": 0.5,
                "confidence_score": 0.0,
                "errors": ["Exception: boom"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "VALID"
        assert pred.confidence == pytest.approx(0.5)
        assert "abstention" not in pred.reason
