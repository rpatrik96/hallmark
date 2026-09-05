"""Tests for the bibtex-check (bibtex-updater) baseline wrapper.

Covers ``_parse_jsonl_output`` across the tool's output-contract versions:
pre-1.2.0 records, 1.2.0 realness records (``confidence_score`` /
``abstained``), and post-1.2.0 records carrying ``coverage_incomplete`` and
``p_valid``.  No subprocess or network — JSONL output is faked on disk,
following the pattern of ``TestParseJsonlToRaw`` in
``test_llm_tool_augmented.py``.

Also covers the batch-level sanity check (``assess_batch_health``), added after a
2026-09-02 wifi outage made bibtex-check return ``not_found`` for 2,500
consecutive references: every source lookup failed DNS resolution, and nothing in
HALLMARK noticed that a whole batch had arrived with no database evidence behind
it.

These tests live in their own module (not ``test_baselines.py``) because that
module is skipped entirely when the optional ``openai`` dependency is absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from hallmark.baselines.bibtexupdater import (
    MIN_BATCH_FOR_HEALTH_CHECK,
    NOT_FOUND_SHARE_THRESHOLD,
    STATUS_TO_CONFIDENCE,
    STATUS_TO_LABEL,
    _parse_jsonl_output,
    assess_batch_health,
    run_bibtex_check_with_health,
    run_bibtex_check_with_status,
)
from hallmark.dataset.schema import BlindEntry, Prediction


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
        # UNCERTAIN, not VALID: the lookup never completed, so the entry is
        # unanswered rather than cleared. Writing it as a committed VALID is
        # what made ``coverage`` read 1.0 on runs full of abstentions
        # (tests/test_abstention_is_not_a_verdict.py).
        assert pred.label == "UNCERTAIN"
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
        """coverage_incomplete only rewrites not_found; api_error is already an
        abstention by status and keeps its p_valid-derived confidence."""
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
        # ``coverage_incomplete`` still rewrites nothing here -- api_error is an
        # abstention on its own account, by status, and is reported as one.
        assert pred.label == "UNCERTAIN"
        assert pred.confidence == pytest.approx(0.5)
        assert "throttling" not in pred.reason


class TestFieldRendering:
    """``mismatched_fields`` vs ``unconfirmed_fields`` in ``reason`` (issue #37).

    Through bibtex-updater 1.10.3 a field the checker declined to compare was
    reported under ``mismatched_fields``; 1.11.0 narrows that key to real
    contradictions and moves abstentions to the additive ``unconfirmed_fields``.
    The label is derived from ``status`` alone, so only ``reason`` changes, and
    the two kinds of finding must read differently to a human auditing it.
    """

    def test_both_keys_present_render_distinctly(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "k",
                "status": "unconfirmed",
                "p_valid": 0.5,
                "mismatched_fields": ["year"],
                "unconfirmed_fields": ["venue"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.reason.startswith("Status: unconfirmed")
        assert "Mismatched: ['year']" in pred.reason
        assert "Unconfirmed (not compared): ['venue']" in pred.reason
        # The contradiction is listed before the abstention.
        assert pred.reason.index("Mismatched:") < pred.reason.index("Unconfirmed (not compared):")

    def test_pre_1_11_record_without_unconfirmed_key(self, tmp_path: Path) -> None:
        """Older releases have no ``unconfirmed_fields``; the reason reads
        exactly as before."""
        records: list[dict[str, Any]] = [
            {
                "key": "k",
                "status": "venue_mismatch",
                "confidence": 0.8,
                "mismatched_fields": ["venue"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "HALLUCINATED"
        assert pred.reason == "Status: venue_mismatch; Mismatched: ['venue']"
        assert "Unconfirmed" not in pred.reason

    def test_only_unconfirmed_present_is_not_called_a_mismatch(self, tmp_path: Path) -> None:
        """The 1.11.0 case the issue is about: a venue abstention must no
        longer be rendered as ``Mismatched: ['venue']``."""
        records: list[dict[str, Any]] = [
            {
                "key": "k",
                "status": "unconfirmed",
                "p_valid": 0.5,
                "mismatched_fields": [],
                "unconfirmed_fields": ["venue"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert "Mismatched" not in pred.reason
        assert "Unconfirmed (not compared): ['venue']" in pred.reason

    def test_both_keys_empty_add_nothing(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "k",
                "status": "verified",
                "p_valid": 0.94,
                "mismatched_fields": [],
                "unconfirmed_fields": [],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.reason == "Status: verified"

    def test_null_field_lists_are_tolerated(self, tmp_path: Path) -> None:
        records: list[dict[str, Any]] = [
            {
                "key": "k",
                "status": "verified",
                "mismatched_fields": None,
                "unconfirmed_fields": None,
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.reason == "Status: verified"

    def test_label_is_unaffected_by_field_lists(self, tmp_path: Path) -> None:
        """Scope check from the issue: the label comes from ``status`` only."""
        records: list[dict[str, Any]] = [
            {
                "key": "k",
                "status": "verified",
                "p_valid": 0.9,
                "mismatched_fields": ["venue"],
                "unconfirmed_fields": ["year"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "VALID"


class TestTransportStatusMapping:
    """A failed lookup is an abstention, never evidence of fabrication."""

    def test_network_error_is_a_conservative_valid(self) -> None:
        assert STATUS_TO_LABEL["network_error"] == "VALID"
        assert STATUS_TO_CONFIDENCE["network_error"] == 0.30

    def test_coverage_incomplete_status_is_a_conservative_valid(self) -> None:
        assert STATUS_TO_LABEL["coverage_incomplete"] == "VALID"
        assert STATUS_TO_CONFIDENCE["coverage_incomplete"] == 0.45

    def test_network_error_record_never_parses_to_hallucinated(self, tmp_path: Path) -> None:
        """The status the upgraded tool emits for a DNS/connection failure."""
        records: list[dict[str, Any]] = [
            {
                "key": "n",
                "status": "network_error",
                "confidence": 0.0,
                "mismatched_fields": [],
                "api_sources": [],
                "errors": ["crossref: [Errno 8] nodename nor servname provided"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        # The point of this test is that a transport failure is never evidence
        # of fabrication. UNCERTAIN says that more precisely than VALID did.
        assert pred.label == "UNCERTAIN"
        assert pred.label != "HALLUCINATED"

    def test_unknown_future_status_never_parses_to_hallucinated(self, tmp_path: Path) -> None:
        """Whatever the sibling tool names its new transport status, an unmapped
        status falls through to conservative VALID rather than a fabrication
        verdict."""
        records: list[dict[str, Any]] = [
            {
                "key": "u",
                "status": "transport_failure_some_future_name",
                "confidence": 0.0,
                "mismatched_fields": [],
                "api_sources": [],
                "errors": ["dns failure"],
            }
        ]
        (pred,) = _parse(tmp_path, records)
        assert pred.label == "VALID"


class TestAssessBatchHealth:
    """The 2026-09-02 incident: 85-98% ``not_found`` with no database behind it."""

    def test_poisoned_batch_trips_the_detector(self) -> None:
        statuses = ["not_found"] * 95 + ["verified"] * 5
        health = assess_batch_health(statuses)
        assert health.suspected_transport_failure
        assert health.not_found == 95
        assert health.no_evidence_share == pytest.approx(0.95)

    def test_healthy_batch_does_not_trip_the_detector(self) -> None:
        """Healthy runs: ~52% verified, 1-3% not_found."""
        statuses = ["verified"] * 52 + ["not_found"] * 3 + ["unconfirmed"] * 45
        health = assess_batch_health(statuses)
        assert not health.suspected_transport_failure
        assert health.not_found_share == pytest.approx(0.03)

    def test_all_network_error_batch_trips_the_detector(self) -> None:
        """Same outage seen through the upgraded tool's own status."""
        health = assess_batch_health(["network_error"] * 100)
        assert health.suspected_transport_failure
        assert health.transport_error == 100
        assert health.not_found == 0

    def test_coverage_incomplete_status_counts_as_no_evidence(self) -> None:
        health = assess_batch_health(["coverage_incomplete"] * 100)
        assert health.suspected_transport_failure
        assert health.coverage_incomplete == 100

    def test_mixed_failure_shapes_accumulate(self) -> None:
        """A partially-upgraded pipeline splits the same outage across statuses;
        neither share alone crosses the threshold, together they do."""
        statuses = ["not_found"] * 20 + ["network_error"] * 20 + ["verified"] * 60
        health = assess_batch_health(statuses)
        assert health.not_found_share == pytest.approx(0.20)
        assert health.transport_error_share == pytest.approx(0.20)
        assert health.suspected_transport_failure

    def test_small_batch_is_not_judged(self) -> None:
        """Below the minimum batch size the share is noise, not signal."""
        statuses = ["not_found"] * (MIN_BATCH_FOR_HEALTH_CHECK - 1)
        health = assess_batch_health(statuses)
        assert not health.suspected_transport_failure
        assert health.no_evidence_share == pytest.approx(1.0)

    def test_empty_batch_is_safe(self) -> None:
        health = assess_batch_health([])
        assert not health.suspected_transport_failure
        assert health.no_evidence_share == 0.0

    def test_threshold_boundary_is_exclusive(self) -> None:
        statuses = ["not_found"] * 30 + ["verified"] * 70
        health = assess_batch_health(statuses)
        assert health.no_evidence_share == pytest.approx(NOT_FOUND_SHARE_THRESHOLD)
        assert not health.suspected_transport_failure

    def test_missing_sentinel_is_not_counted(self) -> None:
        """A timeout is already reported by the subprocess runner; it is a
        different failure and must not be read as a transport outage."""
        health = assess_batch_health(["missing"] * 100)
        assert not health.suspected_transport_failure

    def test_warning_message_blames_the_lookup_path_not_the_bibliography(
        self,
    ) -> None:
        health = assess_batch_health(["not_found"] * 98 + ["verified"] * 2)
        message = health.warning_message()
        assert "98/100" in message
        assert "98.0%" in message
        assert "do not" in message.lower()
        assert "checkpoint" in message
        assert "invented papers" in message


def _blind(key: str) -> BlindEntry:
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields={"title": "T", "author": "A", "year": "2024"},
        raw_bibtex=f"@article{{{key}, title={{T}}}}",
    )


def _fake_subprocess(status: str) -> Any:
    def _run(entries: list[BlindEntry], **_kw: Any) -> list[Prediction]:
        return [
            Prediction(
                bibtex_key=e.bibtex_key,
                label=STATUS_TO_LABEL.get(status, "VALID"),  # type: ignore[arg-type]
                confidence=STATUS_TO_CONFIDENCE.get(status, 0.5),
                reason=f"Status: {status}",
            )
            for e in entries
        ]

    return _run


class TestBatchHealthPlumbing:
    """The signal reaches the caller, not only the log."""

    def test_poisoned_run_logs_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            "hallmark.baselines.bibtexupdater._run_bibtex_check_subprocess",
            _fake_subprocess("not_found"),
        )
        entries = [_blind(f"k{i}") for i in range(40)]
        with caplog.at_level(logging.WARNING, logger="hallmark.baselines.bibtexupdater"):
            _, status_map = run_bibtex_check_with_status(entries, skip_prescreening=True)
        assert set(status_map.values()) == {"not_found"}
        assert any("do not" in r.message.lower() for r in caplog.records)
        assert any("40/40" in r.message for r in caplog.records)

    def test_healthy_run_logs_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            "hallmark.baselines.bibtexupdater._run_bibtex_check_subprocess",
            _fake_subprocess("verified"),
        )
        entries = [_blind(f"k{i}") for i in range(40)]
        with caplog.at_level(logging.WARNING, logger="hallmark.baselines.bibtexupdater"):
            run_bibtex_check_with_status(entries, skip_prescreening=True)
        assert not [r for r in caplog.records if "checkpoint" in r.message]

    def test_with_health_returns_the_flag_a_caller_gates_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "hallmark.baselines.bibtexupdater._run_bibtex_check_subprocess",
            _fake_subprocess("network_error"),
        )
        entries = [_blind(f"k{i}") for i in range(40)]
        predictions, status_map, health = run_bibtex_check_with_health(
            entries, skip_prescreening=True
        )
        assert len(predictions) == 40
        assert set(status_map.values()) == {"network_error"}
        assert health.suspected_transport_failure
        assert health.transport_error == 40
        assert all(p.label != "HALLUCINATED" for p in predictions)

    def test_with_health_is_quiet_on_a_healthy_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "hallmark.baselines.bibtexupdater._run_bibtex_check_subprocess",
            _fake_subprocess("verified"),
        )
        entries = [_blind(f"k{i}") for i in range(40)]
        _, _, health = run_bibtex_check_with_health(entries, skip_prescreening=True)
        assert not health.suspected_transport_failure
