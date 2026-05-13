"""Tests for the DB-first cascade baseline."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from hallmark.baselines.cascade import (
    ROUTE_TO_STAGE2,
    STAGE1_VERIFIED,
    STATUS_TO_TYPE,
    _aggressive_fallback,
    _stage1_predict,
    run_cascade,
)
from hallmark.dataset.schema import BlindEntry, Prediction


def _entry(key: str) -> BlindEntry:
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields={"title": "T", "author": "A", "year": "2024"},
        raw_bibtex=f"@article{{{key}, title={{T}}}}",
    )


def _raw(key: str, label: str = "VALID", confidence: float = 0.7) -> Prediction:
    return Prediction(bibtex_key=key, label=label, confidence=confidence, reason="raw")


# ---------------------------------------------------------------------------
# Stage 1 routing tests (no subprocess, no LLM)
# ---------------------------------------------------------------------------


def test_status_to_type_has_no_overlap_with_route_to_stage2() -> None:
    """A status cannot be both a Stage 1 mismatch and a Stage 2 deferral."""
    assert STATUS_TO_TYPE.keys().isdisjoint(ROUTE_TO_STAGE2)
    assert STATUS_TO_TYPE.keys().isdisjoint(STAGE1_VERIFIED)
    assert STAGE1_VERIFIED.isdisjoint(ROUTE_TO_STAGE2)


def test_status_to_type_values_are_valid_hallucination_types() -> None:
    from hallmark.dataset.schema import HallucinationType

    valid = {t.value for t in HallucinationType}
    for status, mapped_type in STATUS_TO_TYPE.items():
        assert mapped_type in valid, f"{status} maps to invalid type {mapped_type!r}"


def test_stage1_verified_emits_valid_with_high_confidence() -> None:
    entry = _entry("k1")
    raw = _raw("k1", label="VALID", confidence=0.5)
    out = _stage1_predict(entry, raw, "verified")
    assert out is not None
    assert out.label == "VALID"
    assert out.confidence >= 0.85
    assert out.cascade_stage == "stage1_db"
    assert out.predicted_hallucination_type is None


def test_stage1_mismatch_emits_hallucinated_with_predicted_type() -> None:
    entry = _entry("k2")
    raw = _raw("k2", label="HALLUCINATED", confidence=0.5)
    out = _stage1_predict(entry, raw, "doi_not_found")
    assert out is not None
    assert out.label == "HALLUCINATED"
    assert out.predicted_hallucination_type == "fabricated_doi"
    assert out.cascade_stage == "stage1_db"


def test_stage1_route_to_stage2_returns_none() -> None:
    entry = _entry("k3")
    raw = _raw("k3", label="VALID", confidence=0.3)
    for status in ROUTE_TO_STAGE2:
        assert _stage1_predict(entry, raw, status) is None, f"{status} should defer"


def test_stage1_prescreening_override_is_honored() -> None:
    entry = _entry("k4")
    raw = _raw("k4", label="HALLUCINATED", confidence=0.95)
    out = _stage1_predict(entry, raw, "prescreening_override")
    assert out is not None
    assert out.label == "HALLUCINATED"
    assert out.cascade_stage == "prescreening"
    assert out.source == "prescreening"


# ---------------------------------------------------------------------------
# Aggressive fallback
# ---------------------------------------------------------------------------


def test_aggressive_fallback_keeps_hallucinated() -> None:
    p = Prediction(
        bibtex_key="k",
        label="HALLUCINATED",
        confidence=0.6,
        predicted_hallucination_type="fabricated_doi",
        cascade_stage="stage2_diagnosis",
    )
    out = _aggressive_fallback(p)
    assert out is p


def test_aggressive_fallback_keeps_high_confidence_valid() -> None:
    p = Prediction(
        bibtex_key="k",
        label="VALID",
        confidence=0.85,
        cascade_stage="stage2_diagnosis",
    )
    out = _aggressive_fallback(p)
    assert out.label == "VALID"


def test_aggressive_fallback_promotes_uncertain_to_hallucinated() -> None:
    p = Prediction(
        bibtex_key="k",
        label="UNCERTAIN",
        confidence=0.5,
        cascade_stage="stage2_diagnosis",
    )
    out = _aggressive_fallback(p)
    assert out.label == "HALLUCINATED"
    assert out.predicted_hallucination_type == "plausible_fabrication"
    assert out.confidence == 0.55


def test_aggressive_fallback_promotes_low_confidence_valid() -> None:
    p = Prediction(
        bibtex_key="k",
        label="VALID",
        confidence=0.3,
        cascade_stage="stage2_diagnosis",
    )
    out = _aggressive_fallback(p)
    assert out.label == "HALLUCINATED"
    assert out.predicted_hallucination_type == "plausible_fabrication"


# ---------------------------------------------------------------------------
# End-to-end cascade tests with mocked Stage 1 + Stage 2
# ---------------------------------------------------------------------------


def _mock_stage1(
    status_per_key: dict[str, str],
    label_per_key: dict[str, str] | None = None,
    confidence: float = 0.7,
) -> Any:
    """Build a fake replacement for ``run_bibtex_check_with_status``."""

    def _impl(entries: list[BlindEntry], **_kw: Any) -> tuple[list[Prediction], dict[str, str]]:
        preds = []
        for e in entries:
            status = status_per_key.get(e.bibtex_key, "missing")
            if label_per_key and e.bibtex_key in label_per_key:
                label = label_per_key[e.bibtex_key]
            elif status in {"verified", "published_version_exists", "url_verified"}:
                label = "VALID"
            elif status in STATUS_TO_TYPE:
                label = "HALLUCINATED"
            else:
                label = "VALID"
            preds.append(
                Prediction(
                    bibtex_key=e.bibtex_key,
                    label=label,
                    confidence=confidence,
                    reason=f"status={status}",
                )
            )
        return preds, dict(status_per_key)

    return _impl


def _mock_stage2(verdicts: dict[str, dict[str, Any]]) -> Any:
    """Mock stage 2 baseline by registering a fake into the registry."""
    from hallmark.baselines.registry import _REGISTRY, BaselineInfo

    def _runner(entries: list[BlindEntry], **_kw: Any) -> list[Prediction]:
        out = []
        for e in entries:
            v = verdicts.get(
                e.bibtex_key,
                {"label": "UNCERTAIN", "confidence": 0.5},
            )
            out.append(
                Prediction(
                    bibtex_key=e.bibtex_key,
                    label=v["label"],
                    confidence=v.get("confidence", 0.7),
                    predicted_hallucination_type=v.get("predicted_hallucination_type"),
                    reason=v.get("reason", "stage2"),
                )
            )
        return out

    name = "_test_stage2_mock"
    _REGISTRY[name] = BaselineInfo(name=name, description="mock", runner=_runner)
    return name


def test_cascade_stage1_verified_short_circuits_no_stage2() -> None:
    entries = [_entry("a"), _entry("b")]
    stage2_called = []

    def _stage2_runner(es: list[BlindEntry], **_kw: Any) -> list[Prediction]:
        stage2_called.append(es)
        return []

    from hallmark.baselines.registry import _REGISTRY, BaselineInfo

    _REGISTRY["_test_should_not_run"] = BaselineInfo(
        name="_test_should_not_run", description="x", runner=_stage2_runner
    )

    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1({"a": "verified", "b": "verified"}),
    ):
        preds = run_cascade(entries, stage2_baseline="_test_should_not_run")

    assert not stage2_called
    assert all(p.label == "VALID" for p in preds)
    assert all(p.cascade_stage == "stage1_db" for p in preds)


def test_cascade_stage1_mismatch_short_circuits_with_predicted_type() -> None:
    entries = [_entry("a")]
    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1({"a": "venue_mismatch"}, label_per_key={"a": "HALLUCINATED"}),
    ):
        preds = run_cascade(entries, stage2_baseline="_test_should_not_run")

    assert preds[0].label == "HALLUCINATED"
    assert preds[0].predicted_hallucination_type == "wrong_venue"
    assert preds[0].cascade_stage == "stage1_db"


def test_cascade_routes_api_error_to_stage2() -> None:
    entries = [_entry("a")]
    name = _mock_stage2(
        {
            "a": {
                "label": "HALLUCINATED",
                "confidence": 0.9,
                "predicted_hallucination_type": "plausible_fabrication",
            }
        }
    )
    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1({"a": "api_error"}),
    ):
        preds = run_cascade(entries, stage2_baseline=name)

    assert preds[0].label == "HALLUCINATED"
    assert preds[0].cascade_stage == "stage2_diagnosis"
    assert preds[0].predicted_hallucination_type == "plausible_fabrication"


def test_cascade_aggressive_promotes_stage2_uncertain() -> None:
    entries = [_entry("a")]
    name = _mock_stage2({"a": {"label": "UNCERTAIN", "confidence": 0.5}})
    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1({"a": "not_found"}),
    ):
        conservative = run_cascade(entries, stage2_baseline=name, aggressive=False)
        aggressive = run_cascade(entries, stage2_baseline=name, aggressive=True)

    assert conservative[0].label == "UNCERTAIN"
    assert aggressive[0].label == "HALLUCINATED"
    assert aggressive[0].predicted_hallucination_type == "plausible_fabrication"


def test_cascade_aggressive_keeps_stage1_decisions() -> None:
    """Stage 1 (verified or definite mismatch) should not be touched by aggressive mode."""
    entries = [_entry("ok"), _entry("bad")]
    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1(
            {"ok": "verified", "bad": "doi_not_found"},
            label_per_key={"ok": "VALID", "bad": "HALLUCINATED"},
        ),
    ):
        preds = run_cascade(entries, stage2_baseline="_test_should_not_run", aggressive=True)

    by_key = {p.bibtex_key: p for p in preds}
    assert by_key["ok"].label == "VALID"
    assert by_key["ok"].cascade_stage == "stage1_db"
    assert by_key["bad"].label == "HALLUCINATED"
    assert by_key["bad"].predicted_hallucination_type == "fabricated_doi"


def test_cascade_unavailable_stage2_returns_uncertain() -> None:
    """If the requested Stage 2 baseline isn't registered, deferred entries become UNCERTAIN."""
    entries = [_entry("a")]
    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1({"a": "api_error"}),
    ):
        preds = run_cascade(entries, stage2_baseline="nonexistent_baseline_xyz")

    assert preds[0].label == "UNCERTAIN"


def test_cascade_preserves_input_order() -> None:
    entries = [_entry(k) for k in ["c", "a", "b"]]
    with patch(
        "hallmark.baselines.cascade.run_bibtex_check_with_status",
        _mock_stage1({"a": "verified", "b": "verified", "c": "verified"}),
    ):
        preds = run_cascade(entries, stage2_baseline="_test_should_not_run")
    assert [p.bibtex_key for p in preds] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# Schema back-compat
# ---------------------------------------------------------------------------


def test_prediction_round_trip_with_cascade_fields() -> None:
    p = Prediction(
        bibtex_key="k",
        label="HALLUCINATED",
        confidence=0.9,
        predicted_hallucination_type="fabricated_doi",
        cascade_stage="stage1_db",
    )
    line = p.to_json()
    p2 = Prediction.from_json(line)
    assert p2.predicted_hallucination_type == "fabricated_doi"
    assert p2.cascade_stage == "stage1_db"


def test_prediction_round_trip_without_cascade_fields() -> None:
    p = Prediction(bibtex_key="k", label="VALID", confidence=0.9)
    line = p.to_json()
    p2 = Prediction.from_json(line)
    assert p2.predicted_hallucination_type is None
    assert p2.cascade_stage is None


def test_prediction_rejects_predicted_type_with_valid_label() -> None:
    with pytest.raises(ValueError, match="predicted_hallucination_type"):
        Prediction(
            bibtex_key="k",
            label="VALID",
            confidence=0.9,
            predicted_hallucination_type="fabricated_doi",
        )


def test_prediction_rejects_invalid_predicted_type() -> None:
    with pytest.raises(ValueError, match="Invalid predicted_hallucination_type"):
        Prediction(
            bibtex_key="k",
            label="HALLUCINATED",
            confidence=0.9,
            predicted_hallucination_type="not_a_real_type",
        )
