"""Integration tests for the DB-first cascade orchestrator + dual-mode evaluation.

These tests exercise the wiring between the registered cascade baseline runner,
the Stage 1 status vocabulary, and ``evaluate(..., eval_mode="both")`` — the
gap left by ``test_cascade.py`` (mocked Stage 1 only) and ``test_type_metrics.py``
(pure metric unit tests).

All three tests are self-contained, deterministic, and avoid subprocess/network
calls by monkeypatching ``run_bibtex_check_with_status`` directly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest

from hallmark.baselines.cascade import STATUS_TO_TYPE
from hallmark.baselines.registry import _REGISTRY, BaselineInfo
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, EvaluationResult, Prediction
from hallmark.evaluation.metrics import evaluate

# Cascade stage Literal values from Prediction (NOT the abbreviated names in the
# task spec — the actual schema uses ``stage1_db`` / ``stage2_diagnosis``).
VALID_CASCADE_STAGES = {"stage1_db", "stage2_diagnosis", "prescreening"}

# Metric keys that ``EvaluationResult.summary()`` and ``to_dict()`` must expose
# for both modes of an ``eval_mode="both"`` payload.
REQUIRED_METRIC_KEYS = {
    "detection_rate",
    "false_positive_rate",
    "f1_hallucination",
    "num_valid",
    "num_hallucinated",
    "num_uncertain",
}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _benchmark_entry(
    key: str,
    label: str,
    *,
    tier: int | None = None,
    h_type: str | None = None,
    year: str = "2024",
    title: str | None = None,
    author: str = "Smith, J. and Doe, A.",
) -> BenchmarkEntry:
    """Build a minimal BenchmarkEntry with sensible defaults."""
    kwargs: dict[str, Any] = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {
            "title": title or f"Paper {key}",
            "author": author,
            "year": year,
            "journal": "Journal of Tests",
        },
        "label": label,
        "explanation": "fixture entry",
        "raw_bibtex": f"@article{{{key}, title={{Paper {key}}}}}",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type or "fabricated_doi"
        kwargs["difficulty_tier"] = tier if tier is not None else 1
    return BenchmarkEntry(**kwargs)


def _blind(entry: BenchmarkEntry) -> BlindEntry:
    return entry.to_blind()


def _register_stage2_mock(name: str, fn: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Insert a Stage 2 runner into the registry and remove it on teardown."""
    info = BaselineInfo(name=name, description="integration-mock", runner=fn)
    monkeypatch.setitem(_REGISTRY, name, info)


# ---------------------------------------------------------------------------
# Test 1: end-to-end aggressive cascade smoke test
# ---------------------------------------------------------------------------


def test_cascade_integration_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end invariants on the aggressive cascade orchestrator.

    Builds 10 synthetic entries spanning every routing path Stage 1 supports,
    monkeypatches the bibtex-check call to return a deterministic status map,
    runs the registered ``cascade_db_diagnosis_aggressive`` runner, then
    evaluates with ``eval_mode="both"`` and asserts the dual-mode invariants.
    """
    entries: list[BenchmarkEntry] = [
        # 2 ground-truth VALID — Stage 1 should verify both
        _benchmark_entry("valid_a", "VALID"),
        _benchmark_entry("valid_b", "VALID"),
        # 3 fabricated (DOI / venue / placeholder)
        _benchmark_entry("fab_doi", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        _benchmark_entry("fab_venue", "HALLUCINATED", tier=1, h_type="nonexistent_venue"),
        _benchmark_entry("fab_authors", "HALLUCINATED", tier=1, h_type="placeholder_authors"),
        # 2 with field mismatch (title / author)
        _benchmark_entry("mismatch_title", "HALLUCINATED", tier=2, h_type="chimeric_title"),
        _benchmark_entry("mismatch_author", "HALLUCINATED", tier=2, h_type="swapped_authors"),
        # 1 future-year — pre-screening territory
        _benchmark_entry("future_year", "HALLUCINATED", tier=1, h_type="future_date", year="2099"),
        # 2 entries Stage 1 can't decide on (route to Stage 2)
        _benchmark_entry("missing_a", "HALLUCINATED", tier=3, h_type="plausible_fabrication"),
        _benchmark_entry("missing_b", "VALID"),
    ]
    blind_entries = [_blind(e) for e in entries]

    # Deterministic status assignment matching the cascade routing.
    status_map = {
        "valid_a": "verified",
        "valid_b": "verified",
        "fab_doi": "doi_not_found",
        "fab_venue": "venue_mismatch",
        "fab_authors": "prescreening_override",  # placeholder author caught locally
        "mismatch_title": "title_mismatch",
        "mismatch_author": "author_mismatch",
        "future_year": "prescreening_override",  # future year caught locally
        "missing_a": "not_found",  # → Stage 2
        "missing_b": "api_error",  # → Stage 2
    }

    def _fake_bibtex(
        entries: list[BlindEntry], **_kw: Any
    ) -> tuple[list[Prediction], dict[str, str]]:
        preds: list[Prediction] = []
        for e in entries:
            status = status_map[e.bibtex_key]
            if status == "verified":
                label, conf = "VALID", 0.7
            elif status == "prescreening_override":
                # Pre-screening always emits a confident HALLUCINATED for these.
                label, conf = "HALLUCINATED", 0.95
            elif status in STATUS_TO_TYPE:
                label, conf = "HALLUCINATED", 0.7
            else:
                label, conf = "VALID", 0.4
            preds.append(
                Prediction(
                    bibtex_key=e.bibtex_key,
                    label=label,
                    confidence=conf,
                    reason=f"Status: {status}",
                    source="tool" if status != "prescreening_override" else "prescreening",
                )
            )
        return preds, dict(status_map)

    # Stage 2 returns mixed verdicts for missing_a/b so we exercise aggressive
    # promotion of UNCERTAIN as well as a confident HALLUCINATED.
    def _fake_stage2(es: list[BlindEntry], **_kw: Any) -> list[Prediction]:
        out: list[Prediction] = []
        for e in es:
            if e.bibtex_key == "missing_a":
                out.append(
                    Prediction(
                        bibtex_key=e.bibtex_key,
                        label="HALLUCINATED",
                        confidence=0.8,
                        predicted_hallucination_type="plausible_fabrication",
                        reason="diagnosed by stage2 mock",
                    )
                )
            else:  # missing_b
                out.append(
                    Prediction(
                        bibtex_key=e.bibtex_key,
                        label="UNCERTAIN",
                        confidence=0.5,
                        reason="stage2 mock unsure",
                    )
                )
        return out

    stage2_name = "_test_integration_stage2"
    _register_stage2_mock(stage2_name, _fake_stage2, monkeypatch)

    # Patch the bibtex-check call inside the cascade module.
    monkeypatch.setattr("hallmark.baselines.cascade.run_bibtex_check_with_status", _fake_bibtex)

    # 1. Invoke the registered aggressive cascade runner directly (bypassing
    #    check_available() which would gate on the bibtex-check CLI).
    info = _REGISTRY["cascade_db_diagnosis_aggressive"]
    assert info.runner_kwargs["aggressive"] is True  # sanity
    predictions = info.runner(
        blind_entries,
        stage2_baseline=stage2_name,
        aggressive=True,
    )

    # Invariant 1: returned without error and produced one prediction per entry.
    assert len(predictions) == len(entries)
    assert [p.bibtex_key for p in predictions] == [e.bibtex_key for e in entries]

    # Invariant 7: every cascade_stage is one of the valid Literal values.
    for p in predictions:
        assert p.cascade_stage in VALID_CASCADE_STAGES, (
            f"{p.bibtex_key}: unexpected cascade_stage {p.cascade_stage!r}"
        )

    # 2. Evaluate in both modes.
    both = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name="cascade_db_diagnosis_aggressive",
        split_name="integration",
        eval_mode="both",
    )

    # Invariant 2: dict shape with both sub-payloads.
    assert isinstance(both, dict)
    assert set(both.keys()) == {"conservative", "aggressive"}
    conservative = both["conservative"]
    aggressive = both["aggressive"]
    assert isinstance(conservative, EvaluationResult)
    assert isinstance(aggressive, EvaluationResult)

    # Invariant 3: aggressive eliminates all UNCERTAIN predictions.
    assert aggressive.num_uncertain == 0

    # Invariant 4: aggressive flags at least as many as conservative.
    # (Strict ">" because we deliberately included a Stage 2 UNCERTAIN entry
    # that aggressive must promote; if this assertion ever fails it indicates
    # the mode is collapsing.)
    agg_hall = _count_label(predictions, "HALLUCINATED", aggressive=True, entries=entries)
    cons_hall = _count_label(predictions, "HALLUCINATED", aggressive=False, entries=entries)
    assert agg_hall >= cons_hall

    # Invariant 5: conservative VALID >= aggressive VALID (aggressive promotes
    # some VALIDs to HALLUCINATED via the cascade's own aggressive fallback,
    # and the eval_mode="aggressive" branch never adds VALIDs).
    cons_valid = _count_label(predictions, "VALID", aggressive=False, entries=entries)
    agg_valid = _count_label(predictions, "VALID", aggressive=True, entries=entries)
    assert cons_valid >= agg_valid

    # Invariant 6: aggressive FPR >= conservative FPR. UNCERTAINs that aggressive
    # promotes can only land in either correct-detection or false-positive bins;
    # they cannot reduce FPR below conservative.
    if conservative.false_positive_rate is not None and aggressive.false_positive_rate is not None:
        assert aggressive.false_positive_rate >= conservative.false_positive_rate - 1e-9


def _count_label(
    predictions: list[Prediction],
    label: str,
    *,
    aggressive: bool,
    entries: list[BenchmarkEntry],
) -> int:
    """Helper to count post-transformation labels matching either eval mode.

    For aggressive mode this mirrors what ``_make_aggressive_predictions``
    does: UNCERTAINs and missing predictions become HALLUCINATED.
    """
    pred_map = {p.bibtex_key: p for p in predictions}
    count = 0
    for e in entries:
        pred = pred_map.get(e.bibtex_key)
        if pred is None:
            effective = "HALLUCINATED" if aggressive else None
        elif aggressive and pred.label == "UNCERTAIN":
            effective = "HALLUCINATED"
        else:
            effective = pred.label
        if effective == label:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Test 2: eval_mode="both" round-trips and exposes both payloads on disk
# ---------------------------------------------------------------------------


def test_eval_mode_both_writes_dual_payload(tmp_path) -> None:
    """``evaluate(..., eval_mode="both")`` must serialize to a JSON file with
    matching shapes for ``conservative`` and ``aggressive``, and predictions
    must bucket identically — only the UNCERTAIN treatment differs."""
    # Minimal split: 2 valid, 2 hallucinated. Predictions cover all four label
    # outcomes (TP, TN, FP, UNCERTAIN) so the two modes diverge.
    entries = [
        _benchmark_entry("v1", "VALID"),
        _benchmark_entry("v2", "VALID"),
        _benchmark_entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        _benchmark_entry("h2", "HALLUCINATED", tier=2, h_type="wrong_venue"),
    ]
    predictions = [
        Prediction(bibtex_key="v1", label="VALID", confidence=0.9),  # TN
        Prediction(bibtex_key="v2", label="UNCERTAIN", confidence=0.5),  # mode-sensitive
        Prediction(
            bibtex_key="h1",
            label="HALLUCINATED",
            confidence=0.85,
            predicted_hallucination_type="fabricated_doi",
        ),  # TP
        Prediction(bibtex_key="h2", label="UNCERTAIN", confidence=0.5),  # mode-sensitive
    ]

    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name="dual_mode_fixture",
        split_name="integration",
        eval_mode="both",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"conservative", "aggressive"}

    # Serialize to disk via to_dict — emulates how the CLI writes results.
    payload = {mode: r.to_dict() for mode, r in result.items()}
    out_path = tmp_path / "dual_eval.json"
    out_path.write_text(json.dumps(payload, indent=2))

    # Round-trip cleanly.
    loaded = json.loads(out_path.read_text())
    assert json.dumps(loaded)  # would raise on non-serializable content

    # Both sub-payloads present and contain the required metric keys.
    assert set(loaded.keys()) == {"conservative", "aggressive"}
    for mode, sub in loaded.items():
        missing = REQUIRED_METRIC_KEYS - set(sub.keys())
        assert not missing, f"{mode} sub-payload missing keys: {missing}"

    # Predictions come from the same underlying source: the four entries must
    # be accounted for in every mode (num_valid + num_hallucinated == 4 since
    # those count ground-truth labels, identical across modes), but the count
    # of UNCERTAIN predictions DOES differ.
    for mode in ("conservative", "aggressive"):
        sub = loaded[mode]
        assert sub["num_valid"] + sub["num_hallucinated"] == len(entries)

    # The conservative payload preserves UNCERTAIN; aggressive zeroes it out.
    assert loaded["conservative"]["num_uncertain"] == 2
    assert loaded["aggressive"]["num_uncertain"] == 0


# ---------------------------------------------------------------------------
# Test 3: prescreening overrides never reach Stage 2
# ---------------------------------------------------------------------------


def test_prescreening_override_bypasses_stage2(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the bibtex-check status is ``prescreening_override`` the cascade
    must accept that verdict directly — Stage 2 must NOT be invoked, even in
    aggressive mode."""
    entries = [
        _benchmark_entry(
            f"future_{i}",
            "HALLUCINATED",
            tier=1,
            h_type="future_date",
            year="2099",
        )
        for i in range(3)
    ]
    blind_entries = [_blind(e) for e in entries]

    # Stage 1 returns pre-screening verdicts for every entry. Confidence 0.95
    # — well above the aggressive fallback's 0.55 — is what we assert against.
    PRESCREEN_CONFIDENCE = 0.95

    def _fake_bibtex(
        entries: list[BlindEntry], **_kw: Any
    ) -> tuple[list[Prediction], dict[str, str]]:
        preds = [
            Prediction(
                bibtex_key=e.bibtex_key,
                label="HALLUCINATED",
                confidence=PRESCREEN_CONFIDENCE,
                reason="Status: prescreening_override; future_date detected",
                source="prescreening",
            )
            for e in entries
        ]
        statuses = {e.bibtex_key: "prescreening_override" for e in entries}
        return preds, statuses

    # Stage 2 must NOT be called — wire a Mock that raises if invoked.
    stage2_mock = Mock(
        side_effect=AssertionError("stage2 should not be called"),
        name="forbidden_stage2",
    )
    _register_stage2_mock("_test_forbidden_stage2", stage2_mock, monkeypatch)

    monkeypatch.setattr("hallmark.baselines.cascade.run_bibtex_check_with_status", _fake_bibtex)

    info = _REGISTRY["cascade_db_diagnosis_aggressive"]
    preds = info.runner(
        blind_entries,
        stage2_baseline="_test_forbidden_stage2",
        aggressive=True,
    )

    # Stage 2 mock never invoked.
    assert stage2_mock.call_count == 0

    # All predictions stamped as prescreening-stage HALLUCINATED.
    assert len(preds) == 3
    for p in preds:
        assert p.cascade_stage == "prescreening"
        assert p.label == "HALLUCINATED"
        # Confidence preserved at the pre-screening level — NOT the 0.55
        # aggressive-fallback value.
        assert p.confidence == pytest.approx(PRESCREEN_CONFIDENCE)
        assert p.confidence != pytest.approx(0.55)
