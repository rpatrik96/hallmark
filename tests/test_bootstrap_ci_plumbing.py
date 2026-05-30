"""Tests for the bootstrap-CI / paired-significance reporting plumbing (task #9).

Covers:
* ``compute_persisted_cis`` — populates the ``*_ci`` fields from per-entry
  predictions, deterministically; returns null + a recorded reason when no
  predictions are supplied.
* ``paired_significance`` — pairwise p-values are deterministic and order
  invariant.
* The ``scripts/compute_bootstrap_ci.py`` filename/split helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import (
    CI_FIELD_FOR_METRIC,
    PERSISTED_CI_METRICS,
    compute_persisted_cis,
    paired_significance,
)

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
requires_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")

# Make scripts/ importable for the helper-function tests.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# --- Helpers ---------------------------------------------------------------


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


def _toy_dataset(n_hall: int = 20, n_valid: int = 20):
    """Return (entries, good_predictions) with varied confidences (ECE-friendly)."""
    entries: list[BenchmarkEntry] = []
    preds: list[Prediction] = []
    for i in range(n_hall):
        k = f"h{i}"
        tier = (i % 3) + 1
        entries.append(_entry(k, "HALLUCINATED", tier=tier))
        # Detect ~80%, varied confidences so ECE is meaningful.
        label = "HALLUCINATED" if i % 5 != 0 else "VALID"
        preds.append(_pred(k, label, confidence=0.6 + 0.3 * (i % 4) / 3))
    for i in range(n_valid):
        k = f"v{i}"
        entries.append(_entry(k, "VALID"))
        label = "VALID" if i % 6 != 0 else "HALLUCINATED"  # ~17% FPR
        preds.append(_pred(k, label, confidence=0.55 + 0.4 * (i % 3) / 2))
    return entries, preds


# --- compute_persisted_cis -------------------------------------------------


def test_ci_field_map_covers_all_persisted_metrics():
    for metric in PERSISTED_CI_METRICS:
        assert metric in CI_FIELD_FOR_METRIC


@requires_numpy
def test_compute_persisted_cis_populates_fields():
    entries, preds = _toy_dataset()
    out = compute_persisted_cis(entries, preds, n_bootstrap=200, seed=42)

    for field_name in (
        "detection_rate_ci",
        "fpr_ci",
        "f1_hallucination_ci",
        "tier_weighted_f1_ci",
        "mcc_ci",
    ):
        ci = out[field_name]
        assert isinstance(ci, list) and len(ci) == 2, field_name
        lo, hi = ci
        assert lo <= hi

    prov = out["ci_provenance"]
    assert prov["computed"] is True
    assert prov["source"] == "per_entry_predictions"
    assert prov["seed"] == 42
    assert prov["n_bootstrap"] == 200
    assert prov["num_predictions"] == len(preds)


@requires_numpy
def test_compute_persisted_cis_is_deterministic():
    entries, preds = _toy_dataset()
    a = compute_persisted_cis(entries, preds, n_bootstrap=150, seed=7)
    b = compute_persisted_cis(entries, preds, n_bootstrap=150, seed=7)
    assert a["detection_rate_ci"] == b["detection_rate_ci"]
    assert a["mcc_ci"] == b["mcc_ci"]
    assert a["f1_hallucination_ci"] == b["f1_hallucination_ci"]


@requires_numpy
def test_compute_persisted_cis_seed_changes_result():
    entries, preds = _toy_dataset()
    a = compute_persisted_cis(entries, preds, n_bootstrap=150, seed=1)
    b = compute_persisted_cis(entries, preds, n_bootstrap=150, seed=2)
    # Different seeds should (with overwhelming probability) give different bounds.
    assert a["detection_rate_ci"] != b["detection_rate_ci"]


def test_compute_persisted_cis_none_predictions_records_reason():
    entries, _ = _toy_dataset()
    out = compute_persisted_cis(entries, None)
    for field_name in (
        "detection_rate_ci",
        "fpr_ci",
        "f1_hallucination_ci",
        "tier_weighted_f1_ci",
        "ece_ci",
        "mcc_ci",
    ):
        assert out[field_name] is None
    prov = out["ci_provenance"]
    assert prov["computed"] is False
    assert "no per-entry predictions" in prov["reason"]


def test_compute_persisted_cis_empty_predictions_records_reason():
    entries, _ = _toy_dataset()
    out = compute_persisted_cis(entries, [])
    assert out["detection_rate_ci"] is None
    assert out["ci_provenance"]["computed"] is False


@requires_numpy
def test_compute_persisted_cis_fpr_null_when_no_valid_entries():
    # Stress-like split: all HALLUCINATED, FPR is undefined.
    entries = [_entry(f"h{i}", "HALLUCINATED", tier=(i % 3) + 1) for i in range(15)]
    preds = [_pred(f"h{i}", "HALLUCINATED", confidence=0.8) for i in range(15)]
    out = compute_persisted_cis(entries, preds, n_bootstrap=100, seed=42)
    assert out["fpr_ci"] is None
    assert out["detection_rate_ci"] is not None


@requires_numpy
def test_compute_persisted_cis_ece_null_with_few_distinct_confidences():
    entries, _ = _toy_dataset()
    # Binary tool: only 0.0/1.0 confidences -> ECE unreliable -> null CI.
    preds = []
    for e in entries:
        conf = 1.0
        label = e.label
        preds.append(_pred(e.bibtex_key, label, confidence=conf))
    out = compute_persisted_cis(entries, preds, n_bootstrap=100, seed=42)
    assert out["ece_ci"] is None


# --- paired_significance ---------------------------------------------------


@requires_numpy
def test_paired_significance_detects_difference():
    entries, good = _toy_dataset()
    # Weak tool: predict everything VALID -> much lower F1.
    weak = [_pred(e.bibtex_key, "VALID", confidence=0.6) for e in entries]
    sig = paired_significance(entries, {"good": good, "weak": weak}, n_bootstrap=300, seed=42)
    assert "good_vs_weak" in sig
    rec = sig["good_vs_weak"]
    assert rec["observed_diff"] > 0  # good has higher F1
    assert rec["p_value"] <= 0.05
    assert "cohens_h" in rec


@requires_numpy
def test_paired_significance_is_order_invariant_and_deterministic():
    entries, good = _toy_dataset()
    weak = [_pred(e.bibtex_key, "VALID", confidence=0.6) for e in entries]
    a = paired_significance(entries, {"good": good, "weak": weak}, n_bootstrap=200, seed=9)
    b = paired_significance(entries, {"weak": weak, "good": good}, n_bootstrap=200, seed=9)
    assert a == b  # sorted tools + per-pair seed => order invariant


@requires_numpy
def test_paired_significance_skips_tools_without_predictions():
    entries, good = _toy_dataset()
    weak = [_pred(e.bibtex_key, "VALID", confidence=0.6) for e in entries]
    sig = paired_significance(
        entries, {"good": good, "weak": weak, "empty": []}, n_bootstrap=100, seed=42
    )
    # "empty" must not appear in any pair key.
    assert all("empty" not in key for key in sig)


def test_paired_significance_rejects_unknown_metric():
    entries, good = _toy_dataset()
    with pytest.raises(ValueError):
        paired_significance(entries, {"good": good}, metric="not_a_metric")


# --- script helpers --------------------------------------------------------


def test_infer_split_from_name():
    import compute_bootstrap_ci as cbc

    assert cbc.infer_split_from_name("llm_openai_dev_public.json") == "dev_public"
    assert cbc.infer_split_from_name("doi_only_test_public.json") == "test_public"
    assert cbc.infer_split_from_name("cascade_db_diagnosis_stress_test.json") == "stress_test"
    assert cbc.infer_split_from_name("manifest.json") is None
    assert cbc.infer_split_from_name("random_thing.json") is None


def test_load_predictions_canonical_rejects_non_canonical(tmp_path):
    import compute_bootstrap_ci as cbc

    # Tool-native raw schema ({"key","status"}) is NOT canonical -> returns None.
    p = tmp_path / "raw.jsonl"
    p.write_text('{"key": "abc", "status": "hallucinated", "confidence": 0.7}\n')
    assert cbc.load_predictions_canonical(p) is None


def test_load_predictions_canonical_parses_canonical(tmp_path):
    import compute_bootstrap_ci as cbc

    p = tmp_path / "preds.jsonl"
    p.write_text(
        '{"bibtex_key": "a", "label": "HALLUCINATED", "confidence": 0.9}\n'
        '{"bibtex_key": "b", "label": "VALID", "confidence": 0.8}\n'
    )
    preds = cbc.load_predictions_canonical(p)
    assert preds is not None
    assert len(preds) == 2
    assert preds[0].bibtex_key == "a"
