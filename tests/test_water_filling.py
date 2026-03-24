"""Tests for hallmark.evaluation.water_filling."""

from __future__ import annotations

import math

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.water_filling import (
    compute_tier_detection_rates,
    normalized_shannon_entropy,
    water_filling_analysis,
    water_filling_gini,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# One representative type per tier
_T1_TYPE = "fabricated_doi"
_T2_TYPE = "chimeric_title"
_T3_TYPE = "near_miss_title"


def _entry(
    key: str,
    label: str,
    tier: int | None = None,
    h_type: str | None = None,
) -> BenchmarkEntry:
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2024"},
        "label": label,
        "explanation": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type or _T1_TYPE
        kwargs["difficulty_tier"] = tier or 1
    return BenchmarkEntry(**kwargs)


def _pred(key: str, label: str, confidence: float = 0.9) -> Prediction:
    return Prediction(bibtex_key=key, label=label, confidence=confidence)


# ---------------------------------------------------------------------------
# compute_tier_detection_rates
# ---------------------------------------------------------------------------


def test_compute_tier_drs_perfect():
    """Tool detects every hallucination → DR=1.0 for all represented tiers."""
    entries = [
        _entry("t1a", "HALLUCINATED", tier=1, h_type=_T1_TYPE),
        _entry("t2a", "HALLUCINATED", tier=2, h_type=_T2_TYPE),
        _entry("t3a", "HALLUCINATED", tier=3, h_type=_T3_TYPE),
    ]
    preds = [
        _pred("t1a", "HALLUCINATED"),
        _pred("t2a", "HALLUCINATED"),
        _pred("t3a", "HALLUCINATED"),
    ]
    drs = compute_tier_detection_rates(entries, preds)
    assert drs[1] == pytest.approx(1.0)
    assert drs[2] == pytest.approx(1.0)
    assert drs[3] == pytest.approx(1.0)


def test_compute_tier_drs_tier1_only():
    """Tool detects only Tier 1 → DR(T1)=1.0, DR(T2)=0.0, DR(T3)=0.0."""
    entries = [
        _entry("t1a", "HALLUCINATED", tier=1, h_type=_T1_TYPE),
        _entry("t1b", "HALLUCINATED", tier=1, h_type="nonexistent_venue"),
        _entry("t2a", "HALLUCINATED", tier=2, h_type=_T2_TYPE),
        _entry("t3a", "HALLUCINATED", tier=3, h_type=_T3_TYPE),
    ]
    preds = [
        _pred("t1a", "HALLUCINATED"),
        _pred("t1b", "HALLUCINATED"),
        _pred("t2a", "VALID"),
        _pred("t3a", "VALID"),
    ]
    drs = compute_tier_detection_rates(entries, preds)
    assert drs[1] == pytest.approx(1.0)
    assert drs[2] == pytest.approx(0.0)
    assert drs[3] == pytest.approx(0.0)


def test_compute_tier_drs_excludes_valid():
    """VALID entries don't contribute to any tier's DR."""
    entries = [
        _entry("v1", "VALID"),
        _entry("v2", "VALID"),
        _entry("h1", "HALLUCINATED", tier=1, h_type=_T1_TYPE),
    ]
    preds = [
        _pred("v1", "HALLUCINATED"),  # false positive — should not affect DR
        _pred("v2", "HALLUCINATED"),  # false positive — should not affect DR
        _pred("h1", "HALLUCINATED"),  # true positive
    ]
    drs = compute_tier_detection_rates(entries, preds)
    assert drs[1] == pytest.approx(1.0)
    # Tiers 2 and 3 should not appear (no hallucinated entries there)
    assert 2 not in drs
    assert 3 not in drs


def test_compute_tier_drs_excludes_uncertain():
    """UNCERTAIN predictions don't count as correct detections."""
    entries = [
        _entry("t1a", "HALLUCINATED", tier=1, h_type=_T1_TYPE),
        _entry("t1b", "HALLUCINATED", tier=1, h_type=_T1_TYPE),
    ]
    preds = [
        _pred("t1a", "HALLUCINATED"),
        Prediction(bibtex_key="t1b", label="UNCERTAIN", confidence=0.5),
    ]
    drs = compute_tier_detection_rates(entries, preds)
    # Only t1a counts; t1b is uncertain → DR = 0.5
    assert drs[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# water_filling_gini
# ---------------------------------------------------------------------------


def test_gini_uniform():
    """Equal DRs across tiers → Gini ≈ 0."""
    tier_drs = {1: 0.8, 2: 0.8, 3: 0.8}
    assert water_filling_gini(tier_drs) == pytest.approx(0.0)


def test_gini_concentrated():
    """One tier has all the detection, others zero → Gini close to max."""
    tier_drs = {1: 1.0, 2: 0.0, 3: 0.0}
    gini = water_filling_gini(tier_drs)
    # With n=3 and values [1, 0, 0]: sum of |xi - xj| = 1+1+1+0+1+0 = 4 (counting both orders)
    # Actually: |1-1|=0, |1-0|=1, |1-0|=1, |0-1|=1, |0-0|=0, |0-1|=1,
    #           |0-1|=1, |0-0|=0, |0-1|=1 → sum=6 (3x3 grid)
    # Wait: (i,j) over 3x3 = 9 pairs: sum = 0+1+1+1+0+0+1+0+0 = 4... let me just check range
    # Formula: G = pairwise_sum / (2 * n * total) = 6 / (2*3*1) = 1.0 but with 0+0 duplicates
    # Regardless, a concentrated distribution should have high Gini (> 0.5)
    assert gini > 0.5


def test_gini_all_zero():
    """All DRs zero → Gini = 0 (degenerate case)."""
    tier_drs = {1: 0.0, 2: 0.0, 3: 0.0}
    assert water_filling_gini(tier_drs) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# normalized_shannon_entropy
# ---------------------------------------------------------------------------


def test_entropy_uniform():
    """Equal DRs → normalized entropy = 1.0."""
    tier_drs = {1: 0.5, 2: 0.5, 3: 0.5}
    assert normalized_shannon_entropy(tier_drs) == pytest.approx(1.0)


def test_entropy_concentrated():
    """All detection in one tier → entropy = 0.0."""
    tier_drs = {1: 1.0, 2: 0.0, 3: 0.0}
    assert normalized_shannon_entropy(tier_drs) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# water_filling_analysis
# ---------------------------------------------------------------------------


def _make_entries_all_tiers() -> list[BenchmarkEntry]:
    """Return a small set covering all 3 tiers."""
    return [
        _entry("t1a", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        _entry("t1b", "HALLUCINATED", tier=1, h_type="future_date"),
        _entry("t2a", "HALLUCINATED", tier=2, h_type="chimeric_title"),
        _entry("t2b", "HALLUCINATED", tier=2, h_type="wrong_venue"),
        _entry("t3a", "HALLUCINATED", tier=3, h_type="near_miss_title"),
        _entry("t3b", "HALLUCINATED", tier=3, h_type="plausible_fabrication"),
        _entry("v1", "VALID"),
    ]


def test_water_filling_analysis_flags_ratio():
    """Tool that catches T1 but misses T3 gets flagged (ratio > threshold)."""
    entries = _make_entries_all_tiers()
    preds = [
        _pred("t1a", "HALLUCINATED"),
        _pred("t1b", "HALLUCINATED"),
        _pred("t2a", "VALID"),
        _pred("t2b", "VALID"),
        _pred("t3a", "VALID"),
        _pred("t3b", "VALID"),
        _pred("v1", "VALID"),
    ]
    profiles = water_filling_analysis(
        entries, {"tool": preds}, ratio_threshold=1.5, gini_threshold=0.99
    )
    assert profiles["tool"].is_water_filling is True
    assert math.isinf(profiles["tool"].tier_ratio)


def test_water_filling_analysis_flags_gini():
    """Tool with high Gini coefficient gets flagged even if ratio threshold not set low."""
    entries = _make_entries_all_tiers()
    # Detect all T1, nothing else
    preds = [
        _pred("t1a", "HALLUCINATED"),
        _pred("t1b", "HALLUCINATED"),
        _pred("t2a", "VALID"),
        _pred("t2b", "VALID"),
        _pred("t3a", "VALID"),
        _pred("t3b", "VALID"),
    ]
    profiles = water_filling_analysis(
        entries, {"tool": preds}, ratio_threshold=9999.0, gini_threshold=0.3
    )
    assert profiles["tool"].is_water_filling is True
    assert profiles["tool"].gini_coefficient > 0.3


def test_water_filling_analysis_uniform_tool():
    """Tool with equal detection across tiers is not flagged."""
    entries = _make_entries_all_tiers()
    preds = [
        _pred("t1a", "HALLUCINATED"),
        _pred("t1b", "HALLUCINATED"),
        _pred("t2a", "HALLUCINATED"),
        _pred("t2b", "HALLUCINATED"),
        _pred("t3a", "HALLUCINATED"),
        _pred("t3b", "HALLUCINATED"),
    ]
    profiles = water_filling_analysis(
        entries, {"tool": preds}, ratio_threshold=3.0, gini_threshold=0.3
    )
    profile = profiles["tool"]
    assert profile.is_water_filling is False
    assert profile.tier_ratio == pytest.approx(1.0)
    assert profile.gini_coefficient == pytest.approx(0.0)


def test_tier_ratio_t3_zero():
    """T3 DR=0, T1 DR>0 → ratio=inf, flagged."""
    entries = _make_entries_all_tiers()
    preds = [
        _pred("t1a", "HALLUCINATED"),
        _pred("t1b", "HALLUCINATED"),
        _pred("t2a", "HALLUCINATED"),
        _pred("t2b", "HALLUCINATED"),
        _pred("t3a", "VALID"),
        _pred("t3b", "VALID"),
    ]
    profiles = water_filling_analysis(
        entries, {"tool": preds}, ratio_threshold=3.0, gini_threshold=0.3
    )
    profile = profiles["tool"]
    assert math.isinf(profile.tier_ratio)
    assert profile.is_water_filling is True


def test_tier_ratio_both_zero():
    """T1 DR=0 and T3 DR=0 → ratio=1.0, not flagged (assuming Gini also low)."""
    entries = _make_entries_all_tiers()
    preds = [
        _pred("t1a", "VALID"),
        _pred("t1b", "VALID"),
        _pred("t2a", "HALLUCINATED"),
        _pred("t2b", "HALLUCINATED"),
        _pred("t3a", "VALID"),
        _pred("t3b", "VALID"),
    ]
    profiles = water_filling_analysis(
        entries, {"tool": preds}, ratio_threshold=3.0, gini_threshold=0.99
    )
    profile = profiles["tool"]
    assert profile.tier_ratio == pytest.approx(1.0)
    assert profile.is_water_filling is False
