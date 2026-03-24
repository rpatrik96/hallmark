"""Tests for hallmark.evaluation.power."""

from __future__ import annotations

import pytest

from hallmark.dataset.schema import BenchmarkEntry
from hallmark.evaluation.power import (
    mde_two_proportion,
    required_n,
    subtype_power_audit,
    z_score,
)


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


class TestZScore:
    def test_standard_values(self):
        # z(0.025) ≈ 1.96
        assert z_score(0.025) == pytest.approx(1.96, abs=0.01)
        # z(0.1) ≈ 1.282
        assert z_score(0.10) == pytest.approx(1.282, abs=0.01)
        # z(0.5) = 0
        assert z_score(0.5) == pytest.approx(0.0, abs=0.01)

    def test_symmetry(self):
        assert z_score(0.025) == pytest.approx(-z_score(0.975), abs=0.01)

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            z_score(0.0)
        with pytest.raises(ValueError):
            z_score(1.0)


class TestMDE:
    def test_known_value_n100(self):
        # n=100, p0=0.5 → MDE ≈ 0.198
        mde = mde_two_proportion(100, p0=0.5)
        assert mde == pytest.approx(0.198, abs=0.01)

    def test_known_value_n785(self):
        # n=785 → MDE ≈ 0.071 (z_alpha + z_power ≈ 2.80, sqrt(0.5/785) ≈ 0.0252)
        mde = mde_two_proportion(785, p0=0.5)
        assert mde == pytest.approx(0.071, abs=0.01)

    def test_larger_n_smaller_mde(self):
        assert mde_two_proportion(1000) < mde_two_proportion(100)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError):
            mde_two_proportion(0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            mde_two_proportion(-10)


class TestRequiredN:
    def test_inverse_of_mde(self):
        # Round-trip: required_n(mde(n)) ≈ n (within ceiling)
        for n in [50, 100, 200, 500]:
            mde = mde_two_proportion(n)
            n_back = required_n(mde)
            assert abs(n_back - n) <= 1, f"n={n}, mde={mde}, n_back={n_back}"

    def test_known_value(self):
        # delta=0.10, p0=0.5 → n ≈ 393
        n = required_n(0.10)
        assert 390 <= n <= 400

    def test_zero_delta_raises(self):
        with pytest.raises(ValueError):
            required_n(0.0)


class TestSubtypePowerAudit:
    def test_flags_underpowered(self):
        # 10 entries of one type → MDE > 0.20 → underpowered
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=1, h_type="fabricated_doi") for i in range(10)
        ]
        results = subtype_power_audit(entries)
        assert "fabricated_doi" in results
        assert results["fabricated_doi"].underpowered is True
        assert results["fabricated_doi"].n == 10
        assert results["fabricated_doi"].mde > 0.20

    def test_not_underpowered_large_n(self):
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=1, h_type="fabricated_doi") for i in range(500)
        ]
        results = subtype_power_audit(entries)
        assert results["fabricated_doi"].underpowered is False

    def test_counts_only_hallucinated(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        ]
        results = subtype_power_audit(entries)
        assert results["fabricated_doi"].n == 1

    def test_custom_target_deltas(self):
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=1, h_type="fabricated_doi") for i in range(100)
        ]
        results = subtype_power_audit(entries, target_deltas=[0.05, 0.30])
        req = results["fabricated_doi"].required_n_by_delta
        assert 0.05 in req
        assert 0.30 in req
        assert req[0.05] > req[0.30]  # smaller delta needs more samples

    def test_tier_populated(self):
        entries = [
            _entry(f"h{i}", "HALLUCINATED", tier=3, h_type="near_miss_title") for i in range(50)
        ]
        results = subtype_power_audit(entries)
        assert results["near_miss_title"].tier == 3
