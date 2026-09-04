"""Tests for scripts/compute_base_rate_precision.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "compute_base_rate_precision.py"

_spec = importlib.util.spec_from_file_location("compute_base_rate_precision", _SCRIPT)
assert _spec is not None and _spec.loader is not None
brp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brp)


class TestPrecisionAtPrevalence:
    def test_matches_bayes_rule_by_hand(self):
        # DR .99, FPR .16, prevalence 1%: TP = .0099, FP = .1584.
        assert brp.precision_at_prevalence(0.99, 0.16, 0.01) == pytest.approx(
            0.0099 / (0.0099 + 0.1584)
        )

    def test_perfect_specificity_gives_perfect_precision(self):
        """No false positives means every flag is a true finding, at any rate."""
        for p in (0.001, 0.5, 0.99):
            assert brp.precision_at_prevalence(0.8, 0.0, p) == pytest.approx(1.0)

    def test_precision_rises_monotonically_with_prevalence(self):
        """The whole point of the table: DR and FPR are fixed, precision is not."""
        vals = [brp.precision_at_prevalence(0.9, 0.2, p) for p in brp.DEFAULT_PREVALENCES]
        assert vals == sorted(vals)
        assert vals[0] < 0.05, "at a 0.1% base rate precision must be very low"
        assert vals[-1] > 0.85, "at the benchmark's own base rate it must look good"

    def test_zero_prevalence_yields_zero_precision(self):
        """With nothing to find, every flag is false."""
        assert brp.precision_at_prevalence(0.99, 0.16, 0.0) == 0.0

    def test_degenerate_detector_yields_zero(self):
        """A detector that flags nothing has no precision to report."""
        assert brp.precision_at_prevalence(0.0, 0.0, 0.5) == 0.0


class TestLoadResults:
    def test_skips_runs_without_an_fpr(self, tmp_path):
        """A split with no valid entries reports FPR None.

        Substituting zero there would manufacture perfect precision at every
        prevalence, so such a run must be dropped rather than defaulted.
        """
        (tmp_path / "with_fpr_test_public.json").write_text(
            json.dumps({"detection_rate": 0.9, "false_positive_rate": 0.1})
        )
        (tmp_path / "no_fpr_stress_test.json").write_text(
            json.dumps({"detection_rate": 0.97, "false_positive_rate": None})
        )
        out = brp.load_results(tmp_path)
        assert set(out) == {"with_fpr_test_public"}

    def test_ignores_manifest_and_unparseable_files(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"detection_rate": 1.0, "false_positive_rate": 0.0})
        )
        (tmp_path / "broken_test_public.json").write_text("{not json")
        (tmp_path / "list_test_public.json").write_text("[1, 2, 3]")
        assert brp.load_results(tmp_path) == {}
