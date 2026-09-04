"""Tests for selective prediction and calibration.

Several of these pin a *reason* rather than a number: that the curve is ordered
by certainty and not by class, that an API failure is not counted as an
abstention, and that AURC is refused across disjoint coverage domains. Those are
the properties that make the metric mean what it claims, and they are what a
later refactor is most likely to quietly undo.
"""

from __future__ import annotations

import pathlib

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.selective import (
    NOT_A_MEASUREMENT,
    abstention_breakdown,
    brier_decomposition,
    calibration_report,
    compare_aurc,
    format_reliability_diagram,
    format_risk_coverage,
    is_error_fallback,
    is_unevaluated,
    not_a_measurement,
    p_hallucinated,
    rejection_score,
    risk_coverage_curve,
    run_made_no_decisions,
)


def entry(key: str, label: str = "VALID", tier: int | None = None) -> BenchmarkEntry:
    kwargs = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "A. Author", "year": "2024"},
        "label": label,
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = "chimeric_title"
        kwargs["difficulty_tier"] = tier or 2
    return BenchmarkEntry(**kwargs)


def pred(key: str, label: str = "VALID", conf: float = 0.9, reason: str = "") -> Prediction:
    return Prediction(bibtex_key=key, label=label, confidence=conf, reason=reason)


def unrun(key: str) -> Prediction:
    """What ``fallback_predictions`` produces when the tool itself is unavailable.

    VALID at confidence 0.5 with no marker in ``reason`` — indistinguishable
    from a real verdict except for the flag, which is the point. Set here rather
    than passed to the constructor so these tests pin the ``getattr`` contract
    the metrics read through, and pass whether or not the field has landed in
    the schema.
    """
    p = Prediction(bibtex_key=key, label="VALID", confidence=0.5, reason="Tool unavailable")
    p.evaluated = False
    return p


class TestProbabilityTransform:
    """`confidence` is P(label is correct), not P(HALLUCINATED)."""

    def test_hallucinated_confidence_is_taken_directly(self):
        assert p_hallucinated(pred("a", "HALLUCINATED", 0.9)) == pytest.approx(0.9)

    def test_valid_confidence_is_inverted(self):
        # a VALID prediction at 0.9 claims the entry is valid, so P(hallucinated) is 0.1
        assert p_hallucinated(pred("a", "VALID", 0.9)) == pytest.approx(0.1)

    def test_uncertain_carries_no_signal(self):
        assert p_hallucinated(pred("a", "UNCERTAIN", 0.9)) == pytest.approx(0.5)

    def test_rejection_ranks_by_certainty_not_by_class(self):
        """A confident VALID and a confident HALLUCINATED are equally retainable."""
        assert rejection_score(pred("a", "VALID", 0.95)) == pytest.approx(
            rejection_score(pred("b", "HALLUCINATED", 0.95))
        )

    def test_uncertain_is_rejected_first(self):
        assert rejection_score(pred("a", "UNCERTAIN", 0.9)) == pytest.approx(0.0)
        assert rejection_score(pred("b", "VALID", 0.6)) > 0.0


class TestRiskCoverage:
    def test_perfect_tool_has_zero_risk_everywhere(self):
        entries = [entry("a", "VALID"), entry("b", "HALLUCINATED")]
        preds = {"a": pred("a", "VALID", 0.99), "b": pred("b", "HALLUCINATED", 0.99)}
        curve = risk_coverage_curve(entries, preds)
        assert curve.aurc == pytest.approx(0.0)
        assert curve.risk_at_full_coverage == pytest.approx(0.0)

    def test_informative_confidence_beats_uninformative(self):
        """A tool whose errors are its least confident calls scores a lower AURC."""
        entries = [entry(f"e{i}", "VALID") for i in range(10)]
        informative, blind = {}, {}
        for i in range(10):
            wrong = i >= 8  # the last two are errors
            label = "HALLUCINATED" if wrong else "VALID"
            # informative: the tool is least certain exactly where it is wrong
            informative[f"e{i}"] = pred(f"e{i}", label, 0.55 if wrong else 0.99)
            # blind: it is MOST certain where it is wrong, the realistic failure
            blind[f"e{i}"] = pred(f"e{i}", label, 0.99 if wrong else 0.55)
        assert (
            risk_coverage_curve(entries, informative).aurc
            < risk_coverage_curve(entries, blind).aurc
        )

    def test_uncertain_counts_as_an_error_when_retained(self):
        """At full coverage the tool has to commit; declining is not a right answer."""
        entries = [entry("a", "HALLUCINATED")]
        curve = risk_coverage_curve(entries, {"a": pred("a", "UNCERTAIN", 0.5)})
        assert curve.risk_at_full_coverage == pytest.approx(1.0)

    def test_missing_predictions_are_not_credited(self):
        entries = [entry("a", "VALID"), entry("b", "VALID")]
        curve = risk_coverage_curve(entries, {"a": pred("a", "VALID", 0.9)})
        assert curve.n_scored == 1
        assert curve.n_missing == 1

    def test_empty_input_does_not_raise(self):
        curve = risk_coverage_curve([], {})
        assert curve.points == [] and curve.aurc == 0.0


class TestErrorFallbacks:
    """An API failure is not a decision to abstain."""

    def test_marker_is_recognised(self):
        assert is_error_fallback(pred("a", "UNCERTAIN", 0.5, "[Error fallback] timeout"))
        assert not is_error_fallback(pred("a", "UNCERTAIN", 0.5, "insufficient evidence"))

    def test_excluded_from_the_curve_but_counted(self):
        entries = [entry("a", "VALID"), entry("b", "HALLUCINATED")]
        preds = {
            "a": pred("a", "VALID", 0.99),
            "b": pred("b", "UNCERTAIN", 0.5, "[Error fallback] 502"),
        }
        curve = risk_coverage_curve(entries, preds)
        assert curve.n_scored == 1
        assert curve.n_error_fallbacks == 1
        # excluding them must not silently improve the reported risk
        assert curve.risk_at_full_coverage == pytest.approx(0.0)

    def test_including_them_makes_the_tool_look_worse(self):
        entries = [entry("a", "VALID"), entry("b", "HALLUCINATED")]
        preds = {
            "a": pred("a", "VALID", 0.99),
            "b": pred("b", "UNCERTAIN", 0.5, "[Error fallback] 502"),
        }
        kept = risk_coverage_curve(entries, preds, exclude_non_decisions=False)
        assert kept.n_scored == 2
        assert kept.risk_at_full_coverage > 0.0

    def test_abstention_breakdown_separates_the_two(self):
        preds = [
            pred("a", "UNCERTAIN", 0.5, "[Error fallback] 502"),
            pred("b", "UNCERTAIN", 0.5, "conflicting sources"),
            pred("c", "VALID", 0.9),
        ]
        out = abstention_breakdown(preds)
        assert out["n_uncertain"] == 2
        assert out["n_error_fallbacks"] == 1
        assert out["n_genuine_abstentions"] == 1


class TestAURCComparability:
    """AURC over different coverage domains is a different integral."""

    def _curve(self, n_scored: int, n_entries: int):
        entries = [entry(f"e{i}", "VALID") for i in range(n_entries)]
        preds = {f"e{i}": pred(f"e{i}", "VALID", 0.9) for i in range(n_scored)}
        return risk_coverage_curve(entries, preds)

    def test_refuses_to_rank_across_disjoint_domains(self):
        wide = self._curve(500, 500)
        narrow = self._curve(68, 500)  # the real case: answered 68 of 500
        out = compare_aurc({"wide": wide, "narrow": narrow})
        assert out["ranking"] is None
        assert "not comparable" in out["reason"]

    def test_ranks_when_domains_agree(self):
        a, b = self._curve(100, 100), self._curve(100, 100)
        out = compare_aurc({"a": a, "b": b})
        assert out["ranking"] is not None
        assert set(out["aurc"]) == {"a", "b"}

    def test_reports_ranges_when_it_refuses(self):
        out = compare_aurc({"wide": self._curve(500, 500), "narrow": self._curve(68, 500)})
        assert "coverage_ranges" in out


class TestBrierDecomposition:
    def test_identity_holds(self):
        """brier = reliability - resolution + uncertainty, to binning error."""
        pairs = [(0.9, True), (0.8, True), (0.3, False), (0.1, False), (0.6, True)]
        d = brier_decomposition(pairs, n_bins=5)
        assert d.brier == pytest.approx(d.reliability - d.resolution + d.uncertainty, abs=1e-9)

    def test_perfect_calibration_has_near_zero_reliability(self):
        pairs = [(1.0, True)] * 50 + [(0.0, False)] * 50
        d = brier_decomposition(pairs)
        assert d.reliability == pytest.approx(0.0, abs=1e-9)
        assert d.skill == pytest.approx(1.0)

    def test_confident_and_wrong_is_penalised(self):
        pairs = [(0.99, False)] * 50 + [(0.01, True)] * 50
        d = brier_decomposition(pairs)
        assert d.brier > 0.9
        assert d.skill < 0.0

    def test_empty_input_does_not_raise(self):
        assert brier_decomposition([]).brier == 0.0


class TestCalibrationReport:
    def test_flagged_calibration_sees_what_the_aggregate_hides(self):
        """Confident wrong accusations, drowned by a correct VALID majority.

        This is the shape measured on a real screening run: the aggregate looks
        healthy while every flag is a confident mistake.
        """
        entries = [entry(f"v{i}", "VALID") for i in range(96)]
        preds = {f"v{i}": pred(f"v{i}", "VALID", 0.95) for i in range(96)}
        for i in range(4):  # four confident false accusations
            entries.append(entry(f"f{i}", "VALID"))
            preds[f"f{i}"] = pred(f"f{i}", "HALLUCINATED", 0.97)

        rep = calibration_report(entries, preds)
        assert rep.n_flagged == 4
        # every flag is wrong, so the flagged Brier is near its worst
        assert rep.flagged.brier > 0.9
        # while the overall figure stays comfortable
        assert rep.overall.brier < 0.1

    def test_per_tier_is_reported(self):
        entries = [entry("a", "HALLUCINATED", tier=1), entry("b", "HALLUCINATED", tier=3)]
        preds = {"a": pred("a", "HALLUCINATED", 0.9), "b": pred("b", "VALID", 0.9)}
        rep = calibration_report(entries, preds)
        assert set(rep.per_tier) <= {1, 2, 3}
        assert 1 in rep.per_tier and 3 in rep.per_tier

    def test_error_fallbacks_excluded_and_counted(self):
        entries = [entry("a", "VALID"), entry("b", "VALID")]
        preds = {
            "a": pred("a", "VALID", 0.9),
            "b": pred("b", "UNCERTAIN", 0.5, "[Error fallback]"),
        }
        rep = calibration_report(entries, preds)
        assert rep.n_error_fallbacks == 1


class TestRendering:
    def test_diagram_renders_without_bins(self):
        assert "no scored" in format_reliability_diagram([])

    def test_curve_renders_and_names_its_domain(self):
        entries = [entry(f"e{i}", "VALID") for i in range(20)]
        preds = {f"e{i}": pred(f"e{i}", "VALID", 0.9) for i in range(20)}
        out = format_risk_coverage(risk_coverage_curve(entries, preds))
        assert "AURC" in out and "coverage" in out

    def test_curve_reports_excluded_fallbacks_in_its_header(self):
        entries = [entry("a", "VALID"), entry("b", "VALID")]
        preds = {
            "a": pred("a", "VALID", 0.9),
            "b": pred("b", "UNCERTAIN", 0.5, "[Error fallback]"),
        }
        out = format_risk_coverage(risk_coverage_curve(entries, preds))
        assert "error fallbacks excluded" in out


class TestUnevaluatedIsNotADecision:
    """A tool that never ran must not score as a tool that answered.

    ``fallback_predictions`` writes VALID at 0.5 with no marker, so before the
    ``evaluated`` flag a fully timed-out run entered the curve as a set of
    confident correct answers. Four such runs shipped as pre-screening
    ablations.
    """

    def test_flag_is_read_and_defaults_to_evaluated(self):
        assert is_unevaluated(unrun("a"))
        # A prediction written before the field existed cannot be reclassified.
        assert not is_unevaluated(pred("b", "VALID", 0.9))

    def test_excluded_from_the_curve_and_counted_apart_from_fallbacks(self):
        entries = [entry("a", "HALLUCINATED"), entry("b"), entry("c")]
        preds = {
            "a": pred("a", "HALLUCINATED", 0.9),
            "b": unrun("b"),
            "c": pred("c", "UNCERTAIN", 0.5, "[Error fallback] 502"),
        }
        curve = risk_coverage_curve(entries, preds)
        assert curve.n_scored == 1
        assert curve.n_unevaluated == 1
        assert curve.n_error_fallbacks == 1

    def test_a_null_run_scores_perfectly_if_it_is_not_refused(self):
        """The defect, stated as a test: silence looks like a flawless tool."""
        entries = [entry("a"), entry("b")]
        preds = {"a": unrun("a"), "b": unrun("b")}
        kept = risk_coverage_curve(entries, preds, exclude_non_decisions=False)
        assert kept.aurc == 0.0  # a perfect selective predictor, on no evidence

    def test_a_run_that_decided_nothing_is_refused(self):
        entries = [entry("a"), entry("b")]
        preds = {"a": unrun("a"), "b": unrun("b")}
        with pytest.raises(ValueError, match="decided nothing"):
            risk_coverage_curve(entries, preds)

    def test_having_nothing_to_evaluate_is_not_evaluating_nothing(self):
        assert risk_coverage_curve([], {}).points == []
        assert not run_made_no_decisions([])

    def test_run_predicate_covers_both_mechanisms(self):
        assert run_made_no_decisions([unrun("a"), unrun("b")])
        assert run_made_no_decisions([pred("a", "UNCERTAIN", 0.5, "[Error fallback] 502")])
        assert not run_made_no_decisions([unrun("a"), pred("b", "HALLUCINATED", 0.9)])

    def test_breakdown_counts_them_though_their_label_is_valid(self):
        out = abstention_breakdown([unrun("a"), pred("b", "UNCERTAIN", 0.5), pred("c")])
        assert out["n_unevaluated"] == 1
        assert out["n_uncertain"] == 1  # the unevaluated one is VALID, not UNCERTAIN
        assert out["n_genuine_abstentions"] == 1

    def test_calibration_excludes_and_counts_them(self):
        entries = [entry("a", "HALLUCINATED"), entry("b")]
        preds = {"a": pred("a", "HALLUCINATED", 0.9), "b": unrun("b")}
        rep = calibration_report(entries, preds)
        assert rep.n_unevaluated == 1
        assert rep.n_flagged == 1

    def test_curve_header_names_them(self):
        entries = [entry("a", "HALLUCINATED"), entry("b")]
        preds = {"a": pred("a", "HALLUCINATED", 0.9), "b": unrun("b")}
        out = format_risk_coverage(risk_coverage_curve(entries, preds))
        assert "1 never evaluated" in out


class TestRunsThatAreNotMeasurements:
    """Named runs the flag cannot catch, because they predate it.

    A prediction set written before ``Prediction.evaluated`` reads as fully
    evaluated, so the four quarantined pre-screening ablations would score as
    real arms. The register is the honest option until they are re-run;
    inferring the null signature is not, and ``harc_with_s2key_dev_public``
    is why: it reports ``mean_api_calls`` 0.0 and is a real evaluation.
    """

    def test_lookup_tolerates_a_filename(self):
        name = "harc_no_prescreening_dev_public"
        assert not_a_measurement(name) == not_a_measurement(f"{name}.json")
        assert not_a_measurement("bibtexupdater_dev_public") is None
        assert not_a_measurement(None) is None

    def test_a_real_run_is_not_registered(self):
        """The one with half the null signature and a real result."""
        assert not_a_measurement("harc_with_s2key_dev_public") is None

    def test_the_degenerate_baseline_scores_normally(self):
        """``always_valid`` predicts VALID at confidence 1.0 for every entry.

        DR 0.0 and FPR 0.0 with no API calls is its correct output, so it is the
        case a null-signature rule would silently discard. It carries no marker
        and it ran, so nothing here excludes it: its curve is flat at the base
        rate, which is the honest description of a constant baseline.
        """
        entries = [entry("a", "HALLUCINATED"), entry("b"), entry("c")]
        preds = {k: pred(k, "VALID", 1.0) for k in ("a", "b", "c")}
        curve = risk_coverage_curve(entries, preds, run_name="always_valid_dev_public")
        assert curve.unscoreable is None
        assert curve.n_scored == 3
        assert curve.risk_at_full_coverage == pytest.approx(1 / 3)

    def test_scoring_a_registered_run_raises(self):
        entries = [entry("a", "HALLUCINATED")]
        preds = {"a": pred("a", "VALID", 0.5)}
        with pytest.raises(ValueError, match="not a measurement"):
            risk_coverage_curve(entries, preds, run_name="harc_no_prescreening_dev_public")
        with pytest.raises(ValueError, match="not a measurement"):
            calibration_report(entries, preds, run_name="harc_no_prescreening_dev_public.json")

    def test_batch_callers_get_it_marked_instead(self):
        """One dead arm must not take a whole comparison down."""
        entries = [entry("a", "HALLUCINATED")]
        preds = {"a": pred("a", "VALID", 0.5)}
        curve = risk_coverage_curve(
            entries, preds, run_name="harc_no_prescreening_dev_public", strict=False
        )
        assert curve.unscoreable
        assert curve.points == []
        assert "not a measurement" in format_risk_coverage(curve)

    def test_a_run_that_decided_nothing_can_also_be_marked_rather_than_raised(self):
        entries = [entry("a"), entry("b")]
        preds = {"a": unrun("a"), "b": unrun("b")}
        curve = risk_coverage_curve(entries, preds, strict=False)
        assert curve.unscoreable and curve.n_unevaluated == 2

    def test_register_names_only_quarantined_runs(self):
        """Shrink-only: a re-run arm comes back to the results dir, and then the
        entry is a false record rather than a caveat."""
        results_dir = pathlib.Path(__file__).resolve().parent.parent / "data/v1.2/baseline_results"
        if not results_dir.is_dir():
            pytest.skip("real baseline_results dir not present")
        back = sorted(n for n in NOT_A_MEASUREMENT if (results_dir / f"{n}.json").exists())
        assert not back, (
            f"registered as not a measurement but present among the released results: {back}. "
            "If they were re-run, remove them from NOT_A_MEASUREMENT."
        )
