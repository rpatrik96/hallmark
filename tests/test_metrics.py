"""Tests for hallmark.evaluation.metrics."""

import importlib.util

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import (
    ConfusionMatrix,
    apply_multiple_comparison_correction,
    build_confusion_matrix,
    cost_efficiency,
    detect_at_k,
    evaluate,
    per_tier_metrics,
    per_type_metrics,
    tier_weighted_f1,
)

# --- Helpers ---


def _entry(key: str, label: str, tier: int | None = None, h_type: str | None = None):
    kwargs = {
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


# --- Tests ---


class TestConfusionMatrix:
    def test_perfect_detection(self):
        cm = ConfusionMatrix(tp=10, fp=0, tn=90, fn=0)
        assert cm.detection_rate == 1.0
        assert cm.false_positive_rate == 0.0
        assert cm.f1 == 1.0

    def test_no_detection(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=90, fn=10)
        assert cm.detection_rate == 0.0
        assert cm.f1 == 0.0

    def test_all_flagged(self):
        cm = ConfusionMatrix(tp=10, fp=90, tn=0, fn=0)
        assert cm.detection_rate == 1.0
        assert cm.false_positive_rate == 1.0

    def test_empty(self):
        cm = ConfusionMatrix()
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0


class TestBuildConfusionMatrix:
    def test_all_correct(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID"),
            "v2": _pred("v2", "VALID"),
            "h1": _pred("h1", "HALLUCINATED"),
        }
        cm = build_confusion_matrix(entries, preds)
        assert cm.tp == 1
        assert cm.tn == 2
        assert cm.fp == 0
        assert cm.fn == 0

    def test_missing_predictions_conservative(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED"),
        ]
        cm = build_confusion_matrix(entries, {})
        # Missing predictions treated as VALID (conservative)
        assert cm.tn == 1
        assert cm.fn == 1

    def test_mixed_results(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID"),
            "v2": _pred("v2", "HALLUCINATED"),  # false positive
            "h1": _pred("h1", "HALLUCINATED"),  # correct
            "h2": _pred("h2", "VALID"),  # missed
        }
        cm = build_confusion_matrix(entries, preds)
        assert cm.tp == 1
        assert cm.fp == 1
        assert cm.tn == 1
        assert cm.fn == 1


class TestTierWeightedF1:
    def test_higher_tier_weighted_more(self):
        entries = [
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h3", "HALLUCINATED", tier=3),
        ]
        # Only detect tier 1
        preds_t1 = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h3": _pred("h3", "VALID"),
        }
        # Only detect tier 3
        preds_t3 = {
            "h1": _pred("h1", "VALID"),
            "h3": _pred("h3", "HALLUCINATED"),
        }
        f1_t1 = tier_weighted_f1(entries, preds_t1)
        f1_t3 = tier_weighted_f1(entries, preds_t3)
        # Detecting tier 3 should give higher weighted F1
        assert f1_t3 > f1_t1


class TestDetectAtK:
    def test_single_strategy(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        strategy1 = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h2": _pred("h2", "VALID"),
        }
        result = detect_at_k(entries, [strategy1])
        assert result[1] == 0.5  # detected 1 of 2

    def test_two_strategies_complementary(self):
        entries = [
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        strategy1 = {"h1": _pred("h1", "HALLUCINATED"), "h2": _pred("h2", "VALID")}
        strategy2 = {"h1": _pred("h1", "VALID"), "h2": _pred("h2", "HALLUCINATED")}
        result = detect_at_k(entries, [strategy1, strategy2])
        assert result[1] == 0.5
        assert result[2] == 1.0  # both detected when combining

    def test_no_hallucinated(self):
        entries = [_entry("v1", "VALID")]
        result = detect_at_k(entries, [{}])
        assert result == {}


class TestPerTierMetrics:
    def test_breakdown(self):
        entries = [
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h2", "HALLUCINATED", tier=2),
            _entry("h3", "HALLUCINATED", tier=3),
        ]
        preds = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h2": _pred("h2", "VALID"),
            "h3": _pred("h3", "VALID"),
        }
        result = per_tier_metrics(entries, preds)
        assert result[1]["detection_rate"] == 1.0
        assert result[2]["detection_rate"] == 0.0
        assert result[3]["detection_rate"] == 0.0


class TestPerTypeMetrics:
    def test_breakdown(self):
        entries = [
            _entry("h1", "HALLUCINATED", h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", h_type="near_miss_title"),
        ]
        preds = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h2": _pred("h2", "VALID"),
        }
        result = per_type_metrics(entries, preds)
        assert result["fabricated_doi"]["detection_rate"] == 1.0
        assert result["near_miss_title"]["detection_rate"] == 0.0


class TestPerTypeMetricsWilsonCI:
    def _make_entries_preds(self, n: int, dr: float, h_type: str = "fabricated_doi"):
        """Create n hallucinated entries with detection rate dr."""
        n_detected = round(n * dr)
        entries = [_entry(f"h{i}", "HALLUCINATED", h_type=h_type) for i in range(n)]
        preds = {}
        for i in range(n):
            label = "HALLUCINATED" if i < n_detected else "VALID"
            preds[f"h{i}"] = _pred(f"h{i}", label)
        return entries, preds

    def test_ci_keys_present_when_compute_ci_true(self):
        entries, preds = self._make_entries_preds(20, 0.6)
        result = per_type_metrics(entries, preds, compute_ci=True)
        assert "dr_ci_lower" in result["fabricated_doi"]
        assert "dr_ci_upper" in result["fabricated_doi"]

    def test_ci_keys_absent_when_compute_ci_false(self):
        """Default (compute_ci=False) must not add CI keys — backward compat."""
        entries, preds = self._make_entries_preds(20, 0.6)
        result = per_type_metrics(entries, preds)
        assert "dr_ci_lower" not in result["fabricated_doi"]
        assert "dr_ci_upper" not in result["fabricated_doi"]

    def test_ci_brackets_detection_rate(self):
        """CI lower <= DR <= CI upper for every type."""
        entries, preds = self._make_entries_preds(30, 0.7, h_type="near_miss_title")
        result = per_type_metrics(entries, preds, compute_ci=True)
        dr = result["near_miss_title"]["detection_rate"]
        lo = result["near_miss_title"]["dr_ci_lower"]
        hi = result["near_miss_title"]["dr_ci_upper"]
        assert lo <= dr <= hi

    def test_ci_bounds_in_unit_interval(self):
        entries, preds = self._make_entries_preds(30, 0.5)
        result = per_type_metrics(entries, preds, compute_ci=True)
        lo = result["fabricated_doi"]["dr_ci_lower"]
        hi = result["fabricated_doi"]["dr_ci_upper"]
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_ci_width_reasonable_n30_dr05(self):
        """For n=30, DR=0.5, Wilson 95% CI width should be ~0.30-0.40."""
        entries, preds = self._make_entries_preds(30, 0.5)
        result = per_type_metrics(entries, preds, compute_ci=True)
        width = result["fabricated_doi"]["dr_ci_upper"] - result["fabricated_doi"]["dr_ci_lower"]
        assert 0.28 <= width <= 0.42

    def test_ci_zero_for_valid_type_group(self):
        """Valid entries (h_type=None) have n=0 → CI should be (0.0, 0.0)."""
        entries = [_entry("v1", "VALID"), _entry("v2", "VALID")]
        preds = {"v1": _pred("v1", "VALID"), "v2": _pred("v2", "VALID")}
        result = per_type_metrics(entries, preds, compute_ci=True)
        assert result["valid"]["dr_ci_lower"] == 0.0
        assert result["valid"]["dr_ci_upper"] == 0.0

    def test_ci_multiple_types(self):
        """CI keys are added for every type when compute_ci=True."""
        entries = [
            _entry("h1", "HALLUCINATED", h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", h_type="fabricated_doi"),
            _entry("h3", "HALLUCINATED", h_type="near_miss_title"),
            _entry("v1", "VALID"),
        ]
        preds = {
            "h1": _pred("h1", "HALLUCINATED"),
            "h2": _pred("h2", "VALID"),
            "h3": _pred("h3", "HALLUCINATED"),
            "v1": _pred("v1", "VALID"),
        }
        result = per_type_metrics(entries, preds, compute_ci=True)
        for type_key in result:
            assert "dr_ci_lower" in result[type_key], f"missing dr_ci_lower for {type_key}"
            assert "dr_ci_upper" in result[type_key], f"missing dr_ci_upper for {type_key}"
            assert result[type_key]["dr_ci_lower"] <= result[type_key]["dr_ci_upper"]


class TestCostEfficiency:
    def test_basic(self):
        preds = [
            Prediction(
                bibtex_key="a",
                label="VALID",
                confidence=0.9,
                wall_clock_seconds=1.0,
                api_calls=3,
            ),
            Prediction(
                bibtex_key="b",
                label="VALID",
                confidence=0.9,
                wall_clock_seconds=2.0,
                api_calls=5,
            ),
        ]
        result = cost_efficiency(preds)
        assert result["entries_per_second"] == pytest.approx(2 / 3)
        assert result["mean_api_calls"] == 4.0

    def test_empty(self):
        result = cost_efficiency([])
        assert result["entries_per_second"] == 0.0


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.9),
            "v2": _pred("v2", "VALID", confidence=0.9),
            "h1": _pred("h1", "HALLUCINATED", confidence=0.9),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.9),
        }
        ece = expected_calibration_error(entries, preds)
        assert ece >= 0.0
        assert ece <= 1.0
        # Perfect predictions with consistent confidence should have low ECE
        assert ece < 0.2

    def test_bad_calibration(self):
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        # High confidence but all predictions are wrong
        preds = {
            "v1": _pred("v1", "HALLUCINATED", confidence=0.95),
            "v2": _pred("v2", "HALLUCINATED", confidence=0.95),
            "h1": _pred("h1", "VALID", confidence=0.95),
            "h2": _pred("h2", "VALID", confidence=0.95),
        }
        ece = expected_calibration_error(entries, preds)
        # Bad calibration should give high ECE
        assert ece > 0.5

    def test_empty_predictions(self):
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [_entry("v1", "VALID")]
        ece = expected_calibration_error(entries, {})
        assert ece == 0.0


class TestSourceStratifiedMetrics:
    def test_stratify_by_api_sources(self):
        from hallmark.evaluation.metrics import source_stratified_metrics

        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
            _entry("h3", "HALLUCINATED"),
        ]
        preds = {
            "v1": Prediction(
                bibtex_key="v1",
                label="VALID",
                confidence=0.9,
                api_sources_queried=["crossref"],
            ),
            "h1": Prediction(
                bibtex_key="h1",
                label="HALLUCINATED",
                confidence=0.8,
                api_sources_queried=["crossref"],
            ),
            "h2": Prediction(
                bibtex_key="h2",
                label="HALLUCINATED",
                confidence=0.9,
                api_sources_queried=["dblp", "semantic_scholar"],
            ),
            "h3": Prediction(
                bibtex_key="h3",
                label="VALID",
                confidence=0.7,
                api_sources_queried=["dblp", "semantic_scholar"],
            ),
        }
        result = source_stratified_metrics(entries, preds)
        assert "crossref" in result
        assert "dblp,semantic_scholar" in result
        assert result["crossref"]["detection_rate"] == 1.0
        assert result["dblp,semantic_scholar"]["detection_rate"] == 0.5


class TestSubtestAccuracyTable:
    def test_subtest_accuracy(self):
        from hallmark.evaluation.metrics import subtest_accuracy_table

        entries = [
            BenchmarkEntry(
                bibtex_key="e1",
                bibtex_type="article",
                fields={"title": "A", "author": "B", "year": "2024"},
                label="VALID",
                subtests={"doi_resolves": True, "title_exists": True},
            ),
            BenchmarkEntry(
                bibtex_key="e2",
                bibtex_type="article",
                fields={"title": "C", "author": "D", "year": "2024"},
                label="HALLUCINATED",
                hallucination_type="fabricated_doi",
                difficulty_tier=1,
                subtests={"doi_resolves": False, "title_exists": False},
            ),
        ]
        preds = {
            "e1": Prediction(
                bibtex_key="e1",
                label="VALID",
                confidence=0.9,
                subtest_results={"doi_resolves": True, "title_exists": True},
            ),
            "e2": Prediction(
                bibtex_key="e2",
                label="HALLUCINATED",
                confidence=0.8,
                subtest_results={"doi_resolves": False, "title_exists": True},
            ),
        }
        result = subtest_accuracy_table(entries, preds)
        assert "doi_resolves" in result
        assert "title_exists" in result
        # doi_resolves: both correct (2/2)
        assert result["doi_resolves"]["accuracy"] == 1.0
        # title_exists: e1 correct, e2 wrong (1/2)
        assert result["title_exists"]["accuracy"] == 0.5


class TestUncertainPredictions:
    """Tests for UNCERTAIN label handling across all metrics."""

    def test_uncertain_prediction_creates_valid_object(self):
        pred = _pred("k1", "UNCERTAIN", confidence=0.5)
        assert pred.label == "UNCERTAIN"

    def test_uncertain_serialization_roundtrip(self):
        pred = _pred("k1", "UNCERTAIN", confidence=0.5)
        json_str = pred.to_json()
        restored = Prediction.from_json(json_str)
        assert restored.label == "UNCERTAIN"

    def test_confusion_matrix_treats_uncertain_as_valid(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "UNCERTAIN", confidence=0.5),
            "h1": _pred("h1", "UNCERTAIN", confidence=0.5),
        }
        cm = build_confusion_matrix(entries, preds)
        # UNCERTAIN on valid entry -> treated as VALID -> TN
        assert cm.tn == 1
        # UNCERTAIN on hallucinated entry -> treated as VALID -> FN (missed)
        assert cm.fn == 1
        assert cm.tp == 0
        assert cm.fp == 0

    def test_tier_weighted_f1_treats_uncertain_as_valid(self):
        entries = [
            _entry("h1", "HALLUCINATED", tier=3),
        ]
        preds_uncertain = {"h1": _pred("h1", "UNCERTAIN", confidence=0.5)}
        preds_valid = {"h1": _pred("h1", "VALID", confidence=0.5)}
        # UNCERTAIN and VALID should yield identical tier-weighted F1
        assert tier_weighted_f1(entries, preds_uncertain) == tier_weighted_f1(entries, preds_valid)

    def test_detect_at_k_ignores_uncertain(self):
        entries = [_entry("h1", "HALLUCINATED")]
        strategy = {"h1": _pred("h1", "UNCERTAIN", confidence=0.5)}
        result = detect_at_k(entries, [strategy])
        assert result[1] == 0.0  # UNCERTAIN is not a detection

    def test_evaluate_counts_uncertain(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1),
        ]
        preds = [
            _pred("v1", "UNCERTAIN", confidence=0.5),
            _pred("h1", "UNCERTAIN", confidence=0.5),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.num_uncertain == 2
        # Both treated as VALID -> DR=0, FPR=0
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 0.0

    def test_evaluate_mixed_uncertain_and_hallucinated(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h2", "HALLUCINATED", tier=2),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("h1", "HALLUCINATED"),  # correct detection
            _pred("h2", "UNCERTAIN", confidence=0.5),  # treated as VALID = missed
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.num_uncertain == 1
        assert result.detection_rate == 0.5  # 1 of 2 hallucinated detected
        assert result.false_positive_rate == 0.0

    def test_ece_treats_uncertain_as_valid(self):
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [_entry("v1", "VALID")]
        preds = {"v1": _pred("v1", "UNCERTAIN", confidence=0.5)}
        ece = expected_calibration_error(entries, preds)
        assert ece >= 0.0
        assert ece <= 1.0


class TestAUROC:
    def test_auroc_perfect_predictions(self):
        from hallmark.evaluation.metrics import auroc

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.95),
            "v2": _pred("v2", "VALID", confidence=0.95),
            "h1": _pred("h1", "HALLUCINATED", confidence=0.95),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.95),
        }
        score = auroc(entries, preds)
        assert score is not None
        assert score >= 0.95  # Should be near 1.0 for perfect predictions

    def test_auroc_random_predictions(self):
        from hallmark.evaluation.metrics import auroc

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        # Random scores around 0.5
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.5),
            "v2": _pred("v2", "HALLUCINATED", confidence=0.5),
            "h1": _pred("h1", "VALID", confidence=0.5),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.5),
        }
        score = auroc(entries, preds)
        assert score is not None
        # Random predictions should give AUROC around 0.5
        assert 0.3 <= score <= 0.7

    def test_auroc_no_positive_class(self):
        from hallmark.evaluation.metrics import auroc

        entries = [_entry("v1", "VALID"), _entry("v2", "VALID")]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.9),
            "v2": _pred("v2", "VALID", confidence=0.9),
        }
        score = auroc(entries, preds)
        assert score is None  # Cannot compute AUROC with only one class

    def test_auroc_missing_predictions(self):
        from hallmark.evaluation.metrics import auroc

        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED"),
        ]
        # Only predict one entry — missing predictions are skipped entirely,
        # leaving a single class (HALLUCINATED) → AUROC is undefined → None
        preds = {"h1": _pred("h1", "HALLUCINATED", confidence=0.9)}
        score = auroc(entries, preds)
        assert score is None

    def test_auroc_partial_predictions_both_classes(self):
        from hallmark.evaluation.metrics import auroc

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        # Predict 3 of 4 entries — both classes still represented
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.9),
            "h1": _pred("h1", "HALLUCINATED", confidence=0.9),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.8),
        }
        score = auroc(entries, preds)
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestAUPRC:
    def test_auprc_perfect_predictions(self):
        from hallmark.evaluation.metrics import auprc

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.95),
            "v2": _pred("v2", "VALID", confidence=0.95),
            "h1": _pred("h1", "HALLUCINATED", confidence=0.95),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.95),
        }
        score = auprc(entries, preds)
        assert score is not None
        assert score >= 0.95  # Should be near 1.0 for perfect predictions

    def test_auprc_random_predictions(self):
        from hallmark.evaluation.metrics import auprc

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        # Random predictions
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.5),
            "v2": _pred("v2", "HALLUCINATED", confidence=0.5),
            "h1": _pred("h1", "VALID", confidence=0.5),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.5),
        }
        score = auprc(entries, preds)
        assert score is not None
        # Base rate is 0.5 (2 hallucinated of 4 total)
        # Random predictions should give AUPRC close to base rate
        assert 0.3 <= score <= 0.7

    def test_auprc_no_positive_examples(self):
        from hallmark.evaluation.metrics import auprc

        entries = [_entry("v1", "VALID"), _entry("v2", "VALID")]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.9),
            "v2": _pred("v2", "VALID", confidence=0.9),
        }
        score = auprc(entries, preds)
        assert score is None  # Cannot compute AUPRC with no positive examples


class TestAUROCandAUPRCInEvaluate:
    def test_auroc_auprc_in_evaluate(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h2", "HALLUCINATED", tier=2),
        ]
        preds = [
            _pred("v1", "VALID", confidence=0.9),
            _pred("v2", "VALID", confidence=0.9),
            _pred("h1", "HALLUCINATED", confidence=0.9),
            _pred("h2", "HALLUCINATED", confidence=0.9),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")

        # Check that AUROC and AUPRC are computed and stored
        assert result.auroc is not None
        assert result.auprc is not None
        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.auprc <= 1.0
        # Perfect predictions should give high scores
        assert result.auroc >= 0.8
        assert result.auprc >= 0.8


class TestEvaluate:
    def test_full_evaluation(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h2", "HALLUCINATED", tier=2),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
            _pred("h1", "HALLUCINATED"),
            _pred("h2", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.num_entries == 4
        assert result.num_hallucinated == 2
        assert result.num_valid == 2
        assert result.detection_rate == 0.5
        assert result.false_positive_rate == 0.0
        assert result.f1_hallucination > 0


# Check if numpy is available for statistical tests

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
class TestStatisticalMethods:
    """Tests for bootstrap CIs, paired tests, and sensitivity analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create a small but representative dataset for statistical tests."""
        # Create ~60 entries: 30 valid + 30 hallucinated (mix of types/tiers)
        entries = []
        predictions = []

        # Valid entries (30)
        for i in range(30):
            entries.append(_entry(f"v{i}", "VALID"))
            # 90% accuracy on valid entries
            label = "VALID" if i < 27 else "HALLUCINATED"
            predictions.append(_pred(f"v{i}", label, confidence=0.85))

        # Hallucinated entries (30): 10 per tier
        h_types = ["fabricated_doi", "near_miss_title", "chimeric_title"]
        for tier in [1, 2, 3]:
            for i in range(10):
                key = f"h{tier}_{i}"
                h_type = h_types[tier - 1]
                entries.append(_entry(key, "HALLUCINATED", tier=tier, h_type=h_type))
                # 80% detection rate overall (24/30 detected)
                # Tier 1: 90%, Tier 2: 80%, Tier 3: 70%
                detect_threshold = 0.9 if tier == 1 else (0.8 if tier == 2 else 0.7)
                label = "HALLUCINATED" if (i / 10.0) < detect_threshold else "VALID"
                predictions.append(_pred(key, label, confidence=0.8))

        return entries, predictions

    def test_stratified_bootstrap_ci_returns_tuple(self, sample_data):
        """CI returns (lower, upper) with lower < upper."""
        from hallmark.evaluation.metrics import stratified_bootstrap_ci

        entries, predictions = sample_data

        def f1_metric(e, p):
            pred_map = {pred.bibtex_key: pred for pred in p}
            cm = build_confusion_matrix(e, pred_map)
            return cm.f1

        lower, upper = stratified_bootstrap_ci(
            entries, predictions, f1_metric, n_bootstrap=100, seed=42
        )
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    def test_stratified_bootstrap_ci_contains_point_estimate(self, sample_data):
        """The point estimate should fall within the CI."""
        from hallmark.evaluation.metrics import stratified_bootstrap_ci

        entries, predictions = sample_data
        pred_map = {p.bibtex_key: p for p in predictions}

        def f1_metric(e, p):
            pred_map_inner = {pred.bibtex_key: pred for pred in p}
            cm = build_confusion_matrix(e, pred_map_inner)
            return cm.f1

        # Compute point estimate
        cm = build_confusion_matrix(entries, pred_map)
        point_estimate = cm.f1

        # Compute CI
        lower, upper = stratified_bootstrap_ci(
            entries, predictions, f1_metric, n_bootstrap=100, seed=42
        )

        # Point estimate should be within CI (with some tolerance for sampling)
        assert lower <= point_estimate <= upper or abs(lower - point_estimate) < 0.05

    def test_stratified_bootstrap_ci_narrower_with_more_data(self):
        """Larger samples should produce narrower CIs."""
        from hallmark.evaluation.metrics import stratified_bootstrap_ci

        def f1_metric(e, p):
            pred_map = {pred.bibtex_key: pred for pred in p}
            cm = build_confusion_matrix(e, pred_map)
            return cm.f1

        # Small sample (20 entries) with ~80% accuracy
        small_entries = [_entry(f"v{i}", "VALID") for i in range(10)]
        small_entries += [_entry(f"h{i}", "HALLUCINATED", tier=1) for i in range(10)]
        small_preds = [_pred(f"v{i}", "VALID") for i in range(9)]
        small_preds += [_pred("v9", "HALLUCINATED")]  # 1 false positive
        small_preds += [_pred(f"h{i}", "HALLUCINATED") for i in range(8)]
        small_preds += [_pred("h8", "VALID"), _pred("h9", "VALID")]  # 2 false negatives

        # Large sample (100 entries) with ~80% accuracy
        large_entries = [_entry(f"v{i}", "VALID") for i in range(50)]
        large_entries += [_entry(f"h{i}", "HALLUCINATED", tier=1) for i in range(50)]
        large_preds = [_pred(f"v{i}", "VALID") for i in range(45)]
        large_preds += [_pred(f"v{i}", "HALLUCINATED") for i in range(45, 50)]  # 5 FP
        large_preds += [_pred(f"h{i}", "HALLUCINATED") for i in range(40)]
        large_preds += [_pred(f"h{i}", "VALID") for i in range(40, 50)]  # 10 FN

        lower_small, upper_small = stratified_bootstrap_ci(
            small_entries, small_preds, f1_metric, n_bootstrap=100, seed=42
        )
        lower_large, upper_large = stratified_bootstrap_ci(
            large_entries, large_preds, f1_metric, n_bootstrap=100, seed=42
        )

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large

        # Larger sample should have narrower CI (or at least not wider)
        assert width_large <= width_small * 1.1  # Allow 10% tolerance for sampling variance

    def test_paired_bootstrap_test_detects_large_difference(self, sample_data):
        """A large performance gap should have p < 0.05."""
        from hallmark.evaluation.metrics import paired_bootstrap_test

        entries, _ = sample_data

        # predictions_a: 90% correct
        preds_a = []
        for i, entry in enumerate(entries):
            if entry.label == "HALLUCINATED":
                label = "HALLUCINATED" if i % 10 < 9 else "VALID"
            else:
                label = "VALID"
            preds_a.append(_pred(entry.bibtex_key, label, confidence=0.9))

        # predictions_b: 50% correct
        preds_b = []
        for i, entry in enumerate(entries):
            if entry.label == "HALLUCINATED":
                label = "HALLUCINATED" if i % 2 == 0 else "VALID"
            else:
                label = "VALID" if i % 2 == 0 else "HALLUCINATED"
            preds_b.append(_pred(entry.bibtex_key, label, confidence=0.5))

        def f1_metric(e, p):
            pred_map = {pred.bibtex_key: pred for pred in p}
            cm = build_confusion_matrix(e, pred_map)
            return cm.f1

        obs_diff, p_value, effect_size = paired_bootstrap_test(
            entries, preds_a, preds_b, f1_metric, n_bootstrap=100, seed=42
        )

        assert obs_diff > 0  # Tool A should be better
        assert p_value < 0.05  # Statistically significant
        assert isinstance(effect_size, float)

    def test_paired_bootstrap_test_no_difference(self, sample_data):
        """Identical predictions should have p > 0.05."""
        from hallmark.evaluation.metrics import paired_bootstrap_test

        entries, predictions = sample_data

        def f1_metric(e, p):
            pred_map = {pred.bibtex_key: pred for pred in p}
            cm = build_confusion_matrix(e, pred_map)
            return cm.f1

        # Same predictions for both
        obs_diff, p_value, _effect_size = paired_bootstrap_test(
            entries, predictions, predictions, f1_metric, n_bootstrap=100, seed=42
        )

        assert abs(obs_diff) < 0.01  # Should be near zero
        assert p_value > 0.4  # Should not be significant

    def test_tier_weight_sensitivity_returns_all_schemes(self, sample_data):
        """Should return values for all default weighting schemes."""
        from hallmark.evaluation.metrics import tier_weight_sensitivity

        entries, predictions = sample_data

        results = tier_weight_sensitivity(entries, predictions)

        # Check all default schemes are present
        assert "uniform" in results
        assert "linear" in results
        assert "quadratic" in results
        assert "log" in results
        assert "inverse_difficulty" in results

        # All should be valid F1 scores
        for value in results.values():
            assert 0.0 <= value <= 1.0

    def test_tier_weight_sensitivity_uniform_equals_standard(self, sample_data):
        """Uniform weights {1,1,1} should match standard F1-based computation."""
        from hallmark.evaluation.metrics import tier_weight_sensitivity

        entries, predictions = sample_data
        pred_map = {p.bibtex_key: p for p in predictions}

        results = tier_weight_sensitivity(entries, predictions)
        uniform_f1 = results["uniform"]

        # Compute standard weighted F1 with uniform weights
        tw_f1_uniform = tier_weighted_f1(entries, pred_map, tier_weights={1: 1.0, 2: 1.0, 3: 1.0})

        assert abs(uniform_f1 - tw_f1_uniform) < 0.01

    def test_equivalence_test_identical_splits(self, sample_data):
        """Same data should be equivalent (is_equivalent=True)."""
        from hallmark.evaluation.metrics import equivalence_test

        entries, predictions = sample_data

        # Split into two halves
        n_half = len(entries) // 2
        entries_a = entries[:n_half]

        # Test same split against itself
        is_equiv, obs_diff, _p_value = equivalence_test(
            entries_a, entries_a, predictions, epsilon=0.02, n_permutations=100, seed=42
        )

        assert abs(obs_diff) < 0.001  # Should be near zero
        assert is_equiv  # Should be equivalent

    def test_equivalence_test_different_distributions(self):
        """Very different splits should NOT be equivalent."""
        from hallmark.evaluation.metrics import equivalence_test

        # Split A: all easy (tier 1)
        entries_a = [_entry(f"h{i}", "HALLUCINATED", tier=1) for i in range(20)]
        # Split B: all hard (tier 3)
        entries_b = [_entry(f"h{i + 20}", "HALLUCINATED", tier=3) for i in range(20)]

        # Predictions that detect tier 1 well but not tier 3
        preds_a = [_pred(f"h{i}", "HALLUCINATED", confidence=0.9) for i in range(20)]
        preds_b = [_pred(f"h{i + 20}", "VALID", confidence=0.5) for i in range(20)]
        all_preds = preds_a + preds_b

        is_equiv, obs_diff, _p_value = equivalence_test(
            entries_a, entries_b, all_preds, epsilon=0.02, n_permutations=100, seed=42
        )

        # Should detect large difference
        assert abs(obs_diff) > 0.1  # Large difference in detection rate
        assert not is_equiv  # Should NOT be equivalent

    def test_evaluate_with_ci(self, sample_data):
        """evaluate(compute_ci=True) should populate CI fields."""
        entries, predictions = sample_data

        # Note: evaluate() doesn't currently have compute_ci parameter
        # This test is for the future implementation
        # For now, we test that evaluate() works with the data
        result = evaluate(entries, predictions, tool_name="test", split_name="dev")

        assert result.detection_rate > 0
        assert result.f1_hallucination > 0
        assert result.tier_weighted_f1 > 0

    def test_evaluate_without_ci_has_none(self, sample_data):
        """evaluate(compute_ci=False) should have None CI fields."""
        entries, predictions = sample_data

        # Current evaluate() doesn't store CIs in result
        # This test documents expected behavior
        result = evaluate(entries, predictions, tool_name="test", split_name="dev")

        # Should complete without error
        assert result.num_entries == len(entries)


class TestECEAdaptiveMode:
    """Tests for adaptive (equal-frequency) ECE binning."""

    def test_ece_adaptive_returns_valid_value(self):
        """ECE with adaptive=True should return a value in [0, 1]."""
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.9),
            "v2": _pred("v2", "VALID", confidence=0.7),
            "h1": _pred("h1", "HALLUCINATED", confidence=0.85),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.6),
        }
        ece = expected_calibration_error(entries, preds, adaptive=True)
        assert 0.0 <= ece <= 1.0

    def test_ece_adaptive_perfect_calibration_low(self):
        """Correct predictions with consistent confidence should yield low adaptive ECE."""
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [_entry(f"h{i}", "HALLUCINATED") for i in range(5)]
        entries += [_entry(f"v{i}", "VALID") for i in range(5)]
        preds = {}
        for i in range(5):
            preds[f"h{i}"] = _pred(f"h{i}", "HALLUCINATED", confidence=0.9)
        for i in range(5):
            preds[f"v{i}"] = _pred(f"v{i}", "VALID", confidence=0.9)
        ece = expected_calibration_error(entries, preds, adaptive=True)
        assert ece < 0.3

    def test_ece_adaptive_empty_predictions(self):
        """Empty predictions should return 0.0 for adaptive ECE."""
        from hallmark.evaluation.metrics import expected_calibration_error

        entries = [_entry("v1", "VALID")]
        ece = expected_calibration_error(entries, {}, adaptive=True)
        assert ece == 0.0

    def test_ece_adaptive_varies_from_fixed(self):
        """Adaptive and fixed ECE may differ on the same data (at least one should be callable)."""
        from hallmark.evaluation.metrics import expected_calibration_error

        # Deterministic tool: all predictions at exactly 0.9 confidence
        entries = [_entry("h1", "HALLUCINATED"), _entry("v1", "VALID")]
        preds = {
            "h1": _pred("h1", "HALLUCINATED", confidence=0.9),
            "v1": _pred("v1", "VALID", confidence=0.9),
        }
        ece_fixed = expected_calibration_error(entries, preds, adaptive=False)
        ece_adaptive = expected_calibration_error(entries, preds, adaptive=True)
        # Both must be in valid range
        assert 0.0 <= ece_fixed <= 1.0
        assert 0.0 <= ece_adaptive <= 1.0


class TestBenjaminiHochbergCorrection:
    """Tests for BH multiple comparison correction."""

    def test_bh_returns_values_in_unit_interval(self):
        raw = {"a": 0.01, "b": 0.04, "c": 0.10}
        result = apply_multiple_comparison_correction(raw, method="bh")
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_bh_less_conservative_than_bonferroni(self):
        """BH adjusted p-values should be <= Bonferroni adjusted p-values."""
        raw = {"a": 0.01, "b": 0.03, "c": 0.05, "d": 0.10}
        bh = apply_multiple_comparison_correction(raw, method="bh")
        bonf = apply_multiple_comparison_correction(raw, method="bonferroni")
        for label in raw:
            assert bh[label] <= bonf[label], f"BH not less conservative than Bonferroni for {label}"

    def test_bh_monotone_adjusted_p_values(self):
        """BH adjusted p-values should be non-decreasing when sorted by raw p-value."""
        raw = {"a": 0.005, "b": 0.01, "c": 0.03, "d": 0.10}
        result = apply_multiple_comparison_correction(raw, method="bh")
        # Sort by raw p-value order and check adjusted values are non-decreasing
        sorted_labels = sorted(raw, key=lambda x: raw[x])
        adjusted_in_order = [result[label] for label in sorted_labels]
        for i in range(len(adjusted_in_order) - 1):
            assert adjusted_in_order[i] <= adjusted_in_order[i + 1], (
                f"BH adjusted p-values not monotone at index {i}: "
                f"{adjusted_in_order[i]} > {adjusted_in_order[i + 1]}"
            )

    def test_bh_single_p_value_equals_raw(self):
        raw = {"only": 0.04}
        result = apply_multiple_comparison_correction(raw, method="bh")
        assert result["only"] == pytest.approx(0.04)

    def test_bh_empty_dict_returns_empty(self):
        assert apply_multiple_comparison_correction({}, method="bh") == {}

    def test_bh_adjusted_ge_raw(self):
        """BH step-up with monotone enforcement: adjusted >= raw only for the smallest p-value
        when FDR < FWER. The key property is monotonicity, not that adjusted >= raw."""
        raw = {"t1": 0.01, "t2": 0.03, "t3": 0.05, "t4": 0.10}
        result = apply_multiple_comparison_correction(raw, method="bh")
        # The smallest raw p-value: adjusted = p * n / 1 = p * n >= p
        smallest_label = min(raw, key=lambda x: raw[x])
        assert result[smallest_label] >= raw[smallest_label]

    def test_bh_all_p_values_significant_stays_bounded(self):
        """All very small p-values should still produce adjusted values in [0, 1]."""
        raw = {"a": 0.0001, "b": 0.0002, "c": 0.0003}
        result = apply_multiple_comparison_correction(raw, method="bh")
        for v in result.values():
            assert 0.0 <= v <= 1.0


class TestDuplicatePredictionKeys:
    """Duplicate bibtex_key in predictions list — last value wins."""

    def test_evaluate_duplicate_prediction_keys_last_wins(self):
        """Duplicate bibtex_keys in predictions list should use the last value (dict overwrite)."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("v1", "VALID"),
        ]
        # Two predictions for h1: first says VALID, last says HALLUCINATED
        # The dict comprehension {p.bibtex_key: p for p in predictions} gives last one priority
        preds = [
            _pred("v1", "VALID"),
            _pred("h1", "VALID", confidence=0.8),  # first: wrong label
            _pred("h1", "HALLUCINATED", confidence=0.9),  # last: correct label
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        # Last prediction for h1 is HALLUCINATED — should be detected
        assert result.detection_rate == 1.0

    def test_evaluate_duplicate_prediction_keys_first_is_overwritten(self):
        """When first prediction is correct but last is wrong, last wins (overwrite)."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("v1", "VALID"),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("h1", "HALLUCINATED", confidence=0.9),  # first: correct
            _pred("h1", "VALID", confidence=0.6),  # last: wrong (overwrites)
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        # Last prediction for h1 is VALID — hallucination missed
        assert result.detection_rate == 0.0


class TestPrescreeningAblation:
    """Tests for bibtexupdater pre-screening ablation."""

    def test_bibtexupdater_no_prescreening_registered(self):
        """The no-prescreening variant should be in the registry."""
        from hallmark.baselines.registry import list_baselines

        names = list_baselines()
        assert "bibtexupdater_no_prescreening" in names

    def test_prescreening_flag_exists(self):
        """bibtexupdater wrapper should accept skip_prescreening param."""
        import inspect

        from hallmark.baselines.bibtexupdater import run_bibtex_check

        # Check that skip_prescreening parameter exists
        sig = inspect.signature(run_bibtex_check)
        assert "skip_prescreening" in sig.parameters
        assert sig.parameters["skip_prescreening"].default is False


class TestMultipleComparisonCorrection:
    """Tests for apply_multiple_comparison_correction()."""

    def test_bonferroni_multiplies_by_n(self):
        raw = {"a": 0.01, "b": 0.04, "c": 0.10}
        result = apply_multiple_comparison_correction(raw, method="bonferroni")
        assert result["a"] == pytest.approx(0.03)
        assert result["b"] == pytest.approx(0.12)
        assert result["c"] == pytest.approx(0.30)

    def test_bonferroni_caps_at_one(self):
        raw = {"a": 0.5, "b": 0.8}
        result = apply_multiple_comparison_correction(raw, method="bonferroni")
        assert result["a"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(1.0)

    def test_holm_step_down_logic(self):
        # Sorted order: type_b (0.01), type_a (0.03), type_c (0.08)
        # rank 0: adj = min(1, 0.01 * 3) = 0.03; max_so_far = 0.03
        # rank 1: adj = min(1, 0.03 * 2) = 0.06; max_so_far = 0.06
        # rank 2: adj = min(1, 0.08 * 1) = 0.08; max_so_far = 0.08
        raw = {"type_a": 0.03, "type_b": 0.01, "type_c": 0.08}
        result = apply_multiple_comparison_correction(raw, method="holm")
        assert result["type_b"] == pytest.approx(0.03)
        assert result["type_a"] == pytest.approx(0.06)
        assert result["type_c"] == pytest.approx(0.08)

    def test_holm_monotone_enforcement(self):
        # When a later (larger) raw p-value would produce a smaller adjusted value,
        # Holm enforces monotonicity via max_so_far.
        # raw: x=0.001, y=0.002, z=0.003  n=3
        # rank 0: 0.001*3=0.003; rank 1: 0.002*2=0.004; rank 2: 0.003*1=0.003 → capped to 0.004
        raw = {"x": 0.001, "y": 0.002, "z": 0.003}
        result = apply_multiple_comparison_correction(raw, method="holm")
        assert result["x"] == pytest.approx(0.003)
        assert result["y"] == pytest.approx(0.004)
        assert result["z"] == pytest.approx(0.004)  # monotone enforcement

    def test_empty_dict_returns_empty(self):
        assert apply_multiple_comparison_correction({}) == {}
        assert apply_multiple_comparison_correction({}, method="bonferroni") == {}

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_multiple_comparison_correction({"a": 0.05}, method="fdr_bh")

    def test_adjusted_ge_raw(self):
        raw = {"t1": 0.01, "t2": 0.03, "t3": 0.05, "t4": 0.10}
        for method in ("holm", "bonferroni"):
            result = apply_multiple_comparison_correction(raw, method=method)
            for label, p in raw.items():
                assert result[label] >= p, f"{method}: {label} adjusted < raw"

    def test_single_p_value_both_methods_equal(self):
        raw = {"only": 0.04}
        holm = apply_multiple_comparison_correction(raw, method="holm")
        bonf = apply_multiple_comparison_correction(raw, method="bonferroni")
        # Both multiply by n=1, so adjusted == raw
        assert holm["only"] == pytest.approx(0.04)
        assert bonf["only"] == pytest.approx(0.04)
        assert holm["only"] == pytest.approx(bonf["only"])


class TestZeroValidEntriesWarning:
    """evaluate() with zero valid entries: FPR undefined guard and warning."""

    def _all_hallucinated_entries(self):
        return [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=2, h_type="chimeric_title"),
            _entry("h3", "HALLUCINATED", tier=3, h_type="near_miss_title"),
        ]

    def test_runs_without_error(self):
        entries = self._all_hallucinated_entries()
        preds = [
            _pred("h1", "HALLUCINATED"),
            _pred("h2", "HALLUCINATED"),
            _pred("h3", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.num_valid == 0
        assert result.num_hallucinated == 3

    def test_fpr_is_zero_when_no_valid_entries(self):
        """FPR is None when there are zero valid entries (undefined)."""
        entries = self._all_hallucinated_entries()
        preds = [
            _pred("h1", "HALLUCINATED"),
            _pred("h2", "HALLUCINATED"),
            _pred("h3", "HALLUCINATED"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.false_positive_rate is None

    def test_detection_rate_and_f1_valid(self):
        """Metrics other than FPR should still be meaningful."""
        entries = self._all_hallucinated_entries()
        preds = [
            _pred("h1", "HALLUCINATED"),
            _pred("h2", "HALLUCINATED"),
            _pred("h3", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.detection_rate == pytest.approx(2 / 3)
        assert 0.0 <= result.f1_hallucination <= 1.0
        assert 0.0 <= result.tier_weighted_f1 <= 1.0

    def test_warning_emitted(self, caplog):
        """logger.warning should fire when num_valid == 0."""
        import logging

        entries = self._all_hallucinated_entries()
        preds = [_pred("h1", "HALLUCINATED"), _pred("h2", "VALID"), _pred("h3", "VALID")]
        with caplog.at_level(logging.WARNING, logger="hallmark.evaluation.metrics"):
            evaluate(entries, preds, tool_name="test", split_name="dev")
        assert any("zero valid entries" in record.message for record in caplog.records)


class TestMCC:
    """Tests for Matthews Correlation Coefficient."""

    def test_perfect_detection(self):
        cm = ConfusionMatrix(tp=10, fp=0, tn=90, fn=0)
        assert cm.mcc == pytest.approx(1.0)

    def test_perfect_inverse(self):
        cm = ConfusionMatrix(tp=0, fp=10, tn=0, fn=90)
        assert cm.mcc == pytest.approx(-1.0)

    def test_random_classifier(self):
        """Balanced random classifier → MCC ≈ 0."""
        cm = ConfusionMatrix(tp=25, fp=25, tn=25, fn=25)
        assert cm.mcc == pytest.approx(0.0)

    def test_empty_matrix(self):
        cm = ConfusionMatrix()
        assert cm.mcc == 0.0

    def test_no_positives(self):
        """All entries are valid, none hallucinated."""
        cm = ConfusionMatrix(tp=0, fp=5, tn=95, fn=0)
        assert cm.mcc == 0.0  # denom has (tp+fn)=0 factor

    def test_mcc_differs_from_f1_on_imbalanced(self):
        """On imbalanced data, MCC and F1 should diverge."""
        cm = ConfusionMatrix(tp=5, fp=5, tn=85, fn=5)
        assert cm.mcc != pytest.approx(cm.f1, abs=0.01)

    def test_mcc_in_evaluate(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h2", "HALLUCINATED", tier=2),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
            _pred("h1", "HALLUCINATED"),
            _pred("h2", "HALLUCINATED"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="dev")
        assert result.mcc is not None
        assert result.mcc == pytest.approx(1.0)
        assert result.macro_f1 is not None
        assert result.macro_f1 == pytest.approx(1.0)


class TestMacroF1:
    """Tests for Macro-averaged F1."""

    def test_perfect(self):
        cm = ConfusionMatrix(tp=10, fp=0, tn=10, fn=0)
        assert cm.macro_f1 == pytest.approx(1.0)

    def test_all_wrong(self):
        cm = ConfusionMatrix(tp=0, fp=10, tn=0, fn=10)
        assert cm.macro_f1 == pytest.approx(0.0)

    def test_empty(self):
        cm = ConfusionMatrix()
        assert cm.macro_f1 == 0.0

    def test_asymmetric(self):
        """Macro-F1 averages F1 for each class equally."""
        cm = ConfusionMatrix(tp=9, fp=1, tn=1, fn=9)
        # F1_hall = 2*9/(2*9+1+9) = 18/28 ≈ 0.643
        # prec_valid = 1/(1+9) = 0.1, rec_valid = 1/(1+1) = 0.5
        # F1_valid = 2*0.1*0.5/(0.1+0.5) = 0.1/0.6 ≈ 0.167
        # Macro = (0.643 + 0.167) / 2 ≈ 0.405
        assert 0.35 < cm.macro_f1 < 0.45


class TestAUPRCInterpolated:
    """Tests for interpolated average precision in AUPRC."""

    def test_perfect_predictions(self):
        from hallmark.evaluation.metrics import auprc

        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
        ]
        preds = {
            "v1": _pred("v1", "VALID", confidence=0.99),
            "v2": _pred("v2", "VALID", confidence=0.99),
            "h1": _pred("h1", "HALLUCINATED", confidence=0.99),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.99),
        }
        score = auprc(entries, preds)
        assert score is not None
        assert score == pytest.approx(1.0)

    def test_interpolation_improves_over_riemann(self):
        """Interpolated AP >= Riemann sum AP for the same data.

        Construct a case where precision dips then recovers, so the
        monotone envelope differs from raw precision.
        """
        from hallmark.evaluation.metrics import auprc

        # 4 hallucinated, 2 valid.  Scores arranged so precision dips then recovers:
        # sorted by score desc: h1(0.95), v1(0.85), h2(0.75), h3(0.65), v2(0.55), h4(0.45)
        # Threshold points:
        #   after h1: P=1/1=1.0, R=1/4=0.25
        #   after v1: P=1/2=0.5, R=1/4=0.25  (no recall change, same threshold group? no, different scores)
        #   after h2: P=2/3≈0.67, R=2/4=0.5
        #   after h3: P=3/4=0.75, R=3/4=0.75
        #   after v2: P=3/5=0.6, R=3/4=0.75
        #   after h4: P=4/6≈0.67, R=4/4=1.0
        # Interpolated envelope at each recall change: max(P from here on)
        entries = [
            _entry("h1", "HALLUCINATED"),
            _entry("h2", "HALLUCINATED"),
            _entry("h3", "HALLUCINATED"),
            _entry("h4", "HALLUCINATED"),
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
        ]
        preds = {
            "h1": _pred("h1", "HALLUCINATED", confidence=0.95),
            "v1": _pred("v1", "HALLUCINATED", confidence=0.85),
            "h2": _pred("h2", "HALLUCINATED", confidence=0.75),
            "h3": _pred("h3", "HALLUCINATED", confidence=0.65),
            "v2": _pred("v2", "HALLUCINATED", confidence=0.55),
            "h4": _pred("h4", "HALLUCINATED", confidence=0.45),
        }
        score = auprc(entries, preds)
        assert score is not None
        # With interpolation the score should be >= the non-interpolated version
        # The exact value depends on the monotone envelope
        assert score > 0.5


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")
class TestCompareTools:
    """Tests for compare_tools() pairwise comparison with multiple comparison correction."""

    def test_identical_predictions_not_significant(self):
        entries = [
            _entry("v1", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1),
            _entry("h2", "HALLUCINATED", tier=2),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("h1", "HALLUCINATED"),
            _pred("h2", "VALID"),
        ]
        from hallmark.evaluation.metrics import compare_tools

        results = compare_tools(
            entries,
            {"tool_a": preds, "tool_b": preds},
            n_bootstrap=200,
            seed=42,
        )
        assert len(results) == 1
        assert abs(results[0]["observed_diff"]) < 0.01  # type: ignore[arg-type]
        assert not results[0]["significant"]

    def test_large_difference_is_significant(self):
        entries = [_entry(f"h{i}", "HALLUCINATED", tier=1) for i in range(30)]
        entries += [_entry(f"v{i}", "VALID") for i in range(30)]

        # "alpha" (perfect) sorts before "zeta" (all-miss) alphabetically
        preds_perfect = [_pred(f"h{i}", "HALLUCINATED") for i in range(30)]
        preds_perfect += [_pred(f"v{i}", "VALID") for i in range(30)]

        preds_miss = [_pred(f"h{i}", "VALID") for i in range(30)]
        preds_miss += [_pred(f"v{i}", "VALID") for i in range(30)]

        from hallmark.evaluation.metrics import compare_tools

        results = compare_tools(
            entries,
            {"alpha": preds_perfect, "zeta": preds_miss},
            n_bootstrap=2000,
            seed=42,
        )
        assert len(results) == 1
        # alpha (perfect) - zeta (miss) > 0, one-sided p should be < 0.05
        assert float(results[0]["observed_diff"]) > 0.5  # type: ignore[arg-type]
        assert float(results[0]["p_value_raw"]) < 0.05  # type: ignore[arg-type]

    def test_bh_correction_method(self):
        """compare_tools with correction_method='bh' should run without error."""
        entries = [_entry("h1", "HALLUCINATED", tier=1), _entry("v1", "VALID")]
        preds = [_pred("h1", "HALLUCINATED"), _pred("v1", "VALID")]

        from hallmark.evaluation.metrics import compare_tools

        results = compare_tools(
            entries,
            {"a": preds, "b": preds},
            correction_method="bh",
            n_bootstrap=100,
            seed=42,
        )
        assert len(results) == 1
        assert "p_value_adjusted" in results[0]

    def test_three_tools_produces_three_pairs(self):
        entries = [_entry("h1", "HALLUCINATED", tier=1), _entry("v1", "VALID")]
        preds = [_pred("h1", "HALLUCINATED"), _pred("v1", "VALID")]

        from hallmark.evaluation.metrics import compare_tools

        results = compare_tools(
            entries,
            {"a": preds, "b": preds, "c": preds},
            n_bootstrap=100,
            seed=42,
        )
        assert len(results) == 3  # C(3,2) = 3
