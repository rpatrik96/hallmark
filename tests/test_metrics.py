"""Tests for hallmark.evaluation.metrics."""

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import (
    ConfusionMatrix,
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
