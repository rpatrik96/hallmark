"""Integration tests for the full evaluate() pipeline."""

import importlib.util

import pytest

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import evaluate

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _make_entry(key, label="HALLUCINATED", h_type="fabricated_doi", tier=1):
    """Create a minimal BenchmarkEntry for testing."""
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {
            "title": f"Test Paper {key}",
            "author": "Test Author",
            "year": "2024",
            "journal": "Test Journal",
        },
        "label": label,
        "generation_method": "test",
        "explanation": "Test entry",
        "source": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type
        kwargs["difficulty_tier"] = tier
    return BenchmarkEntry(**kwargs)


def _make_pred(key, label="HALLUCINATED", confidence=0.9):
    return Prediction(bibtex_key=key, label=label, confidence=confidence, reason="test")


class TestEvaluatePerfect:
    """Perfect predictions → DR=1.0, FPR=0.0, F1≈1.0."""

    def test_perfect_detection_rate(self):
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("h2", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
            _make_entry("v2", "VALID"),
        ]
        preds = [
            _make_pred("h1", "HALLUCINATED"),
            _make_pred("h2", "HALLUCINATED"),
            _make_pred("v1", "VALID"),
            _make_pred("v2", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="perfect", split_name="test")
        assert result.detection_rate == pytest.approx(1.0)
        assert result.false_positive_rate == pytest.approx(0.0)
        assert result.f1_hallucination == pytest.approx(1.0)

    def test_perfect_counts(self):
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
        ]
        preds = [
            _make_pred("h1", "HALLUCINATED"),
            _make_pred("v1", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="perfect", split_name="test")
        assert result.num_entries == 2
        assert result.num_hallucinated == 1
        assert result.num_valid == 1


class TestEvaluateAllWrong:
    """All predictions inverted → DR=0.0, FPR=1.0."""

    def test_all_wrong_detection_rate(self):
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("h2", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
            _make_entry("v2", "VALID"),
        ]
        preds = [
            _make_pred("h1", "VALID"),
            _make_pred("h2", "VALID"),
            _make_pred("v1", "HALLUCINATED"),
            _make_pred("v2", "HALLUCINATED"),
        ]
        result = evaluate(entries, preds, tool_name="all_wrong", split_name="test")
        assert result.detection_rate == pytest.approx(0.0)
        assert result.false_positive_rate == pytest.approx(1.0)


class TestEvaluateMixed:
    """Mixed predictions with known TP/FP/TN/FN → exact metric values."""

    def test_exact_metrics(self):
        # 2 hallucinated: detect h1, miss h2 → TP=1, FN=1
        # 2 valid: flag v1 wrongly, keep v2 → FP=1, TN=1
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("h2", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
            _make_entry("v2", "VALID"),
        ]
        preds = [
            _make_pred("h1", "HALLUCINATED"),  # TP
            _make_pred("h2", "VALID"),  # FN
            _make_pred("v1", "HALLUCINATED"),  # FP
            _make_pred("v2", "VALID"),  # TN
        ]
        result = evaluate(entries, preds, tool_name="mixed", split_name="test")
        assert result.detection_rate == pytest.approx(0.5)  # 1/2 hallucinated found
        assert result.false_positive_rate == pytest.approx(0.5)  # 1/2 valid flagged
        # precision=0.5, recall=0.5 → F1=0.5
        assert result.f1_hallucination == pytest.approx(0.5)
        assert result.num_entries == 4


class TestEvaluateEmptyPredictions:
    """Zero-overlap predictions → conservative (all VALID)."""

    def test_zero_overlap_handled_gracefully(self, caplog):
        import logging

        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
        ]
        # Predictions with entirely different keys
        preds = [
            _make_pred("other1", "HALLUCINATED"),
            _make_pred("other2", "VALID"),
        ]
        with caplog.at_level(logging.WARNING, logger="hallmark.evaluation.metrics"):
            result = evaluate(entries, preds, tool_name="empty", split_name="test")
        # Zero overlap should emit a warning
        assert any("overlap" in r.message.lower() for r in caplog.records)
        # Missing predictions treated as VALID → DR=0
        assert result.detection_rate == pytest.approx(0.0)

    def test_empty_prediction_list(self):
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
        ]
        result = evaluate(entries, [], tool_name="empty", split_name="test")
        assert result.detection_rate == pytest.approx(0.0)
        assert result.false_positive_rate == pytest.approx(0.0)
        assert result.num_entries == 2


class TestEvaluateUncertain:
    """UNCERTAIN predictions are excluded from classification metrics (new protocol).

    UNCERTAIN entries count toward coverage but are not included in the confusion
    matrix — they do not contribute to TP, FP, TN, or FN.
    """

    def test_uncertain_excluded_from_confusion_matrix(self):
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("v1", "VALID"),
        ]
        preds = [
            _make_pred("h1", "UNCERTAIN", confidence=0.5),  # excluded → no TP/FN
            _make_pred("v1", "UNCERTAIN", confidence=0.5),  # excluded → no TN/FP
        ]
        result = evaluate(entries, preds, tool_name="uncertain", split_name="test")
        # Both entries excluded from confusion matrix → detection_rate=0.0 (no TPs)
        assert result.detection_rate == pytest.approx(0.0)
        assert result.false_positive_rate == pytest.approx(0.0)
        assert result.num_uncertain == 2

    def test_uncertain_counted_separately(self):
        entries = [
            _make_entry("h1", "HALLUCINATED"),
            _make_entry("h2", "HALLUCINATED"),
        ]
        preds = [
            _make_pred("h1", "HALLUCINATED"),
            _make_pred("h2", "UNCERTAIN", confidence=0.5),  # excluded from metrics
        ]
        result = evaluate(entries, preds, tool_name="uncertain", split_name="test")
        assert result.num_uncertain == 1
        # h2 excluded; only h1 counts → detection_rate = 1/1 = 1.0
        assert result.detection_rate == pytest.approx(1.0)


class TestEvaluateSingleEntry:
    """Single-entry edge case."""

    def test_single_hallucinated_correct(self):
        entries = [_make_entry("h1", "HALLUCINATED")]
        preds = [_make_pred("h1", "HALLUCINATED")]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert result.detection_rate == pytest.approx(1.0)
        assert result.num_entries == 1

    def test_single_hallucinated_missed(self):
        entries = [_make_entry("h1", "HALLUCINATED")]
        preds = [_make_pred("h1", "VALID")]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert result.detection_rate == pytest.approx(0.0)


class TestEvaluateAllValid:
    """No hallucinated entries → detection_rate is meaningless (0 by convention)."""

    def test_no_hallucinations(self):
        entries = [
            _make_entry("v1", "VALID"),
            _make_entry("v2", "VALID"),
        ]
        preds = [
            _make_pred("v1", "VALID"),
            _make_pred("v2", "VALID"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert result.num_hallucinated == 0
        # DR = 0 when no positives (no TP, no FN)
        assert result.detection_rate == pytest.approx(0.0)
        assert result.false_positive_rate == pytest.approx(0.0)


class TestEvaluateAllHallucinated:
    """No valid entries → FPR is None (undefined)."""

    def test_no_valid_entries_fpr_none(self):
        entries = [
            _make_entry("h1", "HALLUCINATED", tier=1),
            _make_entry("h2", "HALLUCINATED", tier=2),
        ]
        preds = [
            _make_pred("h1", "HALLUCINATED"),
            _make_pred("h2", "HALLUCINATED"),
        ]
        result = evaluate(entries, preds, tool_name="test", split_name="test")
        assert result.num_valid == 0
        assert result.false_positive_rate is None
        assert result.detection_rate == pytest.approx(1.0)


class TestDegenerateBaselinesOnRealData:
    """Degenerate baselines (random, always_hallucinated, always_valid) on real dev_public entries."""

    def _load_entries(self, n: int = 10):
        from hallmark.dataset.loader import load_split

        return load_split("dev_public")[:n]

    def test_always_hallucinated_detection_rate_is_one(self):
        from hallmark.baselines.degenerate import always_hallucinated_baseline

        entries = self._load_entries()
        predictions = always_hallucinated_baseline(entries)
        result = evaluate(
            entries, predictions, tool_name="always_hallucinated", split_name="dev_public"
        )

        # Every hallucinated entry is flagged → detection_rate == 1.0
        assert result.detection_rate == pytest.approx(1.0)

    def test_always_valid_detection_rate_is_zero(self):
        from hallmark.baselines.degenerate import always_valid_baseline

        entries = self._load_entries()
        predictions = always_valid_baseline(entries)
        result = evaluate(entries, predictions, tool_name="always_valid", split_name="dev_public")

        # No hallucinated entry is flagged → detection_rate == 0.0
        assert result.detection_rate == pytest.approx(0.0)

    def test_random_baseline_returns_correct_number_of_predictions(self):
        from hallmark.baselines.degenerate import random_baseline

        entries = self._load_entries()
        predictions = random_baseline(entries, seed=0)
        assert len(predictions) == len(entries)

    def test_all_baselines_produce_populated_evaluation_result(self):
        from hallmark.baselines.degenerate import (
            always_hallucinated_baseline,
            always_valid_baseline,
            random_baseline,
        )

        entries = self._load_entries()

        for name, preds in [
            ("always_hallucinated", always_hallucinated_baseline(entries)),
            ("always_valid", always_valid_baseline(entries)),
            ("random", random_baseline(entries, seed=42)),
        ]:
            result = evaluate(entries, preds, tool_name=name, split_name="dev_public")

            # Core fields must not be None
            assert result.detection_rate is not None, f"{name}: detection_rate is None"
            assert result.f1_hallucination is not None, f"{name}: f1_hallucination is None"
            assert result.tier_weighted_f1 is not None, f"{name}: tier_weighted_f1 is None"
            assert result.ece is not None, f"{name}: ece is None"
            # Coverage must be 1.0 — all entries have predictions
            assert result.coverage == pytest.approx(1.0), f"{name}: coverage != 1.0"
            # Entry counts must match
            assert result.num_entries == len(entries), f"{name}: num_entries mismatch"
