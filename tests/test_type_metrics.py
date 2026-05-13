"""Tests for hallucination_type_accuracy, type_confusion_matrix, and cascade_breakdown."""

import math

import pytest

from hallmark.dataset.schema import BenchmarkEntry, HallucinationType, Prediction
from hallmark.evaluation.metrics import (
    cascade_breakdown,
    hallucination_type_accuracy,
    type_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_HT_VALUES = [ht.value for ht in HallucinationType]


def _entry(key: str, label: str, tier: int = 1, h_type: str | None = None) -> BenchmarkEntry:
    kwargs: dict = {
        "bibtex_key": key,
        "bibtex_type": "article",
        "fields": {"title": f"Paper {key}", "author": "Author", "year": "2024"},
        "label": label,
        "explanation": "test",
    }
    if label == "HALLUCINATED":
        kwargs["hallucination_type"] = h_type or "fabricated_doi"
        kwargs["difficulty_tier"] = tier
    return BenchmarkEntry(**kwargs)


def _pred(
    key: str,
    label: str,
    predicted_type: str | None = None,
    cascade_stage: str | None = None,
    confidence: float = 0.9,
) -> Prediction:
    return Prediction(
        bibtex_key=key,
        label=label,
        confidence=confidence,
        predicted_hallucination_type=predicted_type,
        cascade_stage=cascade_stage,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# hallucination_type_accuracy
# ---------------------------------------------------------------------------


class TestHallucinationTypeAccuracy:
    def test_perfect_accuracy(self):
        """GT type matches predicted_type → accuracy 1.0."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=2, h_type="chimeric_title"),
        ]
        preds = [
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),
            _pred("h2", "HALLUCINATED", predicted_type="chimeric_title"),
        ]
        result = hallucination_type_accuracy(entries, preds)
        assert result["overall"] == pytest.approx(1.0)
        assert result["num_evaluated"] == 2
        assert result["num_correct"] == 2

    def test_zero_accuracy(self):
        """All wrong → accuracy 0.0."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=2, h_type="chimeric_title"),
        ]
        preds = [
            _pred("h1", "HALLUCINATED", predicted_type="future_date"),
            _pred("h2", "HALLUCINATED", predicted_type="wrong_venue"),
        ]
        result = hallucination_type_accuracy(entries, preds)
        assert result["overall"] == pytest.approx(0.0)
        assert result["num_evaluated"] == 2
        assert result["num_correct"] == 0

    def test_skips_valid_entries(self):
        """Valid entries don't bring accuracy down."""
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("v2", "VALID"),
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),
        ]
        result = hallucination_type_accuracy(entries, preds)
        assert result["overall"] == pytest.approx(1.0)
        assert result["num_evaluated"] == 1

    def test_skips_missing_predicted_type(self):
        """predicted_hallucination_type=None excluded from count."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=1, h_type="future_date"),
        ]
        preds = [
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),  # correct
            _pred("h2", "HALLUCINATED", predicted_type=None),  # skipped
        ]
        result = hallucination_type_accuracy(entries, preds)
        # Only h1 qualifies (h2 has None type)
        assert result["num_evaluated"] == 1
        assert result["num_correct"] == 1
        assert result["overall"] == pytest.approx(1.0)

    def test_skips_false_negatives(self):
        """Entries predicted VALID (false negatives) are excluded."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", tier=1, h_type="future_date"),
        ]
        preds = [
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),
            _pred("h2", "VALID"),  # false negative — excluded
        ]
        result = hallucination_type_accuracy(entries, preds)
        assert result["num_evaluated"] == 1
        assert result["overall"] == pytest.approx(1.0)

    def test_nan_for_empty_partition(self):
        """Partition with zero qualifying entries → NaN."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),
        ]
        preds = [
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),
        ]
        result = hallucination_type_accuracy(entries, preds)
        # tier_2 and tier_3 have no entries → NaN
        assert math.isnan(result["tier_2"])
        assert math.isnan(result["tier_3"])
        # tier_1 has one correct
        assert result["tier_1"] == pytest.approx(1.0)

    def test_stress_test_partition(self):
        """Stress-test types tracked separately from main types."""
        entries = [
            _entry("h1", "HALLUCINATED", tier=1, h_type="fabricated_doi"),  # main
            _entry("h2", "HALLUCINATED", tier=2, h_type="merged_citation"),  # stress
        ]
        preds = [
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),
            _pred("h2", "HALLUCINATED", predicted_type="merged_citation"),
        ]
        result = hallucination_type_accuracy(entries, preds)
        assert result["main_types_only"] == pytest.approx(1.0)
        assert result["stress_test_only"] == pytest.approx(1.0)
        assert result["overall"] == pytest.approx(1.0)

    def test_accepts_dict_predictions(self):
        """Accepts dict[str, Prediction] as well as list."""
        entries = [_entry("h1", "HALLUCINATED", h_type="fabricated_doi")]
        pred = _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi")
        result_list = hallucination_type_accuracy(entries, [pred])
        result_dict = hallucination_type_accuracy(entries, {"h1": pred})
        assert result_list["overall"] == result_dict["overall"]


# ---------------------------------------------------------------------------
# type_confusion_matrix
# ---------------------------------------------------------------------------


class TestTypeConfusionMatrix:
    def test_shape(self):
        """15 rows x 17 cols (14 ht + valid) x (14 ht + VALID + UNCERTAIN + HALLUCINATED_unknown)."""
        entries = []
        preds = []
        result = type_confusion_matrix(entries, preds)
        all_ht = [ht.value for ht in HallucinationType]
        expected_rows = set(all_ht) | {"valid"}
        expected_cols = set(all_ht) | {"VALID", "UNCERTAIN", "HALLUCINATED_unknown"}
        assert set(result.keys()) == expected_rows
        for row_dict in result.values():
            assert set(row_dict.keys()) == expected_cols

    def test_all_zero_on_empty_input(self):
        result = type_confusion_matrix([], [])
        for row_dict in result.values():
            assert all(v == 0 for v in row_dict.values())

    def test_diagonal_perfect_predictions(self):
        """Perfect type predictions concentrate counts on the diagonal."""
        ht = "fabricated_doi"
        entries = [_entry(f"h{i}", "HALLUCINATED", h_type=ht) for i in range(5)]
        preds = [_pred(f"h{i}", "HALLUCINATED", predicted_type=ht) for i in range(5)]
        result = type_confusion_matrix(entries, preds)
        # All 5 should land in [fabricated_doi][fabricated_doi]
        assert result[ht][ht] == 5
        # No other cell in that row should be nonzero
        for col, v in result[ht].items():
            if col != ht:
                assert v == 0

    def test_valid_entry_missing_pred_counts_as_valid(self):
        """Missing prediction for VALID entry → VALID column."""
        entries = [_entry("v1", "VALID")]
        result = type_confusion_matrix(entries, [])
        assert result["valid"]["VALID"] == 1

    def test_uncertain_prediction_lands_in_uncertain_col(self):
        entries = [_entry("h1", "HALLUCINATED", h_type="fabricated_doi")]
        preds = [_pred("h1", "UNCERTAIN")]
        result = type_confusion_matrix(entries, preds)
        assert result["fabricated_doi"]["UNCERTAIN"] == 1

    def test_hallucinated_no_type_lands_in_unknown_col(self):
        entries = [_entry("h1", "HALLUCINATED", h_type="future_date")]
        preds = [_pred("h1", "HALLUCINATED", predicted_type=None)]
        result = type_confusion_matrix(entries, preds)
        assert result["future_date"]["HALLUCINATED_unknown"] == 1

    def test_row_totals_equal_entries(self):
        """Sum of each row == number of entries with that GT type."""
        entries = [
            _entry("v1", "VALID"),
            _entry("v2", "VALID"),
            _entry("h1", "HALLUCINATED", h_type="fabricated_doi"),
            _entry("h2", "HALLUCINATED", h_type="fabricated_doi"),
            _entry("h3", "HALLUCINATED", h_type="chimeric_title"),
        ]
        preds = [
            _pred("v1", "VALID"),
            _pred("v2", "HALLUCINATED"),
            _pred("h1", "HALLUCINATED", predicted_type="fabricated_doi"),
            _pred("h2", "HALLUCINATED", predicted_type="future_date"),
            _pred("h3", "VALID"),
        ]
        result = type_confusion_matrix(entries, preds)
        assert sum(result["valid"].values()) == 2
        assert sum(result["fabricated_doi"].values()) == 2
        assert sum(result["chimeric_title"].values()) == 1


# ---------------------------------------------------------------------------
# cascade_breakdown
# ---------------------------------------------------------------------------


class TestCascadeBreakdown:
    def test_counts_by_stage(self):
        preds = [
            _pred("a", "HALLUCINATED", cascade_stage="stage1_db"),
            _pred("b", "VALID", cascade_stage="stage1_db"),
            _pred("c", "HALLUCINATED", cascade_stage="stage2_diagnosis"),
            _pred("d", "UNCERTAIN", cascade_stage="prescreening"),
            _pred("e", "VALID"),  # no stage
        ]
        result = cascade_breakdown(preds)
        assert result["stage1_db"]["count"] == 2
        assert result["stage2_diagnosis"]["count"] == 1
        assert result["prescreening"]["count"] == 1
        assert result["none"]["count"] == 1

    def test_fractions_sum_to_one(self):
        preds = [
            _pred("a", "VALID", cascade_stage="stage1_db"),
            _pred("b", "HALLUCINATED", cascade_stage="stage2_diagnosis"),
            _pred("c", "UNCERTAIN"),
        ]
        result = cascade_breakdown(preds)
        total_fraction = sum(v["fraction"] for v in result.values())
        assert total_fraction == pytest.approx(1.0)

    def test_label_tallies(self):
        preds = [
            _pred("a", "HALLUCINATED", cascade_stage="stage1_db"),
            _pred("b", "HALLUCINATED", cascade_stage="stage1_db"),
            _pred("c", "VALID", cascade_stage="stage1_db"),
        ]
        result = cascade_breakdown(preds)
        stage = result["stage1_db"]
        assert stage["HALL"] == 2
        assert stage["VALID"] == 1
        assert stage["UNCERTAIN"] == 0

    def test_empty_predictions(self):
        result = cascade_breakdown([])
        for stage_data in result.values():
            assert stage_data["count"] == 0
            assert stage_data["fraction"] == 0.0

    def test_all_stages_present_in_output(self):
        result = cascade_breakdown([])
        assert set(result.keys()) == {"stage1_db", "stage2_diagnosis", "prescreening", "none"}
