"""An arm that never ran must not enter a results table as zero detections.

When a baseline's tool is missing or its subprocess times out,
``fallback_predictions`` returns ``label="VALID"`` for every entry. Those flow
into the metrics as ordinary predictions, so a tool that never executed is
scored identically to a tool that checked everything and found nothing wrong.
Both produce detection rate 0.0 and false positive rate 0.0.

That is how four pre-screening ablations shipped as null runs. The recorded
cause was "the CLI was not on PATH"; the measured cause, once harcx was on PATH
and keyed, was a subprocess hanging on Google Scholar until its batch timed out.
Downstream the two were indistinguishable, so neither could be told from a
genuine all-valid result.

This is the same found/not-found conflation the source-lookup layer has, one
level up: silence has to be labelled as silence rather than as a finding.
"""

from __future__ import annotations

from hallmark.baselines.common import fallback_predictions
from hallmark.dataset.schema import BlindEntry, Prediction


def entries(n: int = 3) -> list[BlindEntry]:
    return [
        BlindEntry(bibtex_key=f"e{i}", bibtex_type="article", fields={"title": f"T{i}"})
        for i in range(n)
    ]


class TestPredictionsCarryWhetherTheyWereEvaluated:
    def test_a_real_prediction_is_evaluated_by_default(self):
        p = Prediction(bibtex_key="e0", label="VALID")
        assert p.evaluated is True, "existing callers must not have to change"

    def test_fallback_predictions_are_marked_unevaluated(self):
        preds = fallback_predictions(entries(3), reason="Fallback: tool unavailable")
        assert preds != []
        assert all(p.evaluated is False for p in preds)

    def test_fallback_still_carries_a_label_for_schema_compatibility(self):
        # The label stays VALID so nothing downstream breaks on a null; the
        # `evaluated` flag is what tells a consumer not to score it.
        preds = fallback_predictions(entries(2))
        assert all(p.label == "VALID" for p in preds)
        assert all(p.evaluated is False for p in preds)


class TestARunThatCheckedNothingIsDetectable:
    def test_an_all_fallback_run_reports_zero_evaluated(self):
        from hallmark.evaluation.metrics import evaluated_count

        preds = fallback_predictions(entries(5))
        assert evaluated_count(preds) == 0

    def test_a_real_run_reports_its_count(self):
        preds = [Prediction(bibtex_key=f"e{i}", label="VALID") for i in range(4)]
        from hallmark.evaluation.metrics import evaluated_count

        assert evaluated_count(preds) == 4

    def test_a_partial_run_counts_only_what_ran(self):
        from hallmark.evaluation.metrics import evaluated_count

        preds = [
            *fallback_predictions(entries(3)),
            Prediction(bibtex_key="real", label="HALLUCINATED", confidence=0.9),
        ]
        assert evaluated_count(preds) == 1

    def test_the_helper_says_when_a_run_evaluated_nothing(self):
        from hallmark.evaluation.metrics import run_evaluated_nothing

        assert run_evaluated_nothing(fallback_predictions(entries(3))) is True
        assert run_evaluated_nothing([Prediction(bibtex_key="e", label="VALID")]) is False

    def test_an_empty_prediction_set_is_not_called_a_null_run(self):
        # Nothing to evaluate is a different situation from "ran and checked
        # nothing", and conflating them would reintroduce the bug.
        from hallmark.evaluation.metrics import run_evaluated_nothing

        assert run_evaluated_nothing([]) is False


class TestPartialResultsSurvive:
    """A completed batch beside a failed one is real data and must be kept."""

    def test_partial_runs_are_not_discarded(self):
        from hallmark.evaluation.metrics import run_evaluated_nothing

        preds = [
            *fallback_predictions(entries(10)),
            Prediction(bibtex_key="r1", label="HALLUCINATED", confidence=0.9),
            Prediction(bibtex_key="r2", label="VALID", confidence=0.8),
        ]
        assert run_evaluated_nothing(preds) is False
        assert len(preds) == 12


def _base() -> dict:
    """Minimal required fields for an EvaluationResult."""
    return dict(
        tool_name="t",
        split_name="s",
        num_entries=10,
        num_hallucinated=5,
        num_valid=5,
        detection_rate=0.0,
        false_positive_rate=0.0,
        f1_hallucination=0.0,
        tier_weighted_f1=0.0,
    )


class TestRunLevelTriState:
    """``num_evaluated`` distinguishes three states; the entry flag only two.

    ``Prediction.evaluated`` defaults True, so a prediction deserialised from a
    file written before this change reads as evaluated. That is a real limit and
    it is why historical nulls must be excluded by name rather than detected.

    At run level the distinction survives, because ``num_evaluated`` defaults to
    None rather than to a number: None means the run predates the field, 0 means
    the run ran and evaluated nothing. A reproducer reading a results JSON can
    tell those apart, which is the level where it matters.
    """

    def test_a_result_predating_the_field_reads_as_none(self):
        from hallmark.dataset.schema import EvaluationResult

        assert EvaluationResult(**_base()).num_evaluated is None

    def test_a_null_run_reads_as_zero_not_none(self):
        from hallmark.dataset.schema import EvaluationResult

        assert EvaluationResult(**_base(), num_evaluated=0).num_evaluated == 0

    def test_the_entry_flag_cannot_make_that_distinction(self):
        from hallmark.dataset.schema import Prediction

        p = Prediction(bibtex_key="k", label="VALID")
        historical = {k: v for k, v in p.to_dict().items() if k != "evaluated"}
        assert Prediction(**historical).evaluated is True, (
            "documented limitation: absence of the key reads as evaluated, so "
            "historical results must be excluded by name, never detected"
        )
