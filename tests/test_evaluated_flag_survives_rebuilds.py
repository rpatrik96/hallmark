"""``Prediction.evaluated`` must survive every place a prediction is rebuilt.

The flag says the tool never ran for an entry. Four code paths copy a prediction
field by field into a new ``Prediction`` and were written before the field
existed, so they reset it to the default ``True``: pre-screening's override and
confirm branches, the cascade's aggressive promotion, the aggressive eval-mode
remap, and the both-variants backfill. Through any of them a run that measured
nothing comes out looking measured. The worst case is the cascade's promotion,
which turns an all-fallback run into an all-HALLUCINATED run at detection rate
1.0 with every entry marked evaluated.
"""

from __future__ import annotations

from hallmark.baselines.cascade import _aggressive_fallback
from hallmark.baselines.common import fallback_predictions, run_with_prescreening
from hallmark.baselines.prescreening import PreScreenResult, merge_with_predictions
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, Prediction
from hallmark.evaluation.metrics import _make_aggressive_predictions, run_evaluated_nothing


def _blind(key: str) -> BlindEntry:
    return BlindEntry(bibtex_key=key, bibtex_type="article", fields={"title": key, "year": "2020"})


def _hit(key: str, confidence: float = 0.95) -> list[PreScreenResult]:
    return [PreScreenResult("HALLUCINATED", confidence, "Fake DOI prefix", "doi_check")]


def _unevaluated(key: str, label: str = "VALID", confidence: float = 0.5) -> Prediction:
    return Prediction(
        bibtex_key=key, label=label, confidence=confidence, reason="backfill", evaluated=False
    )


class TestPrescreeningMergeKeepsTheFlag:
    def test_override_of_an_unevaluated_valid_stays_unevaluated(self):
        merged = merge_with_predictions([_blind("a")], [_unevaluated("a")], {"a": _hit("a")})
        assert merged[0].label == "HALLUCINATED"
        assert merged[0].evaluated is False

    def test_confirm_with_stronger_prescreening_stays_unevaluated(self):
        tool = _unevaluated("a", label="HALLUCINATED", confidence=0.6)
        merged = merge_with_predictions([_blind("a")], [tool], {"a": _hit("a", 0.95)})
        assert merged[0].evaluated is False

    def test_confirm_with_stronger_tool_stays_unevaluated(self):
        tool = _unevaluated("a", label="HALLUCINATED", confidence=0.99)
        merged = merge_with_predictions([_blind("a")], [tool], {"a": _hit("a", 0.6)})
        assert merged[0].evaluated is False

    def test_an_entry_the_tool_never_returned_is_unevaluated(self):
        with_hit = merge_with_predictions([_blind("a")], [], {"a": _hit("a")})
        without = merge_with_predictions([_blind("b")], [], {})
        assert with_hit[0].evaluated is False
        assert without[0].evaluated is False

    def test_a_real_prediction_stays_evaluated(self):
        real = Prediction(bibtex_key="a", label="VALID", confidence=0.9, reason="verified")
        merged = merge_with_predictions([_blind("a")], [real], {"a": _hit("a")})
        assert merged[0].evaluated is True


def test_a_null_run_with_prescreening_on_is_still_a_null_run():
    """The default HaRC path: every batch times out, pre-screening flags some.

    Pre-screening is a benchmark-side check, not the tool. Its detections are
    recorded in ``source``; they must not make the tool look like it ran.
    """
    entries = [_blind("a"), _blind("b"), _blind("c")]
    hits = {"a": _hit("a")}
    import hallmark.baselines.prescreening as ps

    original = ps.prescreen_entries
    ps.prescreen_entries = lambda es, reference_year=None: hits
    try:
        preds = run_with_prescreening(entries, lambda es: [], skip_prescreening=False)
    finally:
        ps.prescreen_entries = original
    by_key = {p.bibtex_key: p for p in preds}
    assert by_key["a"].label == "HALLUCINATED" and by_key["a"].source == "prescreening_override"
    assert all(p.evaluated is False for p in preds)
    assert run_evaluated_nothing(preds) is True


def test_the_cascades_aggressive_promotion_keeps_the_flag():
    fallback = fallback_predictions([_blind("a")], reason="Fallback: tool unavailable")[0]
    promoted = _aggressive_fallback(fallback)
    assert promoted.label == "HALLUCINATED"
    assert promoted.evaluated is False, "a null run must not become DR 1.0 of evaluated detections"


class TestAggressiveEvalModeKeepsTheFlag:
    def _entries(self) -> list[BenchmarkEntry]:
        return [
            BenchmarkEntry(
                bibtex_key=k,
                bibtex_type="article",
                fields={"title": k},
                label="HALLUCINATED",
                hallucination_type="fabricated_doi",
                difficulty_tier=1,
            )
            for k in ("a", "b")
        ]

    def test_an_unevaluated_uncertain_stays_unevaluated_when_remapped(self):
        remapped = _make_aggressive_predictions(
            self._entries(),
            [_unevaluated("a", label="UNCERTAIN"), _unevaluated("b", label="UNCERTAIN")],
        )
        assert all(p.label == "HALLUCINATED" for p in remapped)
        assert all(p.evaluated is False for p in remapped)

    def test_a_missing_prediction_is_synthesised_as_unevaluated(self):
        remapped = _make_aggressive_predictions(self._entries(), [])
        assert all(p.evaluated is False for p in remapped)


def test_both_variants_backfill_is_marked_unevaluated():
    import hallmark.baselines.prescreening as ps
    from hallmark.baselines.common import run_baseline_both_variants

    original = ps.prescreen_entries
    ps.prescreen_entries = lambda es, reference_year=None: {}
    try:
        both = run_baseline_both_variants([_blind("a"), _blind("b")], lambda es: [])
    finally:
        ps.prescreen_entries = original
    assert all(p.evaluated is False for p in both["without_prescreening"])
    assert run_evaluated_nothing(both["without_prescreening"]) is True
