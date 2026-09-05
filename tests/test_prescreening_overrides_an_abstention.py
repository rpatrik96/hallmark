"""Pre-screening must be able to override an abstention.

Since #49, bibtex-updater's abstentions parse to ``UNCERTAIN`` instead of a
committed ``VALID``. ``merge_with_predictions`` only ever overrode ``VALID``, so
a fake DOI or a future year that pre-screening had found was dropped whenever
the tool had abstained on that entry. On the released raw runs that is 25 true
positives on ``dev_public`` and 19 on ``test_public`` -- all ``unconfirmed``,
all gold HALLUCINATED -- that a fresh run under the new mapping loses. The
cascade compounds it: the status dict never says ``prescreening_override`` for
them, so they route to Stage 2 and spend LLM budget on a verdict the local
check had already reached.
"""

from __future__ import annotations

from hallmark.baselines import bibtexupdater
from hallmark.baselines.prescreening import PreScreenResult, merge_with_predictions
from hallmark.dataset.schema import BlindEntry, Prediction


def _entry(key: str = "k1") -> BlindEntry:
    return BlindEntry(bibtex_key=key, bibtex_type="article", fields={"title": "T", "year": "2031"})


def _abstention(key: str = "k1") -> Prediction:
    return Prediction(
        bibtex_key=key,
        label="UNCERTAIN",
        confidence=0.5,
        reason="Status: unconfirmed; Abstention: no source answered, so this is not a verdict",
    )


def _hit(key: str = "k1") -> dict[str, list[PreScreenResult]]:
    return {
        key: [
            PreScreenResult(
                label="HALLUCINATED",
                confidence=0.95,
                reason="Future year 2031",
                check_name="year_bounds",
            )
        ]
    }


def test_an_abstention_is_overridden_by_a_prescreening_detection():
    merged = merge_with_predictions([_entry()], [_abstention()], _hit())
    assert merged[0].label == "HALLUCINATED"
    assert merged[0].source == "prescreening_override"
    assert "Future year 2031" in merged[0].reason


def test_an_abstention_with_no_prescreening_signal_stays_an_abstention():
    no_signal = {"k1": [PreScreenResult("UNKNOWN", 0.0, "no signal", "doi_check")]}
    merged = merge_with_predictions([_entry()], [_abstention()], no_signal)
    assert merged[0].label == "UNCERTAIN"


def test_a_committed_valid_is_still_overridden():
    """The pre-existing behaviour, pinned so the fix does not narrow it."""
    valid = Prediction(bibtex_key="k1", label="VALID", confidence=0.9, reason="Status: verified")
    merged = merge_with_predictions([_entry()], [valid], _hit())
    assert merged[0].label == "HALLUCINATED"
    assert merged[0].source == "prescreening_override"


def test_the_status_dict_reports_the_override_so_the_cascade_credits_it(monkeypatch):
    """``run_bibtex_check_with_status`` must say ``prescreening_override``.

    ``cascade._stage1_predict`` honours that status as a Stage 1 verdict and
    routes ``unconfirmed`` to Stage 2; which one the entry gets is decided here.
    """
    monkeypatch.setattr(
        bibtexupdater, "_run_bibtex_check_subprocess", lambda entries, **kw: [_abstention()]
    )
    monkeypatch.setattr(
        "hallmark.baselines.prescreening.prescreen_entries",
        lambda entries, reference_year=None: _hit(),
    )
    predictions, status = bibtexupdater.run_bibtex_check_with_status([_entry()])
    assert predictions[0].label == "HALLUCINATED"
    assert status["k1"] == "prescreening_override"
