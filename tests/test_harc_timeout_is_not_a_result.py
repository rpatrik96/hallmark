"""A harcx run that checked nothing must not look like a clean result.

Adapted from the fix/metric-correctness version, which raised. The assertions
are the same behaviour; only the mechanism changed, because refusing the run
also threw away partial results.

``_run_harcx_batch`` returns ``({}, set(), True)`` on ``TimeoutExpired`` -- no
flags AND an empty ``checked`` set. Unchecked entries are backfilled downstream,
so a total hang emerges as an all-VALID prediction set scoring DR 0.0 /
FPR 0.0 with zero API calls, which is indistinguishable from a tool that ran
fine and flagged nothing. Four such files shipped as pre-screening ablations.

The cause is not only a missing binary, which is what the null files were first
attributed to. harcx queries Google Scholar through ``scholarly``; Scholar blocks
automated access aggressively and the library retries rather than failing, so a
single entry can exceed 150 seconds. At ``batch_size`` 20 every batch then
exceeds ``batch_timeout``, every batch contributes an empty ``checked`` set, and
the run reaches zero having made real HTTP requests during pre-screening -- which
is exactly why it reads as a clean measurement.

Same conflation as an API error written into a result file as a verdict, one
level up: an infrastructure event must not be able to wear the shape of a
measurement.
"""

from __future__ import annotations

import subprocess

from hallmark.baselines import harc
from hallmark.dataset.schema import BlindEntry


def _entries(n: int) -> list[BlindEntry]:
    return [
        BlindEntry(
            bibtex_key=f"k{i}",
            bibtex_type="article",
            fields={"title": f"Paper {i}", "author": "A. Author", "year": "2024"},
        )
        for i in range(n)
    ]


def test_a_run_that_checked_nothing_is_marked_unevaluated(monkeypatch):
    """Every batch timing out is a failed run, not a run with no findings.

    Originally this asserted a RuntimeError. Refusing outright also discarded
    partial runs, so the run now returns predictions carrying evaluated=False
    and the metrics layer declines to score them.
    """
    monkeypatch.setattr(harc, "_run_harcx_batch", lambda *a, **k: ({}, set(), True))
    preds = harc._run_harc_batches(
        _entries(40),
        harcx_bin="/fake/harcx",
        author_threshold=0.6,
        check_urls=False,
        api_key=None,
        batch_size=20,
        batch_timeout=1.0,
        total_timeout=60.0,
    )
    # _run_harc_batches returns only the entries harcx actually checked, so a
    # total hang gives an empty list here. The backfill -- and the flag that
    # marks it -- happens one level up, in run_with_prescreening.
    assert preds == []


def test_the_failure_names_the_mechanism(monkeypatch, caplog):
    """A bare failure would send the next person back to the same dead end."""
    import logging

    monkeypatch.setattr(harc, "_run_harcx_batch", lambda *a, **k: ({}, set(), True))
    with caplog.at_level(logging.ERROR):
        harc._run_harc_batches(
            _entries(40),
            harcx_bin="/fake/harcx",
            author_threshold=0.6,
            check_urls=False,
            api_key=None,
            batch_size=20,
            batch_timeout=1.0,
            total_timeout=60.0,
        )
    message = caplog.text
    assert "checked 0 of" in message
    assert "batch-size" in message or "batch_timeout" in message, "and what to try"


def test_a_partial_run_still_returns(monkeypatch):
    """Partial results are real results. Only a total loss is refused."""
    calls = {"n": 0}

    def _batch(_bin, batch, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {}, {e.bibtex_key for e in batch}, False
        return {}, set(), True

    monkeypatch.setattr(harc, "_run_harcx_batch", _batch)
    preds = harc._run_harc_batches(
        _entries(40),
        harcx_bin="/fake/harcx",
        author_threshold=0.6,
        check_urls=False,
        api_key=None,
        batch_size=20,
        batch_timeout=1.0,
        total_timeout=60.0,
    )
    assert len(preds) == 20, "the completed batch must survive the failed one"
    from hallmark.evaluation.metrics import run_evaluated_nothing

    assert run_evaluated_nothing(preds) is False, "a partial run is still a run"


def test_a_clean_empty_run_is_not_refused(monkeypatch):
    """A tool that checked everything and flagged nothing is a valid result."""
    monkeypatch.setattr(
        harc,
        "_run_harcx_batch",
        lambda _bin, batch, *a, **k: ({}, {e.bibtex_key for e in batch}, False),
    )
    preds = harc._run_harc_batches(
        _entries(20),
        harcx_bin="/fake/harcx",
        author_threshold=0.6,
        check_urls=False,
        api_key=None,
        batch_size=20,
        batch_timeout=1.0,
        total_timeout=60.0,
    )
    assert len(preds) == 20
    assert all(p.label == "VALID" for p in preds)
    assert all(p.evaluated for p in preds), (
        "a tool that checked everything and flagged nothing is a real result, "
        "and must not be confused with one that never ran"
    )


def test_no_entries_is_not_an_error(monkeypatch):
    monkeypatch.setattr(harc, "_run_harcx_batch", lambda *a, **k: ({}, set(), True))
    assert (
        harc._run_harc_batches(
            [],
            harcx_bin="/fake/harcx",
            author_threshold=0.6,
            check_urls=False,
            api_key=None,
            batch_size=20,
            batch_timeout=1.0,
            total_timeout=60.0,
        )
        == []
    )


def test_subprocess_timeout_yields_the_empty_shape(monkeypatch):
    """The real code path, with subprocess.run forced to time out."""

    def _raise(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="harcx", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _raise)
    flagged, checked, timed_out = harc._run_harcx_batch(
        "/fake/harcx", _entries(3), 0.6, False, None, 1.0
    )
    assert timed_out is True
    assert flagged == {} and checked == set()


def test_the_backfill_one_level_up_carries_the_flag(monkeypatch):
    """Where the all-VALID set is actually manufactured, it is marked.

    This is the level that matters: `_run_harc_batches` returning [] is honest,
    but the caller turns that into a full set of VALID predictions, and without
    the flag that set is what reaches the metrics looking like a clean run.
    """
    from hallmark.baselines.common import run_with_prescreening
    from hallmark.evaluation.metrics import run_evaluated_nothing

    entries = _entries(20)
    preds = run_with_prescreening(
        entries,
        lambda es: [],  # the tool checked nothing
        skip_prescreening=True,
        backfill_reason="HaRC: entry not checked (timeout or missing)",
    )
    assert len(preds) == 20, "every entry still gets a prediction"
    assert all(p.label == "VALID" for p in preds)
    assert all(p.evaluated is False for p in preds)
    assert run_evaluated_nothing(preds) is True
