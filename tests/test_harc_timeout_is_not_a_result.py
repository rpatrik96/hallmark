"""A harcx run that checked nothing must not look like a clean result.

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

import pytest

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


def test_a_run_that_checked_nothing_raises(monkeypatch):
    """Every batch timing out is a failed run, not a run with no findings."""
    monkeypatch.setattr(harc, "_run_harcx_batch", lambda *a, **k: ({}, set(), True))
    with pytest.raises(RuntimeError, match="checked 0 of"):
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


def test_the_error_names_the_mechanism(monkeypatch):
    """A bare 'it failed' would send the next person back to the same dead end."""
    monkeypatch.setattr(harc, "_run_harcx_batch", lambda *a, **k: ({}, set(), True))
    with pytest.raises(RuntimeError) as excinfo:
        harc._run_harc_batches(
            _entries(20),
            harcx_bin="/fake/harcx",
            author_threshold=0.6,
            check_urls=False,
            api_key=None,
            batch_size=20,
            batch_timeout=1.0,
            total_timeout=60.0,
        )
    message = str(excinfo.value)
    assert "DR 0.0" in message, "the error must say what the silent failure looked like"
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
