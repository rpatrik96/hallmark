"""A transient HTTP status must never become a detection.

Three implementations resolve a DOI and they disagreed. ``prescreening`` drew
the lines correctly; ``doi_only.check_doi`` and ``subtests.check_doi_resolves``
both treated *any* non-200 as proof the DOI was fabricated, so ``doi_only``
scored the citation HALLUCINATED at confidence 0.75 on a 202.

That is not a corner case. Sampling 150 VALID entries carrying a DOI, 56
returned HTTP 202 and one returned 403 — IEEE and ACM landing pages applying bot
mitigation *after* doi.org redirected successfully. The released ``doi_only``
false-positive rate was therefore substantially a measurement of a publisher's
bot policy on the day the run happened.

The rule all three now share: only doi.org answering 404 or 410 *itself* is
evidence of non-registration. A 404 from the redirect target means the DOI
resolved and the landing page is broken. Everything else is indeterminate.
"""

from __future__ import annotations

import httpx
import pytest

from hallmark.baselines import doi_only
from hallmark.baselines.prescreening import check_doi_resolves as prescreen_doi
from hallmark.dataset.schema import BlindEntry
from hallmark.evaluation import subtests

#: (status, has_redirect_history, verdict) where verdict is True = resolves,
#: False = does not exist, None = indeterminate.
CASES = [
    (200, False, True),
    (202, False, None),  # IEEE bot mitigation after a successful redirect
    (301, True, None),
    (400, False, None),
    (403, False, None),  # ACM bot block
    (404, False, False),  # doi.org itself: genuinely unregistered
    (404, True, None),  # redirect target, not doi.org: DOI is registered
    (410, False, False),
    (429, False, None),  # rate limited
    (500, False, None),
    (503, False, None),
]


def _response(status: int, history: bool) -> httpx.Response:
    request = httpx.Request("HEAD", "https://doi.org/10.1000/x")
    resp = httpx.Response(status_code=status, request=request)
    if history:
        prior = httpx.Response(
            status_code=302, request=httpx.Request("HEAD", "https://doi.org/10.1000/x")
        )
        resp.history = [prior]
    return resp


@pytest.mark.parametrize(("status", "history", "expected"), CASES)
def test_doi_only_matches_the_shared_rule(monkeypatch, status, history, expected):
    monkeypatch.setattr(httpx, "head", lambda *a, **k: _response(status, history))
    resolves, detail = doi_only.check_doi("10.1000/x")
    assert resolves is expected, f"HTTP {status} (history={history}) -> {detail}"


@pytest.mark.parametrize(("status", "history", "expected"), CASES)
def test_subtests_matches_the_shared_rule(monkeypatch, status, history, expected):
    monkeypatch.setattr(httpx, "head", lambda *a, **k: _response(status, history))
    result = subtests.check_doi_resolves("10.1000/x")
    assert result.passed is expected, f"HTTP {status} (history={history}): {result.detail}"


@pytest.mark.parametrize(("status", "history", "expected"), CASES)
def test_all_three_implementations_agree(monkeypatch, status, history, expected):
    """The three must not drift apart again."""
    monkeypatch.setattr(httpx, "head", lambda *a, **k: _response(status, history))

    from_doi_only = doi_only.check_doi("10.1000/x")[0]
    from_subtests = subtests.check_doi_resolves("10.1000/x").passed

    entry = BlindEntry(
        bibtex_key="k", bibtex_type="article", fields={"doi": "10.1000/x", "title": "T"}
    )
    pre = prescreen_doi(entry)
    from_prescreen = {"VALID": True, "HALLUCINATED": False, "UNKNOWN": None}[pre.label]

    assert from_doi_only is expected
    assert from_subtests is expected
    assert from_prescreen is expected, (
        f"prescreening disagrees on HTTP {status} (history={history}): {pre.reason}"
    )


def test_a_transient_status_never_yields_a_hallucinated_prediction(monkeypatch):
    """The end-to-end consequence: no flag from a 202."""
    monkeypatch.setattr(httpx, "head", lambda *a, **k: _response(202, False))
    entries = [
        BlindEntry(
            bibtex_key="k1",
            bibtex_type="article",
            fields={"doi": "10.1109/CVPR.2020.00001", "title": "T", "year": "2020"},
        )
    ]
    preds = doi_only.run_doi_only(entries, skip_prescreening=True)
    assert [p.label for p in preds] == ["VALID"]
    assert "indeterminate" in (preds[0].reason or "")


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadError("read failed"),
        httpx.ConnectTimeout("timed out"),
    ],
)
def test_transport_failures_are_indeterminate_not_crashes(monkeypatch, exc):
    """A transport failure must abstain, not abort the run.

    ``check_doi`` caught only TimeoutException and ConnectError, so a
    RemoteProtocolError -- an ordinary server disconnect -- propagated out of the
    baseline and killed a full-split evaluation partway through. It happened on
    the first real re-run after the 202 fix.
    """

    def _raise(*_a, **_k):
        raise exc

    monkeypatch.setattr(httpx, "head", _raise)
    resolves, detail = doi_only.check_doi("10.1000/x")
    assert resolves is None
    assert type(exc).__name__ in detail

    result = subtests.check_doi_resolves("10.1000/x")
    assert result.passed is None
