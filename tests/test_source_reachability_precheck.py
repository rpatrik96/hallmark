"""A source that is not answering must stop a run before it starts, not after.

The prescreening ablation was discarded three times in one day. Twice the cause
was a source outage the run could only discover at the end: ``bibtex-check``
exits 5, the wrapper raises ``SourceOutageError``, and ninety minutes are gone.
The last attempt reported 244 of 1,119 entries with an incomplete lookup, 230 of
them DBLP.

These tests pin the classification, because the interesting cases are the ones
where a status code answers a question nobody asked -- the same shape as the
HTTP-202 defect in the DOI check, where a publisher's bot mitigation was read as
proof a reference did not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_source_reachability import PROBES, main, probe


def _response(status: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    return r


def test_throttled_is_reachable_not_down():
    """429 means the service answered and asked us to slow down.

    Calling it an outage would block every run against a busy source; pacing is
    the fix, and the ablation got *faster* once it was paced (2.4 s/entry
    against 5.7).
    """
    with patch("httpx.get", return_value=_response(429)) as get:
        result = probe("openalex", PROBES["openalex"], 5.0, None)
    assert result.ok is True
    assert result.status == 429
    assert "throttled" in result.detail
    assert get.call_count == 1, "a 429 is an answer; retrying it wastes the budget"


def test_server_error_is_down_after_a_retry():
    """The live DBLP failure mode: HTTP 500, twice."""
    with patch("httpx.get", return_value=_response(500)) as get:
        result = probe("dblp", PROBES["dblp"], 5.0, None)
    assert result.ok is False
    assert get.call_count == 2, "one flake must not fail the gate; a persistent fault must"


def test_a_single_flake_does_not_fail_the_gate():
    with patch("httpx.get", side_effect=[_response(500), _response(200)]):
        result = probe("dblp", PROBES["dblp"], 5.0, None)
    assert result.ok is True


def test_transport_failure_is_down():
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("no route")):
        result = probe("dblp", PROBES["dblp"], 5.0, None)
    assert result.ok is False
    assert result.status is None
    assert "ConnectError" in result.detail


def test_required_source_down_exits_nonzero(capsys):
    """The whole point: a run gated on this must not start."""
    with patch("httpx.get", return_value=_response(500)):
        code = main_with_args(["--require", "dblp", "--timeout", "0.1"])
    assert code == 1
    assert "dblp" in capsys.readouterr().err


def test_all_required_up_exits_zero():
    with patch("httpx.get", return_value=_response(200)):
        assert main_with_args(["--require", "dblp,openalex", "--timeout", "0.1"]) == 0


def test_unknown_source_never_reports_a_pass(capsys):
    """A typo in --require must not read as 'nothing was down'."""
    assert main_with_args(["--require", "dblpp"]) == 2
    assert "unknown source" in capsys.readouterr().err


def test_report_only_mode_never_fails():
    with patch("httpx.get", return_value=_response(500)):
        assert main_with_args(["--timeout", "0.1"]) == 0


def main_with_args(argv: list[str]) -> int:
    with (
        patch.object(sys, "argv", ["check_source_reachability.py", *argv]),
        patch("time.sleep"),
    ):
        return main()


def test_every_probe_asks_for_a_real_record():
    """Not a bare root.

    Several of these serve 200 from a CDN at the root while the query path
    behind it is down, which is exactly the status-answering-the-wrong-question
    failure this script exists to avoid.
    """
    for source, url in PROBES.items():
        assert "?" in url or url.count("/") > 3, f"{source} probes a bare root: {url}"
