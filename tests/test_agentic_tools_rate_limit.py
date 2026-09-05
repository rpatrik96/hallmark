"""Tests for the per-service rate limiter in `_agentic_tools`.

Covers the failure mode measured in a 2026-09 Stage-2 run: arXiv answered 21 requests
and rejected 356 while CrossRef and OpenAlex stayed clean, because the source publishes
its limit per caller and Stage 2 called it from 6-8 worker threads that only ever
backed off *after* being refused.
"""

from __future__ import annotations

import threading
import time

import pytest

from hallmark.baselines import _agentic_tools
from hallmark.baselines._agentic_tools import (
    _DEFAULT_SERVICE_RATES,
    _configured_rate,
    _limiter_for,
    _pace,
    _rate_limiters,
    _ServiceRateLimiter,
)


@pytest.fixture
def fresh_limiters():
    """Give each test an empty limiter registry, and leave one behind."""
    _rate_limiters.clear()
    yield
    _rate_limiters.clear()


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace monotonic time and sleep so pacing is checked without waiting for it."""

    class _Clock:
        def __init__(self) -> None:
            self.now = 1000.0
            self.slept: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(_agentic_tools.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(_agentic_tools.time, "sleep", clock.sleep)
    return clock


class TestAdmissionRate:
    def test_first_request_is_not_delayed(self, fake_clock):
        limiter = _ServiceRateLimiter(20.0)
        assert limiter.acquire() == 0.0
        assert fake_clock.slept == []

    def test_admissions_are_one_interval_apart(self, fake_clock):
        """20/min means a request every 3s, which is what arXiv's ceiling buys."""
        limiter = _ServiceRateLimiter(20.0)
        waits = [limiter.acquire() for _ in range(4)]
        assert waits == [0.0, 3.0, 3.0, 3.0]

    def test_time_already_spent_counts_against_the_interval(self, fake_clock):
        """A slow request pays part of its own pacing; only the remainder is waited out."""
        limiter = _ServiceRateLimiter(20.0)
        limiter.acquire()
        fake_clock.now += 2.0
        assert limiter.acquire() == pytest.approx(1.0)

    def test_an_idle_gap_does_not_accrue_a_burst(self, fake_clock):
        """Waiting a minute does not buy twenty requests at once — the next one is free,
        the one after it still waits a full interval."""
        limiter = _ServiceRateLimiter(20.0)
        limiter.acquire()
        fake_clock.now += 60.0
        assert limiter.acquire() == 0.0
        assert limiter.acquire() == 3.0

    @pytest.mark.parametrize("rate", [0.0, -5.0])
    def test_a_non_positive_rate_means_unpaced(self, fake_clock, rate):
        limiter = _ServiceRateLimiter(rate)
        assert [limiter.acquire() for _ in range(5)] == [0.0] * 5

    def test_set_rate_keeps_the_pacing_already_accrued(self, fake_clock):
        """Re-reading the env mid-run must not hand out a free request."""
        limiter = _ServiceRateLimiter(20.0)
        limiter.acquire()
        limiter.set_rate(60.0)
        assert limiter.acquire() == 3.0  # the interval booked under the old rate
        assert limiter.acquire() == 1.0  # then the new one


class TestBudgetIsSharedAcrossThreads:
    def test_two_threads_on_one_service_share_one_budget(self, fresh_limiters, monkeypatch):
        """The defect: 6-8 workers each pacing themselves is 6-8x the caller's budget.

        Runs on the real clock, because the threads would race on a fake one. 600/min
        keeps it to half a second of wall time while still discriminating: six
        admissions through one budget take five intervals, where two independent
        budgets would take two and a half.
        """
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", "600")
        interval = 0.1
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            for _ in range(3):
                _pace("arxiv")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        started = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - started

        # Five intervals for six admissions, minus a margin for clock granularity.
        assert elapsed >= 5 * interval * 0.95
        assert elapsed < 5 * interval + 2.0

    def test_the_same_service_resolves_to_the_same_limiter(self, fresh_limiters):
        assert _limiter_for("arxiv") is _limiter_for("arxiv")

    def test_different_services_have_separate_budgets(self, fresh_limiters, fake_clock):
        """arXiv being paced must not slow CrossRef, which was clean in the same run."""
        for _ in range(3):
            _pace("arxiv")
        assert _pace("crossref") == 0.0


class TestEnvOverride:
    def test_default_applies_when_unset(self, fresh_limiters, monkeypatch):
        monkeypatch.delenv("HALLMARK_ARXIV_RATE", raising=False)
        assert _configured_rate("arxiv") == _DEFAULT_SERVICE_RATES["arxiv"] == 20.0

    def test_env_override_is_honoured(self, fresh_limiters, monkeypatch):
        """A run sharded across two processes halves each process's share."""
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", "10")
        assert _configured_rate("arxiv") == 10.0
        assert _limiter_for("arxiv").rate_per_minute == 10.0

    def test_override_reaches_the_pacing(self, fresh_limiters, fake_clock, monkeypatch):
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", "10")
        _pace("arxiv")
        assert _pace("arxiv") == 6.0

    def test_every_service_has_its_own_knob(self, fresh_limiters, monkeypatch):
        for service, default in _DEFAULT_SERVICE_RATES.items():
            monkeypatch.setenv(f"HALLMARK_{service.upper()}_RATE", "7")
            assert _configured_rate(service) == 7.0
            monkeypatch.delenv(f"HALLMARK_{service.upper()}_RATE")
            assert _configured_rate(service) == default

    def test_a_change_mid_run_updates_the_shared_limiter(self, fresh_limiters, monkeypatch):
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", "10")
        limiter = _limiter_for("arxiv")
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", "5")
        assert _limiter_for("arxiv") is limiter
        assert limiter.rate_per_minute == 5.0

    def test_an_unknown_service_is_unpaced(self, fresh_limiters, fake_clock):
        assert _configured_rate("nowhere") == 0.0
        assert [_pace("nowhere") for _ in range(5)] == [0.0] * 5


class TestBadEnvValueDoesNotCrash:
    @pytest.mark.parametrize(
        "value", ["", "   ", "twenty", "20/min", "nan", "inf", "-inf", "0", "-3"]
    )
    def test_unusable_values_fall_back_to_the_default(self, fresh_limiters, monkeypatch, value):
        """A mistyped knob must cost the run its override, never the run."""
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", value)
        assert _configured_rate("arxiv") == 20.0

    def test_a_bad_value_still_paces(self, fresh_limiters, fake_clock, monkeypatch):
        monkeypatch.setenv("HALLMARK_ARXIV_RATE", "twenty")
        _pace("arxiv")
        assert _pace("arxiv") == 3.0

    def test_a_fractional_rate_is_accepted(self, fresh_limiters, monkeypatch):
        """Sharding four ways over a 10/min source lands on a non-integer share."""
        monkeypatch.setenv("HALLMARK_SEMANTICSCHOLAR_RATE", "2.5")
        assert _configured_rate("semanticscholar") == 2.5


class _StubResponse:
    """Minimal stand-in for an httpx.Response holding an empty result set."""

    def __init__(self, payload: dict | None = None, text: str = "") -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


_EMPTY_ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

# One canned empty response per source, keyed by the tool that fetches it.
_TOOL_CASES = [
    ("resolve_doi", "crossref", {"message": {}}, "", {"doi": "10.1234/abc"}),
    ("search_crossref", "crossref", {"message": {"items": []}}, "", {"query": "q"}),
    ("search_openalex", "openalex", {"results": []}, "", {"query": "q"}),
    ("search_arxiv", "arxiv", {}, _EMPTY_ATOM, {"query": "q"}),
    ("search_semantic_scholar", "semanticscholar", {"data": []}, "", {"query": "q"}),
]


class TestEveryRequestIsPaced:
    """Each source-calling tool must pass through the limiter before it fetches.

    This is the regression guard: the defect was one tool reaching a source with no
    pacing at all, and a tool added later would repeat it silently.
    """

    @pytest.mark.parametrize(
        ("tool_name", "service", "payload", "text", "kwargs"),
        _TOOL_CASES,
        ids=[case[0] for case in _TOOL_CASES],
    )
    def test_tool_paces_its_service(
        self, fresh_limiters, monkeypatch, tool_name, service, payload, text, kwargs
    ):
        import httpx

        paced: list[str] = []
        monkeypatch.setattr(_agentic_tools, "_pace", lambda name: paced.append(name) or 0.0)
        monkeypatch.setattr(
            httpx, "get", lambda *a, **kw: _StubResponse(payload, text), raising=True
        )
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        monkeypatch.delenv("S2_API_KEY", raising=False)

        getattr(_agentic_tools, tool_name)(**kwargs)
        assert paced == [service]

    def test_every_source_tool_is_covered(self):
        """Fails when a tool is added to the registry without a pacing case here."""
        covered = {case[0] for case in _TOOL_CASES} | {"verify_with_bibtex_updater"}
        assert set(_agentic_tools.TOOL_REGISTRY) == covered
