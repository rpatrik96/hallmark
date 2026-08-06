"""Tests for Retry-After parsing and throttle detection in `_agentic_tools`.

Covers the failure mode behind the 2026-07 cascade run, where 173 Semantic Scholar
and 149 OpenAlex lookups were lost to HTTP 429: the servers stated when to return,
and the client ignored them in favour of a fixed 1s/2s/4s schedule.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from hallmark.baselines import _agentic_tools
from hallmark.baselines._agentic_tools import (
    _openalex_exhausted,
    _openalex_key_candidates,
    _parse_retry_after,
    _raise_for_throttle,
    _retire_openalex_key,
    _seconds_until_utc_midnight,
)
from hallmark.baselines._cache import RateLimitedError


class _Resp:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers: dict[str, Any] = headers or {}


# Observed on the keyless pool during the 2026-08-04 GPT-5.4 dev run: bursts asked for
# 1-60s, exhausted quota asked for 27087-52660s (each the seconds left until midnight UTC).
_BURST_RETRY_AFTER = 60.0
_CAP_RETRY_AFTER = 27087.0


@pytest.fixture
def openalex_state(monkeypatch):
    """Give each test a clean retirement map and one configured key."""
    _openalex_exhausted.clear()
    monkeypatch.setenv("OPENALEX_API_KEY", "testkey1234")
    yield
    _openalex_exhausted.clear()


class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert _parse_retry_after("30") == 30.0

    def test_http_date(self):
        """RFC 9110 also permits an absolute date, which must resolve to a delay."""
        future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=42)
        parsed = _parse_retry_after(future.strftime("%a, %d %b %Y %H:%M:%S GMT"))
        assert parsed is not None
        assert 30 <= parsed <= 45

    def test_past_date_yields_none(self):
        """An already-elapsed date means no wait is owed."""
        past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=60)
        assert _parse_retry_after(past.strftime("%a, %d %b %Y %H:%M:%S GMT")) is None

    @pytest.mark.parametrize("value", [None, "", "   ", "soon", "not-a-date"])
    def test_unparseable_yields_none(self, value):
        """Callers fall back to exponential backoff rather than crashing."""
        assert _parse_retry_after(value) is None


class TestRaiseForThrottle:
    @pytest.mark.parametrize("code", [429, 503])
    def test_throttle_codes_raise_with_delay(self, code):
        with pytest.raises(RateLimitedError) as exc_info:
            _raise_for_throttle(_Resp(code, {"Retry-After": "30"}), "TestSource")
        assert exc_info.value.retry_after == 30.0

    def test_throttle_without_header_still_raises(self):
        """Still a rate limit, just with no instruction — retry_after is None."""
        with pytest.raises(RateLimitedError) as exc_info:
            _raise_for_throttle(_Resp(429), "TestSource")
        assert exc_info.value.retry_after is None

    @pytest.mark.parametrize("code", [200, 404, 500])
    def test_non_throttle_codes_pass_through(self, code):
        """Other statuses are left to the caller's own error handling."""
        assert _raise_for_throttle(_Resp(code), "TestSource") is None

    def test_none_response_passes_through(self):
        """A missing response is not a throttle; the caller reports it separately."""
        assert _raise_for_throttle(None, "TestSource") is None


class TestOpenAlexFailover:
    """Credential rotation for OpenAlex: keyless first, keys as paid fallback.

    The keyless mailto pool is free and generous, so it is spent before the metered
    key. Retirements carry a deadline rather than lasting the whole process, so a run
    that outlives a midnight-UTC reset picks the free pool back up.
    """

    def test_keyless_is_preferred_over_the_key(self, openalex_state):
        assert list(_openalex_key_candidates()) == [None, "testkey1234"]

    def test_burst_throttle_does_not_fail_over(self, openalex_state):
        """1-60s means "slow down", not "you are out" — the backoff layer owns it."""
        assert _retire_openalex_key(None, _BURST_RETRY_AFTER, 429) is False
        assert list(_openalex_key_candidates()) == [None, "testkey1234"]

    def test_missing_retry_after_does_not_fail_over(self, openalex_state):
        """With no stated deadline there is nothing to distinguish a cap from a burst."""
        assert _retire_openalex_key(None, None, 429) is False
        assert list(_openalex_key_candidates()) == [None, "testkey1234"]

    def test_daily_cap_retires_keyless_and_falls_over_to_the_key(self, openalex_state):
        assert _retire_openalex_key(None, _CAP_RETRY_AFTER, 429) is True
        assert list(_openalex_key_candidates()) == ["testkey1234"]

    def test_both_capped_leaves_no_candidates(self, openalex_state):
        _retire_openalex_key(None, _CAP_RETRY_AFTER, 429)
        _retire_openalex_key("testkey1234", _CAP_RETRY_AFTER, 429)
        assert list(_openalex_key_candidates()) == []

    @pytest.mark.parametrize("code", [402, 403])
    def test_credit_exhaustion_retires_until_utc_midnight(self, openalex_state, code):
        """A spent metered key states no Retry-After, so assume the daily reset."""
        assert _retire_openalex_key("testkey1234", None, code) is True
        assert list(_openalex_key_candidates()) == [None]
        remaining = _openalex_exhausted["testkey1234"] - _agentic_tools.time.monotonic()
        assert remaining == pytest.approx(_seconds_until_utc_midnight(), abs=5.0)

    def test_server_outage_never_retires(self, openalex_state):
        """503 hits every credential alike; failing over just repeats the request."""
        assert _retire_openalex_key(None, _CAP_RETRY_AFTER, 503) is False
        assert list(_openalex_key_candidates()) == [None, "testkey1234"]

    def test_keyless_returns_to_the_front_once_its_deadline_lapses(
        self, openalex_state, monkeypatch
    ):
        """The overnight case: capped at 23:00, quota back at 02:00, key stops being spent."""
        clock = 1000.0
        monkeypatch.setattr(_agentic_tools.time, "monotonic", lambda: clock)
        _retire_openalex_key(None, _CAP_RETRY_AFTER, 429)
        assert list(_openalex_key_candidates()) == ["testkey1234"]

        clock += _CAP_RETRY_AFTER - 1  # one second short of the reset
        assert list(_openalex_key_candidates()) == ["testkey1234"]

        clock += 2  # past it
        assert list(_openalex_key_candidates()) == [None, "testkey1234"]
        assert _openalex_exhausted == {}

    def test_no_configured_key_leaves_only_the_polite_pool(self, openalex_state, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY")
        assert list(_openalex_key_candidates()) == [None]

    def test_multiple_keys_are_tried_in_order(self, openalex_state, monkeypatch):
        """Comma-separated keys let a run fail over again without a code change."""
        monkeypatch.setenv("OPENALEX_API_KEY", "first1234, second5678 ,")
        assert list(_openalex_key_candidates()) == [None, "first1234", "second5678"]
