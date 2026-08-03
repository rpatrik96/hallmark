"""Tests for Retry-After parsing and throttle detection in `_agentic_tools`.

Covers the failure mode behind the 2026-07 cascade run, where 173 Semantic Scholar
and 149 OpenAlex lookups were lost to HTTP 429: the servers stated when to return,
and the client ignored them in favour of a fixed 1s/2s/4s schedule.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from hallmark.baselines._agentic_tools import _parse_retry_after, _raise_for_throttle
from hallmark.baselines._cache import RateLimitedError


class _Resp:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers: dict[str, Any] = headers or {}


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
