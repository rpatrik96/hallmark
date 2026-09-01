"""Tests for hallmark.baselines._cache."""

from __future__ import annotations

import logging
import time

import pytest

from hallmark.baselines._cache import (
    MAX_RETRY_AFTER_SECONDS,
    NonRetryableError,
    RateLimitedError,
    cached_call,
    clear_cache,
    content_hash,
    retry_with_backoff,
)


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("a") != content_hash("b")


class TestCachedCall:
    def test_cache_hit(self, tmp_path):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return 42

        r1 = cached_call("test_ns", "key1", fn, cache_dir=tmp_path)
        r2 = cached_call("test_ns", "key1", fn, cache_dir=tmp_path)
        assert r1 == 42
        assert r2 == 42
        assert call_count == 1  # second call was a cache hit

    def test_cache_miss_different_keys(self, tmp_path):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return call_count

        r1 = cached_call("test_ns", "key1", fn, cache_dir=tmp_path)
        r2 = cached_call("test_ns", "key2", fn, cache_dir=tmp_path)
        assert r1 == 1
        assert r2 == 2

    def test_different_namespaces(self, tmp_path):
        cached_call("ns_a", "key1", lambda: "a", cache_dir=tmp_path)
        cached_call("ns_b", "key1", lambda: "b", cache_dir=tmp_path)
        # Each namespace is independent
        r_a = cached_call("ns_a", "key1", lambda: "should_not_run", cache_dir=tmp_path)
        r_b = cached_call("ns_b", "key1", lambda: "should_not_run", cache_dir=tmp_path)
        assert r_a == "a"
        assert r_b == "b"


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        result = retry_with_backoff(lambda: 42, max_retries=3, base_delay=0.01)
        assert result == 42

    def test_succeeds_after_retries(self):
        attempts = 0

        def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("transient")
            return "ok"

        result = retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert attempts == 3

    def test_exhausts_retries(self):
        def always_fail():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    def test_non_retryable_fails_on_first_attempt(self):
        """Deterministic client-side faults must not be retried.

        A decode failure recurs identically on every attempt, so the backoff
        schedule only burns wall-clock and disguises a code defect as a flaky API.
        """
        attempts = 0

        def deterministic_fault():
            nonlocal attempts
            attempts += 1
            raise NonRetryableError("mislabelled response body")

        with pytest.raises(NonRetryableError, match="mislabelled"):
            retry_with_backoff(deterministic_fault, max_retries=3, base_delay=10.0)

        assert attempts == 1, "NonRetryableError must not be retried"

    def test_honours_server_retry_after(self, monkeypatch):
        """A server-stated Retry-After replaces our exponential guess.

        Retrying sooner than instructed is guaranteed to be refused again, which is
        how a 7-second budget lost calls to endpoints asking for 30.
        """
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

        def throttled():
            raise RateLimitedError("HTTP 429", retry_after=30.0)

        with pytest.raises(RateLimitedError):
            retry_with_backoff(throttled, max_retries=3, base_delay=1.0)

        assert slept == [30.0, 30.0, 30.0], "should wait as instructed, not 1/2/4"

    def test_retry_after_is_capped(self, monkeypatch):
        """An implausibly long Retry-After is capped so a run cannot stall."""
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

        def throttled():
            raise RateLimitedError("HTTP 429", retry_after=3600.0)

        with pytest.raises(RateLimitedError):
            retry_with_backoff(throttled, max_retries=1, base_delay=1.0)

        assert slept == [MAX_RETRY_AFTER_SECONDS]

    def test_falls_back_to_exponential_without_header(self, monkeypatch):
        """With no Retry-After, the original exponential schedule is unchanged."""
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

        def throttled():
            raise RateLimitedError("HTTP 429", retry_after=None)

        with pytest.raises(RateLimitedError):
            retry_with_backoff(throttled, max_retries=3, base_delay=1.0)

        assert slept == [1.0, 2.0, 4.0]

    def test_non_retryable_logged_as_error(self, caplog):
        """It is logged at ERROR, since it signals our bug rather than a sick upstream."""

        def deterministic_fault():
            raise NonRetryableError("mislabelled response body")

        with caplog.at_level(logging.ERROR), pytest.raises(NonRetryableError):
            retry_with_backoff(deterministic_fault, max_retries=3, base_delay=0.01)

        assert [r for r in caplog.records if r.levelno == logging.ERROR], (
            "expected an ERROR-level log record"
        )

    def test_selective_exception_types(self):
        """Only retries on specified exception types."""

        def wrong_type():
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            retry_with_backoff(
                wrong_type,
                max_retries=3,
                base_delay=0.01,
                exceptions=(ConnectionError,),
            )


class TestClearCache:
    def test_clear_removes_cache(self, tmp_path):
        cached_call("to_clear", "k", lambda: 1, cache_dir=tmp_path)
        clear_cache("to_clear", cache_dir=tmp_path)
        # After clearing, fn should be called again
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return 2

        result = cached_call("to_clear", "k", fn, cache_dir=tmp_path)
        assert result == 2
        assert call_count == 1
