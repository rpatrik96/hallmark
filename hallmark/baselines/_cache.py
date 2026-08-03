"""Disk cache and retry utilities for API-based baselines.

Provides deterministic caching (keyed by entry content hash) so that
repeated evaluation runs do not re-query external APIs, and exponential
backoff retry to handle transient network errors gracefully.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import shelve
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

_HAS_FCNTL: bool = importlib.util.find_spec("fcntl") is not None

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hallmark"

# Upper bound on a honoured ``Retry-After``. Servers occasionally answer with windows
# of an hour or more; sleeping that long would stall an evaluation run, so we wait at
# most this and let the normal retry budget expire.
MAX_RETRY_AFTER_SECONDS = 60.0

_BENCHMARK_VERSION = "1.0"

# Sentinel object used to distinguish "key absent" from a cached falsy value.
_MISSING: object = object()

# Emit the no-fcntl warning at most once per process.
_fcntl_warning_emitted: bool = False


def _cache_dir() -> Path:
    d = _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def content_hash(data: str) -> str:
    """SHA-256 hex digest of the input string."""
    return hashlib.sha256(data.encode()).hexdigest()


def _locked_shelve_read(db_path: str, key: str) -> Any:
    """Read from shelve with a shared (read) file lock.

    Returns ``_MISSING`` when the key is absent so that callers can
    distinguish a missing entry from a cached falsy value (``None``,
    ``False``, ``0``, ``""``).
    """
    global _fcntl_warning_emitted
    lock_path = db_path + ".lock"
    if _HAS_FCNTL:
        import fcntl as _fcntl

        with open(lock_path, "a") as lock_file:
            _fcntl.flock(lock_file, _fcntl.LOCK_SH)
            try:
                with shelve.open(db_path) as db:
                    return db.get(key, _MISSING)
            finally:
                _fcntl.flock(lock_file, _fcntl.LOCK_UN)
    else:
        if not _fcntl_warning_emitted:
            logger.warning(
                "fcntl not available on this platform; concurrent cache access is not protected"
            )
            _fcntl_warning_emitted = True
        with shelve.open(db_path) as db:
            return db.get(key, _MISSING)


def _locked_shelve_write(db_path: str, key: str, value: Any) -> None:
    """Write to shelve with an exclusive (write) file lock."""
    global _fcntl_warning_emitted
    lock_path = db_path + ".lock"
    if _HAS_FCNTL:
        import fcntl as _fcntl

        with open(lock_path, "a") as lock_file:
            _fcntl.flock(lock_file, _fcntl.LOCK_EX)
            try:
                with shelve.open(db_path) as db:
                    db[key] = value
            finally:
                _fcntl.flock(lock_file, _fcntl.LOCK_UN)
    else:
        if not _fcntl_warning_emitted:
            logger.warning(
                "fcntl not available on this platform; concurrent cache access is not protected"
            )
            _fcntl_warning_emitted = True
        with shelve.open(db_path) as db:
            db[key] = value


def cached_call(
    namespace: str,
    key: str,
    fn: Callable[[], T],
    cache_dir: Path | None = None,
) -> T:
    """Return cached result for *key* under *namespace*, or call *fn* and cache it.

    Args:
        namespace: Logical grouping (e.g., baseline name). Becomes the shelve filename.
        key: Cache key (typically ``content_hash(entry_bibtex)``).
        fn: Zero-argument callable to invoke on cache miss.
        cache_dir: Override cache directory (default: ``~/.cache/hallmark``).

    Returns:
        The cached or freshly computed value.
    """
    d = cache_dir or _cache_dir()
    versioned_namespace = f"{namespace}:v{_BENCHMARK_VERSION}"
    db_path = str(d / versioned_namespace)

    cached = _locked_shelve_read(db_path, key)
    if cached is not _MISSING:
        logger.debug("Cache hit: %s/%s", versioned_namespace, key[:12])
        return cached  # type: ignore[return-value,no-any-return]

    result: T = fn()
    _locked_shelve_write(db_path, key, result)
    logger.debug("Cache miss: %s/%s — stored", versioned_namespace, key[:12])
    return result


class RateLimitedError(RuntimeError):
    """An upstream throttle (HTTP 429/503) that told us how long to wait.

    Servers answering 429 usually include a ``Retry-After`` header stating when the
    caller may return.  Ignoring it and falling back to a fixed exponential schedule
    means retrying *before* the window reopens and exhausting the budget while being
    told exactly how to succeed — the shape of the 173 Semantic Scholar and 149
    OpenAlex failures in the 2026-07 cascade run.

    Attributes:
        retry_after: Seconds to wait as instructed by the server, or ``None`` when the
            header was absent or unparseable (callers then fall back to exponential
            backoff).
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NonRetryableError(RuntimeError):
    """An error that is guaranteed to recur identically on every attempt.

    Raised for deterministic, client-side faults (e.g. decoding a response body
    we ourselves mislabelled) as opposed to transient network conditions such as
    HTTP 429, timeouts, or connection resets.  :func:`retry_with_backoff` re-raises
    these immediately instead of sleeping through a backoff schedule that cannot
    change the outcome.

    Failing fast also keeps the diagnosis honest: four retries and a backoff make
    a code defect look like a struggling upstream API.
    """


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Retry *fn* with exponential backoff on failure.

    Three classes of failure are treated differently:

    - :class:`NonRetryableError` — deterministic client-side fault; re-raised on the
      first attempt, since retrying cannot change the outcome.
    - :class:`RateLimitedError` carrying ``retry_after`` — the server stated when to
      return, so that wait is honoured (capped at :data:`MAX_RETRY_AFTER_SECONDS`)
      in place of the exponential schedule.
    - Everything else — ordinary transient errors; exponential backoff.

    Args:
        fn: Zero-argument callable.
        max_retries: Maximum number of retry attempts (total calls = max_retries + 1).
        base_delay: Initial delay in seconds; doubles each retry.
        exceptions: Exception types to catch and retry on.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    delay = base_delay
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except NonRetryableError as exc:
            # Deterministic client-side fault — retrying cannot change the result.
            # Logged at ERROR (not WARNING) because it signals a bug in our code,
            # not an unhealthy upstream API.
            logger.error("Non-retryable error, failing immediately: %s", exc)
            raise
        except exceptions as exc:
            last_exc = exc
            # Honour a server-stated Retry-After in place of our own guess: retrying
            # sooner than instructed is guaranteed to be refused again.
            wait = delay
            requested = getattr(exc, "retry_after", None)
            if isinstance(requested, (int, float)) and requested > 0:
                wait = min(float(requested), MAX_RETRY_AFTER_SECONDS)
                if float(requested) > MAX_RETRY_AFTER_SECONDS:
                    logger.warning(
                        "Server asked for %.0fs, capping wait at %.0fs",
                        float(requested),
                        MAX_RETRY_AFTER_SECONDS,
                    )
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed (%s), retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
                delay *= 2
            else:
                logger.error(
                    "All %d attempts failed for %s",
                    max_retries + 1,
                    fn,
                )

    raise last_exc  # type: ignore[misc]


def clear_cache(namespace: str | None = None, cache_dir: Path | None = None) -> None:
    """Remove cached entries for *namespace*, or all entries if *namespace* is None.

    The versioned shelve files (e.g. ``doi_only:v1.0``) are matched by prefix
    when *namespace* is given, so callers pass the bare namespace name (e.g.
    ``"doi_only"``) without the version suffix.
    """
    d = cache_dir or _cache_dir()
    if namespace is None:
        candidates = list(d.iterdir())
    else:
        versioned_prefix = f"{namespace}:v"
        candidates = [p for p in d.iterdir() if p.name.startswith(versioned_prefix)]

    removed: set[Path] = set()
    for p in candidates:
        # shelve may append .db/.dir/.bak/.dat; group by stem to avoid double-logging
        stem = p
        for suffix in (".db", ".dir", ".bak", ".dat"):
            if p.name.endswith(suffix):
                stem = p.with_suffix("")
                break
        if stem in removed:
            continue
        # Remove all shelve-related files for this stem
        for suffix in ("", ".db", ".dir", ".bak", ".dat", ".lock"):
            target = Path(str(stem) + suffix) if suffix else stem
            if target.exists():
                target.unlink()
                logger.info("Removed cache file: %s", target)
        removed.add(stem)
