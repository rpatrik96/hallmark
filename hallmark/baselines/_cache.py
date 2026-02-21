"""Disk cache and retry utilities for API-based baselines.

Provides deterministic caching (keyed by entry content hash) so that
repeated evaluation runs do not re-query external APIs, and exponential
backoff retry to handle transient network errors gracefully.
"""

from __future__ import annotations

import hashlib
import logging
import shelve
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hallmark"


def _cache_dir() -> Path:
    d = _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def content_hash(data: str) -> str:
    """SHA-256 hex digest of the input string."""
    return hashlib.sha256(data.encode()).hexdigest()


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
    db_path = str(d / namespace)

    with shelve.open(db_path) as db:
        if key in db:
            logger.debug("Cache hit: %s/%s", namespace, key[:12])
            result: T = db[key]  # type: ignore[assignment]
            return result

        result = fn()
        db[key] = result
        logger.debug("Cache miss: %s/%s — stored", namespace, key[:12])
        return result


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Retry *fn* with exponential backoff on failure.

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
        except exceptions as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed (%s), retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(
                    "All %d attempts failed for %s",
                    max_retries + 1,
                    fn,
                )

    raise last_exc  # type: ignore[misc]


def clear_cache(namespace: str, cache_dir: Path | None = None) -> None:
    """Remove all cached entries for *namespace*."""
    d = cache_dir or _cache_dir()
    db_path = d / namespace
    for suffix in ("", ".db", ".dir", ".bak", ".dat"):
        p = db_path.with_suffix(suffix) if suffix else db_path
        if p.exists():
            p.unlink()
            logger.info("Removed cache file: %s", p)
