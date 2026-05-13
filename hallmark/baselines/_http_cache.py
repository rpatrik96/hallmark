"""HTTP-response caching infrastructure for reproducible HALLMARK eval runs.

Backed by ``requests-cache`` (SQLite). When enabled, baselines that issue HTTP
calls reuse cached responses instead of re-querying remote APIs — supporting
the reproducibility story for re-runs.

Usage::

    from hallmark.baselines._http_cache import http_cache

    with http_cache(Path("/tmp/hallmark.sqlite")):
        run_baselines(...)

If ``cache_path`` is ``None``, the context manager is a no-op (regular
network calls). Otherwise it monkey-patches ``httpx`` so that requests
go through ``requests-cache`` for the duration of the block.

Notes:
    - ``requests-cache`` is an *optional* dependency. The context manager
      raises ``ImportError`` only when invoked with a non-None path; calling
      with ``None`` never imports the optional package.
    - Cache TTL is 30 days. Match key is full URL + headers, excluding
      ``User-Agent`` (which varies per release).
    - The patch is a thin shim: it replaces ``httpx.get/head/post/request``
      with a function that delegates to a ``requests_cache.CachedSession``.
      We restore the originals on exit, even if the body raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# 30 days in seconds.
_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60

# Headers we want to ignore when matching cached responses.
_IGNORED_HEADERS = ("User-Agent",)


@contextmanager
def http_cache(cache_path: Path | None) -> Iterator[None]:
    """Install a SQLite-backed HTTP response cache for the duration of the block.

    Args:
        cache_path: filesystem path to the SQLite cache file. If ``None``,
            the context manager is a no-op (no caching is installed).

    Raises:
        ImportError: if ``cache_path`` is non-None but ``requests-cache`` is
            not installed.
    """
    if cache_path is None:
        # No-op path — yield immediately so callers can always use this CM.
        yield
        return

    try:
        import httpx
        import requests_cache
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "http_cache requires the optional 'requests-cache' dependency. "
            "Install with: pip install 'hallmark[reproducibility]' "
            "or: pip install requests-cache"
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip the .sqlite extension: requests_cache appends one itself.
    backend_name = str(cache_path)
    if backend_name.endswith(".sqlite"):
        backend_name = backend_name[: -len(".sqlite")]

    session = requests_cache.CachedSession(
        cache_name=backend_name,
        backend="sqlite",
        expire_after=_CACHE_TTL_SECONDS,
        match_headers=True,
        ignored_parameters=list(_IGNORED_HEADERS),
    )

    # Save originals so we can restore on exit.
    originals: dict[str, Any] = {
        "get": httpx.get,
        "head": httpx.head,
        "post": httpx.post,
        "request": httpx.request,
    }

    def _to_httpx_response(resp: Any) -> Any:
        """Adapt a ``requests.Response`` to a duck-typed ``httpx.Response``-ish object."""
        # Build an httpx.Response so callers using .status_code/.text/.json() work.
        content = resp.content if hasattr(resp, "content") else b""
        headers = dict(resp.headers) if hasattr(resp, "headers") else {}
        return httpx.Response(
            status_code=int(resp.status_code),
            headers=headers,
            content=content,
        )

    def _cached_request(method: str, url: str, **kwargs: Any) -> Any:
        # Drop httpx-only kwargs requests doesn't recognise.
        kwargs.pop("follow_redirects", None)
        timeout = kwargs.pop("timeout", None)
        if timeout is not None:
            kwargs["timeout"] = timeout
        # User-Agent header is intentionally excluded from cache key.
        headers = kwargs.get("headers") or {}
        clean_headers = {k: v for k, v in headers.items() if k not in _IGNORED_HEADERS}
        kwargs["headers"] = clean_headers
        resp = session.request(method.upper(), url, **kwargs)
        return _to_httpx_response(resp)

    def _patched_get(url: str, **kwargs: Any) -> Any:
        return _cached_request("GET", url, **kwargs)

    def _patched_head(url: str, **kwargs: Any) -> Any:
        return _cached_request("HEAD", url, **kwargs)

    def _patched_post(url: str, **kwargs: Any) -> Any:
        return _cached_request("POST", url, **kwargs)

    def _patched_request(method: str, url: str, **kwargs: Any) -> Any:
        return _cached_request(method, url, **kwargs)

    httpx.get = _patched_get  # type: ignore[assignment]
    httpx.head = _patched_head  # type: ignore[assignment]
    httpx.post = _patched_post  # type: ignore[assignment]
    httpx.request = _patched_request  # type: ignore[assignment]

    try:
        yield
    finally:
        # Restore originals even on exception.
        for name, fn in originals.items():
            setattr(httpx, name, fn)
        session.close()
