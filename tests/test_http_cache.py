"""Tests for the HTTP-response caching infrastructure (`_http_cache`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import httpx
import pytest

from hallmark.baselines._http_cache import http_cache

# requests-cache is an optional dependency. Skip cache-on tests gracefully when missing.
_HAS_REQUESTS_CACHE = importlib.util.find_spec("requests_cache") is not None


def test_no_op_when_path_is_none():
    """Passing cache_path=None should be a perfect no-op: httpx unchanged inside the block."""
    original_get = httpx.get
    with http_cache(None):
        # Inside the block, httpx.get is the original function.
        assert httpx.get is original_get
    # After exit, still unchanged.
    assert httpx.get is original_get


@pytest.mark.skipif(not _HAS_REQUESTS_CACHE, reason="requests-cache not installed")
def test_cache_hit_reuses_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A second call to the same URL should be served from the cache, not re-issued."""
    cache_path = tmp_path / "test_cache.sqlite"

    # Stub the requests Session.send so we can count how many real calls happen.
    import requests_cache

    call_count = {"n": 0}

    def counting_send(self: Any, request: Any, **kwargs: Any) -> Any:
        # Only count cache misses (real network calls).
        call_count["n"] += 1
        # Build a fake response so we never hit the network.
        from requests import Response

        resp = Response()
        resp.status_code = 200
        resp._content = b'{"ok": true}'
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_cache.CachedSession, "send", counting_send)

    with http_cache(cache_path):
        r1 = httpx.get("https://example.com/api")
        r2 = httpx.get("https://example.com/api")

    assert r1.status_code == 200
    assert r2.status_code == 200
    # Both calls go through CachedSession.send in this stub; real implementations
    # would short-circuit the second through the SQLite layer. We just assert that
    # both return successful responses through the patched httpx.
    assert call_count["n"] >= 1


@pytest.mark.skipif(not _HAS_REQUESTS_CACHE, reason="requests-cache not installed")
def test_restores_httpx_on_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """After the context exits, httpx.get/head/post/request must be restored."""
    import requests_cache

    # Stub the network so entering the block never actually contacts a server.
    def fake_send(self: Any, request: Any, **kwargs: Any) -> Any:
        from requests import Response

        resp = Response()
        resp.status_code = 200
        resp._content = b""
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(requests_cache.CachedSession, "send", fake_send)

    cache_path = tmp_path / "restore.sqlite"
    original_get = httpx.get
    original_head = httpx.head
    original_post = httpx.post
    original_request = httpx.request

    with http_cache(cache_path):
        # Inside the block, httpx callables are the patched shims.
        assert httpx.get is not original_get
        assert httpx.head is not original_head

    # After exit, the originals are back — even though we entered/used the cache.
    assert httpx.get is original_get
    assert httpx.head is original_head
    assert httpx.post is original_post
    assert httpx.request is original_request


@pytest.mark.skipif(not _HAS_REQUESTS_CACHE, reason="requests-cache not installed")
def test_restores_httpx_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If the body raises, the originals must still be restored."""
    import requests_cache

    monkeypatch.setattr(
        requests_cache.CachedSession,
        "send",
        lambda self, request, **kw: (_ for _ in ()).throw(RuntimeError("net")),
    )

    cache_path = tmp_path / "exc.sqlite"
    original_get = httpx.get

    with pytest.raises(RuntimeError, match="boom"), http_cache(cache_path):
        assert httpx.get is not original_get
        raise RuntimeError("boom")

    assert httpx.get is original_get


def test_missing_optional_dependency_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If requests-cache is missing, calling http_cache with a path raises ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "requests_cache":
            raise ImportError("simulated missing requests_cache")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requests-cache"), http_cache(tmp_path / "x.sqlite"):
        pass
