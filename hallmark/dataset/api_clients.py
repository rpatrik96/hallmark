"""Consolidated HTTP/API clients for external bibliographic data sources.

Single location for all network I/O against CrossRef, Semantic Scholar, DBLP,
and arXiv. All clients use httpx and share the retry helper below.
"""

from __future__ import annotations

import logging
import ssl
import time

import httpx

from hallmark.dataset.text_utils import title_similarity as _title_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------

CROSSREF_API_BASE = "https://api.crossref.org/works"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
DBLP_API_BASE = "https://dblp.org/search/publ/api"
ARXIV_API_BASE = "https://export.arxiv.org/api/query"


# ---------------------------------------------------------------------------
# SSL context
# ---------------------------------------------------------------------------


def get_ssl_context() -> ssl.SSLContext:
    """Return an SSL context using certifi CA bundle when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs: object,
) -> httpx.Response | None:
    """Make an HTTP request with exponential backoff on transient failures.

    Returns the response on success, or None after exhausting retries.
    """
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.request(method, url, **kwargs)  # type: ignore[arg-type]
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if attempt == max_retries:
                logger.error("Request failed after %d attempts: %s: %s", max_retries + 1, url, e)
                return None
            logger.warning(
                "Attempt %d failed for %s: %s, retrying in %.1fs", attempt + 1, url, e, delay
            )
            time.sleep(delay)
            delay *= 2
    return None


# ---------------------------------------------------------------------------
# CrossRef client
# ---------------------------------------------------------------------------


class CrossRefClient:
    def __init__(
        self,
        user_agent: str = "HALLMARK/1.0",
        rate_limit: float = 1.0,
        timeout: float = 20.0,
    ) -> None:
        self._user_agent = user_agent
        self._rate_limit = rate_limit
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._user_agent}

    def query_by_doi(self, doi: str) -> dict | None:
        """Fetch CrossRef metadata for a DOI. Returns the ``message`` dict or None."""
        time.sleep(self._rate_limit)
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(
                client,
                "GET",
                f"{CROSSREF_API_BASE}/{doi}",
                headers=self._headers(),
            )
        if resp is None:
            return None
        try:
            msg = resp.json().get("message")
            return msg if isinstance(msg, dict) else None
        except Exception:
            return None

    def query_by_title(self, title: str, rows: int = 3) -> list[dict]:
        """Search CrossRef by title. Returns the ``items`` list (may be empty)."""
        time.sleep(self._rate_limit)
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(
                client,
                "GET",
                CROSSREF_API_BASE,
                params={"query.bibliographic": title, "rows": rows},
                headers=self._headers(),
            )
        if resp is None:
            return []
        try:
            items: list[dict] = resp.json().get("message", {}).get("items", [])
            return items
        except Exception:
            return []

    def verify_doi(self, doi: str) -> bool:
        """Return True if the DOI resolves in CrossRef."""
        return self.query_by_doi(doi) is not None

    def verify_title(self, title: str, min_similarity: float = 0.5) -> dict | None:
        """Return the best-matching CrossRef item whose title similarity >= min_similarity, or None."""
        items = self.query_by_title(title, rows=3)
        best: dict | None = None
        best_score = 0.0
        for item in items:
            cr_title = (item.get("title") or [""])[0]
            score = _title_similarity(title, cr_title)
            if score > best_score:
                best_score = score
                best = item
        return best if best_score >= min_similarity else None


# ---------------------------------------------------------------------------
# Semantic Scholar client
# ---------------------------------------------------------------------------


class SemanticScholarClient:
    def __init__(
        self,
        user_agent: str = "HALLMARK/1.0",
        rate_limit: float = 1.0,
        timeout: float = 20.0,
    ) -> None:
        self._user_agent = user_agent
        self._rate_limit = rate_limit
        self._timeout = timeout

    def query_by_title(self, title: str, limit: int = 5) -> list[dict]:
        """Search S2 by title. Returns the ``data`` list (may be empty)."""
        time.sleep(self._rate_limit)
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(
                client,
                "GET",
                f"{S2_API_BASE}/paper/search",
                params={"query": title, "limit": limit},
                headers={"User-Agent": self._user_agent},
            )
        if resp is None:
            return []
        try:
            data: list[dict] = resp.json().get("data", [])
            return data
        except Exception:
            return []

    def verify_title(self, title: str) -> bool:
        """Return True if S2 returns at least one result for the title."""
        return len(self.query_by_title(title, limit=1)) > 0


# ---------------------------------------------------------------------------
# DBLP client
# ---------------------------------------------------------------------------


class DBLPClient:
    def __init__(
        self,
        user_agent: str = "HALLMARK/1.0",
        rate_limit: float = 1.0,
        timeout: float = 20.0,
    ) -> None:
        self._user_agent = user_agent
        self._rate_limit = rate_limit
        self._timeout = timeout

    def search(self, query: str, max_results: int = 100) -> list[dict]:
        """Run an arbitrary DBLP publication search. Returns list of hit info dicts."""
        time.sleep(self._rate_limit)
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "h": min(max_results, 1000),
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(
                client,
                "GET",
                DBLP_API_BASE,
                params=params,
                headers={"User-Agent": self._user_agent},
            )
        if resp is None:
            return []
        try:
            data = resp.json()
        except Exception as e:
            logger.error("DBLP JSON decode failed for query '%s': %s", query, e)
            return []
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        return [h.get("info", {}) for h in hits if "info" in h]

    def search_venue_year(
        self,
        venue_key: str,
        year: int,
        max_results: int = 100,
    ) -> list[dict]:
        """Search DBLP for a venue stream in a given year.

        Uses ``stream:<venue_key>: year:<year>`` which reliably covers 2024+
        (the older ``venue:<key>/<year>`` format returns 0 results for recent years).
        """
        return self.search(f"stream:{venue_key}: year:{year}", max_results=max_results)
