"""Consolidated HTTP/API clients for external bibliographic data sources.

Single location for all network I/O against CrossRef, Semantic Scholar, DBLP,
and arXiv. All clients use httpx and share the retry helper below.
"""

from __future__ import annotations

import logging
import ssl
import time
from collections.abc import Callable

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
BIORXIV_API_BASE = "https://api.biorxiv.org"
PUBMED_API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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


# ---------------------------------------------------------------------------
# bioRxiv / medRxiv client
# ---------------------------------------------------------------------------


class BioRxivClient:
    """Thin wrapper over the bioRxiv/medRxiv public API.

    Endpoints used:
        /details/{server}/{interval}/{cursor}     — list papers in a date range
        /details/{server}/{doi}/na/json           — fetch a single paper by DOI

    The ``server`` parameter is "biorxiv" or "medrxiv".
    """

    def __init__(
        self,
        server: str = "biorxiv",
        user_agent: str = "HALLMARK/1.0",
        rate_limit: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        if server not in ("biorxiv", "medrxiv"):
            raise ValueError(f"server must be 'biorxiv' or 'medrxiv', got {server!r}")
        self._server = server
        self._user_agent = user_agent
        self._rate_limit = rate_limit
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._user_agent}

    def list_papers(
        self,
        from_date: str,
        to_date: str,
        max_results: int = 100,
    ) -> list[dict]:
        """List papers posted between from_date and to_date (YYYY-MM-DD).

        Returns the merged ``collection`` records across cursors. The bioRxiv
        API returns at most 100 records per cursor; we paginate until we have
        ``max_results`` records or the API stops returning more.
        """
        results: list[dict] = []
        cursor = 0
        total: int | None = None
        while len(results) < max_results:
            time.sleep(self._rate_limit)
            url = f"{BIORXIV_API_BASE}/details/{self._server}/{from_date}/{to_date}/{cursor}"
            with httpx.Client(timeout=self._timeout) as client:
                resp = _request_with_retry(client, "GET", url, headers=self._headers())
            if resp is None:
                break
            try:
                data = resp.json()
            except Exception as e:
                logger.error("bioRxiv JSON decode failed (%s): %s", url, e)
                break
            batch = data.get("collection", []) or []
            if not batch:
                break
            results.extend(batch)
            cursor += len(batch)
            # bioRxiv reports the total in the messages block. Stop when cursor
            # passes total or when we've collected enough.
            if total is None:
                msgs = data.get("messages") or []
                if msgs and isinstance(msgs[0], dict):
                    try:
                        total = int(msgs[0].get("total", 0))
                    except (TypeError, ValueError):
                        total = None
            if total is not None and cursor >= total:
                break
        return results[:max_results]

    def get_paper_by_doi(self, doi: str) -> dict | None:
        """Fetch a single paper record by DOI (returns the latest version)."""
        time.sleep(self._rate_limit)
        url = f"{BIORXIV_API_BASE}/details/{self._server}/{doi}/na/json"
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(client, "GET", url, headers=self._headers())
        if resp is None:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        coll = data.get("collection", []) or []
        return coll[-1] if coll else None

    def find_withdrawn_papers_parallel(
        self,
        from_date: str,
        to_date: str,
        workers: int = 8,
        on_batch: Callable[[list[dict], int, int], None] | None = None,
    ) -> list[dict]:
        """Same as ``find_withdrawn_papers`` but pages in parallel via threads.

        Strategy: fetch cursor=0 to learn the total count, then dispatch all
        remaining cursor offsets across a thread pool. Bandwidth-bound, so
        threads are sufficient (no GIL contention on httpx I/O).

        ``on_batch(hits, completed, total_pages)`` is invoked after each page
        completes — useful for live progress logging or partial-checkpoint
        writes during a long scan.
        """
        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pat = re.compile(
            r"^\s*(withdrawal statement|\[?withdrawn)|has been withdrawn",
            re.IGNORECASE,
        )

        def _fetch_page(cursor: int) -> tuple[int, list[dict], int | None]:
            url = f"{BIORXIV_API_BASE}/details/{self._server}/{from_date}/{to_date}/{cursor}"
            with httpx.Client(timeout=self._timeout) as client:
                resp = _request_with_retry(client, "GET", url, headers=self._headers())
            if resp is None:
                return cursor, [], None
            try:
                data = resp.json()
            except Exception:
                return cursor, [], None
            batch = data.get("collection", []) or []
            total = None
            msgs = data.get("messages") or []
            if msgs and isinstance(msgs[0], dict):
                try:
                    total = int(msgs[0].get("total", 0))
                except (TypeError, ValueError):
                    total = None
            hits = [r for r in batch if pat.search((r.get("abstract") or "")[:300])]
            return cursor, hits, total

        # Probe the total first so we know how many pages to dispatch.
        _first_cursor, first_hits, total = _fetch_page(0)
        if total is None or total <= 0:
            return first_hits

        # bioRxiv pages are 30 records each.
        page_size = 30
        offsets = list(range(page_size, total, page_size))
        all_hits = list(first_hits)
        if on_batch is not None:
            on_batch(first_hits, 1, len(offsets) + 1)

        completed = 1
        total_pages = len(offsets) + 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch_page, off): off for off in offsets}
            for f in as_completed(futures):
                _cursor, hits, _ = f.result()
                completed += 1
                if hits:
                    all_hits.extend(hits)
                if on_batch is not None:
                    on_batch(hits, completed, total_pages)
        return all_hits

    def find_withdrawn_papers(
        self,
        from_date: str,
        to_date: str,
        max_results: int | None = None,
    ) -> list[dict]:
        """Scan a date range and return only records whose abstract carries a
        withdrawal marker.

        bioRxiv/medRxiv do not expose a withdrawn-only endpoint. Withdrawn
        papers are flagged by replacing (or prefixing) the abstract with a
        "Withdrawal Statement" or "[Withdrawn]" notice. We page through all
        records in the window and filter on the abstract head.

        Withdrawn rates observed empirically:
            bioRxiv  ~0.25% of all submissions
            medRxiv  ~0.4%  of all submissions

        Parameters
        ----------
        max_results:
            Stop early once this many withdrawn records have been found.
            None means scan the whole window.
        """
        import re

        pat = re.compile(
            r"^\s*(withdrawal statement|\[?withdrawn)|has been withdrawn",
            re.IGNORECASE,
        )
        out: list[dict] = []
        cursor = 0
        total: int | None = None
        while True:
            time.sleep(self._rate_limit)
            url = f"{BIORXIV_API_BASE}/details/{self._server}/{from_date}/{to_date}/{cursor}"
            with httpx.Client(timeout=self._timeout) as client:
                resp = _request_with_retry(client, "GET", url, headers=self._headers())
            if resp is None:
                break
            try:
                data = resp.json()
            except Exception as e:
                logger.error("bioRxiv JSON decode failed (%s): %s", url, e)
                break
            batch = data.get("collection", []) or []
            if not batch:
                break

            for rec in batch:
                ab = (rec.get("abstract") or "")[:300]
                if pat.search(ab):
                    out.append(rec)
                    if max_results is not None and len(out) >= max_results:
                        return out

            cursor += len(batch)
            if total is None:
                msgs = data.get("messages") or []
                if msgs and isinstance(msgs[0], dict):
                    try:
                        total = int(msgs[0].get("total", 0))
                    except (TypeError, ValueError):
                        total = None
            if total is not None and cursor >= total:
                break
        return out


# ---------------------------------------------------------------------------
# PubMed (NCBI E-utilities) client
# ---------------------------------------------------------------------------


class PubMedClient:
    """Minimal NCBI E-utilities client.

    Uses esearch + esummary in JSON mode. An optional API key raises the rate
    limit from 3 req/s to 10 req/s but is not required.
    """

    def __init__(
        self,
        api_key: str | None = None,
        tool: str = "HALLMARK",
        email: str | None = None,
        rate_limit: float = 0.4,
        timeout: float = 30.0,
        user_agent: str = "HALLMARK/1.0",
    ) -> None:
        self._api_key = api_key
        self._tool = tool
        self._email = email
        self._rate_limit = rate_limit
        self._timeout = timeout
        self._user_agent = user_agent

    def _common_params(self) -> dict[str, str]:
        params = {"tool": self._tool, "retmode": "json"}
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email
        return params

    def esearch(
        self, term: str, retmax: int = 100, mindate: str | None = None, maxdate: str | None = None
    ) -> list[str]:
        """Run an E-utilities search; returns a list of PMIDs."""
        time.sleep(self._rate_limit)
        params = {**self._common_params(), "db": "pubmed", "term": term, "retmax": str(retmax)}
        if mindate:
            params["mindate"] = mindate
            params["datetype"] = "pdat"
        if maxdate:
            params["maxdate"] = maxdate
            params["datetype"] = "pdat"
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(
                client,
                "GET",
                f"{PUBMED_API_BASE}/esearch.fcgi",
                params=params,
                headers={"User-Agent": self._user_agent},
            )
        if resp is None:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        ids = data.get("esearchresult", {}).get("idlist", []) or []
        return [str(x) for x in ids]

    def esummary(self, pmids: list[str]) -> list[dict]:
        """Fetch metadata summaries for a batch of PMIDs."""
        if not pmids:
            return []
        time.sleep(self._rate_limit)
        params = {**self._common_params(), "db": "pubmed", "id": ",".join(pmids)}
        with httpx.Client(timeout=self._timeout) as client:
            resp = _request_with_retry(
                client,
                "GET",
                f"{PUBMED_API_BASE}/esummary.fcgi",
                params=params,
                headers={"User-Agent": self._user_agent},
            )
        if resp is None:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        result = data.get("result", {}) or {}
        uids = result.get("uids", []) or []
        out: list[dict] = []
        for uid in uids:
            rec = result.get(uid)
            if isinstance(rec, dict):
                out.append(rec)
        return out
