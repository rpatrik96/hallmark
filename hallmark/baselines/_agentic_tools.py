"""Pure tool wrappers for the agentic citation-verification baseline.

Each function makes one external API call, normalises the result to a flat
dict ``{authors, title, venue, year, doi}`` (fields truncated to 500 chars),
and raises on error so the caller can apply retry/backoff.

Tool set mirrors bibtex-updater's lookup sources:
- resolve_doi      — CrossRef DOI resolution (reuses prescreening DOI helper)
- search_crossref  — CrossRef title/author search
- search_openalex  — OpenAlex polite pool
- search_arxiv     — arXiv API
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_MAX_FIELD_CHARS = 500
_OPENALEX_MAILTO = "hallmark@example.com"


def _trunc(value: object) -> str:
    """Convert value to str and truncate to _MAX_FIELD_CHARS."""
    s = str(value) if value is not None else ""
    return s[:_MAX_FIELD_CHARS]


def _normalise(raw: dict) -> dict[str, str]:
    """Flatten an arbitrary API response to a canonical 5-field dict."""
    return {
        "authors": _trunc(raw.get("authors", "")),
        "title": _trunc(raw.get("title", "")),
        "venue": _trunc(raw.get("venue", "")),
        "year": _trunc(raw.get("year", "")),
        "doi": _trunc(raw.get("doi", "")),
    }


# ---------------------------------------------------------------------------
# resolve_doi
# ---------------------------------------------------------------------------


def resolve_doi(doi: str) -> dict[str, str]:
    """Resolve a DOI via CrossRef API and return normalised metadata.

    Args:
        doi: Raw DOI string (may include https://doi.org/ prefix).

    Returns:
        Normalised metadata dict.

    Raises:
        ValueError: DOI malformed or not found (404).
        RuntimeError: Network/API error.
    """
    import httpx

    doi_match = re.search(r"10\.\d+/[^\s]+", doi)
    if not doi_match:
        raise ValueError(f"Malformed DOI: {doi!r}")
    normalized_doi = doi_match.group(0)

    url = f"https://api.crossref.org/works/{urllib.parse.quote(normalized_doi, safe='/')}"
    headers = {"User-Agent": f"HALLMARK/1.0 (mailto:{_OPENALEX_MAILTO})"}

    try:
        resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
    except httpx.RequestError as exc:
        raise RuntimeError(f"Network error resolving DOI {normalized_doi}: {exc}") from exc

    if resp.status_code == 404:
        raise ValueError(f"DOI not found: {normalized_doi}")
    if resp.status_code != 200:
        raise RuntimeError(f"CrossRef returned HTTP {resp.status_code} for {normalized_doi}")

    data = resp.json().get("message", {})
    authors_list = data.get("author", [])
    authors_str = "; ".join(
        f"{a.get('family', '')} {a.get('given', '')}".strip() for a in authors_list
    )
    title_list = data.get("title", [])
    title_str = title_list[0] if title_list else ""
    container = data.get("container-title", [])
    venue_str = container[0] if container else ""
    date_parts = data.get("published", {}).get("date-parts", [[]])
    year_str = str(date_parts[0][0]) if date_parts and date_parts[0] else ""

    return _normalise(
        {
            "authors": authors_str,
            "title": title_str,
            "venue": venue_str,
            "year": year_str,
            "doi": normalized_doi,
        }
    )


# ---------------------------------------------------------------------------
# search_crossref
# ---------------------------------------------------------------------------


def search_crossref(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search CrossRef by bibliographic query string.

    Args:
        query: Free-text query (title, authors, etc.).
        limit: Maximum number of results to return.

    Returns:
        List of normalised metadata dicts (may be empty).

    Raises:
        RuntimeError: Network/API error.
    """
    import httpx

    params = {
        "query.bibliographic": query,
        "rows": str(min(limit, 20)),
        "mailto": _OPENALEX_MAILTO,
    }
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": f"HALLMARK/1.0 (mailto:{_OPENALEX_MAILTO})"}

    try:
        resp = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
    except httpx.RequestError as exc:
        raise RuntimeError(f"CrossRef search network error: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"CrossRef search returned HTTP {resp.status_code}")

    items = resp.json().get("message", {}).get("items", [])
    results = []
    for item in items[:limit]:
        authors_list = item.get("author", [])
        authors_str = "; ".join(
            f"{a.get('family', '')} {a.get('given', '')}".strip() for a in authors_list
        )
        title_list = item.get("title", [])
        title_str = title_list[0] if title_list else ""
        container = item.get("container-title", [])
        venue_str = container[0] if container else ""
        date_parts = item.get("published", {}).get("date-parts", [[]])
        year_str = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
        doi_str = item.get("DOI", "")
        results.append(
            _normalise(
                {
                    "authors": authors_str,
                    "title": title_str,
                    "venue": venue_str,
                    "year": year_str,
                    "doi": doi_str,
                }
            )
        )
    return results


# ---------------------------------------------------------------------------
# search_openalex
# ---------------------------------------------------------------------------


def search_openalex(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search OpenAlex (polite pool) by title/author query.

    Args:
        query: Free-text query.
        limit: Maximum number of results.

    Returns:
        List of normalised metadata dicts.

    Raises:
        RuntimeError: Network/API error.
    """
    import httpx

    params = {
        "search": query,
        "per-page": str(min(limit, 25)),
        "mailto": _OPENALEX_MAILTO,
        "select": "title,authorships,primary_location,publication_year,doi",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)

    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
    except httpx.RequestError as exc:
        raise RuntimeError(f"OpenAlex search network error: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"OpenAlex returned HTTP {resp.status_code}")

    items = resp.json().get("results", [])
    results = []
    for item in items[:limit]:
        authors_list = item.get("authorships", [])
        authors_str = "; ".join(a.get("author", {}).get("display_name", "") for a in authors_list)
        title_str = item.get("title", "") or ""
        source = (item.get("primary_location", {}) or {}).get("source") or {}
        venue_str = source.get("display_name", "") or ""
        year_str = str(item.get("publication_year", "")) if item.get("publication_year") else ""
        doi_raw = item.get("doi", "") or ""
        doi_str = re.sub(r"^https?://doi\.org/", "", doi_raw)
        results.append(
            _normalise(
                {
                    "authors": authors_str,
                    "title": title_str,
                    "venue": venue_str,
                    "year": year_str,
                    "doi": doi_str,
                }
            )
        )
    return results


# ---------------------------------------------------------------------------
# search_arxiv
# ---------------------------------------------------------------------------


def search_arxiv(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search arXiv via the Atom API.

    Args:
        query: Free-text query (title/author terms).
        limit: Maximum number of results.

    Returns:
        List of normalised metadata dicts (venue = "arXiv").

    Raises:
        RuntimeError: Network/API error.
    """
    import xml.etree.ElementTree as ET

    import httpx

    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(min(limit, 20)),
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)

    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
    except httpx.RequestError as exc:
        raise RuntimeError(f"arXiv search network error: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"arXiv returned HTTP {resp.status_code}")

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)
    results = []
    for entry in entries[:limit]:
        title_el = entry.find("atom:title", ns)
        title_str = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

        authors = [
            (name_el.text or "").strip()
            for a in entry.findall("atom:author", ns)
            if (name_el := a.find("atom:name", ns)) is not None
        ]
        authors_str = "; ".join(authors)

        published_el = entry.find("atom:published", ns)
        year_str = ""
        if published_el is not None and published_el.text:
            year_str = published_el.text[:4]

        doi_str = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "doi":
                doi_str = re.sub(r"^https?://doi\.org/", "", link.get("href", ""))
                break

        results.append(
            _normalise(
                {
                    "authors": authors_str,
                    "title": title_str,
                    "venue": "arXiv",
                    "year": year_str,
                    "doi": doi_str,
                }
            )
        )
    return results


# ---------------------------------------------------------------------------
# Tool dispatch table (used by the agentic runner)
# ---------------------------------------------------------------------------


def verify_with_bibtex_updater(bibtex: str) -> dict[str, str]:
    """Run ``bibtex-check`` on a single BibTeX entry and return its verdict.

    Exposes the external ``bibtex-updater`` CLI (``bibtex-check``) as an
    agent-callable tool.  The model receives the structured verdict
    (status, mismatched fields, APIs consulted, confidence) and decides
    whether to trust, override, or corroborate it with other tools.

    Args:
        bibtex: A raw BibTeX entry string (one entry, any @type).

    Returns:
        Flat dict with keys ``status``, ``confidence``, ``mismatched_fields``,
        ``api_sources``, ``errors``.  All values are strings (empty if absent).

    Raises:
        RuntimeError: ``bibtex-check`` not on PATH, or subprocess failure.
    """
    import json as _json
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    bin_path = shutil.which("bibtex-check")
    if bin_path is None:
        raise RuntimeError(
            "bibtex-check not found on PATH. Install with: pipx install bibtex-updater"
        )

    with tempfile.TemporaryDirectory() as td:
        bib_path = Path(td) / "input.bib"
        jsonl_path = Path(td) / "out.jsonl"
        bib_path.write_text(
            bibtex if bibtex.lstrip().startswith("@") else f"@misc{{x,\n{bibtex}\n}}"
        )

        cmd = [
            bin_path,
            str(bib_path),
            "--jsonl",
            str(jsonl_path),
            "--rate-limit",
            "120",
            "--academic-only",
        ]
        s2_key = os.environ.get("S2_API_KEY")
        if s2_key:
            cmd.extend(["--s2-api-key", s2_key])

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=180.0, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("bibtex-check timed out after 180s") from exc

        if not jsonl_path.exists():
            raise RuntimeError("bibtex-check produced no output")

        lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("bibtex-check returned empty output")

        rec = _json.loads(lines[0])
        return {
            "status": _trunc(rec.get("status", "unknown")),
            "confidence": _trunc(rec.get("confidence", "")),
            "mismatched_fields": _trunc(", ".join(rec.get("mismatched_fields", []) or [])),
            "api_sources": _trunc(", ".join(rec.get("api_sources", []) or [])),
            "errors": _trunc("; ".join(str(e) for e in rec.get("errors", []) or [])),
        }


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "resolve_doi",
        "description": (
            "Resolve a DOI via CrossRef and return publication metadata. "
            "Use when the entry has a DOI that may be fabricated or mismatched."
        ),
        "parameters": {
            "type": "object",
            "properties": {"doi": {"type": "string", "description": "The DOI string to resolve."}},
            "required": ["doi"],
        },
    },
    {
        "name": "search_crossref",
        "description": (
            "Search CrossRef by title, author, or other bibliographic terms. "
            "Returns up to `limit` matching publications."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Bibliographic search query (title, authors, etc.).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (1-20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_openalex",
        "description": (
            "Search OpenAlex for publications. Covers a broad corpus including "
            "preprints and grey literature."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (1-25).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_arxiv",
        "description": (
            "Search arXiv preprints by title or author. "
            "Useful for preprint_as_published and arxiv_version_mismatch hallucination types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "arXiv search query."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (1-20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]

# Map tool name → callable
TOOL_REGISTRY: dict[str, object] = {
    "resolve_doi": resolve_doi,
    "search_crossref": search_crossref,
    "search_openalex": search_openalex,
    "search_arxiv": search_arxiv,
    "verify_with_bibtex_updater": verify_with_bibtex_updater,
}


# Stand-alone definition for the BTU-only agentic variant.  Kept separate from
# TOOL_DEFINITIONS so the default four-tool suite remains bit-identical.
BTU_TOOL_DEFINITION: dict = {
    "name": "verify_with_bibtex_updater",
    "description": (
        "Verify a BibTeX entry using the bibtex-updater tool, which cross-references "
        "CrossRef, DBLP, and Semantic Scholar and returns a structured status "
        "(verified / not_found / title_mismatch / author_mismatch / etc.) along "
        "with mismatched fields and consulted APIs. Pass the full BibTeX entry as "
        "a single string. You may call this multiple times, but each call has a "
        "non-trivial latency budget."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "bibtex": {
                "type": "string",
                "description": "Full BibTeX entry to verify (e.g. @inproceedings{key, ...}).",
            }
        },
        "required": ["bibtex"],
    },
}
