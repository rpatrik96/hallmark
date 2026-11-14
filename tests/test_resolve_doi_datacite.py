"""resolve_doi falls back to DataCite when Crossref 404s.

arXiv DOIs (10.48550/arXiv.*) are registered with DataCite, not Crossref, so a
Crossref-only resolver reported every real arXiv DOI as "DOI not found" and the
agentic baselines read that as a fabricated DOI.
"""

from __future__ import annotations

import pytest

from hallmark.baselines import _agentic_tools


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


_DATACITE_ARXIV = {
    "data": {
        "attributes": {
            "creators": [
                {"name": "Shende, Yashas", "givenName": "Yashas", "familyName": "Shende"},
                {"name": "Some Consortium", "nameType": "Organizational"},
            ],
            "titles": [{"title": "Oscillators Are All You Need"}],
            "publisher": "arXiv",
            "publicationYear": 2026,
            "container": {},
        }
    }
}


@pytest.fixture
def calls(monkeypatch):
    import httpx

    seen: list[str] = []
    routes: dict[str, _Resp] = {}

    def fake_get(url, *a, **kw):
        seen.append(url)
        for prefix, resp in routes.items():
            if url.startswith(prefix):
                return resp
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(_agentic_tools, "_pace", lambda name: 0.0)
    return seen, routes


def test_crossref_404_falls_back_to_datacite(calls):
    seen, routes = calls
    routes["https://api.crossref.org/"] = _Resp(404)
    routes["https://api.datacite.org/"] = _Resp(200, _DATACITE_ARXIV)

    out = _agentic_tools.resolve_doi("https://doi.org/10.48550/arXiv.2602.12139")

    assert out["title"] == "Oscillators Are All You Need"
    assert out["authors"] == "Shende Yashas; Some Consortium"
    assert out["venue"] == "arXiv"
    assert out["year"] == "2026"
    assert out["doi"] == "10.48550/arXiv.2602.12139"
    assert [u.split("/")[2] for u in seen] == ["api.crossref.org", "api.datacite.org"]


def test_not_found_in_either_registry_still_raises(calls):
    _, routes = calls
    routes["https://api.crossref.org/"] = _Resp(404)
    routes["https://api.datacite.org/"] = _Resp(404)

    with pytest.raises(ValueError, match="DOI not found"):
        _agentic_tools.resolve_doi("10.9999/does-not-exist")


def test_crossref_hit_does_not_query_datacite(calls):
    seen, routes = calls
    routes["https://api.crossref.org/"] = _Resp(
        200, {"message": {"title": ["T"], "author": [], "published": {"date-parts": [[2020]]}}}
    )

    out = _agentic_tools.resolve_doi("10.1234/abc")

    assert out["title"] == "T"
    assert all("datacite" not in u for u in seen)


def test_datacite_fallback_is_paced(calls, monkeypatch):
    _, routes = calls
    routes["https://api.crossref.org/"] = _Resp(404)
    routes["https://api.datacite.org/"] = _Resp(200, _DATACITE_ARXIV)
    paced: list[str] = []
    monkeypatch.setattr(_agentic_tools, "_pace", lambda name: paced.append(name) or 0.0)

    _agentic_tools.resolve_doi("10.48550/arXiv.2602.12139")

    assert paced == ["crossref", "datacite"]
