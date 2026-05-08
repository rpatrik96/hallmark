"""Unit tests for the bioRxiv and PubMed API clients.

Live network calls are mocked. We only verify request/response plumbing here;
end-to-end scraping is covered by `scripts/scrape_crossdomain.py --smoke`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx

from hallmark.dataset.api_clients import BioRxivClient, PubMedClient
from scripts.scrape_crossdomain import (
    _strip_dblp_disambiguators,
    biorxiv_record_to_entry,
    pubmed_summary_to_entry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("GET", "https://example.test"),
    )


# ---------------------------------------------------------------------------
# BioRxivClient
# ---------------------------------------------------------------------------


def test_biorxiv_paginates_until_total_reached() -> None:
    """list_papers should paginate using the messages.total field, not a fixed page size."""
    page1 = {
        "messages": [{"status": "ok", "total": "5"}],
        "collection": [{"doi": f"10.1101/{i}", "title": f"P{i}"} for i in range(3)],
    }
    page2 = {
        "messages": [{"status": "ok", "total": "5"}],
        "collection": [{"doi": f"10.1101/{i}", "title": f"P{i}"} for i in range(3, 5)],
    }

    responses = [page1, page2]

    def fake_request(*args: Any, **kwargs: Any) -> httpx.Response:
        return _mock_response(responses.pop(0))

    client = BioRxivClient(rate_limit=0.0, timeout=1.0)
    with patch("hallmark.dataset.api_clients._request_with_retry", side_effect=fake_request):
        out = client.list_papers("2026-01-01", "2026-02-01", max_results=10)

    assert len(out) == 5, "should fetch all five papers across both pages"
    assert [r["doi"] for r in out] == [f"10.1101/{i}" for i in range(5)]


def test_biorxiv_record_to_entry_handles_semicolon_authors() -> None:
    rec = {
        "title": "A Test Paper.",
        "authors": "Smith, J.; Doe, J.",
        "date": "2026-04-15",
        "doi": "10.1101/2026.04.15.000001",
    }
    entry = biorxiv_record_to_entry(rec, server="biorxiv")
    assert entry is not None
    assert entry.label == "VALID"
    assert entry.fields["author"] == "J. Smith and J. Doe"
    assert entry.fields["title"] == "A Test Paper"
    assert entry.fields["doi"] == "10.1101/2026.04.15.000001"
    assert entry.fields["year"] == "2026"
    assert entry.publication_date == "2026-04-15"


def test_biorxiv_record_to_entry_drops_records_missing_required_fields() -> None:
    # Missing DOI
    assert biorxiv_record_to_entry({"title": "T", "authors": "A, B", "date": "2026-01-01"}) is None
    # Missing title
    assert biorxiv_record_to_entry({"authors": "A, B", "date": "2026-01-01", "doi": "x"}) is None


# ---------------------------------------------------------------------------
# PubMedClient
# ---------------------------------------------------------------------------


def test_pubmed_esearch_returns_pmids() -> None:
    payload = {"esearchresult": {"idlist": ["1", "2", "3"]}}
    client = PubMedClient(rate_limit=0.0)
    with patch(
        "hallmark.dataset.api_clients._request_with_retry",
        return_value=_mock_response(payload),
    ):
        out = client.esearch("test", retmax=3)
    assert out == ["1", "2", "3"]


def test_pubmed_summary_to_entry_extracts_doi_from_articleids() -> None:
    rec = {
        "uid": "12345",
        "title": "A Clinical Study.",
        "pubdate": "2025 Jul 12",
        "authors": [
            {"name": "Smith J", "authtype": "Author"},
            {"name": "Doe A", "authtype": "Author"},
        ],
        "fulljournalname": "Lancet",
        "articleids": [
            {"idtype": "pubmed", "value": "12345"},
            {"idtype": "doi", "value": "10.1016/example"},
        ],
    }
    entry = pubmed_summary_to_entry(rec)
    assert entry is not None
    assert entry.fields["doi"] == "10.1016/example"
    assert entry.fields["pmid"] == "12345"
    assert entry.fields["journal"] == "Lancet"
    assert entry.fields["year"] == "2025"
    # PubMed "Last F" -> "F Last"
    assert "J Smith" in entry.fields["author"]
    assert "A Doe" in entry.fields["author"]


def test_pubmed_summary_to_entry_drops_records_without_authors() -> None:
    rec = {"uid": "1", "title": "T", "pubdate": "2025", "authors": []}
    assert pubmed_summary_to_entry(rec) is None


# ---------------------------------------------------------------------------
# DBLP disambiguator stripper
# ---------------------------------------------------------------------------


def test_strip_dblp_disambiguators() -> None:
    raw = "Xinghua Zhang 0001 and Bowen Yu 0002 and Jane Doe"
    assert _strip_dblp_disambiguators(raw) == "Xinghua Zhang and Bowen Yu and Jane Doe"


def test_strip_dblp_disambiguators_leaves_year_in_titles_alone() -> None:
    # The stripper only removes whitespace-prefixed 4-digit tokens — it should
    # not touch the actual year field, which is stored separately.
    raw = "John Smith and Mary Jones"
    assert _strip_dblp_disambiguators(raw) == raw


# ---------------------------------------------------------------------------
# dblp_hit_to_entry collision avoidance
# ---------------------------------------------------------------------------


def test_dblp_hit_to_entry_keys_are_unique_for_same_surname_year_firstword() -> None:
    """Two different papers by the same surname in the same year with the same
    first title-word must get distinct bibtex_keys."""
    from hallmark.dataset.scraper import dblp_hit_to_entry

    hit_a = {
        "title": "On the Limits of Foundation Models",
        "authors": {"author": [{"text": "Yiming Li"}]},
        "year": "2024",
        "doi": "10.1/aaa",
        "type": "Conference and Workshop Papers",
    }
    hit_b = {
        "title": "On Generalization in Vision Transformers",
        "authors": {"author": [{"text": "Yiming Li"}]},
        "year": "2024",
        "doi": "10.1/bbb",
        "type": "Conference and Workshop Papers",
    }
    a = dblp_hit_to_entry(hit_a, venue_name="NeurIPS")
    b = dblp_hit_to_entry(hit_b, venue_name="NeurIPS")
    assert a is not None and b is not None
    assert a.bibtex_key != b.bibtex_key, (
        f"keys must differ for distinct papers, got {a.bibtex_key!r} == {b.bibtex_key!r}"
    )


def test_dblp_hit_to_entry_strips_disambiguator_in_key() -> None:
    """The bibtex_key must use the real surname, not the DBLP disambiguator suffix."""
    from hallmark.dataset.scraper import dblp_hit_to_entry

    hit = {
        "title": "A Study on Something",
        "authors": {"author": [{"text": "Wei Wang 0001"}]},
        "year": "2024",
        "doi": "10.1/abc",
        "type": "Conference and Workshop Papers",
    }
    e = dblp_hit_to_entry(hit, venue_name="NeurIPS")
    assert e is not None
    assert e.bibtex_key.startswith("Wang2024"), (
        f"key must start with real surname, got {e.bibtex_key!r}"
    )
    assert "0001" not in e.fields["author"]
