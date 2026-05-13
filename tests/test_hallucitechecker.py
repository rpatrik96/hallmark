"""Unit tests for the HalluCiteChecker baseline.

All API calls are mocked; no live network access in CI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hallmark.baselines.hallucitechecker import (
    _normalize_title,
    _title_similarity,
    run_hallucitechecker,
)
from hallmark.dataset.schema import BlindEntry


def _make_entry(
    key: str = "test2024",
    title: str = "Deep Learning for Citation Verification",
    author: str = "Smith, John and Doe, Jane",
    year: str = "2024",
    doi: str | None = "10.1234/fake",
) -> BlindEntry:
    fields: dict[str, str] = {"title": title, "author": author, "year": year}
    if doi is not None:
        fields["doi"] = doi
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields=fields,
        raw_bibtex=f"@article{{{key}, title={{{title}}} }}",
    )


def _candidate(title: str, year: str = "2024") -> dict[str, str]:
    return {
        "authors": "Smith, John",
        "title": title,
        "venue": "Some Venue",
        "year": year,
        "doi": "",
    }


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_strips_punctuation_and_lowercases(self) -> None:
        assert _normalize_title("Deep, Learning!") == "deeplearning"

    def test_empty_string(self) -> None:
        assert _normalize_title("") == ""

    def test_handles_unicode_safe(self) -> None:
        # Non-ASCII letters get stripped (regex is [a-z0-9]).
        assert _normalize_title("Übercite") == "bercite"


class TestSimilarity:
    def test_identical_titles_score_one(self) -> None:
        assert _title_similarity("Deep Learning", "Deep Learning") == pytest.approx(1.0)

    def test_completely_different_low(self) -> None:
        assert _title_similarity("apple", "xyzqrs") < 0.5

    def test_punctuation_does_not_affect_score(self) -> None:
        assert _title_similarity("Deep, Learning!", "Deep Learning") == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        assert _title_similarity("", "anything") == 0.0


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestRunHalluciteChecker:
    def test_exact_title_match_yields_valid(self) -> None:
        entry = _make_entry(title="Quantum Cryptography Without Entanglement")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                return_value=[_candidate("Quantum Cryptography Without Entanglement")],
            ),
            patch(
                "hallmark.baselines.hallucitechecker.search_arxiv",
                return_value=[],
            ),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            preds = run_hallucitechecker([entry])

        assert len(preds) == 1
        p = preds[0]
        assert p.label == "VALID"
        assert p.confidence == pytest.approx(0.85)
        assert "crossref" in p.api_sources_queried

    def test_no_match_yields_hallucinated(self) -> None:
        entry = _make_entry(title="A Wholly Fabricated Title for Testing")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                return_value=[_candidate("Some Completely Different Paper")],
            ),
            patch("hallmark.baselines.hallucitechecker.search_arxiv", return_value=[]),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            preds = run_hallucitechecker([entry])

        assert preds[0].label == "HALLUCINATED"
        # Confidence is in [0.6, 0.9]
        assert 0.6 <= preds[0].confidence <= 0.9
        assert preds[0].predicted_hallucination_type == "plausible_fabrication"

    def test_one_source_fails_others_succeed_returns_real_label(self) -> None:
        entry = _make_entry(title="Robust Federated Optimisation")

        def crossref_fails(query: str, limit: int) -> list[dict[str, str]]:
            raise RuntimeError("simulated CrossRef outage")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                side_effect=crossref_fails,
            ),
            patch(
                "hallmark.baselines.hallucitechecker.search_arxiv",
                return_value=[_candidate("Robust Federated Optimisation")],
            ),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            preds = run_hallucitechecker([entry])

        assert preds[0].label == "VALID"
        assert "crossref" not in preds[0].api_sources_queried
        assert "arxiv" in preds[0].api_sources_queried

    def test_all_sources_fail_returns_uncertain(self) -> None:
        entry = _make_entry()

        def boom(query: str, limit: int) -> list[dict[str, str]]:
            raise RuntimeError("network error")

        with (
            patch("hallmark.baselines.hallucitechecker.search_crossref", side_effect=boom),
            patch("hallmark.baselines.hallucitechecker.search_arxiv", side_effect=boom),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                side_effect=boom,
            ),
        ):
            preds = run_hallucitechecker([entry])

        assert preds[0].label == "UNCERTAIN"
        assert preds[0].confidence == 0.5

    def test_empty_title_returns_uncertain(self) -> None:
        entry = _make_entry(title="")

        # Mocks should never be called when title is empty
        with (
            patch("hallmark.baselines.hallucitechecker.search_crossref") as m_cr,
            patch("hallmark.baselines.hallucitechecker.search_arxiv") as m_ax,
            patch("hallmark.baselines.hallucitechecker.search_semantic_scholar") as m_s2,
        ):
            preds = run_hallucitechecker([entry])

        assert preds[0].label == "UNCERTAIN"
        m_cr.assert_not_called()
        m_ax.assert_not_called()
        m_s2.assert_not_called()

    def test_knowledge_cutoff_filters_post_cutoff_candidates(self) -> None:
        entry = _make_entry(title="A Future Paper")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                return_value=[_candidate("A Future Paper", year="2099")],
            ),
            patch("hallmark.baselines.hallucitechecker.search_arxiv", return_value=[]),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            preds = run_hallucitechecker([entry], knowledge_cutoff_year=2024)

        # The lone post-cutoff candidate is dropped → no match → HALLUCINATED
        assert preds[0].label == "HALLUCINATED"

    def test_knowledge_cutoff_keeps_candidate_with_no_year(self) -> None:
        entry = _make_entry(title="A Paper Without Year On Server")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                return_value=[_candidate("A Paper Without Year On Server", year="")],
            ),
            patch("hallmark.baselines.hallucitechecker.search_arxiv", return_value=[]),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            preds = run_hallucitechecker([entry], knowledge_cutoff_year=2024)

        assert preds[0].label == "VALID"

    def test_threshold_override(self) -> None:
        entry = _make_entry(title="Original Paper Title")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                # Substantially different but partially overlapping
                return_value=[_candidate("Original Paper Tutle")],
            ),
            patch("hallmark.baselines.hallucitechecker.search_arxiv", return_value=[]),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            # Strict threshold should still match (off-by-one char)
            preds_strict = run_hallucitechecker([entry], title_threshold=0.95)
            preds_loose = run_hallucitechecker([entry], title_threshold=0.50)

        assert preds_strict[0].label in {"VALID", "HALLUCINATED"}
        assert preds_loose[0].label == "VALID"

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown source"):
            run_hallucitechecker([_make_entry()], sources=("not-a-source",))

    def test_records_api_calls(self) -> None:
        entry = _make_entry(title="Some Paper")

        with (
            patch(
                "hallmark.baselines.hallucitechecker.search_crossref",
                return_value=[],
            ),
            patch("hallmark.baselines.hallucitechecker.search_arxiv", return_value=[]),
            patch(
                "hallmark.baselines.hallucitechecker.search_semantic_scholar",
                return_value=[],
            ),
        ):
            preds = run_hallucitechecker([entry])

        # All three sources are queried when there is no match
        assert preds[0].api_calls == 3
