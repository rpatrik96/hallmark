"""Unit tests for the CheckIfExist baseline (port of Abbonato 2026 Alg. 1).

All API calls are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hallmark.baselines.checkifexist import (
    _compute_confidence,
    _detect_fake_authors,
    _evaluate_candidates,
    _extract_family_name,
    _levenshtein_sim,
    _normalize_string,
    _score_candidate,
    run_checkifexist,
)
from hallmark.dataset.schema import BlindEntry


def _make_entry(
    key: str = "test2024",
    title: str = "Quantum Cryptography Without Entanglement",
    author: str = "Smith, John and Doe, Jane",
    year: str = "2024",
    journal: str = "Journal of Cryptography",
) -> BlindEntry:
    fields = {"title": title, "author": author, "year": year, "journal": journal}
    return BlindEntry(
        bibtex_key=key,
        bibtex_type="article",
        fields=fields,
        raw_bibtex=f"@article{{{key}, title={{{title}}} }}",
    )


def _candidate(
    title: str = "",
    authors: str = "",
    venue: str = "",
    year: str = "2024",
) -> dict[str, str]:
    return {
        "authors": authors,
        "title": title,
        "venue": venue,
        "year": year,
        "doi": "",
    }


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalize_string(self) -> None:
        assert _normalize_string("Hello, World!") == "helloworld"
        assert _normalize_string("") == ""

    def test_levenshtein_sim_identical_is_100(self) -> None:
        assert _levenshtein_sim("abc", "abc") == pytest.approx(100.0)

    def test_levenshtein_sim_empty_both_returns_100(self) -> None:
        assert _levenshtein_sim("", "") == 100.0

    def test_levenshtein_sim_one_empty_returns_zero(self) -> None:
        assert _levenshtein_sim("abc", "") == 0.0

    def test_extract_family_name_comma_format(self) -> None:
        assert _extract_family_name("Smith, John") == "smith"

    def test_extract_family_name_space_format(self) -> None:
        assert _extract_family_name("John Smith") == "smith"

    def test_extract_family_name_empty(self) -> None:
        assert _extract_family_name("") == ""


class TestFakeAuthorDetection:
    def test_flags_capitalized_token_not_in_candidate(self) -> None:
        cand = _candidate(
            title="Optimisation Methods",
            authors="Brown, Alan",
            venue="ICML",
        )
        suspect = _detect_fake_authors("Brown, Alan and Fakeperson, Made", cand)
        assert "Fakeperson" in suspect or "Made" in suspect
        # Real author family is not flagged
        assert "Brown" not in suspect

    def test_real_author_token_corroborated_by_title(self) -> None:
        cand = _candidate(
            title="Smith John Conjecture in Topology",
            authors="",
            venue="",
        )
        # All capitalised query tokens appear in title — none flagged
        assert _detect_fake_authors("Smith, John", cand) == []

    def test_short_lowercase_tokens_ignored(self) -> None:
        cand = _candidate(authors="Brown, Alan", title="X", venue="Y")
        # "and", "the" etc. (lowercase, short) must not be flagged
        assert _detect_fake_authors("the and a", cand) == []


class TestCandidateScoring:
    def test_perfect_match_scores_high(self) -> None:
        query = {
            "title": "Quantum Computing",
            "author": "Smith, John",
            "year": "2024",
            "journal": "Nature",
        }
        cand = _candidate(
            title="Quantum Computing",
            authors="Smith, John",
            venue="Nature",
            year="2024",
        )
        sims = _score_candidate(query, cand)
        assert sims["title"] == pytest.approx(100.0)
        assert sims["author"] == pytest.approx(100.0)
        assert sims["journal"] == pytest.approx(100.0)
        assert sims["year"] == 100.0

    def test_year_within_one_is_match(self) -> None:
        query = {"title": "X", "author": "A", "year": "2023", "journal": ""}
        cand = _candidate(title="X", authors="A", year="2024")
        sims = _score_candidate(query, cand)
        assert sims["year"] == 100.0

    def test_year_off_by_two_is_zero(self) -> None:
        query = {"title": "X", "author": "A", "year": "2020", "journal": ""}
        cand = _candidate(title="X", authors="A", year="2024")
        sims = _score_candidate(query, cand)
        assert sims["year"] == 0.0


class TestConfidence:
    def test_asymmetric_formula_when_title_high_author_low(self) -> None:
        sims = {"title": 95.0, "author": 30.0, "journal": 100.0, "year": 100.0}
        # Asymmetric path: 95 - 0.5 * (100 - 30) = 95 - 35 = 60.0
        # Then -20 author_mismatch penalty (since S_author < 90) → 40
        score = _compute_confidence(sims, {"author_mismatch"}, 0.0, [])
        assert score == pytest.approx(40.0)

    def test_average_formula_when_no_asymmetry(self) -> None:
        sims = {"title": 80.0, "author": 95.0, "journal": 80.0, "year": 100.0}
        # Average path used because S_title NOT > 80 OR S_author NOT < 90
        # → (80 + 95 + 80 + 100) / 4 = 88.75
        score = _compute_confidence(sims, set(), 0.0, [])
        assert score == pytest.approx(88.75)

    def test_penalties_subtract_correctly(self) -> None:
        sims = {"title": 100.0, "author": 100.0, "journal": 100.0, "year": 100.0}
        score = _compute_confidence(
            sims,
            {"title_mismatch", "author_mismatch", "journal_mismatch"},
            0.0,
            [],
        )
        # 100 (avg) - 20 (title) - 20 (author) - 15 (journal) = 45
        assert score == pytest.approx(45.0)

    def test_fake_author_penalty_capped(self) -> None:
        sims = {"title": 100.0, "author": 100.0, "journal": 100.0, "year": 100.0}
        # 5 fake authors → would be 50 raw, capped at 20
        score = _compute_confidence(sims, set(), 0.0, ["A", "B", "C", "D", "E"])
        assert score == pytest.approx(80.0)

    def test_multi_source_bonus_added(self) -> None:
        sims = {"title": 80.0, "author": 95.0, "journal": 80.0, "year": 100.0}
        score = _compute_confidence(sims, set(), 10.0, [])
        assert score == pytest.approx(98.75)


class TestEvaluateCandidates:
    def test_picks_best_candidate(self) -> None:
        query = {
            "title": "The Real Paper",
            "author": "Smith, John",
            "year": "2024",
            "journal": "",
        }
        bad = _candidate(title="Wrong Title", authors="Other, Person", year="2024")
        good = _candidate(title="The Real Paper", authors="Smith, John", year="2024")
        best, sims, _, _ = _evaluate_candidates(query, [bad, good])
        assert best is good
        assert sims["title"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


class TestRunCheckIfExist:
    def test_exact_match_yields_valid(self) -> None:
        entry = _make_entry()

        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                return_value=[
                    _candidate(
                        title="Quantum Cryptography Without Entanglement",
                        authors="Smith, John",
                        venue="Journal of Cryptography",
                        year="2024",
                    )
                ],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        assert preds[0].label == "VALID"
        assert preds[0].confidence > 0.7
        assert "crossref" in preds[0].api_sources_queried

    def test_no_candidates_yields_hallucinated(self) -> None:
        entry = _make_entry(title="A Wholly Made-Up Title For Testing")

        with (
            patch("hallmark.baselines.checkifexist.search_crossref", return_value=[]),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        assert preds[0].label == "HALLUCINATED"
        assert preds[0].predicted_hallucination_type == "plausible_fabrication"

    def test_network_error_in_crossref_falls_back_to_s2(self) -> None:
        entry = _make_entry(title="Robust Federated Optimisation")

        def crossref_fails(q: str, limit: int) -> list[dict[str, str]]:
            raise RuntimeError("CrossRef outage")

        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                side_effect=crossref_fails,
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[
                    _candidate(
                        title="Robust Federated Optimisation",
                        authors="Smith, John",
                        venue="Journal of Cryptography",
                        year="2024",
                    )
                ],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        assert preds[0].label == "VALID"

    def test_knowledge_cutoff_filters_post_cutoff(self) -> None:
        entry = _make_entry(title="A Future Paper")

        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                return_value=[
                    _candidate(
                        title="A Future Paper",
                        authors="Smith, John",
                        year="2099",
                    )
                ],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry], knowledge_cutoff_year=2024)

        # All candidates were post-cutoff; treated as no candidates
        assert preds[0].label == "HALLUCINATED"

    def test_cross_source_intersection_adds_bonus(self) -> None:
        # Title close to candidate but author mismatch triggers fallback
        entry = _make_entry(
            title="Quantum Cryptography",
            author="MadeUp, Person",
        )
        common = "Smith, John"

        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                return_value=[
                    _candidate(title="Quantum Cryptography", authors=common, year="2024")
                ],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[
                    _candidate(title="Quantum Cryptography", authors=common, year="2024")
                ],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_openalex",
                return_value=[
                    _candidate(title="Quantum Cryptography", authors=common, year="2024")
                ],
            ),
        ):
            preds = run_checkifexist([entry])

        # cross_db_agreement subtest should be True when bonus applied
        assert preds[0].subtest_results.get("cross_db_agreement") is True
        assert "openalex" in preds[0].api_sources_queried

    def test_fake_author_heuristic_triggers_fabricated_authors_classification(self) -> None:
        # Real candidate, query introduces a clearly-fabricated capitalised name
        entry = _make_entry(
            title="Optimisation Methods For Deep Learning",
            author="Brown, Alan and Fakeperson, Madeup",
            journal="ICML",
        )
        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                return_value=[
                    _candidate(
                        title="Optimisation Methods For Deep Learning",
                        authors="Brown, Alan",
                        venue="ICML",
                        year="2024",
                    )
                ],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        # Fake author triggers issues which can flip to HALLUCINATED with the
        # placeholder_authors classification.
        assert "fake_authors" in preds[0].reason or "fabricated_authors" in preds[0].reason
        if preds[0].label == "HALLUCINATED":
            assert preds[0].predicted_hallucination_type == "placeholder_authors"

    def test_asymmetric_confidence_when_title_high_author_low(self) -> None:
        # Title matches strongly, author entirely different → asymmetric branch
        entry = _make_entry(
            title="A Very Distinctive Paper Title",
            author="ZZZZZ, ZZZ",
            year="2024",
            journal="",
        )

        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                return_value=[
                    _candidate(
                        title="A Very Distinctive Paper Title",
                        authors="Smith, John",
                        venue="",
                        year="2024",
                    )
                ],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        # The asymmetric formula plus author_mismatch penalty + mismatch issues
        # should usually flip the label away from VALID.
        assert preds[0].label in {"HALLUCINATED", "UNCERTAIN"}

    def test_penalty_system_reduces_score(self) -> None:
        # Construct a query that matches title but everything else fails.
        entry = _make_entry(
            title="Original Paper Title XYZ",
            author="Smith, John",
            year="2010",
            journal="Nature",
        )

        # Candidate has matching title only — author/journal/year all wrong
        bad_candidate = _candidate(
            title="Original Paper Title XYZ",
            authors="Other, Person",
            venue="Wrong Journal Name",
            year="2024",
        )
        with (
            patch(
                "hallmark.baselines.checkifexist.search_crossref",
                return_value=[bad_candidate],
            ),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        # Penalties should drop confidence well below 1.0.
        assert preds[0].confidence < 0.7

    def test_predicted_hallucination_type_is_valid_enum_value(self) -> None:
        from hallmark.dataset.schema import HallucinationType

        entry = _make_entry(title="No-Match Paper For Testing")
        with (
            patch("hallmark.baselines.checkifexist.search_crossref", return_value=[]),
            patch(
                "hallmark.baselines.checkifexist.search_semantic_scholar",
                return_value=[],
            ),
            patch("hallmark.baselines.checkifexist.search_openalex", return_value=[]),
        ):
            preds = run_checkifexist([entry])

        assert preds[0].label == "HALLUCINATED"
        valid_values = {t.value for t in HallucinationType}
        assert preds[0].predicted_hallucination_type in valid_values

    def test_empty_query_returns_uncertain(self) -> None:
        entry = BlindEntry(
            bibtex_key="empty",
            bibtex_type="misc",
            fields={},
            raw_bibtex="@misc{empty}",
        )
        preds = run_checkifexist([entry])
        assert preds[0].label == "UNCERTAIN"
