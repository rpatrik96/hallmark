"""CheckIfExist baseline (port of Abbonato 2026, Algorithm 1).

A cascading 3-source verification baseline. Queries CrossRef first, falls
back to Semantic Scholar if no candidates are returned, then computes
weighted per-field similarities. When the best CrossRef score is low or any
issues are detected, fires fallback queries to Semantic Scholar and OpenAlex
and cross-validates authors across sources.

Notes
-----
* No LaTeX preprocessing — the input ``raw_text`` is already clean BibTeX.
* All similarities are on a 0-100 scale (matches the paper).
* Final ``confidence`` is rescaled to [0, 1] for the HALLMARK Prediction
  schema.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from rapidfuzz import fuzz

from hallmark.baselines._agentic_tools import (
    search_crossref,
    search_openalex,
    search_semantic_scholar,
)
from hallmark.dataset.schema import BlindEntry, HallucinationType, Prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _normalize_string(s: str) -> str:
    """Lowercase + strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _levenshtein_sim(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 100] after normalisation."""
    na, nb = _normalize_string(a), _normalize_string(b)
    if not na and not nb:
        return 100.0
    if not na or not nb:
        return 0.0
    return float(fuzz.ratio(na, nb))


def _extract_family_name(author: str) -> str:
    """Extract a normalised family-name token from an author string.

    Accepts ``"Family, Given"`` and ``"Given Family"`` forms, returns the
    normalised family token (lowercase, alphanumeric).
    """
    if not author:
        return ""
    a = author.strip()
    if "," in a:
        family = a.split(",", 1)[0]
    else:
        # Last whitespace token is the family name
        parts = a.split()
        family = parts[-1] if parts else ""
    return _normalize_string(family)


def _split_authors(authors: str) -> list[str]:
    """Split a multi-author field on common separators."""
    if not authors:
        return []
    parts = re.split(r"\s*(?:;|\band\b|,(?=\s*[A-Z]))\s*", authors)
    return [p.strip() for p in parts if p.strip()]


def _candidate_family_set(candidate: dict[str, str]) -> set[str]:
    """Set of normalised family names from a candidate's authors field."""
    return {
        fn for a in _split_authors(candidate.get("authors", "")) if (fn := _extract_family_name(a))
    }


def _detect_fake_authors(query_authors: str, candidate: dict[str, str]) -> list[str]:
    """Flag capitalised tokens in *query_authors* not corroborated by candidate.

    A query token is considered "fabricated" when it is a capitalised word
    (first letter uppercase, len >= 3) and does NOT appear in:
        (a) candidate title words,
        (b) candidate venue/journal name,
        (c) candidate year, or
        (d) any candidate author's family-name token.
    """
    if not query_authors:
        return []
    cand_title_norm = _normalize_string(candidate.get("title", ""))
    cand_venue_norm = _normalize_string(candidate.get("venue", ""))
    cand_year = (candidate.get("year") or "").strip()
    cand_families = _candidate_family_set(candidate)

    suspect: list[str] = []
    seen: set[str] = set()
    # Tokenise on whitespace/comma/semicolon, keep individual capitalised tokens
    for raw_token in re.split(r"[\s,;]+", query_authors):
        token = raw_token.strip(".")
        if len(token) < 3:
            continue
        if not token[0].isupper():
            continue
        norm = _normalize_string(token)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm in cand_families:
            continue
        if cand_title_norm and norm in cand_title_norm:
            continue
        if cand_venue_norm and norm in cand_venue_norm:
            continue
        if cand_year and norm == cand_year:
            continue
        suspect.append(token)
    return suspect


def _query_year(query: dict[str, str]) -> int | None:
    yr = (query.get("year") or "").strip()
    if not yr:
        return None
    try:
        return int(yr[:4])
    except ValueError:
        return None


def _candidate_year(candidate: dict[str, str]) -> int | None:
    yr = (candidate.get("year") or "").strip()
    if not yr:
        return None
    try:
        return int(yr[:4])
    except ValueError:
        return None


def _filter_by_cutoff(
    candidates: list[dict[str, str]], cutoff_year: int | None
) -> list[dict[str, str]]:
    """Drop candidates with year > cutoff_year. Keep candidates with no year."""
    if cutoff_year is None:
        return candidates
    out: list[dict[str, str]] = []
    for cand in candidates:
        cy = _candidate_year(cand)
        if cy is None or cy <= cutoff_year:
            out.append(cand)
    return out


def _score_candidate(query: dict[str, str], candidate: dict[str, str]) -> dict[str, float]:
    """Compute (S_title, S_author, S_journal, S_year) on a 0-100 scale.

    Uses the *first* author of the query and the *first* author of the
    candidate for S_author. S_journal is set to 100 when both sides lack a
    journal/venue (no information; do not penalise).
    """
    s_title = _levenshtein_sim(query.get("title", ""), candidate.get("title", ""))

    q_authors = _split_authors(query.get("author", ""))
    c_authors = _split_authors(candidate.get("authors", ""))
    if q_authors and c_authors:
        s_author = _levenshtein_sim(q_authors[0], c_authors[0])
    elif not q_authors and not c_authors:
        s_author = 100.0
    else:
        s_author = 0.0

    q_journal = (query.get("journal") or query.get("booktitle") or "").strip()
    c_journal = (candidate.get("venue") or "").strip()
    if q_journal and c_journal:
        s_journal = _levenshtein_sim(q_journal, c_journal)
    elif not q_journal and not c_journal:
        s_journal = 100.0
    else:
        # Only one side has a journal: no contribution either way
        s_journal = 100.0

    qy = _query_year(query)
    cy = _candidate_year(candidate)
    s_year = 100.0 if qy is None or cy is None or abs(qy - cy) <= 1 else 0.0

    return {
        "title": s_title,
        "author": s_author,
        "journal": s_journal,
        "year": s_year,
    }


def _detect_issues(
    query: dict[str, str],
    candidate: dict[str, str],
    similarities: dict[str, float],
    fake_authors: list[str],
) -> set[str]:
    """Issue set used for penalties and fallback triggering."""
    issues: set[str] = set()
    if similarities["title"] < 80:
        issues.add("title_mismatch")
    if similarities["author"] < 90:
        issues.add("author_mismatch")
    q_journal = (query.get("journal") or query.get("booktitle") or "").strip()
    c_journal = (candidate.get("venue") or "").strip()
    if q_journal and c_journal and similarities["journal"] < 80:
        issues.add("journal_mismatch")
    if fake_authors:
        issues.add("fabricated_authors")
    return issues


def _compute_confidence(
    similarities: dict[str, float],
    issues: set[str],
    multi_source_bonus: float,
    fake_authors: list[str],
) -> float:
    """Final 0-100 confidence score per the paper (Equations 2-3 + penalties)."""
    s_t = similarities["title"]
    s_a = similarities["author"]
    s_j = similarities["journal"]
    s_y = similarities["year"]
    score = s_t - 0.5 * (100.0 - s_a) if s_t > 80 and s_a < 90 else (s_t + s_a + s_j + s_y) / 4.0
    score += multi_source_bonus

    if "title_mismatch" in issues:
        score -= 20.0
    if "author_mismatch" in issues:
        score -= 20.0
    if "journal_mismatch" in issues:
        score -= 15.0
    if fake_authors:
        score -= min(20.0, 10.0 * len(fake_authors))
    return score


def _evaluate_candidates(
    query: dict[str, str], candidates: list[dict[str, str]]
) -> tuple[dict[str, str] | None, dict[str, float], set[str], list[str]]:
    """Score every candidate and return (best, similarities, issues, fake_authors)."""
    best: dict[str, str] | None = None
    best_sims: dict[str, float] = {"title": 0.0, "author": 0.0, "journal": 0.0, "year": 0.0}
    best_issues: set[str] = set()
    best_fake: list[str] = []
    best_pre_score = -1.0
    query_authors = query.get("author", "")

    for cand in candidates:
        sims = _score_candidate(query, cand)
        fake = _detect_fake_authors(query_authors, cand)
        issues = _detect_issues(query, cand, sims, fake)
        # Pre-bonus, pre-penalty proxy used only for ranking candidates
        pre_score = (sims["title"] + sims["author"] + sims["journal"] + sims["year"]) / 4.0
        if pre_score > best_pre_score:
            best_pre_score = pre_score
            best = cand
            best_sims = sims
            best_issues = issues
            best_fake = fake
    return best, best_sims, best_issues, best_fake


def _classify_hallucination_type(
    issues: set[str], similarities: dict[str, float], fake_authors: list[str]
) -> str | None:
    """Map dominant issue to a HallucinationType value."""
    valid_types = {t.value for t in HallucinationType}

    def _ok(value: str) -> str | None:
        return value if value in valid_types else None

    if "fabricated_authors" in issues and fake_authors:
        return _ok(HallucinationType.PLACEHOLDER_AUTHORS.value)
    has_title = "title_mismatch" in issues
    has_author = "author_mismatch" in issues
    has_journal = "journal_mismatch" in issues
    if has_author and not has_title and similarities["title"] >= 80:
        return _ok(HallucinationType.AUTHOR_MISMATCH.value)
    if has_title and similarities["author"] >= 80:
        return _ok(HallucinationType.CHIMERIC_TITLE.value)
    if has_journal and not has_title and not has_author:
        return _ok(HallucinationType.WRONG_VENUE.value)
    return _ok(HallucinationType.PLAUSIBLE_FABRICATION.value)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_checkifexist(
    entries: list[BlindEntry],
    *,
    knowledge_cutoff_year: int | None = None,
    crossref_limit: int = 3,
    fallback_threshold: float = 70.0,
    multi_source_bonus: float = 10.0,
    valid_threshold: float = 70.0,
    hallucinated_threshold: float = 50.0,
    **kw: Any,
) -> list[Prediction]:
    """Run the CheckIfExist (Abbonato 2026 Alg. 1) baseline.

    Args:
        entries: Blind entries to verify.
        knowledge_cutoff_year: Drop candidates with year > cutoff (kept when
            year is missing).
        crossref_limit: Top-K CrossRef candidates to fetch per query.
        fallback_threshold: Best score below this triggers S2/OpenAlex fallback.
        multi_source_bonus: β_ms added when ≥2 sources confirm authors.
        valid_threshold: Final score > this maps to VALID.
        hallucinated_threshold: Final score ≤ this maps to HALLUCINATED.
        **kw: Forward-compatible kwargs (ignored).

    Returns:
        One Prediction per input entry.
    """
    del kw

    predictions: list[Prediction] = []

    for entry in entries:
        start = time.time()
        api_calls = 0
        sources_queried: list[str] = []

        title = entry.fields.get("title", "") or ""
        author = entry.fields.get("author", "") or ""
        year = entry.fields.get("year", "") or ""
        first_author = ""
        if author:
            authors_split = _split_authors(author)
            if authors_split:
                first_author = authors_split[0]
        # Build query string: title + first author + year
        parts = [p for p in [title, first_author, year] if p]
        query_text = " ".join(parts).strip()
        query: dict[str, str] = {
            "title": title,
            "author": author,
            "year": year,
            "journal": entry.fields.get("journal", "") or entry.fields.get("booktitle", "") or "",
        }

        if not query_text:
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="UNCERTAIN",
                    confidence=0.5,
                    reason="Empty query (no title/author/year)",
                    wall_clock_seconds=time.time() - start,
                    api_calls=0,
                )
            )
            continue

        # ---- Step 1: CrossRef ------------------------------------------
        crossref_candidates: list[dict[str, str]] = []
        sources_queried.append("crossref")
        api_calls += 1
        try:
            crossref_candidates = search_crossref(query_text, crossref_limit)
        except Exception as exc:
            logger.debug("CheckIfExist CrossRef error: %s", exc)
        crossref_candidates = _filter_by_cutoff(crossref_candidates, knowledge_cutoff_year)

        candidates = list(crossref_candidates)

        # ---- Step 2: S2 fallback when CrossRef returned nothing ---------
        s2_candidates: list[dict[str, str]] = []
        if not candidates:
            sources_queried.append("semantic_scholar")
            api_calls += 1
            try:
                s2_candidates = search_semantic_scholar(query_text, 5)
            except Exception as exc:
                logger.debug("CheckIfExist S2 (initial fallback) error: %s", exc)
            s2_candidates = _filter_by_cutoff(s2_candidates, knowledge_cutoff_year)
            candidates = list(s2_candidates)

        if not candidates:
            elapsed = time.time() - start
            predictions.append(
                Prediction(
                    bibtex_key=entry.bibtex_key,
                    label="HALLUCINATED",
                    confidence=0.7,
                    reason=(
                        f"No candidates returned by {sources_queried}; treating as fabricated."
                    ),
                    api_sources_queried=sources_queried,
                    wall_clock_seconds=elapsed,
                    api_calls=api_calls,
                    predicted_hallucination_type=HallucinationType.PLAUSIBLE_FABRICATION.value,
                )
            )
            continue

        # ---- Step 3: Score candidates ----------------------------------
        best, sims, issues, fake_authors = _evaluate_candidates(query, candidates)
        # Pre-bonus best_score for fallback gate
        pre_score = _compute_confidence(sims, issues, 0.0, fake_authors)

        # ---- Step 4: Cross-source author intersection -------------------
        bonus = 0.0
        confirmed_sources = ["crossref"] if best in crossref_candidates else ["semantic_scholar"]
        if pre_score < fallback_threshold or issues:
            # Run S2 (if not already) + OpenAlex
            if "semantic_scholar" not in sources_queried:
                sources_queried.append("semantic_scholar")
                api_calls += 1
                try:
                    s2_candidates = search_semantic_scholar(query_text, 5)
                except Exception as exc:
                    logger.debug("CheckIfExist S2 fallback error: %s", exc)
                s2_candidates = _filter_by_cutoff(s2_candidates, knowledge_cutoff_year)

            sources_queried.append("openalex")
            api_calls += 1
            openalex_candidates: list[dict[str, str]] = []
            try:
                openalex_candidates = search_openalex(query_text, 5)
            except Exception as exc:
                logger.debug("CheckIfExist OpenAlex error: %s", exc)
            openalex_candidates = _filter_by_cutoff(openalex_candidates, knowledge_cutoff_year)

            crossref_families: set[str] = set()
            for c in crossref_candidates:
                crossref_families |= _candidate_family_set(c)
            s2_families: set[str] = set()
            for c in s2_candidates:
                s2_families |= _candidate_family_set(c)
            openalex_families: set[str] = set()
            for c in openalex_candidates:
                openalex_families |= _candidate_family_set(c)

            non_empty = [
                fams for fams in (crossref_families, s2_families, openalex_families) if fams
            ]
            if len(non_empty) >= 2:
                confirmed = set.intersection(*non_empty) if non_empty else set()
            else:
                confirmed = set()

            if len(confirmed) >= 2:
                bonus = multi_source_bonus
                confirmed_sources = []
                if crossref_families:
                    confirmed_sources.append("crossref")
                if s2_families:
                    confirmed_sources.append("semantic_scholar")
                if openalex_families:
                    confirmed_sources.append("openalex")

        final_score = _compute_confidence(sims, issues, bonus, fake_authors)

        # ---- Step 5: Verdict + label mapping ---------------------------
        label: Literal["VALID", "HALLUCINATED", "UNCERTAIN"]
        if final_score > valid_threshold:
            label = "VALID"
        elif final_score <= hallucinated_threshold:
            label = "HALLUCINATED"
        else:
            label = "UNCERTAIN"

        confidence_clamped = max(0.0, min(1.0, final_score / 100.0))

        predicted_type: str | None = None
        if label == "HALLUCINATED":
            predicted_type = _classify_hallucination_type(issues, sims, fake_authors)

        elapsed = time.time() - start
        reason_parts = [
            f"score={final_score:.1f}",
            f"S_title={sims['title']:.0f}",
            f"S_author={sims['author']:.0f}",
            f"S_journal={sims['journal']:.0f}",
            f"S_year={sims['year']:.0f}",
        ]
        if issues:
            reason_parts.append(f"issues={sorted(issues)}")
        if fake_authors:
            reason_parts.append(f"fake_authors={fake_authors}")
        if bonus:
            reason_parts.append(f"bonus={bonus}")

        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label=label,
                confidence=confidence_clamped,
                reason="; ".join(reason_parts),
                subtest_results={
                    "title_exists": sims["title"] >= 80,
                    "authors_match": sims["author"] >= 90,
                    "venue_correct": sims["journal"] >= 80,
                    "cross_db_agreement": bool(bonus),
                },
                api_sources_queried=confirmed_sources or sources_queried,
                wall_clock_seconds=elapsed,
                api_calls=api_calls,
                predicted_hallucination_type=predicted_type,
            )
        )

    return predictions
