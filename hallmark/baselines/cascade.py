"""DB-first cascade baseline with hallucination-mode diagnosis.

Stage 1: ``bibtexupdater`` (CrossRef / DBLP / Semantic Scholar lookup).
- ``verified`` and similar → emit VALID with high confidence.
- Definite mismatch statuses (``doi_not_found``, ``venue_mismatch``, etc.) →
  emit HALLUCINATED with a status-derived ``predicted_hallucination_type``.
- Ambiguous statuses (``not_found``, ``partial_match``, ``api_error``,
  ``skipped``, ``missing``) → defer to Stage 2.

Stage 2: a configurable LLM diagnoser (default ``llm_agentic_anthropic``) is
asked to decide VALID vs HALLUCINATED and, when HALLUCINATED, to classify the
mode against the 14-type taxonomy.

Aggressive mode: any entry that remains UNCERTAIN after Stage 2 is forced to
HALLUCINATED with type ``plausible_fabrication`` and confidence 0.55. This
implements the "treat DB lookups as gold standard" stance — at the cost of
inflated FPR on legitimately-but-not-yet-indexed entries, which the dual-mode
evaluation surfaces as the DB-indexing-lag tax.
"""

from __future__ import annotations

import logging
from typing import Any

from hallmark.baselines.bibtexupdater import run_bibtex_check_with_status
from hallmark.dataset.schema import BlindEntry, Prediction

logger = logging.getLogger(__name__)

# bibtex-check status → predicted hallucination type (used by Stage 1).
STATUS_TO_TYPE: dict[str, str] = {
    "doi_not_found": "fabricated_doi",
    "future_date": "future_date",
    "invalid_year": "future_date",
    "venue_mismatch": "wrong_venue",
    "author_mismatch": "swapped_authors",
    "year_mismatch": "hybrid_fabrication",
    "title_mismatch": "chimeric_title",
    "preprint_only": "preprint_as_published",
    "url_not_found": "fabricated_doi",
    "url_content_mismatch": "near_miss_title",
    "book_not_found": "plausible_fabrication",
    "working_paper_not_found": "plausible_fabrication",
    # bibtex-updater >=1.2.0 statuses with positive problem evidence
    "arxiv_id_mismatch": "hybrid_fabrication",  # cited arXiv ID → different paper
    "doi_mismatch": "hybrid_fabrication",  # cited DOI → different paper
    "given_name_substitution": "swapped_authors",  # co-author given name swapped
    "title_near_miss": "near_miss_title",  # --strict near-miss title
    "author_truncated": "partial_author_list",  # --strict silent truncation
    # not_found / unconfirmed / partial_match / api_error / skipped / missing → Stage 2
}

# Statuses that Stage 1 treats as definitive VALID.
STAGE1_VERIFIED: set[str] = {
    "verified",
    "published_version_exists",
    "url_verified",
    "url_accessible",
    "book_verified",
    "working_paper_verified",
}

# Statuses routed to Stage 2 for diagnosis.
ROUTE_TO_STAGE2: set[str] = {
    "not_found",
    "partial_match",
    "api_error",
    "skipped",
    "missing",
    # bibtex-updater >=1.2.0 abstention statuses (could-not-verify)
    "unconfirmed",
    "strict_warn_preprint_year",
    "strict_warn_cnv",
}


def _stage1_predict(
    entry: BlindEntry,
    raw_pred: Prediction,
    status: str,
) -> Prediction | None:
    """Decide Stage 1 verdict for one entry, or return None to defer to Stage 2.

    Pre-screening overrides (``status == "prescreening_override"``) are honored
    as Stage 1 decisions — the local heuristic already produced a confident
    verdict and we trust it.
    """
    if status == "prescreening_override":
        return Prediction(
            bibtex_key=raw_pred.bibtex_key,
            label=raw_pred.label,
            confidence=raw_pred.confidence,
            reason=f"[Stage 1: prescreening] {raw_pred.reason}",
            subtest_results=dict(raw_pred.subtest_results),
            api_sources_queried=list(raw_pred.api_sources_queried),
            wall_clock_seconds=raw_pred.wall_clock_seconds,
            api_calls=raw_pred.api_calls,
            source="prescreening",
            predicted_hallucination_type=None,
            cascade_stage="prescreening",
        )

    if status in STAGE1_VERIFIED:
        return Prediction(
            bibtex_key=raw_pred.bibtex_key,
            label="VALID",
            confidence=max(raw_pred.confidence, 0.85),
            reason=f"[Stage 1: bibtex-check {status}] {raw_pred.reason}",
            subtest_results=dict(raw_pred.subtest_results),
            api_sources_queried=list(raw_pred.api_sources_queried),
            wall_clock_seconds=raw_pred.wall_clock_seconds,
            api_calls=raw_pred.api_calls,
            source="tool",
            predicted_hallucination_type=None,
            cascade_stage="stage1_db",
        )

    if status in STATUS_TO_TYPE:
        return Prediction(
            bibtex_key=raw_pred.bibtex_key,
            label="HALLUCINATED",
            confidence=max(raw_pred.confidence, 0.85),
            reason=f"[Stage 1: bibtex-check {status}] {raw_pred.reason}",
            subtest_results=dict(raw_pred.subtest_results),
            api_sources_queried=list(raw_pred.api_sources_queried),
            wall_clock_seconds=raw_pred.wall_clock_seconds,
            api_calls=raw_pred.api_calls,
            source="tool",
            predicted_hallucination_type=STATUS_TO_TYPE[status],
            cascade_stage="stage1_db",
        )

    if status in ROUTE_TO_STAGE2:
        return None

    logger.debug(
        "cascade: unmapped status %r for %s — routing to Stage 2", status, raw_pred.bibtex_key
    )
    return None


def _aggressive_fallback(pred: Prediction) -> Prediction:
    """Force any remaining UNCERTAIN/VALID to HALLUCINATED@0.55 in aggressive mode.

    Only applied to predictions that came out of Stage 2 still UNCERTAIN, OR
    Stage-2 VALID verdicts that the diagnoser couldn't confidently confirm.
    Conservative-mode predictions are passed through unchanged.

    The aggressive policy is "if no DB or diagnoser confidently asserted real,
    treat as fabricated"; we type the residual as ``plausible_fabrication``
    since that is the catch-all for "looks plausible but unverifiable".
    """
    if pred.label == "HALLUCINATED":
        return pred
    if pred.label == "VALID" and pred.confidence >= 0.7:
        # Stage 2 made a confident VALID call — respect it.
        return pred
    return Prediction(
        bibtex_key=pred.bibtex_key,
        label="HALLUCINATED",
        confidence=0.55,
        reason=f"[Aggressive: unverifiable] {pred.reason}",
        subtest_results=dict(pred.subtest_results),
        api_sources_queried=list(pred.api_sources_queried),
        wall_clock_seconds=pred.wall_clock_seconds,
        api_calls=pred.api_calls,
        source=pred.source,
        predicted_hallucination_type="plausible_fabrication",
        cascade_stage=pred.cascade_stage,
    )


def run_cascade(
    entries: list[BlindEntry],
    *,
    stage2_baseline: str = "llm_agentic_anthropic",
    aggressive: bool = False,
    stage2_kwargs: dict[str, Any] | None = None,
    **stage1_kwargs: Any,
) -> list[Prediction]:
    """Run the DB-first cascade with hallucination-mode diagnosis.

    Args:
        entries: blind benchmark entries.
        stage2_baseline: name of the registered baseline used to diagnose
            entries Stage 1 couldn't conclusively classify. Default
            ``llm_agentic_anthropic`` for richest tool coverage.
        aggressive: if True, any entry still UNCERTAIN/low-confidence VALID
            after Stage 2 is forced to HALLUCINATED@0.55 with type
            ``plausible_fabrication``. This implements the "DB-as-gold-standard"
            stance from the reviewer feedback.
        stage2_kwargs: kwargs forwarded to the Stage 2 baseline runner.
        **stage1_kwargs: forwarded to ``run_bibtex_check_with_status``.
    """
    if not entries:
        return []

    stage1_preds, status_map = run_bibtex_check_with_status(entries, **stage1_kwargs)
    pred_by_key = {p.bibtex_key: p for p in stage1_preds}

    final: dict[str, Prediction] = {}
    deferred: list[BlindEntry] = []

    for entry in entries:
        key = entry.bibtex_key
        raw_pred = pred_by_key.get(key)
        status = status_map.get(key, "missing")
        if raw_pred is None:
            deferred.append(entry)
            continue

        verdict = _stage1_predict(entry, raw_pred, status)
        if verdict is None:
            deferred.append(entry)
        else:
            final[key] = verdict

    if deferred:
        stage2_preds = _run_stage2(deferred, stage2_baseline, stage2_kwargs or {})
        for p in stage2_preds:
            tagged = Prediction(
                bibtex_key=p.bibtex_key,
                label=p.label,
                confidence=p.confidence,
                reason=f"[Stage 2: {stage2_baseline}] {p.reason}",
                subtest_results=dict(p.subtest_results),
                api_sources_queried=list(p.api_sources_queried),
                wall_clock_seconds=p.wall_clock_seconds,
                api_calls=p.api_calls,
                source=p.source or "tool",
                predicted_hallucination_type=p.predicted_hallucination_type,
                cascade_stage="stage2_diagnosis",
            )
            final[p.bibtex_key] = tagged

    # Backfill anything Stage 2 didn't return (e.g., baseline crash) as conservative VALID
    # (or aggressive HALLUCINATED below).
    for entry in entries:
        if entry.bibtex_key not in final:
            final[entry.bibtex_key] = Prediction(
                bibtex_key=entry.bibtex_key,
                label="VALID",
                confidence=0.30,
                reason="[Cascade: no Stage 2 verdict — conservative backfill]",
                source="tool",
                cascade_stage="stage2_diagnosis",
            )

    if aggressive:
        for key, pred in list(final.items()):
            if pred.cascade_stage in {"stage2_diagnosis"}:
                final[key] = _aggressive_fallback(pred)

    return [final[entry.bibtex_key] for entry in entries]


def _run_stage2(
    entries: list[BlindEntry],
    stage2_baseline: str,
    stage2_kwargs: dict[str, Any],
) -> list[Prediction]:
    """Dispatch to a registered Stage 2 baseline, calling its runner directly.

    We import the registry lazily and call the runner with the already-blind
    entries to avoid a second ``to_blind()`` conversion.
    """
    from hallmark.baselines.registry import _REGISTRY, check_available

    available, reason = check_available(stage2_baseline)
    if not available:
        logger.warning(
            "cascade Stage 2 baseline %r unavailable (%s) — fallback UNCERTAIN",
            stage2_baseline,
            reason,
        )
        return [
            Prediction(
                bibtex_key=e.bibtex_key,
                label="UNCERTAIN",
                confidence=0.5,
                reason=f"Stage 2 baseline {stage2_baseline} unavailable: {reason}",
                source="tool",
            )
            for e in entries
        ]

    info = _REGISTRY[stage2_baseline]
    merged = dict(info.runner_kwargs)
    merged.update(stage2_kwargs)
    return list(info.runner(entries, **merged))
