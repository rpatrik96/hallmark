"""DB-first cascade baseline with hallucination-mode diagnosis.

Stage 1: ``bibtexupdater`` (CrossRef / DBLP / Semantic Scholar lookup).
- ``verified`` and similar → emit VALID with high confidence.
- Definite mismatch statuses (``doi_not_found``, ``venue_mismatch``, etc.) →
  emit HALLUCINATED with a status-derived ``predicted_hallucination_type``.
- Ambiguous statuses (``not_found``, ``partial_match``, ``api_error``,
  ``network_error``, ``coverage_incomplete``, ``skipped``, ``missing``) → defer
  to Stage 2.
- An unmapped status also defers to Stage 2.  This is deliberate, not incidental:
  a status the wrapper has never seen carries no evidence either way, and the only
  safe direction to fail is towards a second opinion.

Stage 2: a configurable LLM diagnoser (default ``llm_agentic_anthropic``) is
asked to decide VALID vs HALLUCINATED and, when HALLUCINATED, to classify the
mode against the 14-type taxonomy.

Aggressive mode: any entry that remains UNCERTAIN after Stage 2 is forced to
HALLUCINATED with type ``plausible_fabrication`` and confidence 0.55. This
implements the "treat DB lookups as gold standard" stance — at the cost of
inflated FPR on legitimately-but-not-yet-indexed entries, which the dual-mode
evaluation surfaces as the DB-indexing-lag tax.  The promotion is suppressed on a
batch whose Stage 1 produced no database evidence: during an outage there is no
gold standard to treat as one, and promoting the residue would manufacture mass
fabrication verdicts out of failed requests.

Stage 1 statuses are also scored as a batch: ``run_cascade_with_health`` returns
the ``BatchHealth`` record for the bibtex-check invocation so a caller can refuse
to checkpoint a run whose Stage 1 produced no database evidence at all.  That is
the shape a transport outage takes, and without the check the cascade forwards
the whole batch to an expensive Stage 2 that has nothing to work from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hallmark.baselines.bibtexupdater import (
    BatchHealth,
    assess_batch_health,
    run_bibtex_check_with_status,
)
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
    "author_truncated": "partial_author_list",  # silent truncation (default mode post-1.2.0)
    # Post-1.2.0 positive-evidence statuses — decided problems, never Stage 2
    "nonexistent_venue": "nonexistent_venue",  # venue unknown to DBLP/OpenAlex registries
    "unpublished_at_claimed_venue": "preprint_as_published",  # real paper, not at cited venue
    # not_found / unconfirmed / partial_match / api_error / network_error /
    # coverage_incomplete / skipped / missing → Stage 2 (see ROUTE_TO_STAGE2)
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
    # ``not_found`` is meant to be an exhaustive miss — every source consulted
    # completed and returned nothing — but the cascade never treats it as
    # evidence of fabrication on its own, because older tool builds also emit it
    # for lookups that failed in transit. Uncertain either way: defer to Stage 2.
    "not_found",
    "partial_match",
    "api_error",
    "skipped",
    "missing",
    # Transport and coverage failures: the lookup never completed, so the entry
    # carries no evidence in either direction.  These must never reach
    # ``STATUS_TO_TYPE`` — a failed request is not a fabricated reference.
    "network_error",
    "coverage_incomplete",
    # bibtex-updater >=1.2.0 abstention statuses (could-not-verify)
    "unconfirmed",
    "strict_warn_preprint_year",
    "strict_warn_cnv",
}


# Unmapped statuses seen in this process, so a drifted tool vocabulary is
# reported once rather than once per entry.
_WARNED_UNMAPPED_STATUSES: set[str] = set()


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

    # Deliberate open-world default: a status the wrapper has never seen is
    # routed to Stage 2 rather than mapped to a verdict here.  Failing towards a
    # second opinion is the only safe direction — the alternative is inventing
    # evidence from an unknown string.  An unmapped status also means the tool's
    # vocabulary has drifted ahead of ``STATUS_TO_TYPE`` / ``ROUTE_TO_STAGE2``,
    # so the first sighting is a WARNING; the rest stay at DEBUG so a whole batch
    # of them cannot drown out the batch-health warning.
    if status not in _WARNED_UNMAPPED_STATUSES:
        _WARNED_UNMAPPED_STATUSES.add(status)
        logger.warning(
            "cascade: unmapped bibtex-check status %r (first seen on %s) — routing "
            "to Stage 2, never HALLUCINATED; add it to the status maps",
            status,
            raw_pred.bibtex_key,
        )
    else:
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

    The function itself is unchanged and unconditional; whether it runs at all is
    decided in ``run_cascade_with_health``, which skips the promotion entirely on
    a batch flagged by ``BatchHealth.suspected_transport_failure``.
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
        # A promoted fallback is still a fallback: without this an all-fallback
        # run becomes detection rate 1.0 with every entry marked evaluated.
        evaluated=pred.evaluated,
    )


def run_cascade(
    entries: list[BlindEntry],
    *,
    stage2_baseline: str = "llm_agentic_anthropic",
    aggressive: bool = False,
    stage2_kwargs: dict[str, Any] | None = None,
    **stage1_kwargs: Any,
) -> list[Prediction]:
    """Run the DB-first cascade and return predictions only.

    Thin wrapper over ``run_cascade_with_health`` for callers that do not persist
    results; see that function for the arguments and for the batch-health signal
    a checkpointing caller needs.
    """
    predictions, _ = run_cascade_with_health(
        entries,
        stage2_baseline=stage2_baseline,
        aggressive=aggressive,
        stage2_kwargs=stage2_kwargs,
        **stage1_kwargs,
    )
    return predictions


def run_cascade_with_health(
    entries: list[BlindEntry],
    *,
    stage2_baseline: str = "llm_agentic_anthropic",
    aggressive: bool = False,
    stage2_kwargs: dict[str, Any] | None = None,
    **stage1_kwargs: Any,
) -> tuple[list[Prediction], BatchHealth]:
    """Run the DB-first cascade with hallucination-mode diagnosis.

    Args:
        entries: blind benchmark entries.
        stage2_baseline: name of the registered baseline used to diagnose
            entries Stage 1 couldn't conclusively classify. Default
            ``llm_agentic_anthropic`` for richest tool coverage.
        aggressive: if True, any entry still UNCERTAIN/low-confidence VALID
            after Stage 2 is forced to HALLUCINATED@0.55 with type
            ``plausible_fabrication``. This implements the "DB-as-gold-standard"
            stance from the reviewer feedback. The promotion is skipped when
            ``health.suspected_transport_failure`` is set, since a batch with no
            database evidence has no gold standard to stand on.
        stage2_kwargs: kwargs forwarded to the Stage 2 baseline runner.
        **stage1_kwargs: forwarded to ``run_bibtex_check_with_status``.

    Returns:
        A 2-tuple ``(predictions, health)``. ``health`` scores the Stage 1 status
        map: when ``health.suspected_transport_failure`` is True the Stage 1
        lookups produced no database evidence, every Stage 2 verdict in the batch
        rests on nothing, and the caller must not checkpoint the results.

    Note on the published results
    -----------------------------
    The aggressive-mode numbers in the paper were produced before this guard
    existed, and they are not recomputed by it: they are frozen files in
    ``data/v1.2/baseline_results/`` that CI validates by checksum rather than
    re-running.  The guard cannot alter them.  It is also inert on a healthy
    batch by construction: ``suspected_transport_failure`` is False whenever the
    no-evidence share stays under the threshold, and the no-evidence statuses are
    a subset of what defers to Stage 2, so the frozen conservative runs'
    ``cascade_breakdown_stats`` bound the share from above.  For dev_public
    (266/1119 = 23.8% deferred) and test_public (222/831 = 26.7%) the bound is
    below the 30% threshold, so those splits' aggressive numbers cannot change on
    a re-run.

    stress_test defers 53/121 = 43.8%, which is an upper bound only: deferrals
    also include ``unconfirmed``, ``partial_match``, ``skipped`` and ``missing``,
    none of which count toward the no-evidence share.  The raw status dump needed
    to compute the actual share is not in the repo:
    ``bibtexupdater_raw_dev_public.jsonl`` is an unfetched git-lfs pointer and
    there is no stress_test raw dump at all.  A future re-run of the aggressive
    stress_test split is therefore the one place this guard could in principle
    produce different numbers; the frozen file is untouched regardless.
    """
    if not entries:
        return [], assess_batch_health([])

    # The harness injects ``checkpoint_dir`` for baselines that support
    # per-entry resume. Stage 1 (bibtex-check subprocess) has no checkpoint
    # support and would silently swallow it via ``**_kw`` — route it to the
    # Stage 2 LLM baseline instead, under a subdirectory so the top-level
    # resume scan (which globs ``checkpoint_dir/*.jsonl``, non-recursive)
    # does not skip entries before they re-enter the cascade for Stage 1
    # verdicts and stage tagging. On resume, Stage 1 re-runs (cheap,
    # disk-cached by the tool) and Stage 2 returns checkpointed predictions
    # without new LLM calls.
    checkpoint_dir = stage1_kwargs.pop("checkpoint_dir", None)

    stage1_preds, status_map = run_bibtex_check_with_status(entries, **stage1_kwargs)
    pred_by_key = {p.bibtex_key: p for p in stage1_preds}

    # Computed from the status map rather than inside the wrapper so the signal
    # survives whichever Stage 1 path produced it.
    health = assess_batch_health(status_map.values())
    if health.suspected_transport_failure:
        logger.warning(
            "cascade Stage 1 produced no database evidence for %d/%d entries — "
            "every Stage 2 verdict in this batch rests on nothing. Do not "
            "checkpoint these results.",
            health.no_evidence,
            health.total,
        )

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
        merged_stage2: dict[str, Any] = dict(stage2_kwargs or {})
        if checkpoint_dir is not None:
            merged_stage2.setdefault("checkpoint_dir", Path(checkpoint_dir) / "stage2")
        stage2_preds = _run_stage2(deferred, stage2_baseline, merged_stage2)
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
        if health.suspected_transport_failure:
            # "DB-as-gold-standard" needs a DB. With no database evidence behind
            # the batch, promoting the residue to HALLUCINATED would turn an
            # outage into mass fabrication verdicts, so the entries stay as
            # Stage 2 returned them. Inert on a healthy batch: the flag is False
            # for every run that is not degraded, so aggressive mode there is
            # byte-identical to what it was before this guard existed.
            logger.warning(
                "cascade: aggressive promotion suppressed — Stage 1 produced no "
                "database evidence for %d/%d entries, so there is no gold standard "
                "to treat as one. Stage 2 verdicts are returned unpromoted; do not "
                "checkpoint this batch.",
                health.no_evidence,
                health.total,
            )
        else:
            for key, pred in list(final.items()):
                if pred.cascade_stage in {"stage2_diagnosis"}:
                    final[key] = _aggressive_fallback(pred)

    return [final[entry.bibtex_key] for entry in entries], health


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
