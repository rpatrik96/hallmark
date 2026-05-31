#!/usr/bin/env python3
"""Fast, deterministic cascade reconstruction for the btu 1.2.0 regeneration.

The live ``run_cascade`` re-invokes ``bibtex-check`` from scratch for every split
(twice, via the wrapper's prescreening double-call), and the arXiv DOI HEAD checks
for fabricated future-dated DOIs (e.g. ``2602.*``) time out repeatedly, making a
full live cascade run take many hours.

This script reuses the *already-persisted* Stage-1 raw JSONL
(``results/relabel_delta/btu_v1_2_0/bibtexupdater_raw_{split}.jsonl``) produced by
the standalone btu 1.2.0 stage, applies the *exact* ``cascade.py`` Stage-1 routing
logic (verified / definite-mismatch → decided; could-not-verify bucket → deferred),
and runs Stage-2 (Sonnet via OpenRouter) ONLY on the deferred bucket. This is
internally consistent with the standalone btu row (same Stage-1 statuses feed every
metric) and is honest as a dated snapshot — Stage-2 is LLM-drift-prone, Stage-1 is
not.

For ``stress_test`` (no persisted Stage-1 cache), the real ``run_cascade`` is used
but with the persistent SQLite cache so re-runs resume; it is small (121 entries).

Outputs (overwrite):
  data/v1.0/baseline_results/cascade_db_diagnosis_{dev_public,test_public,stress_test}.json
  results/relabel_delta/btu_v1_2_0/cascade_db_diagnosis_{split}_per_entry.jsonl

Usage:
  source /tmp/.s2_env; source /tmp/.or_env
  uv run python scripts/regen_btu_v1_2_0_cascade_fast.py --splits dev_public,test_public,stress_test
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hallmark.baselines.cascade import (  # noqa: E402
    _stage1_predict,
)
from hallmark.dataset.loader import load_split  # noqa: E402
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, Prediction  # noqa: E402
from hallmark.evaluation.metrics import evaluate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("regen_cascade_fast")

RESULTS_DIR = REPO / "data/v1.0/baseline_results"
DELTA_DIR = REPO / "results/relabel_delta/btu_v1_2_0"

BTU_VERSION = "1.2.0"
ENDPOINT_BTU = "bibtex-check CLI (crossref/openalex/semanticscholar/arxiv, --academic-only)"
ENDPOINT_STAGE2 = "openrouter:anthropic/claude-sonnet-4.6"
RATE_LIMIT = 120
WORKERS = 8
STAGE2_WORKERS = 3  # ThreadPool fan-out for the agentic Sonnet Stage-2 diagnoser;
# kept low to avoid self-inflicted Semantic-Scholar / OpenAlex HTTP 429 on the shared IP.


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cached_stage1_with_status(
    entries: list[BlindEntry], raw_jsonl: Path
) -> tuple[list[Prediction], dict[str, str]]:
    """Replicate ``run_bibtex_check_with_status`` from a persisted raw JSONL.

    Reads the persisted standalone-stage per-entry JSONL — which already contains
    the FINAL post-prescreening labels (``pred_label``, ``source``, ``btu_status``)
    that ``run_bibtex_check_with_status`` produced. This avoids re-running the
    prescreening DOI HEAD checks (10s timeout per fabricated future-DOI), which is
    what made the live cascade run for hours. No subprocess / network calls.

    ``raw_jsonl`` points at ``bibtexupdater_raw_{split}.jsonl``; the per-entry file
    sits beside it as ``bibtexupdater_{split}_per_entry.jsonl``.
    """
    split = raw_jsonl.name.removeprefix("bibtexupdater_raw_").removesuffix(".jsonl")
    per_entry_path = raw_jsonl.parent / f"bibtexupdater_{split}_per_entry.jsonl"
    if not per_entry_path.exists():
        raise FileNotFoundError(f"missing persisted per-entry JSONL: {per_entry_path}")

    rows_by_key: dict[str, dict] = {}
    for line in per_entry_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows_by_key[r["bibtex_key"]] = r

    final_predictions: list[Prediction] = []
    status_dict: dict[str, str] = {}
    for e in entries:
        r = rows_by_key.get(e.bibtex_key)
        if r is None:
            # Not in the persisted standalone output at all → treat as missing
            # (conservative VALID backfill; routed to Stage-2 in the cascade).
            final_predictions.append(
                Prediction(
                    bibtex_key=e.bibtex_key,
                    label="VALID",
                    confidence=0.5,
                    reason="Entry not in persisted standalone output",
                    source="tool",
                )
            )
            status_dict[e.bibtex_key] = "missing"
            continue

        final_predictions.append(
            Prediction(
                bibtex_key=e.bibtex_key,
                label=r["pred_label"],
                confidence=float(c) if (c := r.get("pred_confidence")) is not None else 0.5,
                reason=r.get("reason") or "",
                api_sources_queried=r.get("btu_api_sources") or [],
                source=r.get("source") or "tool",
            )
        )
        # Reconstruct the wrapper's status vocabulary:
        # - source=="prescreening_override"  → "prescreening_override"
        # - btu_status is None (no tool record) → "missing"
        # - otherwise the raw btu status
        src = r.get("source")
        btu_status = r.get("btu_status")
        if src == "prescreening_override":
            status_dict[e.bibtex_key] = "prescreening_override"
        elif btu_status is None:
            status_dict[e.bibtex_key] = "missing"
        else:
            status_dict[e.bibtex_key] = btu_status

    return final_predictions, status_dict


def _run_stage2_parallel(
    deferred: list[BlindEntry], stage2_baseline: str, split: str
) -> list[Prediction]:
    """Stage-2 via the ThreadPool + checkpoint-resume wrapper (crash-resilient)."""
    from hallmark.baselines.concurrency import parallel_run_baseline
    from hallmark.baselines.registry import check_available

    available, reason = check_available(stage2_baseline)
    if not available:
        logger.warning(
            "Stage-2 baseline %r unavailable (%s) — UNCERTAIN fallback", stage2_baseline, reason
        )
        return [
            Prediction(bibtex_key=e.bibtex_key, label="UNCERTAIN", confidence=0.5, reason=reason)
            for e in deferred
        ]

    ckpt_dir = DELTA_DIR / split / "stage2_checkpoints"
    return parallel_run_baseline(
        stage2_baseline,
        deferred,
        workers=STAGE2_WORKERS,
        checkpoint_dir=ckpt_dir,
        split=split,
    )


def run_cascade_cached(
    entries: list[BlindEntry], raw_jsonl: Path, stage2_baseline: str, split: str
) -> list[Prediction]:
    """Cascade with Stage-1 served from the persisted raw JSONL (no re-run)."""
    stage1_preds, status_map = cached_stage1_with_status(entries, raw_jsonl)
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

    logger.info(
        "Stage-1 decided=%d  deferred-to-Stage-2=%d (of %d)",
        len(final),
        len(deferred),
        len(entries),
    )

    if deferred:
        stage2_preds = _run_stage2_parallel(deferred, stage2_baseline, split)
        for p in stage2_preds:
            final[p.bibtex_key] = Prediction(
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

    return [final[e.bibtex_key] for e in entries]


def run_cascade_live_stress(entries: list[BlindEntry], stage2_baseline: str) -> list[Prediction]:
    """Real cascade for stress_test (no persisted Stage-1) with a persistent cache."""
    from hallmark.baselines.cascade import run_cascade

    work = DELTA_DIR / "stress_test"
    work.mkdir(parents=True, exist_ok=True)
    cache_path = work / "btu_cache.json"
    return run_cascade(
        entries,
        stage2_baseline=stage2_baseline,
        aggressive=False,
        rate_limit=RATE_LIMIT,
        academic_only=True,
        extra_args=["--workers", str(WORKERS), "--cache-file", str(cache_path)],
    )


def add_provenance(d: dict, splits: list[str]) -> dict:
    d["_provenance"] = {
        "snapshot_date": datetime.date.today().isoformat(),
        "generated_at_utc": _now(),
        "btu_version": BTU_VERSION,
        "endpoint_btu": ENDPOINT_BTU,
        "endpoint_stage2": ENDPOINT_STAGE2,
        "note": (
            "FRESH dated snapshot, NOT a reproduction of the 2026-05-04 published "
            "0.10.0 numbers. Stage-1 btu (DOI/metadata) is not LLM-drift-prone and was "
            "served from the persisted standalone-stage raw JSONL; Stage-2 (Sonnet via "
            "OpenRouter) IS drift-prone — treat as a dated snapshot."
        ),
        "kind": "cascade",
        "splits": splits,
        "stage1_source": "cached_raw_jsonl" if splits[0] != "stress_test" else "live_run",
    }
    return d


def persist_per_entry(path: Path, entries: list[BenchmarkEntry], preds: list[Prediction]) -> None:
    pred_by_key = {p.bibtex_key: p for p in preds}
    with open(path, "w") as f:
        for e in entries:
            p = pred_by_key.get(e.bibtex_key)
            row = {
                "bibtex_key": e.bibtex_key,
                "gold_label": e.label,
                "gold_type": e.hallucination_type,
                "difficulty_tier": e.difficulty_tier,
                "pred_label": p.label if p else None,
                "pred_confidence": p.confidence if p else None,
                "pred_type": p.predicted_hallucination_type if p else None,
                "cascade_stage": getattr(p, "cascade_stage", None) if p else None,
                "source": getattr(p, "source", None) if p else None,
                "reason": p.reason if p else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote per-entry predictions: %s (%d rows)", path, len(entries))


def run_split(split: str, stage2_baseline: str) -> dict:
    entries = load_split(split=split, version="v1.0")
    blind = [e.to_blind() for e in entries]
    logger.info("[%s] cascade: %d entries (Stage-2=%s)", split, len(entries), stage2_baseline)
    t0 = time.time()

    if split == "stress_test":
        preds = run_cascade_live_stress(blind, stage2_baseline)
    else:
        raw_jsonl = DELTA_DIR / f"bibtexupdater_raw_{split}.jsonl"
        if not raw_jsonl.exists():
            raise FileNotFoundError(f"missing persisted Stage-1 raw JSONL: {raw_jsonl}")
        preds = run_cascade_cached(blind, raw_jsonl, stage2_baseline, split)

    result = evaluate(
        entries=entries,
        predictions=preds,
        tool_name="cascade_db_diagnosis",
        split_name=split,
        compute_ci=False,
    )
    out = json.loads(result.to_json())
    out = add_provenance(out, splits=[split])

    agg_path = RESULTS_DIR / f"cascade_db_diagnosis_{split}.json"
    agg_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    logger.info(
        "[%s] cascade DONE in %.1fs: DR=%.4f FPR=%.4f F1=%.4f TWF1=%.4f MCC=%s "
        "AUROC=%s T3F1=%s ECE=%.4f -> %s",
        split,
        time.time() - t0,
        out["detection_rate"],
        out["false_positive_rate"],
        out["f1_hallucination"],
        out["tier_weighted_f1"],
        out.get("mcc"),
        out.get("auroc"),
        out.get("tier3_f1"),
        out.get("ece", 0.0),
        agg_path.name,
    )

    persist_per_entry(
        DELTA_DIR / f"cascade_db_diagnosis_{split}_per_entry.jsonl",
        entries,
        preds,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev_public,test_public,stress_test")
    ap.add_argument(
        "--stage2-baseline",
        default="llm_agentic_openrouter_claude_sonnet_4_6",
    )
    args = ap.parse_args()
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        run_split(sp, args.stage2_baseline)


if __name__ == "__main__":
    main()
