#!/usr/bin/env python3
"""Fresh re-run of bibtex-updater 1.2.0 (CLI: bibtex-check) on dev+test, plus the
DB-first cascade (Stage-1 btu 1.2.0 + Stage-2 OpenRouter-Sonnet) on dev+test+stress.

This is a NEW dated snapshot, not a reproduction of the 2026-05-04 published
0.10.0 numbers. The standalone btu run (DOI/metadata resolution) is NOT
LLM-drift-prone; the cascade Stage-2 (Sonnet via OpenRouter) IS — its output is
a dated snapshot.

Outputs (overwrite):
  data/v1.0/baseline_results/bibtexupdater_{dev,test}_public.json   (aggregate)
  data/v1.0/baseline_results/cascade_db_diagnosis_{dev,test}_public_+_stress.json
  results/relabel_delta/btu_v1_2_0/                                 (per-entry preds + raw)

Checkpointing:
  - bibtex-check uses a persistent --cache-file per split so a re-run resumes.
  - per-entry predictions are streamed to JSONL so partial progress survives.

Usage:
  source /tmp/.s2_env; source /tmp/.or_env
  uv run python scripts/regen_btu_v1_2_0.py --stage btu      # standalone btu only
  uv run python scripts/regen_btu_v1_2_0.py --stage cascade  # cascade only
  uv run python scripts/regen_btu_v1_2_0.py --stage all
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hallmark.baselines.bibtexupdater import (  # noqa: E402
    STATUS_TO_CONFIDENCE,
    STATUS_TO_LABEL,
    _parse_jsonl_output,
    parse_jsonl_to_raw,
)
from hallmark.baselines.common import entries_to_bib, run_with_prescreening  # noqa: E402
from hallmark.dataset.loader import load_split  # noqa: E402
from hallmark.dataset.schema import Prediction  # noqa: E402
from hallmark.evaluation.metrics import evaluate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("regen_btu")

RESULTS_DIR = REPO / "data/v1.0/baseline_results"
DELTA_DIR = REPO / "results/relabel_delta/btu_v1_2_0"
DELTA_DIR.mkdir(parents=True, exist_ok=True)

BTU_VERSION = "1.2.0"
ENDPOINT_BTU = "bibtex-check CLI (crossref/openalex/semanticscholar/arxiv, --academic-only)"
ENDPOINT_STAGE2 = "openrouter:anthropic/claude-sonnet-4.6"

RATE_LIMIT = 120
WORKERS = 8


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_bibtex_check_cached(blind_entries: list, split: str) -> Path:
    """Run bibtex-check 1.2.0 once on the full split with a persistent cache.

    Returns the path to the produced JSONL. Re-running resumes from the cache
    (cache hits skip API calls), so this is the checkpoint mechanism.
    """
    import os

    work = DELTA_DIR / split
    work.mkdir(parents=True, exist_ok=True)
    bib_path = work / "input.bib"
    jsonl_path = work / "btu_raw.jsonl"
    cache_path = work / "btu_cache.json"

    bib_path.write_text(entries_to_bib(blind_entries))

    cmd = [
        "bibtex-check",
        str(bib_path),
        "--jsonl",
        str(jsonl_path),
        "--rate-limit",
        str(RATE_LIMIT),
        "--workers",
        str(WORKERS),
        "--academic-only",
        "--cache-file",
        str(cache_path),
    ]
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        cmd.extend(["--s2-api-key", s2_key])

    redacted = [
        ("***" if prev == "--s2-api-key" else c) for prev, c in zip(["", *cmd], cmd, strict=False)
    ]
    logger.info("[%s] running: %s", split, " ".join(redacted))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    logger.info("[%s] bibtex-check exit=%s elapsed=%.1fs", split, proc.returncode, elapsed)
    if proc.returncode not in (0, 2, 4):
        logger.error("[%s] stderr tail: %s", split, proc.stderr[-2000:])
    if not jsonl_path.exists():
        raise RuntimeError(f"[{split}] bibtex-check produced no JSONL output")
    return jsonl_path


def build_btu_predictions(blind_entries: list, jsonl_path: Path) -> list[Prediction]:
    """Merge raw bibtex-check JSONL with prescreening + backfill (the wrapper path).

    We parse the already-produced JSONL (no re-invocation) and run the standard
    prescreening/backfill merge so the standalone btu aggregate matches the
    released wrapper semantics.
    """
    elapsed_total = 0.0  # wall-clock is recorded by the runner; not needed for labels
    raw_preds = _parse_jsonl_output(jsonl_path, elapsed_total, len(blind_entries))
    pred_by_key = {p.bibtex_key: p for p in raw_preds}

    def _run_tool(tool_entries: list) -> list[Prediction]:
        # Tool already ran; return cached predictions for the requested subset.
        return [pred_by_key[e.bibtex_key] for e in tool_entries if e.bibtex_key in pred_by_key]

    return run_with_prescreening(
        blind_entries,
        _run_tool,
        skip_prescreening=False,
        backfill_reason="Entry not in bibtex-check output",
    )


def persist_per_entry(
    path: Path,
    entries: list,
    predictions: list[Prediction],
    raw_records: dict | None = None,
) -> None:
    """Stream per-entry gold+prediction (+raw status) to JSONL for reproducibility."""
    pred_by_key = {p.bibtex_key: p for p in predictions}
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
            if raw_records is not None:
                rr = raw_records.get(e.bibtex_key)
                if rr is not None:
                    row["btu_status"] = rr.get("status")
                    row["btu_abstained"] = rr.get("abstained")
                    row["btu_confidence"] = rr.get("confidence")
                    row["btu_mismatched_fields"] = rr.get("mismatched_fields")
                    row["btu_api_sources"] = rr.get("api_sources")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote per-entry predictions: %s (%d rows)", path, len(entries))


def add_provenance(result_dict: dict, *, kind: str, splits: list[str]) -> dict:
    result_dict["_provenance"] = {
        "snapshot_date": datetime.date.today().isoformat(),
        "generated_at_utc": _now(),
        "btu_version": BTU_VERSION,
        "endpoint_btu": ENDPOINT_BTU,
        "endpoint_stage2": ENDPOINT_STAGE2 if kind == "cascade" else None,
        "note": (
            "FRESH dated snapshot, NOT a reproduction of the 2026-05-04 published "
            "0.10.0 numbers. Standalone btu (DOI/metadata) is not LLM-drift-prone; "
            "cascade Stage-2 (Sonnet via OpenRouter) IS — treat as a dated snapshot."
        ),
        "kind": kind,
        "splits": splits,
    }
    return result_dict


def run_btu_standalone(split: str) -> dict:
    entries = load_split(split=split, version="v1.0")
    blind = [e.to_blind() for e in entries]
    logger.info("[%s] loaded %d entries", split, len(entries))

    jsonl_path = run_bibtex_check_cached(blind, split)
    raw_records = parse_jsonl_to_raw(jsonl_path)
    preds = build_btu_predictions(blind, jsonl_path)

    # status histogram for the report
    hist: dict[str, int] = {}
    for rr in raw_records.values():
        hist[rr.get("status", "?")] = hist.get(rr.get("status", "?"), 0) + 1
    logger.info("[%s] btu 1.2.0 status histogram: %s", split, dict(sorted(hist.items())))

    result = evaluate(
        entries=entries,
        predictions=preds,
        tool_name="bibtexupdater",
        split_name=split,
        compute_ci=False,
    )
    out = json.loads(result.to_json())
    out = add_provenance(out, kind="btu_standalone", splits=[split])
    out["_btu_status_histogram"] = dict(sorted(hist.items()))

    agg_path = RESULTS_DIR / f"bibtexupdater_{split}.json"
    agg_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    logger.info(
        "[%s] btu standalone: DR=%.4f FPR=%.4f F1=%.4f TWF1=%.4f MCC=%s ECE=%.4f -> %s",
        split,
        out["detection_rate"],
        out["false_positive_rate"],
        out["f1_hallucination"],
        out["tier_weighted_f1"],
        out.get("mcc"),
        out.get("ece", 0.0),
        agg_path.name,
    )

    persist_per_entry(
        DELTA_DIR / f"bibtexupdater_{split}_per_entry.jsonl",
        entries,
        preds,
        raw_records=raw_records,
    )
    # keep the raw JSONL alongside for full reproducibility
    (DELTA_DIR / f"bibtexupdater_raw_{split}.jsonl").write_text(jsonl_path.read_text())
    return out


def run_cascade_split(split: str, stage2_baseline: str) -> dict:
    from hallmark.baselines.cascade import run_cascade

    entries = load_split(split=split, version="v1.0")
    blind = [e.to_blind() for e in entries]
    logger.info("[%s] cascade: %d entries (Stage-2=%s)", split, len(entries), stage2_baseline)

    work = DELTA_DIR / split
    work.mkdir(parents=True, exist_ok=True)
    cache_path = work / "btu_cache.json"

    preds = run_cascade(
        blind,
        stage2_baseline=stage2_baseline,
        aggressive=False,
        # Stage-1 kwargs forwarded to run_bibtex_check_with_status:
        rate_limit=RATE_LIMIT,
        academic_only=True,
        extra_args=["--workers", str(WORKERS), "--cache-file", str(cache_path)],
    )

    result = evaluate(
        entries=entries,
        predictions=preds,
        tool_name="cascade_db_diagnosis",
        split_name=split,
        compute_ci=False,
    )
    out = json.loads(result.to_json())
    out = add_provenance(out, kind="cascade", splits=[split])

    agg_path = RESULTS_DIR / f"cascade_db_diagnosis_{split}.json"
    agg_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    logger.info(
        "[%s] cascade: DR=%.4f FPR=%.4f F1=%.4f TWF1=%.4f MCC=%s AUROC=%s T3F1=%s ECE=%.4f -> %s",
        split,
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
    ap.add_argument("--stage", choices=["btu", "cascade", "all"], default="all")
    ap.add_argument(
        "--splits-btu", default="dev_public,test_public", help="splits for standalone btu"
    )
    ap.add_argument(
        "--splits-cascade",
        default="dev_public,test_public,stress_test",
        help="splits for the cascade",
    )
    ap.add_argument(
        "--stage2-baseline",
        default="llm_agentic_openrouter_claude_sonnet_4_6",
        help="cascade Stage-2 diagnoser (released config: OpenRouter Sonnet 4.6)",
    )
    args = ap.parse_args()

    if args.stage in ("btu", "all"):
        for sp in [s.strip() for s in args.splits_btu.split(",") if s.strip()]:
            run_btu_standalone(sp)

    if args.stage in ("cascade", "all"):
        for sp in [s.strip() for s in args.splits_cascade.split(",") if s.strip()]:
            run_cascade_split(sp, args.stage2_baseline)


if __name__ == "__main__":
    # silence unused-import linters for re-exported maps used in docstring/debug
    _ = (STATUS_TO_LABEL, STATUS_TO_CONFIDENCE)
    main()
