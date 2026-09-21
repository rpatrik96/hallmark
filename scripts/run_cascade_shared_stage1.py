#!/usr/bin/env python3
"""Cascade with a Stage 1 shared across Stage-2 backbones.

Every cascade row in the paper should differ only in its Stage-2 backbone. The
GPT-5.1 and GPT-5.4 cascades each ran their own live Stage 1 (2026-08-06), and
the Sonnet 4.6 cascade replayed an older 2026-05-31 Stage-1 snapshot, so the
rows disagreed on which entries reached Stage 2 at all (reviewer feedback:
"cascade unfair comparison").

This script pins Stage 1 to one reference run. It reads the reference cascade's
per-entry predictions, keeps its ``stage1_db`` / ``prescreening`` verdicts
verbatim, and sends exactly its ``stage2_diagnosis`` bucket to the requested
Stage-2 backbone. With ``--reuse-stage2`` an existing cascade's Stage-2 verdicts
are copied for keys it already diagnosed, so only the missing keys cost API calls
(used to align GPT-5.4 to the GPT-5.1 Stage 1).

Stage 2 keeps the harness defaults (``MAX_TOOL_CALLS = 5``), identical to the
original runs. Its tool lookups are live, as in every agentic run.

Usage:
    # 5-entry smoke test, nothing scored
    uv run python scripts/run_cascade_shared_stage1.py --splits dev_public \\
        --stage2-baseline llm_agentic_openrouter_claude_sonnet_4_6 \\
        --out-dir results/cascade_sonnet46_shared --prefix sonnet46 --limit 5

    # full run
    uv run python scripts/run_cascade_shared_stage1.py \\
        --splits dev_public,test_public,stress_test \\
        --stage2-baseline llm_agentic_openrouter_claude_sonnet_4_6 \\
        --out-dir results/cascade_sonnet46_shared --prefix sonnet46
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

from hallmark.baselines._http_cache import http_cache
from hallmark.baselines.concurrency import parallel_run_baseline
from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STAGE1 = "results/cascade_gpt51/gpt51"
STAGE1_STAGES = {"stage1_db", "prescreening"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cascade_shared_stage1")


def load_preds(path: Path) -> dict[str, Prediction]:
    preds = {}
    for line in path.read_text().splitlines():
        if line.strip():
            p = Prediction.from_dict(json.loads(line))
            preds[p.bibtex_key] = p
    return preds


def tag_stage2(p: Prediction, stage2_baseline: str) -> Prediction:
    """Wrap a raw Stage-2 prediction exactly as ``cascade.run_cascade`` does."""
    return Prediction(
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


def run_split(split: str, args: argparse.Namespace) -> None:
    entries = load_split(split)
    ref = load_preds(REPO / f"{args.stage1_from}_{split}_preds.jsonl")
    missing = {e.bibtex_key for e in entries} - ref.keys()
    if missing:
        raise SystemExit(f"[{split}] reference Stage 1 lacks {len(missing)} entries")

    decided = {k: p for k, p in ref.items() if p.cascade_stage in STAGE1_STAGES}
    deferred = [e for e in entries if e.bibtex_key not in decided]

    reused: dict[str, Prediction] = {}
    if args.reuse_stage2:
        prior = load_preds(REPO / f"{args.reuse_stage2}_{split}_preds.jsonl")
        reused = {
            e.bibtex_key: prior[e.bibtex_key]
            for e in deferred
            if e.bibtex_key in prior and prior[e.bibtex_key].cascade_stage == "stage2_diagnosis"
        }
    to_run = [e.to_blind() for e in deferred if e.bibtex_key not in reused]
    if args.limit:
        to_run = to_run[: args.limit]
    logger.info(
        "[%s] Stage 1 decided=%d  Stage 2 bucket=%d  reused=%d  to run=%d",
        split,
        len(decided),
        len(deferred),
        len(reused),
        len(to_run),
    )

    out_dir = REPO / args.out_dir / ("smoke" if args.limit else "")
    ckpt = out_dir / split / "stage2"
    with http_cache(out_dir / "tools.sqlite"):
        fresh = parallel_run_baseline(
            args.stage2_baseline,
            to_run,
            workers=args.workers,
            checkpoint_dir=ckpt,
            split=split,
        )
    fresh_by_key = {p.bibtex_key: tag_stage2(p, args.stage2_baseline) for p in fresh}

    if args.limit:
        for p in fresh_by_key.values():
            logger.info("  %s %s %.2f %s", p.bibtex_key, p.label, p.confidence, p.reason[:160])
        return

    final: list[Prediction] = []
    for e in entries:
        k = e.bibtex_key
        if k in decided:
            final.append(decided[k])
        elif k in reused:
            final.append(reused[k])
        elif k in fresh_by_key:
            final.append(fresh_by_key[k])
        else:
            raise SystemExit(f"[{split}] no Stage 2 verdict for {k}; re-run to resume")

    with (out_dir / f"{args.prefix}_{split}_preds.jsonl").open("w") as f:
        for p in final:
            f.write(p.to_json() + "\n")

    both = evaluate(
        entries=entries,
        predictions=final,
        tool_name="cascade_db_diagnosis",
        split_name=split,
        eval_mode="both",
    )
    out = {mode: res.to_dict() for mode, res in both.items()}
    out["_provenance"] = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage1_source": f"{args.stage1_from}_{split}_preds.jsonl",
        "stage2_baseline": args.stage2_baseline,
        "stage2_reused_from": args.reuse_stage2,
        "stage1_decided": len(decided),
        "stage2_bucket": len(deferred),
        "stage2_reused": len(reused),
        "stage2_fresh": len(fresh_by_key),
    }
    (out_dir / f"{args.prefix}_{split}.json").write_text(json.dumps(out))
    for mode in ("conservative", "aggressive"):
        r = both[mode]
        # stress_test has no scored VALID entries, so its FPR is None
        fpr = "n/a" if r.false_positive_rate is None else f"{r.false_positive_rate:.3f}"
        logger.info(
            "[%s] %-12s DR=%.3f FPR=%s F1=%.3f unc=%d",
            split,
            mode,
            r.detection_rate,
            fpr,
            r.f1_hallucination,
            r.num_uncertain,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default="dev_public,test_public,stress_test")
    ap.add_argument("--stage2-baseline", required=True)
    ap.add_argument("--stage1-from", default=DEFAULT_STAGE1)
    ap.add_argument("--reuse-stage2", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: run N entries, no scoring")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    for split in args.splits.split(","):
        run_split(split.strip(), args)


if __name__ == "__main__":
    main()
