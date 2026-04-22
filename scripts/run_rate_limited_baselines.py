#!/usr/bin/env python3
"""Run rate-limited baselines (HaRC, verify-citations) with extended timeouts.

Reads `S2_API_KEY` from env (typically from /tmp/.s2_env). Writes evaluation
results to data/v1.0/baseline_results/ and updates manifest.json checksums,
matching the format of scripts/generate_reference_results.py.

Usage:
    source /tmp/.s2_env
    python scripts/run_rate_limited_baselines.py --baseline harc --split dev_public
    python scripts/run_rate_limited_baselines.py --baseline verify_citations --split dev_public
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.registry import run_baseline
from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import EvaluationResult
from hallmark.evaluation.metrics import evaluate
from hallmark.evaluation.validate import compute_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/v1.0/baseline_results")


def _env_meta() -> dict[str, str]:
    import datetime
    import platform
    from importlib.metadata import version

    try:
        hm_version = version("hallmark")
    except Exception:
        hm_version = "unknown"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hallmark_version": hm_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["harc", "verify_citations"])
    parser.add_argument("--split", default="dev_public")
    parser.add_argument("--version", default="v1.0")
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=43200.0,
        help="Total wall-clock seconds (default 12h)",
    )
    parser.add_argument("--batch-timeout", type=float, default=1800.0)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    s2_key = os.environ.get("S2_API_KEY")
    if not s2_key:
        logger.error("S2_API_KEY not set. Source /tmp/.s2_env first.")
        return 1

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s split v%s", args.split, args.version)
    entries = load_split(split=args.split, version=args.version)
    logger.info("Loaded %d entries", len(entries))

    runner_kwargs: dict[str, object] = {
        "api_key": s2_key,
        "batch_size": args.batch_size,
        "batch_timeout": args.batch_timeout,
        "total_timeout": args.total_timeout,
    }

    logger.info("Running %s with total_timeout=%.0fs", args.baseline, args.total_timeout)
    predictions = run_baseline(args.baseline, entries, split=args.split, **runner_kwargs)

    eval_result = evaluate(entries, predictions, tool_name=args.baseline, split_name=args.split)
    assert isinstance(eval_result, EvaluationResult)

    out_path = results_dir / f"{args.baseline}_{args.split}.json"
    out_path.write_text(json.dumps(eval_result.to_dict(), indent=2))
    logger.info("Wrote %s", out_path)

    # Update manifest with SHA-256
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "version": "1.0",
            "description": "Pre-computed baseline reference results.",
            "files": {},
        }

    rel_path = str(out_path.relative_to(results_dir.parent.parent))
    manifest["files"][rel_path] = {
        "sha256": compute_sha256(out_path),
        "baseline": args.baseline,
        "split": args.split,
        "environment": _env_meta(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    def _fmt(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

    coverage = eval_result.coverage if hasattr(eval_result, "coverage") else None
    logger.info(
        "%s on %s: DR=%s FPR=%s F1=%s Coverage=%s",
        args.baseline,
        args.split,
        _fmt(eval_result.detection_rate),
        _fmt(eval_result.false_positive_rate),
        _fmt(eval_result.f1_hallucination),
        _fmt(coverage),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
