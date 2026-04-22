#!/usr/bin/env python3
"""Run an agentic LLM baseline with checkpointing.

Handles:
- llm_agentic_openai / llm_agentic_anthropic (multi-tool tool-use)
- llm_agentic_btu_openai / llm_agentic_btu_anthropic (BTU-only tool-use)

Reads ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY`` from env. Writes results to
``data/v1.0/baseline_results/{baseline}_{split}.json`` and updates
``manifest.json`` checksums, matching other reference result entries.
Per-entry JSONL checkpoint lives in ``results/temporal_checkpoints/`` and
is used to resume partially-completed runs.

Usage:
    source /tmp/.openai_env
    python scripts/run_agentic_baseline.py \\
        --baseline llm_agentic_openai --split dev_public
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.registry import run_baseline
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate
from hallmark.evaluation.validate import compute_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/v1.0/baseline_results")
DEFAULT_CHECKPOINT_DIR = Path("results/temporal_checkpoints")
AGENTIC_BASELINES = {
    "llm_agentic_openai",
    "llm_agentic_anthropic",
    "llm_agentic_btu_openai",
    "llm_agentic_btu_anthropic",
}


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
    parser.add_argument("--baseline", required=True, choices=sorted(AGENTIC_BASELINES))
    parser.add_argument("--split", default="dev_public")
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-db-path", type=Path, default=Path(".cache/agentic_tools.sqlite"))
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="If >0, run on the first N entries only (for smoke-testing).",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s split v%s", args.split, args.version)
    entries = load_split(split=args.split, version=args.version)
    if args.max_entries and args.max_entries > 0:
        entries = entries[: args.max_entries]
    logger.info("Running %s on %d entries", args.baseline, len(entries))

    predictions = run_baseline(
        args.baseline,
        entries,
        split=args.split,
        checkpoint_dir=args.checkpoint_dir,
        cache_db_path=args.cache_db_path,
    )

    eval_result = evaluate(entries, predictions, tool_name=args.baseline, split_name=args.split)
    out_path = args.results_dir / f"{args.baseline}_{args.split}.json"
    out_path.write_text(json.dumps(eval_result.to_dict(), indent=2))
    logger.info("Wrote %s", out_path)

    manifest_path = args.results_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "version": "1.0",
            "description": "Pre-computed baseline reference results.",
            "files": {},
        }
    rel_path = str(out_path.relative_to(args.results_dir.parent.parent))
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
