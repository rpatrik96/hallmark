"""Run GPT-5.4 (OpenAI native) on dev_public and/or test_public splits.

Usage:
    uv run python scripts/run_gpt54_splits.py --split dev_public
    uv run python scripts/run_gpt54_splits.py --split test_public
    uv run python scripts/run_gpt54_splits.py --split dev_public --n 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.registry import run_baseline
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_ID = "gpt-5.4"
BASELINE_NAME = "llm_openai"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-5.4 on a HALLMARK split")
    parser.add_argument(
        "--split",
        choices=["dev_public", "test_public"],
        required=True,
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to first N entries (smoke test). Omit for full run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for result JSONs (default: results/)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for JSONL checkpoints. Defaults to {output_dir}/checkpoints/llm_openai_gpt54_{split}/",
    )
    parser.add_argument(
        "--version",
        default="v1.0",
        help="Dataset version (default: v1.0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_split = args.split  # e.g. "dev_public"
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = output_dir / "checkpoints" / f"llm_openai_gpt54_{safe_split}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading split: {args.split} (version={args.version})")
    entries = load_split(args.split, version=args.version)

    if args.n is not None:
        logger.info(f"Limiting to first {args.n} entries (smoke test mode)")
        entries = entries[: args.n]

    logger.info(f"Entries: {len(entries)}")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    predictions = run_baseline(
        BASELINE_NAME,
        entries,
        split=args.split,
        model=MODEL_ID,
        checkpoint_dir=checkpoint_dir,
    )

    result = evaluate(
        entries,
        predictions,
        tool_name=f"llm_openai_{MODEL_ID.replace('.', '_')}",
        split_name=args.split,
    )

    suffix = f"_smoke{args.n}" if args.n is not None else ""
    out_path = output_dir / f"llm_openai_gpt54_{safe_split}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {out_path}")
    logger.info(f"  Detection Rate:   {result.detection_rate:.3f}")
    logger.info(f"  FPR:              {result.false_positive_rate:.3f}")
    logger.info(f"  F1-Hallucination: {result.f1_hallucination:.3f}")
    if result.ece is not None:
        logger.info(f"  ECE:              {result.ece:.3f}")
    logger.info(f"  Predictions:      {len(predictions)}")


if __name__ == "__main__":
    main()
