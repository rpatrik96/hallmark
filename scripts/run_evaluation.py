#!/usr/bin/env python3
"""Run full evaluation pipeline.

Usage:
    python scripts/run_evaluation.py \
        --split dev_public \
        --baseline doi_only \
        --output results/doi_only_dev.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from hallmark.cli import _run_baseline
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument(
        "--split",
        default="dev_public",
        choices=["dev_public", "test_public", "test_hidden"],
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["doi_only", "bibtexupdater", "llm_openai", "llm_anthropic"],
    )
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        entries = load_split(args.split, args.version, args.data_dir)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    logging.info(f"Loaded {len(entries)} entries from {args.split}")

    predictions = _run_baseline(args.baseline, entries)
    logging.info(f"Got {len(predictions)} predictions from {args.baseline}")

    result = evaluate(entries, predictions, args.baseline, args.split)

    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(result.to_json())
    logging.info(f"Results written to {args.output}")

    # Print summary
    print(f"\nDetection Rate: {result.detection_rate:.3f}")
    print(f"FPR: {result.false_positive_rate:.3f}")
    print(f"F1: {result.f1_hallucination:.3f}")
    print(f"Tier-weighted F1: {result.tier_weighted_f1:.3f}")


if __name__ == "__main__":
    main()
