#!/usr/bin/env python3
"""Generate pre-computed reference results for rate-limited baselines.  [evaluation]

Usage:
    python scripts/generate_reference_results.py --baselines harc,bibtexupdater
    python scripts/generate_reference_results.py --baselines harc --split dev_public

Runs the specified baselines on the full dataset locally, evaluates them,
writes EvaluationResult JSON files, and updates manifest.json with SHA-256
checksums.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.registry import run_baseline
from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import EvaluationResult
from hallmark.evaluation.metrics import evaluate
from hallmark.evaluation.validate import compute_sha256, validate_reference_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/v1.0/baseline_results")


def _get_hallmark_version() -> str:
    """Get installed hallmark version, or 'unknown' if not installed."""
    try:
        from importlib.metadata import version

        return version("hallmark")
    except Exception:
        return "unknown"


def _get_environment_metadata() -> dict[str, str]:
    """Collect environment metadata for manifest reproducibility."""
    import datetime
    import platform

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hallmark_version": _get_hallmark_version(),
    }


def generate(
    baselines: list[str],
    split: str = "dev_public",
    version: str = "v1.0",
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    """Run baselines, evaluate, write results and manifest."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest (preserve existing entries)
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "version": "1.0",
            "description": "Pre-computed baseline reference results.",
            "generated_by": "scripts/generate_reference_results.py",
            "files": {},
        }

    entries = load_split(split=split, version=version)
    logger.info(f"Loaded {len(entries)} entries from {split}")

    for baseline_name in baselines:
        logger.info(f"Running baseline: {baseline_name}")
        try:
            predictions = run_baseline(baseline_name, entries)
        except Exception:
            logger.exception(f"Failed to run {baseline_name}")
            continue

        result: EvaluationResult = evaluate(
            entries=entries,
            predictions=predictions,
            tool_name=baseline_name,
            split_name=split,
        )

        # Write result JSON
        filename = f"{baseline_name}_{split}.json"
        result_path = results_dir / filename
        result_path.write_text(result.to_json())
        logger.info(
            f"  {baseline_name}: F1={result.f1_hallucination:.3f}, "
            f"DR={result.detection_rate:.3f}, "
            f"entries={result.num_entries}"
        )

        # Update manifest
        sha = compute_sha256(result_path)
        manifest["files"][filename] = {
            "sha256": sha,
            "baseline": baseline_name,
            "split": split,
            "num_entries": result.num_entries,
            "f1_hallucination": result.f1_hallucination,
        }

    # Add environment metadata for reproducibility
    manifest["environment"] = _get_environment_metadata()

    # Write updated manifest
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"Manifest updated: {manifest_path}")

    # Self-validate
    vr = validate_reference_results(results_dir, strict=True)
    if vr.passed:
        logger.info("Self-validation passed.")
    else:
        logger.error("Self-validation FAILED:")
        for e in vr.errors:
            logger.error(f"  {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed reference results for rate-limited baselines."
    )
    parser.add_argument(
        "--baselines",
        required=True,
        help="Comma-separated baseline names (e.g. harc,bibtexupdater)",
    )
    parser.add_argument(
        "--split",
        default="dev_public",
        choices=["dev_public", "test_public"],
        help="Benchmark split to evaluate on",
    )
    parser.add_argument("--version", default="v1.0", help="Dataset version")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for results and manifest",
    )

    args = parser.parse_args()
    baselines = [b.strip() for b in args.baselines.split(",")]

    generate(
        baselines=baselines,
        split=args.split,
        version=args.version,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
