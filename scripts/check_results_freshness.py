#!/usr/bin/env python3
"""Guard against stale baseline result artifacts.  [evaluation]

Task #2: aggregate result JSONs in ``data/v1.0/baseline_results/`` score a
specific data split (``data/v1.0/<split>.jsonl``). When the split is relabeled
or regenerated, the result JSONs become stale -- their numbers describe data
that no longer exists. This guard makes that desynchronisation a hard failure:

A result JSON is **stale** if either

1. its mtime is older than the split file it scores (the data changed after the
   result was produced), or
2. its recorded ground-truth counts (``num_entries`` / ``num_hallucinated`` /
   ``num_valid``) disagree with the *current* split data.

The split scored by a result is taken from its ``split_name`` field, falling
back to the ``<tool>_<split>.json`` filename suffix.

Used as a library (``check_freshness``) by the pytest guard and as a CLI in CI:

    python scripts/check_results_freshness.py \
        --results-dir data/v1.0/baseline_results \
        --data-dir data --version v1.0

Exit code 0 when everything is fresh, 1 when any result is stale (CLI). Pass
``--warn-only`` to report without failing (used while results are pending
regeneration in a later stage).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.dataset.loader import DEFAULT_DATA_DIR, SPLIT_PATHS, load_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/v1.0/baseline_results")


@dataclass
class StalenessReport:
    """Per-result-file freshness verdict."""

    result_file: str
    split: str | None
    is_stale: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class FreshnessResult:
    """Aggregate freshness verdict across all checked result files."""

    passed: bool
    reports: list[StalenessReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def stale_files(self) -> list[str]:
        return [r.result_file for r in self.reports if r.is_stale]


def _split_path(data_dir: Path, version: str, split: str) -> Path:
    """Resolve the JSONL path for a split (mirrors loader.SPLIT_PATHS)."""
    return data_dir / version / SPLIT_PATHS[split]


def _infer_split(payload: dict, filename: str) -> str | None:
    """Determine which split a result scores, from its payload then filename."""
    split = payload.get("split_name")
    if isinstance(split, str) and split in SPLIT_PATHS:
        return split
    # Fall back to the filename suffix, longest known split wins.
    stem = Path(filename).stem
    matches = [s for s in SPLIT_PATHS if stem == s or stem.endswith(f"_{s}")]
    return max(matches, key=len) if matches else None


def _split_counts(split: str, version: str, data_dir: Path) -> dict[str, int]:
    """Current ground-truth counts for a split (canaries already filtered)."""
    entries = load_split(split, version, data_dir)
    return {
        "num_entries": len(entries),
        "num_hallucinated": sum(1 for e in entries if e.label == "HALLUCINATED"),
        "num_valid": sum(1 for e in entries if e.label == "VALID"),
    }


def check_freshness(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    *,
    version: str = "v1.0",
    data_dir: str | Path | None = None,
) -> FreshnessResult:
    """Check that every aggregate result JSON is fresh w.r.t. its split.

    Args:
        results_dir: Directory of ``<tool>_<split>.json`` aggregate results.
        version: Dataset version (default ``v1.0``).
        data_dir: Root data directory; defaults to the package ``data/`` dir.

    Returns:
        :class:`FreshnessResult`. ``passed`` is True only if no result is stale
        and no hard error occurred.
    """
    results_dir = Path(results_dir)
    data_root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    result_errors: list[str] = []
    reports: list[StalenessReport] = []

    if not results_dir.is_dir():
        return FreshnessResult(
            passed=False,
            errors=[f"Results directory not found: {results_dir}"],
        )

    counts_cache: dict[str, dict[str, int]] = {}
    mtime_cache: dict[str, float] = {}

    for result_path in sorted(results_dir.glob("*.json")):
        if result_path.name == "manifest.json":
            continue

        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            result_errors.append(f"{result_path.name}: unreadable ({exc})")
            continue

        # A dual-mode payload nests {"conservative": ..., "aggressive": ...}.
        probe = payload.get("conservative", payload) if isinstance(payload, dict) else payload
        if not isinstance(probe, dict):
            result_errors.append(f"{result_path.name}: unexpected JSON shape")
            continue

        split = _infer_split(probe, result_path.name)
        report = StalenessReport(result_file=result_path.name, split=split, is_stale=False)

        if split is None:
            report.is_stale = True
            report.reasons.append("could not determine which split this result scores")
            reports.append(report)
            continue

        split_file = _split_path(data_root, version, split)
        if not split_file.exists():
            report.is_stale = True
            report.reasons.append(f"split file missing: {split_file}")
            reports.append(report)
            continue

        # (1) mtime check: result must not predate the data it scores.
        if split not in mtime_cache:
            mtime_cache[split] = split_file.stat().st_mtime
        split_mtime = mtime_cache[split]
        result_mtime = result_path.stat().st_mtime
        if result_mtime < split_mtime:
            report.is_stale = True
            report.reasons.append(
                f"result mtime ({result_mtime:.0f}) older than split "
                f"{split} mtime ({split_mtime:.0f})"
            )

        # (2) count check: recorded counts must match the current split.
        if split not in counts_cache:
            counts_cache[split] = _split_counts(split, version, data_root)
        current = counts_cache[split]
        for key, expected in current.items():
            recorded = probe.get(key)
            if recorded is not None and recorded != expected:
                report.is_stale = True
                report.reasons.append(
                    f"{key} mismatch: recorded {recorded} != current split {expected}"
                )

        reports.append(report)

    passed = not result_errors and not any(r.is_stale for r in reports)
    return FreshnessResult(passed=passed, reports=reports, errors=result_errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report staleness but exit 0 (use while results are pending regeneration).",
    )
    args = parser.parse_args()

    result = check_freshness(
        args.results_dir,
        version=args.version,
        data_dir=args.data_dir,
    )

    for report in result.reports:
        if report.is_stale:
            logger.error("STALE %s [%s]:", report.result_file, report.split)
            for reason in report.reasons:
                logger.error("    - %s", reason)
        else:
            logger.info("fresh %s [%s]", report.result_file, report.split)

    for err in result.errors:
        logger.error("ERROR: %s", err)

    if result.passed:
        logger.info("All %d result file(s) are fresh.", len(result.reports))
        sys.exit(0)

    stale = result.stale_files
    logger.error(
        "Freshness check FAILED: %d stale file(s)%s.",
        len(stale),
        " + errors" if result.errors else "",
    )
    if args.warn_only:
        logger.warning("--warn-only set: exiting 0 despite staleness.")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
