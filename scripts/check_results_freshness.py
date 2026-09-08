#!/usr/bin/env python3
"""Guard against stale baseline result artifacts.  [evaluation]

Task #2: aggregate result JSONs in ``data/v1.2/baseline_results/`` score a
specific data split (``data/v1.2/<split>.jsonl``). When the split is relabeled
or regenerated, the result JSONs become stale -- their numbers describe data
that no longer exists. This guard makes that desynchronisation a hard failure:

A result JSON is **stale** if either

1. its recorded ``split_sha256`` does not match the split file it scores (the
   data changed after the result was produced), or
2. its recorded ground-truth counts (``num_entries`` / ``num_hallucinated`` /
   ``num_valid``) disagree with the *current* split data.

Check 1 used to compare mtimes, which cannot work in a git repo: git does not
preserve them, so on a fresh clone the ordering is whatever order git wrote
files in. On a clean checkout it called all 46 released results stale off
sub-second differences, which is why CI ran it ``--warn-only`` and the repo test
was ``xfail``-ed -- a guard reporting into a void. Hashing the split file is the
check it was reaching for, and it survives a clone.

A result predating ``split_sha256`` is reported as **unverifiable**, not stale.
Treating it as stale would make the guard red until everything is regenerated,
which is exactly how the previous one came to be switched off; the count check
still applies to it. A result whose ``per_type_metrics`` rows predate
``num_valid``/``precision`` is reported the same way: those rows scored their
false positives inside the type, so their f1 is 2*DR/(1+DR) and their
false-positive rate 0.0, and 39 of the 42 released results carry them.

Files under ``<results-dir>/archive/`` are skipped. A run kept for the record --
a CI sample, a smoke run, a probe -- scores no current split, and parking it
there is how it stops being a staleness report nobody can act on.

The split scored by a result is taken from its ``split_name`` field, falling
back to the ``<tool>_<split>.json`` filename suffix.

Derived tables are the layer above, and had no guard at all until
``check_table_freshness`` was added here: the CSVs under ``tables/`` are
arithmetic over these result JSONs, so a re-run leaves them describing numbers
that no longer exist. ``tables/base_rate_precision.csv`` shipped a doi_only
false-positive rate of 0.2788 against the 0.0417 its source run now reports, and
its precision column -- the one number a deployed user experiences directly --
was computed from it. The mechanism lives in
:mod:`hallmark.evaluation.table_provenance`; this script is where CI calls it,
so both layers fail in one place.

Used as a library (``check_freshness``) by the pytest guard and as a CLI in CI:

    python scripts/check_results_freshness.py \
        --results-dir data/v1.2/baseline_results \
        --data-dir data --version v1.2

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
from hallmark.evaluation.table_provenance import TableReport, check_tables
from hallmark.evaluation.validate import ARCHIVE_DIR_NAME, compute_sha256, iter_result_files

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/v1.2/baseline_results")
DEFAULT_TABLES_DIR = Path("tables")

#: Results known to be stale, with the reason. CI fails on anything stale that is
#: NOT listed here, so a new regression is caught immediately while a known debt
#: does not keep the guard red -- which is what --warn-only was papering over.
#: Ratchet DOWN only: a name leaves this dict when the result is regenerated, and
#: adding one is a decision to ship a number scored against data that has moved.
KNOWN_STALE: dict[str, str] = {
    "llm_openrouter_claude_opus_4_7_dev_public.json": (
        "per-type counts sum to 633 positives (the pre-relabel dev_public) while the "
        "headline counts say 606 / 513: the top-level counts were patched, the per-type "
        "block was not, so the per-type figures describe a different split. The "
        "taxonomy-fold ablation drops this file because its rebuilt MCC (0.6625) does "
        "not reproduce the published 0.6828. Needs a re-run under the current labels."
    ),
    "llm_openrouter_claude_sonnet_4_6_dev_public.json": (
        "per-type counts sum to 633 positives (the pre-relabel dev_public) while the "
        "headline counts say 606 / 513, as for the Opus 4.7 file above. Its rebuilt MCC "
        "happens to land within the ablation's 0.02 tolerance, so it was scored there "
        "as if current. Needs a re-run under the current labels."
    ),
    "harc_with_s2key_dev_public.json": (
        "scored against the pre-relabel ground truth (633 hallucinated / 486 valid "
        "against the current 606 / 513). Full coverage at n=1,119, but its DR 0.209 "
        "and FPR 0.045 are not comparable to current-label results. Needs harcx and "
        "a Semantic Scholar key to re-run."
    ),
}


#: Generated tables known to be stale, with the reason. Same contract as
#: KNOWN_STALE above and the same ratchet: a table leaves this dict when it is
#: regenerated, never by editing the reason.
KNOWN_STALE_TABLES: dict[str, str] = {}

#: Fields a per-type row carries under the current definition. A row written
#: before it has only ``detection_rate``, ``false_positive_rate``, ``f1`` and
#: ``count``: false positives were counted inside the type rather than against
#: the split's valid pool, so every hallucination type reports a false-positive
#: rate of 0.0 and an f1 of 2*DR/(1+DR). The two definitions are not comparable,
#: and the released directory publishes ``per_type_metrics.f1`` under both.
#:
#: A result carrying the old rows is reported **unverifiable**, on the contract
#: that already covers a result predating ``split_sha256``: 39 of the 42
#: released results carry it, and a fatal check tripping on all of them at once
#: is how a guard gets switched off. They are regenerated with the next release.
CURRENT_PER_TYPE_FIELDS: tuple[str, ...] = ("num_valid", "precision")


@dataclass
class TableFreshnessResult:
    """Aggregate freshness verdict across the generated tables."""

    passed: bool
    reports: list[TableReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def stale_tables(self) -> list[str]:
        return [r.table for r in self.reports if r.is_stale]


@dataclass
class StalenessReport:
    """Per-result-file freshness verdict."""

    result_file: str
    split: str | None
    is_stale: bool
    reasons: list[str] = field(default_factory=list)
    #: True when something about the result cannot be checked at all: it
    #: predates ``split_sha256``, or its per-type block predates the current
    #: definition. Reported, never fatal -- otherwise the guard would be red
    #: until every result is regenerated, which is how the previous one ended up
    #: switched off.
    unverifiable: bool = False
    #: The specific gap, so the summary can say which one it is.
    missing_split_hash: bool = False
    superseded_per_type: bool = False


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


def _superseded_per_type_rows(per_type: dict) -> int:
    """Count per-type rows written under the superseded definition.

    A current row names the valid pool it scored false positives against
    (``num_valid``) and reports ``precision``; an older one carries neither.
    """
    return sum(
        1
        for name, row in per_type.items()
        if name != "valid"
        and isinstance(row, dict)
        and not any(key in row for key in CURRENT_PER_TYPE_FIELDS)
    )


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
    version: str = "v1.2",
    data_dir: str | Path | None = None,
) -> FreshnessResult:
    """Check that every aggregate result JSON is fresh w.r.t. its split.

    Args:
        results_dir: Directory of ``<tool>_<split>.json`` aggregate results.
        version: Dataset version (default ``v1.2``).
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
    hash_cache: dict[str, str] = {}

    for result_path in iter_result_files(results_dir):
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

        # (1) content check: the result must name the split revision it scored.
        #
        # This replaced an mtime comparison, which could not work: git does not
        # preserve mtimes, so on any fresh clone the ordering is whatever order
        # git happened to write files in. Run on a clean checkout it reported all
        # 46 released results as stale off sub-second differences, which is why
        # CI carried --warn-only and the repo test was xfail-ed. A hash is the
        # same check the guard was reaching for, and it survives a clone.
        #
        # A result with no ``split_sha256`` is NOT stale: results predating the
        # field cannot be judged this way, and calling them stale would recreate
        # the always-red guard. They are reported so the gap stays visible, and
        # the count check below still applies to them.
        if split not in hash_cache:
            hash_cache[split] = compute_sha256(split_file)
        recorded_hash = probe.get("split_sha256")
        if recorded_hash is None:
            report.unverifiable = True
            report.missing_split_hash = True
            report.reasons.append(
                "no split_sha256 recorded — cannot verify which split revision this scored"
            )
        elif recorded_hash != hash_cache[split]:
            report.is_stale = True
            report.reasons.append(
                f"scored split {split} at {recorded_hash[:12]}… but the current file "
                f"is {hash_cache[split][:12]}…"
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

        # (3) per-type check: the per-type counts must sum to the split's positives.
        #
        # The headline counts can be patched without the per-type block being
        # regenerated, and then (2) passes a result whose per-type figures
        # describe a different split. Two released dev_public results carried
        # per-type counts summing to 633 (the pre-relabel split) under headline
        # counts of 606 / 513, and the fold ablation scored one of them.
        per_type = probe.get("per_type_metrics")
        if isinstance(per_type, dict) and per_type:
            per_type_positives = sum(
                int(m.get("count") or 0)
                for mode, m in per_type.items()
                if mode != "valid" and isinstance(m, dict)
            )
            if per_type_positives != current["num_hallucinated"]:
                report.is_stale = True
                report.reasons.append(
                    f"per_type_metrics counts sum to {per_type_positives} positives != "
                    f"current split {current['num_hallucinated']}"
                )

            # (4) per-type definition check: a row predating num_valid/precision
            # scored its false positives inside the type, so its f1 is
            # 2*DR/(1+DR) and its false-positive rate 0.0. The number is not
            # wrong under the definition that produced it and it is not
            # comparable with the current one, which is why this is reported
            # rather than failed.
            superseded_rows = _superseded_per_type_rows(per_type)
            if superseded_rows:
                report.unverifiable = True
                report.superseded_per_type = True
                report.reasons.append(
                    f"{superseded_rows} per_type_metrics row(s) predate "
                    f"{'/'.join(CURRENT_PER_TYPE_FIELDS)}: their f1 is 2*DR/(1+DR) and their "
                    "false-positive rate 0.0, so they cannot be compared with the current "
                    "definition — regenerated with the next release"
                )

        reports.append(report)

    unexpected = [r for r in reports if r.is_stale and r.result_file not in KNOWN_STALE]
    # A KNOWN_STALE entry that is no longer stale is a stale excuse: the result
    # was regenerated and nobody removed the exemption. Fail on it, so the
    # register can only shrink.
    fixed = [
        name
        for name in KNOWN_STALE
        if any(r.result_file == name and not r.is_stale for r in reports)
    ]
    for name in fixed:
        result_errors.append(
            f"{name}: listed in KNOWN_STALE but is now fresh — remove it from the register"
        )
    passed = not result_errors and not unexpected
    return FreshnessResult(passed=passed, reports=reports, errors=result_errors)


def check_table_freshness(
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    *,
    repo_root: str | Path | None = None,
) -> TableFreshnessResult:
    """Check that every generated table still agrees with the results it came from.

    Args:
        tables_dir: Directory of generated tables.
        results_dir: Directory of the result JSONs they derive from.
        repo_root: Root that recorded provenance paths are relative to.

    Returns:
        :class:`TableFreshnessResult`. ``passed`` is True only when nothing is
        stale outside :data:`KNOWN_STALE_TABLES` and no exemption has gone
        obsolete.
    """
    tables_dir = Path(tables_dir)
    if not tables_dir.is_dir():
        return TableFreshnessResult(
            passed=False, errors=[f"Tables directory not found: {tables_dir}"]
        )

    reports = check_tables(tables_dir, results_dir, repo_root=repo_root)
    errors: list[str] = []
    for name in KNOWN_STALE_TABLES:
        if any(r.table == name and not r.is_stale for r in reports):
            errors.append(
                f"{name}: listed in KNOWN_STALE_TABLES but is now fresh — "
                "remove it from the register"
            )
    unexpected = [r for r in reports if r.is_stale and r.table not in KNOWN_STALE_TABLES]
    return TableFreshnessResult(
        passed=not errors and not unexpected, reports=reports, errors=errors
    )


def _report_tables(result: TableFreshnessResult) -> None:
    """Log the table verdicts in the same vocabulary as the result verdicts."""
    for report in result.reports:
        if report.is_stale:
            logger.error("STALE table %s:", report.table)
            for reason in report.reasons:
                logger.error("    - %s", reason)
        elif report.unverifiable:
            logger.warning("unverifiable table %s: %s", report.table, "; ".join(report.reasons))
        else:
            logger.info("fresh table %s", report.table)
    for err in result.errors:
        logger.error("ERROR: %s", err)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory of aggregate result JSONs to check "
            f"(files under <results-dir>/{ARCHIVE_DIR_NAME}/ are skipped)."
        ),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory of generated tables to check against the results (default: tables/).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Check result JSONs only, leaving derived tables unchecked.",
    )
    parser.add_argument("--version", default="v1.2")
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
    tables = None if args.skip_tables else check_table_freshness(args.tables_dir, args.results_dir)

    for report in result.reports:
        if report.is_stale:
            logger.error("STALE %s [%s]:", report.result_file, report.split)
            for reason in report.reasons:
                logger.error("    - %s", reason)
        elif report.unverifiable:
            logger.warning(
                "unverifiable %s [%s]: %s",
                report.result_file,
                report.split,
                "; ".join(report.reasons),
            )
        else:
            logger.info("fresh %s [%s]", report.result_file, report.split)

    for err in result.errors:
        logger.error("ERROR: %s", err)

    if tables is not None:
        _report_tables(tables)

    unverifiable = [r.result_file for r in result.reports if r.unverifiable]
    no_hash = [r.result_file for r in result.reports if r.missing_split_hash]
    old_per_type = [r.result_file for r in result.reports if r.superseded_per_type]
    if result.passed and (tables is None or tables.passed):
        known = [f for f in result.stale_files if f in KNOWN_STALE]
        verified = len(result.reports) - len(unverifiable)
        if known:
            logger.warning(
                "%d file(s) stale but registered in KNOWN_STALE, pending regeneration: %s",
                len(known),
                ", ".join(known),
            )
        if old_per_type:
            logger.warning(
                "%d file(s) carry per_type_metrics rows written under the superseded "
                "definition (no %s), so their per-type f1 is 2*DR/(1+DR) and their per-type "
                "false-positive rate 0.0: not comparable with the current rows, and "
                "regenerated with the next release.",
                len(old_per_type),
                "/".join(CURRENT_PER_TYPE_FIELDS),
            )
        logger.info(
            "No unexpected staleness: %d of %d result file(s) fresh "
            "(%d fully verified, %d predate split_sha256 and were checked on counts only, "
            "%d carry the superseded per-type definition).",
            len(result.reports) - len(known),
            len(result.reports),
            verified,
            len(no_hash),
            len(old_per_type),
        )
        if tables is not None:
            known_tables = [t for t in tables.stale_tables if t in KNOWN_STALE_TABLES]
            unchecked = [r.table for r in tables.reports if r.unverifiable]
            if known_tables:
                logger.warning(
                    "%d table(s) stale but registered in KNOWN_STALE_TABLES, pending "
                    "regeneration: %s",
                    len(known_tables),
                    ", ".join(known_tables),
                )
            logger.info(
                "Tables: %d of %d checked against their inputs, %d unverifiable "
                "(no recorded provenance — the generator has not called record_table yet).",
                len(tables.reports) - len(unchecked),
                len(tables.reports),
                len(unchecked),
            )
        sys.exit(0)

    stale = result.stale_files
    unexpected = [f for f in stale if f not in KNOWN_STALE]
    known = [f for f in stale if f in KNOWN_STALE]
    if known:
        logger.warning(
            "%d known-stale file(s) pending regeneration: %s", len(known), ", ".join(known)
        )
    logger.error(
        "Freshness check FAILED: %d unexpected stale file(s)%s.",
        len(unexpected),
        " + errors" if result.errors else "",
    )
    if tables is not None and not tables.passed:
        unexpected_tables = [t for t in tables.stale_tables if t not in KNOWN_STALE_TABLES]
        logger.error(
            "Table freshness FAILED: %d unexpected stale table(s)%s%s.",
            len(unexpected_tables),
            f" ({', '.join(unexpected_tables)})" if unexpected_tables else "",
            " + errors" if tables.errors else "",
        )
    if args.warn_only:
        logger.warning("--warn-only set: exiting 0 despite staleness.")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
