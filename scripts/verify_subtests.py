#!/usr/bin/env python3
"""Verify sub-test labels against the schema truth table for consistency.  [analysis]

This is a *static* QA pass: it compares each entry's assigned sub-tests against
``EXPECTED_SUBTESTS`` (the single source of truth in
``hallmark.dataset.schema``) without any API calls. It covers all six
sub-tests — ``doi_resolves``, ``title_exists``, ``authors_match``,
``venue_correct``, ``fields_complete``, ``cross_db_agreement`` — so the QA
surface matches the taxonomy rather than only the three checks the original
script inspected.

Semantics
---------
* HALLUCINATED entries are checked against ``EXPECTED_SUBTESTS[type]``.
* VALID entries are checked against ``VALID_SUBTESTS`` (all checks pass).
* A sub-test is a *mismatch* only when both the expected and the assigned
  value are concrete booleans and they differ. ``None`` on either side means
  "depends on the source entry / not applicable" and is skipped — it is not a
  labeling error.
* Canary / watermark entries (``__canary__`` keys) and entries with empty
  ``subtests`` are skipped.

The script REPORTS inconsistencies; it never edits the (frozen) data files.
For full *live* verification (DOI resolution, title search) use
``hallmark evaluate --verify-subtests``.

Usage::

    python scripts/verify_subtests.py                 # human-readable report
    python scripts/verify_subtests.py --json           # machine-readable
    python scripts/verify_subtests.py --max-mismatches 234  # CI regression gate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from hallmark.dataset.schema import (
    EXPECTED_SUBTESTS,
    SUBTEST_NAMES,
    VALID_SUBTESTS,
    HallucinationType,
)

# Default data splits to scan. Paths are relative to the repository root.
DEFAULT_SPLITS: dict[str, str] = {
    "dev_public": "data/v1.0/dev_public.jsonl",
    "test_public": "data/v1.0/test_public.jsonl",
    "test_hidden": "data/hidden/test_hidden.jsonl",
    "stress_test": "data/v1.0/stress_test.jsonl",
    "test_crossdomain": "data/v1.0/test_crossdomain.jsonl",
}

_VALUE_TO_TYPE: dict[str, HallucinationType] = {t.value: t for t in HallucinationType}


@dataclass
class Mismatch:
    """A single sub-test that disagrees with the schema truth table."""

    split: str
    bibtex_key: str
    label: str
    hallucination_type: str | None
    subtest: str
    assigned: bool | None
    expected: bool | None

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "bibtex_key": self.bibtex_key,
            "label": self.label,
            "hallucination_type": self.hallucination_type,
            "subtest": self.subtest,
            "assigned": self.assigned,
            "expected": self.expected,
        }


@dataclass
class ScanReport:
    """Aggregated result of scanning one or more splits."""

    total_entries: int = 0
    skipped_entries: int = 0  # canary / empty subtests
    total_checks: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    unknown_types: Counter = field(default_factory=Counter)
    # split -> {subtest: count}
    per_split: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # (group, subtest) -> {"a=X->e=Y": count}; group is the type name or "VALID"
    per_type_subtest: dict[tuple[str, str], Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    @property
    def num_mismatches(self) -> int:
        return len(self.mismatches)

    @property
    def agreement(self) -> float:
        if self.total_checks == 0:
            return 100.0
        return (self.total_checks - self.num_mismatches) / self.total_checks * 100.0


def _expected_subtests_for(entry: dict) -> dict[str, bool | None] | None:
    """Return the expected sub-test truth table for *entry*, or None if unknown."""
    if entry.get("label") == "VALID":
        return VALID_SUBTESTS
    h_type = entry.get("hallucination_type")
    enum = _VALUE_TO_TYPE.get(h_type) if h_type is not None else None
    if enum is None:
        return None
    return EXPECTED_SUBTESTS[enum]


def verify_entry_subtests(entry: dict) -> list[Mismatch]:
    """Return the list of sub-test mismatches for a single entry.

    A mismatch is recorded only when the expected and assigned values are both
    concrete booleans and differ. ``None`` (on either side) is treated as
    "not applicable" and skipped.
    """
    subtests = entry.get("subtests") or {}
    expected = _expected_subtests_for(entry)
    if expected is None:
        return []
    split = entry.get("_split", "")
    out: list[Mismatch] = []
    for name in SUBTEST_NAMES:
        exp = expected.get(name)
        got = subtests.get(name)
        if exp is None or got is None:
            continue
        if got != exp:
            out.append(
                Mismatch(
                    split=split,
                    bibtex_key=entry.get("bibtex_key", ""),
                    label=entry.get("label", ""),
                    hallucination_type=entry.get("hallucination_type"),
                    subtest=name,
                    assigned=got,
                    expected=exp,
                )
            )
    return out


def _is_canary(entry: dict) -> bool:
    return str(entry.get("bibtex_key", "")).startswith("__canary__")


def scan_splits(splits: dict[str, Path] | None = None) -> ScanReport:
    """Scan the given splits and return an aggregated :class:`ScanReport`.

    *splits* maps a split name to a path; defaults to :data:`DEFAULT_SPLITS`.
    Missing files are skipped silently so the function works in partial
    checkouts.
    """
    if splits is None:
        splits = {name: Path(p) for name, p in DEFAULT_SPLITS.items()}

    report = ScanReport()
    for split_name, path in splits.items():
        if not Path(path).exists():
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                report.total_entries += 1
                if _is_canary(entry) or not (entry.get("subtests") or {}):
                    report.skipped_entries += 1
                    continue
                expected = _expected_subtests_for(entry)
                if expected is None:
                    report.unknown_types[entry.get("hallucination_type")] += 1
                    report.skipped_entries += 1
                    continue
                subtests = entry.get("subtests") or {}
                for name in SUBTEST_NAMES:
                    exp = expected.get(name)
                    got = subtests.get(name)
                    if exp is None or got is None:
                        continue
                    report.total_checks += 1
                    if got != exp:
                        m = Mismatch(
                            split=split_name,
                            bibtex_key=entry.get("bibtex_key", ""),
                            label=entry.get("label", ""),
                            hallucination_type=entry.get("hallucination_type"),
                            subtest=name,
                            assigned=got,
                            expected=exp,
                        )
                        report.mismatches.append(m)
                        report.per_split[split_name][name] += 1
                        group = (
                            "VALID"
                            if entry.get("label") == "VALID"
                            else (m.hallucination_type or "?")
                        )
                        report.per_type_subtest[(group, name)][f"a={got}->e={exp}"] += 1
    return report


def print_report(report: ScanReport) -> None:
    """Print a human-readable summary of *report* to stdout."""
    print(
        f"Scanned {report.total_entries} entries "
        f"({report.skipped_entries} skipped: canary/empty/unknown-type)."
    )
    print(
        f"Overall agreement: {report.agreement:.1f}% "
        f"({report.num_mismatches}/{report.total_checks} sub-test checks mismatch the truth table)."
    )

    if report.unknown_types:
        print("\nEntries with unknown hallucination_type (not in taxonomy):")
        for ht, n in report.unknown_types.most_common():
            print(f"  {ht!r}: {n}")

    print("\nPer-split mismatch counts (by sub-test):")
    for split_name in sorted(report.per_split):
        counts = report.per_split[split_name]
        total = sum(counts.values())
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {split_name}: {total} ({detail})")

    print("\nResidual inconsistencies by (group, sub-test) and direction:")
    for (group, subtest), directions in sorted(report.per_type_subtest.items()):
        detail = ", ".join(f"{d}: {n}" for d, n in sorted(directions.items()))
        print(f"  {group:24s} {subtest:20s} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify sub-test label consistency")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of text.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=None,
        help=(
            "CI regression gate: exit non-zero if the total mismatch count "
            "exceeds this threshold. Use the current count to lock in state."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root containing the data/ directory (default: cwd).",
    )
    args = parser.parse_args()

    splits = {name: args.data_root / rel for name, rel in DEFAULT_SPLITS.items()}
    report = scan_splits(splits)

    if args.json:
        payload = {
            "total_entries": report.total_entries,
            "skipped_entries": report.skipped_entries,
            "total_checks": report.total_checks,
            "num_mismatches": report.num_mismatches,
            "agreement_pct": round(report.agreement, 4),
            "per_split": {s: dict(c) for s, c in report.per_split.items()},
            "per_type_subtest": {
                f"{g}|{st}": dict(c) for (g, st), c in report.per_type_subtest.items()
            },
            "unknown_types": dict(report.unknown_types),
            "mismatches": [m.as_dict() for m in report.mismatches],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_report(report)

    if args.max_mismatches is not None and report.num_mismatches > args.max_mismatches:
        print(
            f"\nFAIL: {report.num_mismatches} mismatches exceed the allowed "
            f"maximum of {args.max_mismatches}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
