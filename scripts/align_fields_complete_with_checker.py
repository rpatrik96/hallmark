#!/usr/bin/env python3
"""Align a type's ``fields_complete`` labels with the checker that computes it.  [data]

``fields_complete`` is the one sub-test decidable offline: ``check_fields_complete``
tests for missing required fields plus a 4-digit year and a well-formed DOI, and
needs no network. So for this sub-test the label can be verified rather than
asserted, and where the two disagree the checker wins.

Why this exists
---------------
``EXPECTED_SUBTESTS[FUTURE_DATE]["fields_complete"]`` was ``False``, but a
future-dated entry is complete -- it is just wrong about the year, and "2032" is
a perfectly well-formed 4-digit year. Auditing every type against the checker
found ``future_date`` to be the only one that disagreed: the checker passes 96 of
99 ``future_date`` entries against an expectation of ``False``. The taxonomy was
wrong, and entries labelled to match it inherited the error.

This writes the checker's per-entry verdict, not a blanket value: the 3
``future_date`` entries that genuinely fail the checker keep ``False``.

Usage
-----
    uv run python scripts/align_fields_complete_with_checker.py --type future_date
    uv run python scripts/align_fields_complete_with_checker.py --type future_date --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT_PATHS: dict[str, str] = {
    "dev_public": "data/v1.2/dev_public.jsonl",
    "test_public": "data/v1.2/test_public.jsonl",
    "test_crossdomain": "data/v1.2/test_crossdomain.jsonl",
    "stress_test": "data/v1.2/stress_test.jsonl",
    "test_hidden": "data/hidden/test_hidden.jsonl",
}

SUBTEST = "fields_complete"


def process_file(path: Path, h_type: str, apply: bool) -> tuple[list[dict[str, Any]], int]:
    """Return (fix records, entries of this type). Rewrites the file when *apply*."""
    from hallmark.evaluation.subtests import check_fields_complete

    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    entries = [json.loads(ln) for ln in lines]
    fixes: list[dict[str, Any]] = []
    seen = 0
    for entry in entries:
        if entry.get("hallucination_type") != h_type:
            continue
        seen += 1
        subtests = entry.get("subtests")
        if not subtests or SUBTEST not in subtests:
            continue
        verdict = check_fields_complete(
            entry.get("bibtex_type", "article"), entry.get("fields") or {}
        ).passed
        before = subtests[SUBTEST]
        if before == verdict:
            continue
        subtests[SUBTEST] = verdict
        fixes.append(
            {
                "key": entry.get("bibtex_key"),
                "split": path.stem,
                "hallucination_type": h_type,
                "subtest": SUBTEST,
                "before": before,
                "after": verdict,
            }
        )
    if apply and fixes:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return fixes, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", required=True, help="hallucination_type to align")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--log", type=Path, help="write a JSON fix log to this path")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.\n")

    all_fixes: list[dict[str, Any]] = []
    for name, rel in SPLIT_PATHS.items():
        path = REPO_ROOT / rel
        if not path.exists():
            # data/hidden/ is gitignored and absent without the full dataset.
            print(f"  [skip] {rel} not found")
            continue
        fixes, seen = process_file(path, args.type, args.apply)
        all_fixes.extend(fixes)
        print(f"  {name:<20} {len(fixes):>4} changed / {seen:<4} {args.type} entries")

    verb = "changed" if args.apply else "would change"
    print(f"\n  TOTAL: {len(all_fixes)} '{SUBTEST}' values {verb}")
    print("  Written from check_fields_complete per entry, not a blanket value.")

    if args.log and all_fixes:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "description": (
                        f"Aligned {SUBTEST} with check_fields_complete for "
                        f"hallucination_type={args.type}. The taxonomy expectation "
                        "contradicted the checker that computes this sub-test."
                    ),
                    "count": len(all_fixes),
                    "fixes": all_fixes,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"  Fix log written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
