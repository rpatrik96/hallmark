#!/usr/bin/env python3
"""Repair sub-tests that contradict their entry's own hallucination type.  [data]

``EXPECTED_SUBTESTS`` fixes some sub-test outcomes by definition of the type: a
preprint cited as published cannot have cross-database agreement, because the
databases report the venue it was actually published in. Where a shipped entry
asserts the opposite, the entry contradicts its own label.

Scope
-----
Only the ``True -> False`` direction, and only for the (type, subtest) pairs
listed in :data:`CONTRADICTIONS`. Those are the pairs where three independent
sources agree on ``False``: ``EXPECTED_SUBTESTS``, the semantics of the type,
and every entry of that type in the public splits (which carry zero mismatches
of this direction).

Deliberately NOT repaired here
-----------------------------
``future_date`` / ``fields_complete``. When this script was written,
``EXPECTED_SUBTESTS`` said ``False`` for that pair while the real checker
(``check_fields_complete``: required fields present, 4-digit year, well-formed
DOI) returned ``True`` for every well-formed future-dated entry, so the hidden
split, which assigned ``True``, was right and the taxonomy entry was wrong.
Repairing the hidden entries to match the taxonomy would have corrupted correct
data, so the pair was excluded from :data:`CONTRADICTIONS`. The taxonomy has
since been fixed (``EXPECTED_SUBTESTS[FUTURE_DATE]["fields_complete"]`` is now
``True``) and 82 public labels were aligned with the checker's per-entry
verdict; see ``results/reviewer_experiments/fields_complete_future_date_alignment.json``.
The pair stays out of this script because its repair went the other way (public
splits, ``False -> True``) and is already done.

Usage
-----
    uv run python scripts/fix_subtest_type_contradictions.py            # dry run
    uv run python scripts/fix_subtest_type_contradictions.py --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (hallucination_type, subtest) -> value the type forces.
#: Restricted to pairs where the type's semantics, EXPECTED_SUBTESTS and the
#: public splits all agree, so a repair cannot be a taxonomy disagreement.
CONTRADICTIONS: dict[tuple[str, str], bool] = {
    # The databases report the venue the work actually appeared in, so they
    # cannot agree with an entry that names a different one.
    ("preprint_as_published", "cross_db_agreement"): False,
    ("wrong_venue", "cross_db_agreement"): False,
    ("arxiv_version_mismatch", "cross_db_agreement"): False,
}

#: Splits to scan. test_hidden lives outside data/v1.2 and is gitignored.
SPLIT_PATHS: dict[str, str] = {
    "dev_public": "data/v1.2/dev_public.jsonl",
    "test_public": "data/v1.2/test_public.jsonl",
    "test_crossdomain": "data/v1.2/test_crossdomain.jsonl",
    "stress_test": "data/v1.2/stress_test.jsonl",
    "test_hidden": "data/hidden/test_hidden.jsonl",
}


def process_file(path: Path, apply: bool) -> tuple[list[dict[str, Any]], int]:
    """Return (fix records, total entries). Rewrites the file when *apply*."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    entries = [json.loads(ln) for ln in lines]
    fixes: list[dict[str, Any]] = []
    for entry in entries:
        h_type = entry.get("hallucination_type")
        subtests = entry.get("subtests") or {}
        for (want_type, subtest), forced in CONTRADICTIONS.items():
            if h_type != want_type or subtest not in subtests:
                continue
            before = subtests[subtest]
            # Only the contradicting direction. A None (not-applicable) is an
            # abstention, not a contradiction, and is left alone.
            if before is not forced and before is not None:
                subtests[subtest] = forced
                fixes.append(
                    {
                        "key": entry.get("bibtex_key"),
                        "split": path.stem,
                        "hallucination_type": h_type,
                        "subtest": subtest,
                        "before": before,
                        "after": forced,
                    }
                )
    if apply and fixes:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return fixes, len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--log", type=Path, help="write a JSON fix log to this path")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.\n")

    all_fixes: list[dict[str, Any]] = []
    for name, rel in SPLIT_PATHS.items():
        path = REPO_ROOT / rel
        if not path.exists():
            # data/hidden/ is gitignored; say so rather than reporting a clean
            # pass over a split that was never opened.
            print(f"  [skip] {rel} not found")
            continue
        fixes, total = process_file(path, args.apply)
        all_fixes.extend(fixes)
        print(f"  {name:<20} {len(fixes):>4} / {total:<5}")

    verb = "changed" if args.apply else "would change"
    print(f"\n  TOTAL: {len(all_fixes)} sub-test values {verb}")
    print("  Only the listed (type, subtest) pairs are touched; no other field is modified.")

    if args.log and all_fixes:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "description": (
                        "Repaired sub-tests that contradicted their entry's own "
                        "hallucination type. Only the True->False direction, and only "
                        "for pairs where the type's semantics, EXPECTED_SUBTESTS and "
                        "the public splits all agree."
                    ),
                    "excluded": (
                        "future_date/fields_complete: check_fields_complete returns True "
                        "for a future-dated entry with all fields present, so the "
                        "taxonomy is wrong there, not the data."
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
