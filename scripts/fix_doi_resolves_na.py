#!/usr/bin/env python3
"""Correct ``doi_resolves`` where it records a failure for an absent DOI.

The ``subtests`` dict is three-valued (see the scoping rule beside
``SUBTEST_NAMES`` in ``hallmark.dataset.schema``):

* ``True``  — the check ran and passed
* ``False`` — the check ran and FAILED
* ``None``  — not applicable: the field it inspects is absent

Three code sites collapsed this to a boolean, so "this entry has no DOI" was
written as "this entry's DOI is broken" (fixed in ``tier2.py`` and
``scrape_crossdomain.py``; the rest came from ``real_world`` hand annotation
using a citation-level reading). This script repairs the entries already
shipped.

Rule
----
``doi_resolves is False`` **and** the entry has no ``doi`` field  ->  ``None``.

Entries whose ``doi_resolves`` is False *while carrying a real DOI* are left
untouched: those are genuine failed resolutions (``fabricated_doi`` and
friends) and are correct as-is. No other field is modified.

Usage
-----
    uv run python scripts/fix_doi_resolves_na.py                  # dry run
    uv run python scripts/fix_doi_resolves_na.py --apply          # write
    uv run python scripts/fix_doi_resolves_na.py --pool           # non-split files
    uv run python scripts/fix_doi_resolves_na.py --apply --log out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path("data/v1.2")

#: Released splits — entries here are scored against by tools.
#:
#: ``test_hidden`` is the one most scored against, and it was missing from this
#: list until 2026-09-04, so the original pass left exactly 14 VALID entries in
#: it carrying ``doi_resolves: False`` with no ``doi`` field — precisely the
#: defect this script exists to remove, in the split whose labels no external
#: contributor can inspect. It lives outside ``data_dir`` (``data/hidden/`` is
#: gitignored), hence the explicit relative path rather than a bare name.
SPLIT_FILES = ["dev_public", "test_public", "test_crossdomain", "test_hidden"]

#: Splits whose file is not ``<data_dir>/<name>.jsonl``. Paths are relative to
#: the repository root.
SPLIT_PATH_OVERRIDES: dict[str, str] = {
    "test_hidden": "data/hidden/test_hidden.jsonl",
}

#: Curated pools and generation-pipeline provenance; not part of any split.
POOL_FILES = [
    "gptzero_neurips2025",
    "real_world_incidents",
    "llm_generated",
    "llm_generated_deepseek_r1",
    "llm_generated_deepseek_v3",
    "llm_generated_gemini_flash",
    "llm_generated_qwen",
    "llm_openai",
]

SUBTEST = "doi_resolves"
FIELD = "doi"


def needs_fix(entry: dict[str, Any]) -> bool:
    """True when ``doi_resolves`` is False but the entry carries no DOI."""
    subtests = entry.get("subtests") or {}
    return subtests.get(SUBTEST) is False and not (entry.get("fields") or {}).get(FIELD)


def process_file(path: Path, apply: bool) -> tuple[list[dict[str, Any]], int]:
    """Return (fix records, total entries). Writes the file when *apply*."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    entries = [json.loads(ln) for ln in lines]
    fixes: list[dict[str, Any]] = []
    for entry in entries:
        if not needs_fix(entry):
            continue
        entry["subtests"][SUBTEST] = None
        fixes.append(
            {
                "key": entry.get("bibtex_key"),
                "split": path.stem,
                "label": entry.get("label"),
                "hallucination_type": entry.get("hallucination_type"),
                "generation_method": entry.get("generation_method"),
                "subtest": SUBTEST,
                "before": False,
                "after": None,
            }
        )
    if apply and fixes:
        with open(path, "w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return fixes, len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--pool", action="store_true", help="operate on non-split files instead")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--log", type=Path, help="write a JSON fix log to this path")
    args = ap.parse_args()

    targets = POOL_FILES if args.pool else SPLIT_FILES
    scope = "non-split pools" if args.pool else "released splits"
    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.\n")
    print(f"Scope: {scope}\n")

    all_fixes: list[dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parent.parent
    for name in targets:
        override = SPLIT_PATH_OVERRIDES.get(name)
        path = repo_root / override if override else args.data_dir / f"{name}.jsonl"
        if not path.exists():
            # data/hidden/ is gitignored, so test_hidden is absent on any
            # checkout without the full dataset. Say so rather than reporting a
            # clean pass over a split that was never opened.
            print(f"  [skip] {path} not found")
            continue
        fixes, total = process_file(path, args.apply)
        all_fixes.extend(fixes)
        if fixes:
            by_label: dict[str, int] = {}
            for f in fixes:
                by_label[f["label"]] = by_label.get(f["label"], 0) + 1
            detail = ", ".join(f"{v} {k}" for k, v in sorted(by_label.items()))
            print(f"  {name:<28} {len(fixes):>4} / {total:<5} ({detail})")
        else:
            print(f"  {name:<28} {0:>4} / {total:<5}")

    print(
        f"\n  TOTAL: {len(all_fixes)} entries "
        f"{'changed' if args.apply else 'would change'} in {len(targets)} files"
    )
    print(f"  Only '{SUBTEST}' is modified; no other field is touched.")

    if args.log:
        payload = {
            "description": (
                f"doi_resolves fix log ({scope}): set doi_resolves=None for entries "
                "recording a failed resolution while carrying no DOI field. The schema "
                "reserves False for a check that ran and failed, None for not-applicable."
            ),
            "rule": "subtests.doi_resolves is False and 'doi' not in fields -> None",
            "count": len(all_fixes),
            "fixes": all_fixes,
        }
        args.log.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"  Fix log written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
