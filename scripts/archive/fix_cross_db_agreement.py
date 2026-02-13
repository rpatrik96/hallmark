"""Fix cross_db_agreement subtest to break near-perfect label correlation.

For hallucination types where the core paper identity (title + authors) is real
and verifiable across databases, cross_db_agreement should be True:
- wrong_venue: real paper, just wrong venue
- preprint_as_published: real preprint, fabricated venue
- version_confusion: real paper, mixed preprint/conference metadata

This breaks the 99.9% correlation between cross_db_agreement and label,
making the subtest genuinely diagnostic rather than a label proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hallmark.dataset.schema import load_entries, save_entries

# Types where title+authors are real → databases agree on core identity
CROSS_DB_TRUE_TYPES = {"wrong_venue", "preprint_as_published", "version_confusion"}


def fix_split(path: Path) -> tuple[int, int]:
    """Fix cross_db_agreement in a split file. Returns (total_fixed, total_entries)."""
    entries = load_entries(path)
    fixed = 0
    for entry in entries:
        if (
            entry.label == "HALLUCINATED"
            and entry.hallucination_type in CROSS_DB_TRUE_TYPES
            and entry.subtests.get("cross_db_agreement") is False
        ):
            entry.subtests["cross_db_agreement"] = True
            fixed += 1
    save_entries(entries, path)
    return fixed, len(entries)


def main() -> None:
    data_dir = Path("data/v1.0")
    total_fixed = 0

    for split_file in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / split_file
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        fixed, total = fix_split(path)
        total_fixed += fixed
        print(f"  {split_file}: fixed {fixed}/{total} entries")

    print(f"\nTotal cross_db_agreement fixes: {total_fixed}")

    # Verify: count cross_db_agreement distribution
    for split_file in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / split_file
        if not path.exists():
            continue
        entries = load_entries(path)
        valid_true = sum(
            1
            for e in entries
            if e.label == "VALID" and e.subtests.get("cross_db_agreement") is True
        )
        valid_false = sum(
            1
            for e in entries
            if e.label == "VALID" and e.subtests.get("cross_db_agreement") is not True
        )
        hall_true = sum(
            1
            for e in entries
            if e.label == "HALLUCINATED" and e.subtests.get("cross_db_agreement") is True
        )
        hall_false = sum(
            1
            for e in entries
            if e.label == "HALLUCINATED" and e.subtests.get("cross_db_agreement") is not True
        )
        total = len(entries)
        correlation = abs(valid_true + hall_false) / total if total else 0
        print(f"\n  {split_file} cross_db_agreement distribution:")
        print(f"    VALID:        True={valid_true}, False/None={valid_false}")
        print(f"    HALLUCINATED: True={hall_true}, False/None={hall_false}")
        print(f"    Label correlation: {correlation:.1%}")

    if total_fixed == 0:
        print("\nNo entries needed fixing.")
        sys.exit(0)


if __name__ == "__main__":
    main()
