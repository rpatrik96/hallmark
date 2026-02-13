"""Populate missing publication_date fields from fields.year.

128/1006 dev entries lack publication_date, breaking temporal segmentation.
For entries with a year field, set publication_date to YYYY-01-01.
For hallucinated entries without year, use the source entry's year field.
"""

from __future__ import annotations

from pathlib import Path

from hallmark.dataset.schema import load_entries, save_entries


def fix_split(path: Path) -> tuple[int, int]:
    """Fix missing publication_date in a split file."""
    entries = load_entries(path)
    fixed = 0
    for entry in entries:
        if not entry.publication_date:
            year = entry.fields.get("year", "")
            if year and year.isdigit():
                entry.publication_date = f"{year}-01-01"
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

        # Verify
        entries = load_entries(path)
        missing = sum(1 for e in entries if not e.publication_date)
        print(f"  {split_file}: fixed {fixed}/{total} entries, {missing} still missing")

    print(f"\nTotal publication_date fixes: {total_fixed}")


if __name__ == "__main__":
    main()
