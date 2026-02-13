"""Redefine fields_complete subtest to be genuinely discriminative.

Previous definition: True for 99.4% of entries (only False for future_date).
New definition: True if entry has all required BibTeX fields (title, author, year,
venue) AND at least one resolvable identifier (doi or url).

This creates meaningful variation since many entries lack DOI/URL fields.
"""

from __future__ import annotations

from pathlib import Path

from hallmark.dataset.schema import load_entries, save_entries

# Required fields for a complete BibTeX entry
REQUIRED_FIELDS = {"title", "author", "year"}
VENUE_FIELDS = {"booktitle", "journal"}
IDENTIFIER_FIELDS = {"doi", "url"}


def compute_fields_complete(fields: dict[str, str]) -> bool:
    """Check if entry has all required fields + venue + at least one identifier."""
    # Must have title, author, year
    for f in REQUIRED_FIELDS:
        if not fields.get(f, "").strip():
            return False

    # Must have a venue (booktitle or journal)
    has_venue = any(fields.get(f, "").strip() for f in VENUE_FIELDS)
    if not has_venue:
        return False

    # Must have at least one resolvable identifier (doi or url)
    has_identifier = any(fields.get(f, "").strip() for f in IDENTIFIER_FIELDS)
    return has_identifier


def fix_split(path: Path) -> dict[str, int]:
    """Recompute fields_complete for all entries. Returns stats."""
    entries = load_entries(path)
    stats = {"total": len(entries), "true": 0, "false": 0, "changed": 0}

    for entry in entries:
        new_val = compute_fields_complete(entry.fields)
        old_val = entry.subtests.get("fields_complete")
        if old_val != new_val:
            stats["changed"] += 1
        entry.subtests["fields_complete"] = new_val
        if new_val:
            stats["true"] += 1
        else:
            stats["false"] += 1

    save_entries(entries, path)
    return stats


def main() -> None:
    data_dir = Path("data/v1.0")

    for split_file in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / split_file
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue

        stats = fix_split(path)
        pct_true = stats["true"] / stats["total"] * 100
        print(f"  {split_file}: {stats['changed']} changed")
        print(f"    True={stats['true']} ({pct_true:.1f}%), False={stats['false']}")

        # Per-label breakdown
        entries = load_entries(path)
        for label in ["VALID", "HALLUCINATED"]:
            label_entries = [e for e in entries if e.label == label]
            true_count = sum(1 for e in label_entries if e.subtests.get("fields_complete"))
            false_count = len(label_entries) - true_count
            pct = true_count / len(label_entries) * 100 if label_entries else 0
            print(f"    {label}: True={true_count} ({pct:.1f}%), False={false_count}")


if __name__ == "__main__":
    main()
