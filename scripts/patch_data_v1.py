#!/usr/bin/env python3
"""Patch shipped data files to fix known quality issues.

Fixes applied (idempotent — safe to run multiple times):
- fabricated_doi + llm_generated entries → relabeled as plausible_fabrication (tier 3)
- Entries with author="Unknown" (case-insensitive) → dropped
- Entries with booktitle/journal longer than 80 chars → venue replaced with a
  deterministically-chosen valid venue (seeded by bibtex_key for reproducibility)

Counts at time of writing (v1.0):
- 51 relabeled (28 dev, 23 test)
- 2 dropped (dev only)
- 29 venues fixed (15 dev, 14 test)
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hallmark.dataset.generators._pools import VALID_CONFERENCES, VALID_JOURNALS  # noqa: E402
from hallmark.dataset.schema import (  # noqa: E402
    EXPECTED_SUBTESTS,
    DifficultyTier,
    HallucinationType,
)

DATA_DIR = ROOT / "data" / "v1.0"
SPLITS = ["dev_public.jsonl", "test_public.jsonl", "stress_test.jsonl"]

MAX_VENUE_LEN = 80


def deterministic_choice(items: list, key_str: str) -> str:
    """Pick an item from *items* deterministically based on *key_str*."""
    h = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
    return items[h % len(items)]


def patch_entry(entry: dict) -> tuple[dict, bool]:
    """Apply all patches to a single entry.

    Returns (patched_entry, should_keep).  should_keep=False means the entry
    should be dropped from the split.
    """
    # --- Drop author="Unknown" entries ----------------------------------
    author = entry.get("fields", {}).get("author", "")
    if author.strip().lower() == "unknown":
        return entry, False

    # --- Relabel fabricated_doi + llm_generated → plausible_fabrication -
    if (
        entry.get("hallucination_type") == HallucinationType.FABRICATED_DOI.value
        and entry.get("generation_method") == "llm_generated"
    ):
        entry["hallucination_type"] = HallucinationType.PLAUSIBLE_FABRICATION.value
        entry["difficulty_tier"] = DifficultyTier.HARD.value

        # Build expected subtests for plausible_fabrication, preserving any
        # dynamic values (doi_resolves, fields_complete) from the old entry.
        expected: dict[str, bool | None] = dict(
            EXPECTED_SUBTESTS[HallucinationType.PLAUSIBLE_FABRICATION]
        )
        old_subtests = entry.get("subtests", {})
        for dynamic_key in ("doi_resolves", "fields_complete"):
            if dynamic_key in old_subtests:
                expected[dynamic_key] = old_subtests[dynamic_key]
        entry["subtests"] = expected

    # --- Fix garbage venue strings (> MAX_VENUE_LEN chars) --------------
    fields = entry.get("fields", {})
    key = entry.get("bibtex_key", "")

    if "booktitle" in fields and len(fields["booktitle"]) > MAX_VENUE_LEN:
        fields["booktitle"] = deterministic_choice(VALID_CONFERENCES, key + "_venue")

    if "journal" in fields and len(fields["journal"]) > MAX_VENUE_LEN:
        fields["journal"] = deterministic_choice(VALID_JOURNALS, key + "_venue")

    return entry, True


def patch_split(split_path: Path) -> dict:
    """Patch a single split file in-place.  Returns a stats dict."""
    entries: list[dict] = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    stats = {
        "total_before": len(entries),
        "relabeled": 0,
        "dropped": 0,
        "venues_fixed": 0,
    }
    patched: list[dict] = []

    for entry in entries:
        ht_before = entry.get("hallucination_type")
        bt_before = entry.get("fields", {}).get("booktitle", "")
        j_before = entry.get("fields", {}).get("journal", "")

        entry, keep = patch_entry(entry)

        if not keep:
            stats["dropped"] += 1
            continue

        ht_after = entry.get("hallucination_type")
        bt_after = entry.get("fields", {}).get("booktitle", "")
        j_after = entry.get("fields", {}).get("journal", "")

        relabeled = (
            ht_before == HallucinationType.FABRICATED_DOI.value
            and ht_after == HallucinationType.PLAUSIBLE_FABRICATION.value
        )
        if relabeled:
            stats["relabeled"] += 1

        if bt_before != bt_after or j_before != j_after:
            stats["venues_fixed"] += 1

        patched.append(entry)

    stats["total_after"] = len(patched)

    with open(split_path, "w") as f:
        for entry in patched:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return stats


def update_metadata(data_dir: Path, split_stats: dict[str, dict]) -> None:
    """Update metadata.json with corrected entry counts."""
    meta_path = data_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    for split_filename, stats in split_stats.items():
        clean_name = split_filename.replace(".jsonl", "")
        if "splits" in metadata and clean_name in metadata["splits"]:
            metadata["splits"][clean_name]["total"] = stats["total_after"]

    # Recompute grand total from splits that have a "total" field
    if "splits" in metadata:
        metadata["total_entries"] = sum(s.get("total", 0) for s in metadata["splits"].values())

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    print("HALLMARK Data Patch v1.0")
    print("=" * 40)

    all_stats: dict[str, dict] = {}
    for split in SPLITS:
        path = DATA_DIR / split
        if not path.exists():
            print(f"  Skipping {split} (not found)")
            continue
        stats = patch_split(path)
        all_stats[split] = stats
        print(f"\n  {split}:")
        print(f"    Entries : {stats['total_before']} → {stats['total_after']}")
        print(f"    Relabeled fabricated_doi→plausible_fabrication: {stats['relabeled']}")
        print(f"    Dropped (author=Unknown)                       : {stats['dropped']}")
        print(f"    Venues replaced (garbage string > 80 chars)    : {stats['venues_fixed']}")

    update_metadata(DATA_DIR, all_stats)
    print("\n  metadata.json updated")
    print("\nDone.")


if __name__ == "__main__":
    main()
