#!/usr/bin/env python3
"""HALLMARK Dataset Sanitization Script — Phase 1 (P0 fixes).

Addresses all P0 (rejection-causing) issues identified in the dataset hardening plan:
  P0.1: Anonymize bibtex_keys and shuffle entries
  P0.2: Normalize bibtex_type distribution (article/misc → inproceedings)
  P0.3: Strip hallucination-only fields that leak labels
  P0.4: Re-split to separate source-hallucination title overlaps
  P0.5: Constrain hallucination years to valid range (except future_date)
  P0.6: Remove fabricated entries from real_world_incidents.jsonl

Usage:
    python scripts/sanitize_dataset.py [--dry-run] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data/v1.0")
SEED = 42

# Fields that ONLY appear in hallucinated entries — leak labels
HALLUCINATION_ONLY_FIELDS = {
    "address",
    "archiveprefix",
    "edition",
    "editor",
    "eprint",
    "note",
    "number",
    "organization",
    "series",
    "volume",
}

# These fields need special handling (journal → booktitle conversion)
# "pages" and "publisher" are stripped since valid entries never have them
STRIP_FIELDS = HALLUCINATION_ONLY_FIELDS | {"pages", "publisher"}

# Valid year range for non-future_date entries
VALID_YEAR_MIN = 2021
VALID_YEAR_MAX = 2023

# Keys of fabricated entries in real_world_incidents.jsonl
FAKE_REALWORLD_KEYS = {
    "realworld_future_date_pattern",
    "realworld_nonexistent_venue",
    "realworld_fabricated_doi",
    "realworld_hybrid_fabrication",
}

# Target split sizes (from current dataset)
TARGET_SPLITS = {
    "dev_public": {"valid": 450, "hallucinated": 555},
    "test_public": {"valid": 270, "hallucinated": 440},
}


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def save_jsonl(entries: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def fix_bibtex_type(entry: dict) -> dict:
    """P0.2: Normalize article/misc to inproceedings for hallucinated entries."""
    if entry["label"] != "HALLUCINATED":
        return entry

    if entry["bibtex_type"] in ("article", "misc"):
        entry["bibtex_type"] = "inproceedings"

        # Move journal → booktitle if needed
        fields = entry["fields"]
        if "journal" in fields and "booktitle" not in fields:
            fields["booktitle"] = fields.pop("journal")
        elif "journal" in fields and "booktitle" in fields:
            # Has both — drop journal
            del fields["journal"]

        # For misc entries that lack both, add a generic venue
        if "booktitle" not in fields and "journal" not in fields:
            # Use existing venue-like info or leave as-is
            pass

    return entry


def strip_leaky_fields(entry: dict) -> dict:
    """P0.3: Remove fields that only appear in hallucinated entries."""
    fields = entry["fields"]
    for f in STRIP_FIELDS:
        fields.pop(f, None)
    return entry


def fix_temporal_distribution(entry: dict) -> dict:
    """P0.5: Constrain hallucination years to valid range (except future_date)."""
    if entry["label"] != "HALLUCINATED":
        return entry

    # future_date entries are SUPPOSED to have out-of-range years
    if entry.get("hallucination_type") == "future_date":
        return entry

    year_str = entry["fields"].get("year", "")
    if not year_str.isdigit():
        return entry

    year = int(year_str)
    if year < VALID_YEAR_MIN or year > VALID_YEAR_MAX:
        # Map to a random year in the valid range, seeded by the entry key
        rng = random.Random(entry["bibtex_key"])
        new_year = rng.randint(VALID_YEAR_MIN, VALID_YEAR_MAX)
        entry["fields"]["year"] = str(new_year)

    return entry


def build_title_index(entries: list[dict]) -> dict[str, list[int]]:
    """Build mapping from normalized title to list of entry indices."""
    index: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        title = e["fields"].get("title", "").strip().lower()
        if title:
            index[title].append(i)
    return index


def resplit_with_source_separation(
    all_entries: list[dict], seed: int
) -> tuple[list[dict], list[dict]]:
    """P0.4: Re-split ensuring source-hallucination pairs are in different splits.

    Strategy:
    1. Group entries by normalized title
    2. For title groups with both valid + hallucinated: assign ALL to one split
    3. Fill remaining slots from ungrouped entries
    4. Maintain target type distribution
    """
    rng = random.Random(seed)

    valid = [e for e in all_entries if e["label"] == "VALID"]
    hall = [e for e in all_entries if e["label"] == "HALLUCINATED"]

    # Build title → entries mapping
    title_to_valid: dict[str, list[dict]] = defaultdict(list)
    title_to_hall: dict[str, list[dict]] = defaultdict(list)

    for e in valid:
        t = e["fields"].get("title", "").strip().lower()
        title_to_valid[t].append(e)

    for e in hall:
        t = e["fields"].get("title", "").strip().lower()
        title_to_hall[t].append(e)

    # Find overlapping titles (same title in both valid and hallucinated)
    overlap_titles = set(title_to_valid.keys()) & set(title_to_hall.keys())
    print(f"  Overlapping titles (valid + hall sharing title): {len(overlap_titles)}")

    # For each overlapping title: valid goes to one split, hall to the other
    # Strategy: Assign valid entries to dev, hallucinated entries to test (or vice versa)
    # We'll randomize the assignment
    dev_valid: list[dict] = []
    dev_hall: list[dict] = []
    test_valid: list[dict] = []
    test_hall: list[dict] = []

    # Track which entries are already assigned
    assigned_valid: set[str] = set()
    assigned_hall: set[str] = set()

    overlap_list = sorted(overlap_titles)
    rng.shuffle(overlap_list)

    for title in overlap_list:
        v_entries = title_to_valid[title]
        h_entries = title_to_hall[title]

        # Randomly decide: valid→dev + hall→test, or valid→test + hall→dev
        if rng.random() < 0.5:
            dev_valid.extend(v_entries)
            test_hall.extend(h_entries)
        else:
            test_valid.extend(v_entries)
            dev_hall.extend(h_entries)

        for e in v_entries:
            assigned_valid.add(e["bibtex_key"])
        for e in h_entries:
            assigned_hall.add(e["bibtex_key"])

    # Remaining unassigned entries
    remaining_valid = [e for e in valid if e["bibtex_key"] not in assigned_valid]
    remaining_hall = [e for e in hall if e["bibtex_key"] not in assigned_hall]
    rng.shuffle(remaining_valid)
    rng.shuffle(remaining_hall)

    # Fill to target sizes
    dev_valid_needed = TARGET_SPLITS["dev_public"]["valid"] - len(dev_valid)
    test_valid_needed = TARGET_SPLITS["test_public"]["valid"] - len(test_valid)
    dev_hall_needed = TARGET_SPLITS["dev_public"]["hallucinated"] - len(dev_hall)
    test_hall_needed = TARGET_SPLITS["test_public"]["hallucinated"] - len(test_hall)

    # If we over-assigned to one split, move excess to the other
    if dev_valid_needed < 0:
        # Move excess from dev to test
        n_excess = abs(dev_valid_needed)
        excess = dev_valid[len(dev_valid) - n_excess :]
        dev_valid = dev_valid[: len(dev_valid) - n_excess]
        remaining_valid = excess + remaining_valid
        dev_valid_needed = 0
    if test_valid_needed < 0:
        n_excess = abs(test_valid_needed)
        excess = test_valid[len(test_valid) - n_excess :]
        test_valid = test_valid[: len(test_valid) - n_excess]
        remaining_valid = excess + remaining_valid
        test_valid_needed = 0
    if dev_hall_needed < 0:
        n_excess = abs(dev_hall_needed)
        excess = dev_hall[len(dev_hall) - n_excess :]
        dev_hall = dev_hall[: len(dev_hall) - n_excess]
        remaining_hall = excess + remaining_hall
        dev_hall_needed = 0
    if test_hall_needed < 0:
        n_excess = abs(test_hall_needed)
        excess = test_hall[len(test_hall) - n_excess :]
        test_hall = test_hall[: len(test_hall) - n_excess]
        remaining_hall = excess + remaining_hall
        test_hall_needed = 0

    # Recalculate needs
    dev_valid_needed = TARGET_SPLITS["dev_public"]["valid"] - len(dev_valid)
    test_valid_needed = TARGET_SPLITS["test_public"]["valid"] - len(test_valid)
    dev_hall_needed = TARGET_SPLITS["dev_public"]["hallucinated"] - len(dev_hall)
    test_hall_needed = TARGET_SPLITS["test_public"]["hallucinated"] - len(test_hall)

    # Fill remaining valid
    dev_valid.extend(remaining_valid[:dev_valid_needed])
    remaining_valid = remaining_valid[dev_valid_needed:]
    test_valid.extend(remaining_valid[:test_valid_needed])

    # Fill remaining hallucinated — try to maintain type balance
    # Group remaining hall by type
    remaining_by_type: dict[str, list[dict]] = defaultdict(list)
    for e in remaining_hall:
        remaining_by_type[e["hallucination_type"]].append(e)

    # Distribute proportionally
    for _type_name, type_entries in remaining_by_type.items():
        rng.shuffle(type_entries)
        total_needed = dev_hall_needed + test_hall_needed
        if total_needed <= 0:
            break
        n_for_dev = round(len(type_entries) * dev_hall_needed / max(total_needed, 1))
        n_for_dev = min(n_for_dev, dev_hall_needed, len(type_entries))
        n_for_test = min(len(type_entries) - n_for_dev, test_hall_needed)
        dev_hall.extend(type_entries[:n_for_dev])
        test_hall.extend(type_entries[n_for_dev : n_for_dev + n_for_test])
        dev_hall_needed -= n_for_dev
        test_hall_needed -= n_for_test

    dev = dev_valid + dev_hall
    test = test_valid + test_hall

    return dev, test


def anonymize_and_shuffle(entries: list[dict], split_name: str, seed: int) -> list[dict]:
    """P0.1: Anonymize bibtex_keys and shuffle entries."""
    rng = random.Random(seed)
    rng.shuffle(entries)

    for i, entry in enumerate(entries):
        entry["bibtex_key"] = f"hallmark_{split_name}_{i:04d}"

    return entries


def verify_no_title_overlap(dev: list[dict], test: list[dict]) -> int:
    """Verify no valid-hallucinated title overlap within a split."""
    violations = 0
    for split_name, entries in [("dev", dev), ("test", test)]:
        valid_titles = set()
        hall_titles = set()
        for e in entries:
            t = e["fields"].get("title", "").strip().lower()
            if e["label"] == "VALID":
                valid_titles.add(t)
            else:
                hall_titles.add(t)
        overlap = valid_titles & hall_titles
        if overlap:
            print(f"  WARNING: {split_name} still has {len(overlap)} title overlaps!")
            violations += len(overlap)
    return violations


def print_split_stats(entries: list[dict], name: str) -> None:
    valid = [e for e in entries if e["label"] == "VALID"]
    hall = [e for e in entries if e["label"] == "HALLUCINATED"]
    types = Counter(e["hallucination_type"] for e in hall)
    methods = Counter(e["generation_method"] for e in entries)
    btypes = Counter(e["bibtex_type"] for e in entries)

    print(f"\n  {name}: {len(entries)} entries ({len(valid)} valid, {len(hall)} hall)")
    print(f"    bibtex_types: {dict(btypes)}")
    print(f"    gen_methods: {dict(methods)}")
    print(f"    hall_types: {dict(sorted(types.items()))}")

    # Year range check
    valid_years = [int(e["fields"]["year"]) for e in valid if e["fields"].get("year", "").isdigit()]
    hall_years = [int(e["fields"]["year"]) for e in hall if e["fields"].get("year", "").isdigit()]
    future_date_years = [
        int(e["fields"]["year"])
        for e in hall
        if e.get("hallucination_type") == "future_date" and e["fields"].get("year", "").isdigit()
    ]
    non_fd_hall_years = [y for y in hall_years if y not in future_date_years]
    print(f"    valid years: {min(valid_years)}-{max(valid_years)}")
    if non_fd_hall_years:
        non_fd = [
            int(e["fields"]["year"])
            for e in hall
            if e.get("hallucination_type") != "future_date"
            and e["fields"].get("year", "").isdigit()
        ]
        print(f"    hall years (excl future_date): {min(non_fd)}-{max(non_fd)}")
    if future_date_years:
        print(f"    future_date years: {min(future_date_years)}-{max(future_date_years)}")

    # Field presence check
    hall_only_fields: set[str] = set()
    valid_field_set = set()
    for e in valid:
        valid_field_set.update(e["fields"].keys())
    for e in hall:
        for f in e["fields"]:
            if f not in valid_field_set:
                hall_only_fields.add(f)
    if hall_only_fields:
        print(f"    WARNING: hall-only fields remain: {hall_only_fields}")
    else:
        print("    OK: no hall-only fields")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize HALLMARK dataset (P0 fixes)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("HALLMARK Dataset Sanitization — Phase 1 (P0 fixes)")
    print("=" * 70)

    # Load current splits
    print("\n1. Loading data...")
    dev = load_jsonl(DATA_DIR / "dev_public.jsonl")
    test = load_jsonl(DATA_DIR / "test_public.jsonl")
    all_entries = dev + test
    print(f"   Loaded {len(dev)} dev + {len(test)} test = {len(all_entries)} total")

    # P0.6: Fix real_world_incidents.jsonl source file
    print("\n2. P0.6: Fixing real_world_incidents.jsonl...")
    rw_path = DATA_DIR / "real_world_incidents.jsonl"
    if rw_path.exists():
        rw_entries = load_jsonl(rw_path)
        rw_before = len(rw_entries)
        rw_entries = [e for e in rw_entries if e["bibtex_key"] not in FAKE_REALWORLD_KEYS]
        print(
            f"   Removed {rw_before - len(rw_entries)} fabricated entries from real_world_incidents.jsonl"
        )
        if not args.dry_run:
            save_jsonl(rw_entries, rw_path)

    # P0.5: Fix temporal distribution
    print("\n3. P0.5: Fixing temporal distribution...")
    year_fixes = 0
    for entry in all_entries:
        old_year = entry["fields"].get("year", "")
        fix_temporal_distribution(entry)
        if entry["fields"].get("year", "") != old_year:
            year_fixes += 1
    print(f"   Fixed {year_fixes} out-of-range years")

    # P0.2: Fix bibtex_type distribution
    print("\n4. P0.2: Normalizing bibtex_type distribution...")
    type_fixes = 0
    for entry in all_entries:
        old_type = entry["bibtex_type"]
        fix_bibtex_type(entry)
        if entry["bibtex_type"] != old_type:
            type_fixes += 1
    print(f"   Normalized {type_fixes} entries from article/misc → inproceedings")

    # P0.3: Strip hallucination-only fields
    print("\n5. P0.3: Stripping hallucination-only fields...")
    field_strips = 0
    for entry in all_entries:
        old_keys = set(entry["fields"].keys())
        strip_leaky_fields(entry)
        new_keys = set(entry["fields"].keys())
        if old_keys != new_keys:
            field_strips += 1
    print(f"   Stripped leaky fields from {field_strips} entries")

    # Verify no hallucination-only fields remain
    valid_entries = [e for e in all_entries if e["label"] == "VALID"]
    hall_entries = [e for e in all_entries if e["label"] == "HALLUCINATED"]
    valid_field_set = set()
    for e in valid_entries:
        valid_field_set.update(e["fields"].keys())
    remaining_hall_only: set[str] = set()
    for e in hall_entries:
        for f in e["fields"]:
            if f not in valid_field_set:
                remaining_hall_only.add(f)
    if remaining_hall_only:
        print(f"   WARNING: hall-only fields still present: {remaining_hall_only}")
        # Strip these too
        for entry in all_entries:
            for f in remaining_hall_only:
                entry["fields"].pop(f, None)
        print(f"   Stripped additional fields: {remaining_hall_only}")

    # P0.4: Re-split with source separation
    print("\n6. P0.4: Re-splitting with source-hallucination separation...")
    dev_new, test_new = resplit_with_source_separation(all_entries, args.seed)

    # Verify
    violations = verify_no_title_overlap(dev_new, test_new)
    if violations:
        print(f"   ERROR: {violations} title overlaps remain after re-split!")
        sys.exit(1)
    else:
        print("   OK: No same-split title overlaps between valid and hallucinated entries")

    # P0.1: Anonymize keys and shuffle
    print("\n7. P0.1: Anonymizing bibtex_keys and shuffling...")
    dev_new = anonymize_and_shuffle(dev_new, "dev", args.seed)
    test_new = anonymize_and_shuffle(test_new, "test", args.seed + 1)

    # Verify key uniqueness
    all_keys = [e["bibtex_key"] for e in dev_new + test_new]
    dupes = [k for k, c in Counter(all_keys).items() if c > 1]
    if dupes:
        print(f"   ERROR: Duplicate keys: {dupes[:5]}")
        sys.exit(1)
    print(
        f"   Keys anonymized: hallmark_dev_0000..hallmark_dev_{len(dev_new) - 1:04d}, "
        f"hallmark_test_0000..hallmark_test_{len(test_new) - 1:04d}"
    )

    # Print statistics
    print("\n" + "=" * 70)
    print("Final split statistics:")
    print_split_stats(dev_new, "dev_public")
    print_split_stats(test_new, "test_public")

    # Save
    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print("\n8. Writing sanitized splits...")
        save_jsonl(dev_new, DATA_DIR / "dev_public.jsonl")
        save_jsonl(test_new, DATA_DIR / "test_public.jsonl")
        print("   Written: dev_public.jsonl, test_public.jsonl, real_world_incidents.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
