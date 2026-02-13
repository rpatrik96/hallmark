#!/usr/bin/env python3
"""Wave 2 Integration: Merge enriched files, LLM-generated, and real-world entries.

Steps:
1. Replace dev/test with DOI-enriched versions
2. Split LLM-generated entries across dev/test/hidden (~60/50/40)
3. Split real-world entries across dev/test/hidden (~15/10/10 or proportional)
4. Validate no duplicate bibtex_keys
5. Update metadata.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hallmark.dataset.schema import HALLUCINATION_TIER_MAP, BenchmarkEntry, HallucinationType

DATA_DIR = Path(__file__).parent.parent / "data" / "v1.0"
HIDDEN_DIR = Path(__file__).parent.parent / "data" / "hidden"

# Split ratios for new entries (dev, test, hidden)
SPLIT_RATIOS = {"dev_public": 0.4, "test_public": 0.33, "test_hidden": 0.27}


def load_jsonl(path: Path) -> list[BenchmarkEntry]:
    """Load entries from a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))
    return entries


def save_jsonl(entries: list[BenchmarkEntry], path: Path) -> None:
    """Save entries to a JSONL file."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    print(f"  Saved {len(entries)} entries to {path.name}")


def get_all_keys(entries: list[BenchmarkEntry]) -> set[str]:
    """Get all bibtex_keys from entries."""
    return {e.bibtex_key for e in entries}


def split_entries_stratified(
    entries: list[BenchmarkEntry],
    split_prefix: str,
) -> dict[str, list[BenchmarkEntry]]:
    """Split entries across dev/test/hidden, stratified by hallucination_type.

    Assigns new bibtex_keys with split-specific prefixes.
    """
    # Group by hallucination type
    by_type: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        ht = entry.hallucination_type or "unknown"
        by_type[ht].append(entry)

    splits: dict[str, list[BenchmarkEntry]] = {
        "dev_public": [],
        "test_public": [],
        "test_hidden": [],
    }

    for ht, type_entries in by_type.items():
        random.shuffle(type_entries)
        n = len(type_entries)

        # Proportional split
        n_dev = max(1, round(n * SPLIT_RATIOS["dev_public"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test_public"]))
        dev_entries = type_entries[:n_dev]
        test_entries = type_entries[n_dev : n_dev + n_test]
        hidden_entries = type_entries[n_dev + n_test :]

        # Assign split-specific keys
        for i, entry in enumerate(dev_entries):
            entry.bibtex_key = f"{split_prefix}_{ht}_dev_{i}"
            splits["dev_public"].append(entry)
        for i, entry in enumerate(test_entries):
            entry.bibtex_key = f"{split_prefix}_{ht}_test_{i}"
            splits["test_public"].append(entry)
        for i, entry in enumerate(hidden_entries):
            entry.bibtex_key = f"{split_prefix}_{ht}_hidden_{i}"
            splits["test_hidden"].append(entry)

    return splits


def compute_metadata(entries: list[BenchmarkEntry]) -> dict:
    """Compute metadata statistics for a list of entries."""
    total = len(entries)
    valid = sum(1 for e in entries if e.label == "VALID")
    hallucinated = total - valid

    tier_dist: Counter[str] = Counter()
    type_dist: Counter[str] = Counter()
    gen_method_dist: Counter[str] = Counter()

    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            ht = HallucinationType(e.hallucination_type)
            tier = HALLUCINATION_TIER_MAP[ht].value
            tier_dist[str(tier)] += 1
            type_dist[e.hallucination_type] += 1
        gen_method_dist[e.generation_method or "synthetic"] += 1

    return {
        "total": total,
        "valid": valid,
        "hallucinated": hallucinated,
        "tier_distribution": dict(sorted(tier_dist.items())),
        "type_distribution": dict(sorted(type_dist.items())),
        "generation_method_distribution": dict(sorted(gen_method_dist.items())),
    }


def main() -> int:
    random.seed(42)  # Reproducibility

    print("=" * 70)
    print("HALLMARK Wave 2: Dataset Integration")
    print("=" * 70)

    # Step 1: Load base files (enriched if available, otherwise originals)
    print("\n--- Step 1: Load base files ---")

    dev_enriched = DATA_DIR / "dev_public_enriched.jsonl"
    test_enriched = DATA_DIR / "test_public_enriched.jsonl"

    if dev_enriched.exists():
        dev = load_jsonl(dev_enriched)
        print(f"  Loaded dev_public (enriched): {len(dev)} entries")
    else:
        dev = load_jsonl(DATA_DIR / "dev_public.jsonl")
        print(f"  Loaded dev_public (original): {len(dev)} entries")

    if test_enriched.exists():
        test = load_jsonl(test_enriched)
        print(f"  Loaded test_public (enriched): {len(test)} entries")
    else:
        test = load_jsonl(DATA_DIR / "test_public.jsonl")
        print(f"  Loaded test_public (original): {len(test)} entries")

    hidden_path = HIDDEN_DIR / "test_hidden.jsonl"
    if hidden_path.exists():
        hidden = load_jsonl(hidden_path)
        print(f"  Loaded test_hidden: {len(hidden)} entries")
    else:
        print("  WARNING: test_hidden.jsonl not found, starting empty")
        hidden = []

    # Step 2: Load LLM-generated entries
    print("\n--- Step 2: Load new entries ---")

    llm_path = DATA_DIR / "llm_generated.jsonl"
    llm_entries: list[BenchmarkEntry] = []
    if llm_path.exists():
        llm_entries = load_jsonl(llm_path)
        print(f"  LLM-generated: {len(llm_entries)} entries")
    else:
        print("  No LLM-generated entries found (skipping)")

    # Load real-world entries
    rw_path = DATA_DIR / "real_world_incidents.jsonl"
    rw_entries: list[BenchmarkEntry] = []
    if rw_path.exists():
        rw_entries = load_jsonl(rw_path)
        print(f"  Real-world: {len(rw_entries)} entries")
    else:
        print("  No real-world entries found (skipping)")

    if not llm_entries and not rw_entries:
        print("\nNo new entries to integrate. Done.")
        return 0

    # Step 3: Split new entries across splits
    print("\n--- Step 3: Split new entries ---")

    if llm_entries:
        llm_splits = split_entries_stratified(llm_entries, "llm")
        for split_name, entries in llm_splits.items():
            print(f"  LLM → {split_name}: {len(entries)} entries")
    else:
        llm_splits = {"dev_public": [], "test_public": [], "test_hidden": []}

    if rw_entries:
        rw_splits = split_entries_stratified(rw_entries, "rw")
        for split_name, entries in rw_splits.items():
            print(f"  Real-world → {split_name}: {len(entries)} entries")
    else:
        rw_splits = {"dev_public": [], "test_public": [], "test_hidden": []}

    # Step 4: Merge into base splits
    print("\n--- Step 4: Merge ---")

    dev.extend(llm_splits["dev_public"])
    dev.extend(rw_splits["dev_public"])

    test.extend(llm_splits["test_public"])
    test.extend(rw_splits["test_public"])

    hidden.extend(llm_splits["test_hidden"])
    hidden.extend(rw_splits["test_hidden"])

    print(f"  dev_public: {len(dev)} entries")
    print(f"  test_public: {len(test)} entries")
    print(f"  test_hidden: {len(hidden)} entries")

    # Step 5: Validate NEW entries don't collide
    print("\n--- Step 5: Validate ---")

    new_keys: set[str] = set()
    for split_entries in [llm_splits, rw_splits]:
        for entries_list in split_entries.values():
            new_keys.update(e.bibtex_key for e in entries_list)

    dev_keys = get_all_keys(dev)
    test_keys = get_all_keys(test)
    hidden_keys = get_all_keys(hidden)

    # Check new entries don't collide across splits (pre-existing dupes are tolerated)
    new_cross_split = set()
    for e in llm_splits["dev_public"] + rw_splits["dev_public"]:
        if e.bibtex_key in test_keys or e.bibtex_key in hidden_keys:
            new_cross_split.add(e.bibtex_key)
    for e in llm_splits["test_public"] + rw_splits["test_public"]:
        if e.bibtex_key in dev_keys or e.bibtex_key in hidden_keys:
            new_cross_split.add(e.bibtex_key)
    for e in llm_splits["test_hidden"] + rw_splits["test_hidden"]:
        if e.bibtex_key in dev_keys or e.bibtex_key in test_keys:
            new_cross_split.add(e.bibtex_key)

    if new_cross_split:
        print(f"  ERROR: {len(new_cross_split)} NEW entry keys collide across splits!")
        for k in list(new_cross_split)[:5]:
            print(f"    - {k}")
        return 1

    # Report pre-existing cross-split dupes as info
    pre_existing_dupes = (
        (dev_keys & test_keys) | (dev_keys & hidden_keys) | (test_keys & hidden_keys)
    )
    pre_existing_dupes -= new_keys
    if pre_existing_dupes:
        print(
            f"  INFO: {len(pre_existing_dupes)} pre-existing cross-split duplicate keys (not from new entries)"
        )

    print(f"  New entries validated: {len(new_keys)} keys, no collisions")

    # Step 6: Save updated files
    print("\n--- Step 6: Save ---")

    save_jsonl(dev, DATA_DIR / "dev_public.jsonl")
    save_jsonl(test, DATA_DIR / "test_public.jsonl")
    save_jsonl(hidden, HIDDEN_DIR / "test_hidden.jsonl")

    # Step 7: Update metadata.json
    print("\n--- Step 7: Update metadata ---")

    metadata_path = DATA_DIR / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata["splits"]["dev_public"].update(compute_metadata(dev))
    metadata["splits"]["dev_public"]["file"] = "dev_public.jsonl"

    metadata["splits"]["test_public"].update(compute_metadata(test))
    metadata["splits"]["test_public"]["file"] = "test_public.jsonl"

    metadata["splits"]["test_hidden"].update(compute_metadata(hidden))
    metadata["splits"]["test_hidden"]["file"] = "test_hidden.jsonl"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")

    print(f"  Updated {metadata_path.name}")

    # Summary
    print("\n" + "=" * 70)
    print("Integration complete!")
    print(f"  Total entries: {len(dev) + len(test) + len(hidden)}")
    print(f"  dev_public: {len(dev)}")
    print(f"  test_public: {len(test)}")
    print(f"  test_hidden: {len(hidden)}")

    new_total = len(llm_entries) + len(rw_entries)
    print(
        f"  New entries added: {new_total} (LLM: {len(llm_entries)}, Real-world: {len(rw_entries)})"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
