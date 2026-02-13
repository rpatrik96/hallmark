#!/usr/bin/env python3
"""Wave 3 Integration: DOI-enriched base + GPTZero NeurIPS 2025 entries.

Steps:
1. Replace dev/test with DOI-enriched versions (DBLP enrichment)
2. Split GPTZero entries across dev/test/hidden (stratified by type)
3. Validate no duplicate bibtex_keys
4. Update metadata.json
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

SPLIT_RATIOS = {"dev_public": 0.4, "test_public": 0.33, "test_hidden": 0.27}


def load_jsonl(path: Path) -> list[BenchmarkEntry]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_json(line))
    return entries


def save_jsonl(entries: list[BenchmarkEntry], path: Path) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    print(f"  Saved {len(entries)} entries to {path.name}")


def split_entries_stratified(
    entries: list[BenchmarkEntry],
    split_prefix: str,
) -> dict[str, list[BenchmarkEntry]]:
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
        n_dev = max(1, round(n * SPLIT_RATIOS["dev_public"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test_public"]))

        dev_entries = type_entries[:n_dev]
        test_entries = type_entries[n_dev : n_dev + n_test]
        hidden_entries = type_entries[n_dev + n_test :]

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
    random.seed(42)

    print("=" * 70)
    print("HALLMARK Wave 3: DOI Enrichment + GPTZero Integration")
    print("=" * 70)

    # Step 1: Load DOI-enriched base files
    print("\n--- Step 1: Load DOI-enriched base files ---")

    dev_enriched = DATA_DIR / "dev_public_enriched.jsonl"
    test_enriched = DATA_DIR / "test_public_enriched.jsonl"

    if not dev_enriched.exists() or not test_enriched.exists():
        print("  ERROR: Enriched files not found. Run enrich_valid_dois.py first.")
        return 1

    dev = load_jsonl(dev_enriched)
    print(f"  Loaded dev_public (enriched): {len(dev)} entries")

    test = load_jsonl(test_enriched)
    print(f"  Loaded test_public (enriched): {len(test)} entries")

    hidden_path = HIDDEN_DIR / "test_hidden.jsonl"
    if hidden_path.exists():
        hidden = load_jsonl(hidden_path)
        print(f"  Loaded test_hidden: {len(hidden)} entries")
    else:
        print("  WARNING: test_hidden.jsonl not found")
        hidden = []

    # Step 2: Load GPTZero entries
    print("\n--- Step 2: Load GPTZero entries ---")

    gptzero_path = DATA_DIR / "gptzero_neurips2025.jsonl"
    if not gptzero_path.exists():
        print("  ERROR: gptzero_neurips2025.jsonl not found")
        return 1

    gptzero_entries = load_jsonl(gptzero_path)
    print(f"  GPTZero NeurIPS 2025: {len(gptzero_entries)} entries")

    # Step 3: Split GPTZero entries
    print("\n--- Step 3: Split GPTZero entries ---")

    gptzero_splits = split_entries_stratified(gptzero_entries, "gptzero")
    for split_name, entries in gptzero_splits.items():
        print(f"  GPTZero → {split_name}: {len(entries)} entries")

    # Step 4: Merge
    print("\n--- Step 4: Merge ---")

    dev.extend(gptzero_splits["dev_public"])
    test.extend(gptzero_splits["test_public"])
    hidden.extend(gptzero_splits["test_hidden"])

    print(f"  dev_public: {len(dev)} entries")
    print(f"  test_public: {len(test)} entries")
    print(f"  test_hidden: {len(hidden)} entries")
    print(f"  Total: {len(dev) + len(test) + len(hidden)} entries")

    # Step 5: Validate
    print("\n--- Step 5: Validate ---")

    new_keys = set()
    for entries_list in gptzero_splits.values():
        new_keys.update(e.bibtex_key for e in entries_list)

    dev_keys = {e.bibtex_key for e in dev}
    test_keys = {e.bibtex_key for e in test}
    hidden_keys = {e.bibtex_key for e in hidden}

    collisions = set()
    for e in gptzero_splits["dev_public"]:
        if e.bibtex_key in test_keys or e.bibtex_key in hidden_keys:
            collisions.add(e.bibtex_key)
    for e in gptzero_splits["test_public"]:
        if e.bibtex_key in dev_keys or e.bibtex_key in hidden_keys:
            collisions.add(e.bibtex_key)
    for e in gptzero_splits["test_hidden"]:
        if e.bibtex_key in dev_keys or e.bibtex_key in test_keys:
            collisions.add(e.bibtex_key)

    if collisions:
        print(f"  ERROR: {len(collisions)} key collisions!")
        return 1

    print(f"  Validated: {len(new_keys)} new keys, no collisions")

    # Step 6: Save
    print("\n--- Step 6: Save ---")

    save_jsonl(dev, DATA_DIR / "dev_public.jsonl")
    save_jsonl(test, DATA_DIR / "test_public.jsonl")
    save_jsonl(hidden, HIDDEN_DIR / "test_hidden.jsonl")

    # Step 7: Update metadata
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
    total = len(dev) + len(test) + len(hidden)
    print("\n" + "=" * 70)
    print("Integration complete!")
    print(f"  Total entries: {total}")
    print(f"  dev_public: {len(dev)}")
    print(f"  test_public: {len(test)}")
    print(f"  test_hidden: {len(hidden)}")
    print(f"  GPTZero entries added: {len(gptzero_entries)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
