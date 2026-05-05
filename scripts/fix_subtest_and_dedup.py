#!/usr/bin/env python3
"""One-pass data quality fixes for HALLMARK benchmark splits.

Tasks:
  1. Fix wrong_venue / preprint_as_published: cross_db_agreement=false, venue_correct=false
  2. Fix arxiv_version_mismatch: venue_correct=false, cross_db_agreement=false
  3. Fix merged_citation in stress_test: venue_correct=false
  4. Cross-split dedup: remove HALLUCINATED test/stress entries whose title matches a VALID dev entry
  5. Fill empty explanations for VALID entries with "Valid entry scraped from DBLP"
  6. Fix comma-separated author fields (replace ', ' with ' and ')
  7. Update metadata.json
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark/data/v1.0")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_title(entry: dict) -> str:
    return entry.get("fields", {}).get("title", "").strip().lower()


def fix_subtests_venue_types(entry: dict) -> int:
    """Task 1: wrong_venue and preprint_as_published need cross_db_agreement=false, venue_correct=false."""
    htype = entry.get("hallucination_type")
    changes = 0
    if htype in ("wrong_venue", "preprint_as_published"):
        subtests = entry.setdefault("subtests", {})
        if subtests.get("cross_db_agreement") is True:
            subtests["cross_db_agreement"] = False
            changes += 1
        if subtests.get("venue_correct") is not False:
            subtests["venue_correct"] = False
            changes += 1
    return changes


def fix_subtests_arxiv_version(entry: dict) -> int:
    """Task 2: arxiv_version_mismatch needs venue_correct=false and cross_db_agreement=false."""
    htype = entry.get("hallucination_type")
    changes = 0
    if htype == "arxiv_version_mismatch":
        subtests = entry.setdefault("subtests", {})
        if subtests.get("venue_correct") is not False:
            subtests["venue_correct"] = False
            changes += 1
        if subtests.get("cross_db_agreement") is True:
            subtests["cross_db_agreement"] = False
            changes += 1
    return changes


def fix_subtests_merged_citation(entry: dict) -> int:
    """Task 3: merged_citation needs venue_correct=false."""
    htype = entry.get("hallucination_type")
    changes = 0
    if htype == "merged_citation":
        subtests = entry.setdefault("subtests", {})
        if subtests.get("venue_correct") is not False:
            subtests["venue_correct"] = False
            changes += 1
    return changes


def fix_explanation(entry: dict) -> int:
    """Task 5: Fill empty explanations for VALID entries."""
    if entry.get("label") == "VALID" and not entry.get("explanation", "").strip():
        entry["explanation"] = "Valid entry scraped from DBLP"
        return 1
    return 0


def fix_author_separator(entry: dict) -> int:
    """Task 6: Replace comma-only author lists with BibTeX 'and' separator.

    Only applies when the author field contains ', ' but no ' and '.
    """
    author = entry.get("fields", {}).get("author", "")
    if author and " and " not in author and ", " in author:
        entry["fields"]["author"] = re.sub(r",\s+", " and ", author)
        return 1
    return 0


def compute_split_metadata(entries: list[dict], file: str) -> dict:
    total = len(entries)
    valid = sum(1 for e in entries if e.get("label") == "VALID")
    hallucinated = sum(1 for e in entries if e.get("label") == "HALLUCINATED")

    tier_dist: Counter = Counter()
    type_dist: Counter = Counter()
    gen_dist: Counter = Counter()

    for e in entries:
        tier = e.get("difficulty_tier")
        if tier is not None:
            tier_dist[str(tier)] += 1
        htype = e.get("hallucination_type")
        if htype:
            type_dist[htype] += 1
        gen = e.get("generation_method")
        if gen:
            gen_dist[gen] += 1

    return {
        "file": file,
        "total": total,
        "valid": valid,
        "hallucinated": hallucinated,
        "tier_distribution": dict(sorted(tier_dist.items())),
        "type_distribution": dict(sorted(type_dist.items())),
        "generation_methods": dict(sorted(gen_dist.items())),
    }


def main() -> None:
    dev = load_jsonl(DATA_DIR / "dev_public.jsonl")
    test = load_jsonl(DATA_DIR / "test_public.jsonl")
    stress = load_jsonl(DATA_DIR / "stress_test.jsonl")

    stats = {
        "task1": 0,
        "task2": 0,
        "task3": 0,
        "task4_test": 0,
        "task4_stress": 0,
        "task5": 0,
        "task6": 0,
    }

    # Task 4 prep: build set of titles that appear as VALID in dev
    dev_valid_titles = {get_title(e) for e in dev if e["label"] == "VALID" and get_title(e)}
    print(f"Dev VALID unique titles: {len(dev_valid_titles)}")

    # --- Process dev ---
    new_dev: list[dict] = []
    for entry in dev:
        stats["task1"] += fix_subtests_venue_types(entry)
        stats["task2"] += fix_subtests_arxiv_version(entry)
        stats["task3"] += fix_subtests_merged_citation(entry)
        stats["task5"] += fix_explanation(entry)
        stats["task6"] += fix_author_separator(entry)
        new_dev.append(entry)

    # --- Process test ---
    new_test: list[dict] = []
    for entry in test:
        if entry["label"] == "HALLUCINATED" and get_title(entry) in dev_valid_titles:
            stats["task4_test"] += 1
            continue
        stats["task1"] += fix_subtests_venue_types(entry)
        stats["task2"] += fix_subtests_arxiv_version(entry)
        stats["task3"] += fix_subtests_merged_citation(entry)
        stats["task5"] += fix_explanation(entry)
        stats["task6"] += fix_author_separator(entry)
        new_test.append(entry)

    # --- Process stress ---
    new_stress: list[dict] = []
    for entry in stress:
        if entry["label"] == "HALLUCINATED" and get_title(entry) in dev_valid_titles:
            stats["task4_stress"] += 1
            continue
        stats["task2"] += fix_subtests_arxiv_version(entry)
        stats["task3"] += fix_subtests_merged_citation(entry)
        new_stress.append(entry)

    # Write
    save_jsonl(DATA_DIR / "dev_public.jsonl", new_dev)
    save_jsonl(DATA_DIR / "test_public.jsonl", new_test)
    save_jsonl(DATA_DIR / "stress_test.jsonl", new_stress)

    # Update metadata.json
    metadata_path = DATA_DIR / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    hidden_total = metadata["splits"]["test_hidden"]["total"]
    metadata["total_entries"] = len(new_dev) + len(new_test) + len(new_stress) + hidden_total

    metadata["splits"]["dev_public"] = compute_split_metadata(new_dev, "dev_public.jsonl")
    metadata["splits"]["test_public"] = compute_split_metadata(new_test, "test_public.jsonl")
    metadata["splits"]["stress_test"] = compute_split_metadata(new_stress, "stress_test.jsonl")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")

    # Summary
    print("\n=== Fix Summary ===")
    print(f"Task 1 (wrong_venue/preprint_as_published subtest fields fixed): {stats['task1']}")
    print(f"Task 2 (arxiv_version_mismatch subtest fields fixed):            {stats['task2']}")
    print(f"Task 3 (merged_citation venue_correct fixed):                    {stats['task3']}")
    print(f"Task 4 (test entries removed — title in dev VALID):              {stats['task4_test']}")
    print(
        f"Task 4 (stress entries removed — title in dev VALID):            {stats['task4_stress']}"
    )
    print(f"Task 5 (VALID explanations filled):                              {stats['task5']}")
    print(f"Task 6 (author separators fixed):                                {stats['task6']}")
    print()
    print(f"dev:    {len(dev)} → {len(new_dev)} entries")
    print(f"test:   {len(test)} → {len(new_test)} entries")
    print(f"stress: {len(stress)} → {len(new_stress)} entries")
    print(f"total public: {len(new_dev) + len(new_test) + len(new_stress)}")
    print(f"metadata total_entries: {metadata['total_entries']}")

    # Spot-check verification
    print("\n=== Spot-check verification ===")
    dev_check = load_jsonl(DATA_DIR / "dev_public.jsonl")
    test_check = load_jsonl(DATA_DIR / "test_public.jsonl")
    stress_check = load_jsonl(DATA_DIR / "stress_test.jsonl")

    for split_name, entries in [("dev", dev_check), ("test", test_check), ("stress", stress_check)]:
        for htype in [
            "wrong_venue",
            "preprint_as_published",
            "arxiv_version_mismatch",
            "merged_citation",
        ]:
            subset = [e for e in entries if e.get("hallucination_type") == htype]
            if not subset:
                continue
            cda_true = sum(
                1 for e in subset if e.get("subtests", {}).get("cross_db_agreement") is True
            )
            vc_null = sum(1 for e in subset if e.get("subtests", {}).get("venue_correct") is None)
            vc_true = sum(1 for e in subset if e.get("subtests", {}).get("venue_correct") is True)
            vc_false = sum(1 for e in subset if e.get("subtests", {}).get("venue_correct") is False)
            status = "OK" if cda_true == 0 and vc_null == 0 and vc_true == 0 else "FAIL"
            print(
                f"  [{status}] {split_name}/{htype}: n={len(subset)} "
                f"cda_true={cda_true} vc_null={vc_null} vc_true={vc_true} vc_false={vc_false}"
            )

    # Check no dev VALID titles in test/stress as HALLUCINATED
    dev_valid_titles2 = {get_title(e) for e in dev_check if e["label"] == "VALID" and get_title(e)}
    leaks_test = [
        e for e in test_check if e["label"] == "HALLUCINATED" and get_title(e) in dev_valid_titles2
    ]
    leaks_stress = [
        e
        for e in stress_check
        if e["label"] == "HALLUCINATED" and get_title(e) in dev_valid_titles2
    ]
    print(
        f"  [{'OK' if not leaks_test else 'FAIL'}] test HALLUCINATED with dev VALID title: {len(leaks_test)}"
    )
    print(
        f"  [{'OK' if not leaks_stress else 'FAIL'}] stress HALLUCINATED with dev VALID title: {len(leaks_stress)}"
    )

    # Check explanations
    empty_dev = sum(
        1 for e in dev_check if e["label"] == "VALID" and not e.get("explanation", "").strip()
    )
    empty_test = sum(
        1 for e in test_check if e["label"] == "VALID" and not e.get("explanation", "").strip()
    )
    print(f"  [{'OK' if empty_dev == 0 else 'FAIL'}] dev VALID with empty explanation: {empty_dev}")
    print(
        f"  [{'OK' if empty_test == 0 else 'FAIL'}] test VALID with empty explanation: {empty_test}"
    )

    # Check author separators
    def _has_comma_authors(e: dict) -> bool:
        a = e.get("fields", {}).get("author", "")
        return " and " not in a and ", " in a

    comma_dev = sum(1 for e in dev_check if _has_comma_authors(e))
    comma_test = sum(1 for e in test_check if _has_comma_authors(e))
    print(
        f"  [{'OK' if comma_dev == 0 else 'FAIL'}] dev entries with comma-only author separator: {comma_dev}"
    )
    print(
        f"  [{'OK' if comma_test == 0 else 'FAIL'}] test entries with comma-only author separator: {comma_test}"
    )


if __name__ == "__main__":
    main()
