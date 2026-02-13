#!/usr/bin/env python3
"""Remove mislabeled real papers from LLM-generated entries in splits.

The LLM-generation pipeline produced entries that are verified real papers
(cross_db_agreement=True), meaning they closely match real papers in CrossRef/DBLP
with title_sim >= 0.95 AND auth_jacc >= 0.8. These are not hallucinations but
accurate recalls, so they should not be in the benchmark.

This script identifies and REMOVES these entries from dev_public.jsonl and
test_public.jsonl splits.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    data_dir = Path(__file__).parent.parent / "data" / "v1.0"

    # Step 1: Load mislabeled entries from llm_generated.jsonl
    llm_path = data_dir / "llm_generated.jsonl"
    mislabeled_titles = set()

    print("Loading llm_generated.jsonl...")
    with open(llm_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            # Identify entries where cross_db_agreement=True (verified real papers)
            if entry.get("subtests", {}).get("cross_db_agreement", False):
                title = entry.get("fields", {}).get("title", "")
                if title:
                    mislabeled_titles.add(title)

    print(f"Found {len(mislabeled_titles)} mislabeled entries (cross_db_agreement=True)")

    if not mislabeled_titles:
        print("No mislabeled entries found. Exiting.")
        return 0

    # Step 2: Process each split file
    split_files = [
        data_dir / "dev_public.jsonl",
        data_dir / "test_public.jsonl",
    ]

    total_removed = 0

    for split_path in split_files:
        if not split_path.exists():
            print(f"Skipping {split_path.name} (not found)")
            continue

        print(f"\nProcessing {split_path.name}...")

        # Load entries
        entries = []
        removed_from_split = []

        with open(split_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)

                # Check if this entry should be removed (match by title)
                title = entry.get("fields", {}).get("title", "")
                if entry.get("generation_method") == "llm_generated" and title in mislabeled_titles:
                    removed_from_split.append(entry["bibtex_key"])
                else:
                    entries.append(entry)

        # Report removals
        if removed_from_split:
            print(f"  Removing {len(removed_from_split)} entries:")
            for key in removed_from_split:
                print(f"    - {key}")

            # Write updated file
            with open(split_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"  Updated {split_path.name}: {len(entries)} entries remaining")
            total_removed += len(removed_from_split)
        else:
            print(f"  No mislabeled entries found in {split_path.name}")

    # Step 3: Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Mislabeled entries identified: {len(mislabeled_titles)}")
    print(f"Total entries removed from splits: {total_removed}")
    print(f"Not found in splits: {len(mislabeled_titles) - total_removed}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
