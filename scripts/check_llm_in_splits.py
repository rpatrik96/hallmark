#!/usr/bin/env python3
"""Check if llm_generated entries are in splits and compare metadata."""

import json
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "v1.0"

# Load all llm_generated entries
llm_entries = []
with open(data_dir / "llm_generated.jsonl") as f:
    for line in f:
        if line.strip():
            llm_entries.append(json.loads(line))

print(f"Total in llm_generated.jsonl: {len(llm_entries)}")

# Check mislabeled count
mislabeled = [e for e in llm_entries if e.get("subtests", {}).get("cross_db_agreement", False)]
print(f"Mislabeled (cross_db_agreement=True): {len(mislabeled)}")

# Count LLM entries in splits
dev_llm = 0
test_llm = 0

with open(data_dir / "dev_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                dev_llm += 1

with open(data_dir / "test_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                test_llm += 1

print(f"\nLLM entries in dev_public: {dev_llm}")
print(f"LLM entries in test_public: {test_llm}")
print(f"Total in splits: {dev_llm + test_llm}")
print(f"\nDifference: {len(llm_entries) - (dev_llm + test_llm)}")

# Check if any entry titles match
print("\n" + "=" * 60)
print("Checking for title matches between llm_generated and splits...")
print("=" * 60)

llm_titles = {e.get("fields", {}).get("title", ""): e for e in mislabeled}

dev_matches = 0
test_matches = 0

with open(data_dir / "dev_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                title = entry.get("fields", {}).get("title", "")
                if title in llm_titles:
                    dev_matches += 1
                    print(f"\nDEV MATCH: {entry['bibtex_key']}")
                    print(f"  Title: {title[:60]}...")

with open(data_dir / "test_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                title = entry.get("fields", {}).get("title", "")
                if title in llm_titles:
                    test_matches += 1
                    print(f"\nTEST MATCH: {entry['bibtex_key']}")
                    print(f"  Title: {title[:60]}...")

print(f"\n{'=' * 60}")
print(f"Mislabeled entries found in dev_public by title: {dev_matches}")
print(f"Mislabeled entries found in test_public by title: {test_matches}")
print(f"Total mislabeled in splits: {dev_matches + test_matches}")
