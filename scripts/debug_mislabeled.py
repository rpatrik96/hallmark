#!/usr/bin/env python3
"""Debug script to understand why mislabeled entries weren't found in splits."""

import json
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "v1.0"

# Get mislabeled keys
mislabeled_keys = set()
with open(data_dir / "llm_generated.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("subtests", {}).get("cross_db_agreement", False):
                mislabeled_keys.add(entry["bibtex_key"])

print(f"Mislabeled keys: {len(mislabeled_keys)}")
print(f"First 3: {sorted(mislabeled_keys)[:3]}")

# Check dev_public.jsonl
dev_llm_keys = set()
with open(data_dir / "dev_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                dev_llm_keys.add(entry["bibtex_key"])

print(f"\nLLM-generated entries in dev_public: {len(dev_llm_keys)}")
print(f"First 3: {sorted(dev_llm_keys)[:3]}")

# Check test_public.jsonl
test_llm_keys = set()
with open(data_dir / "test_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                test_llm_keys.add(entry["bibtex_key"])

print(f"\nLLM-generated entries in test_public: {len(test_llm_keys)}")
print(f"First 3: {sorted(test_llm_keys)[:3]}")

# Check overlap
dev_overlap = mislabeled_keys & dev_llm_keys
test_overlap = mislabeled_keys & test_llm_keys

print(f"\nDev overlap: {len(dev_overlap)}")
if dev_overlap:
    print(f"Examples: {sorted(dev_overlap)[:5]}")

print(f"\nTest overlap: {len(test_overlap)}")
if test_overlap:
    print(f"Examples: {sorted(test_overlap)[:5]}")

print(f"\nTotal mislabeled in splits: {len(dev_overlap) + len(test_overlap)}")
