#!/usr/bin/env python3
"""Verify that mislabeled entries have been removed from splits."""

import json
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "v1.0"

# Load mislabeled titles
mislabeled_titles = set()
with open(data_dir / "llm_generated.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("subtests", {}).get("cross_db_agreement", False):
                title = entry.get("fields", {}).get("title", "")
                if title:
                    mislabeled_titles.add(title)

print(f"Mislabeled entries in llm_generated.jsonl: {len(mislabeled_titles)}")

# Check dev_public.jsonl
dev_llm = 0
dev_mislabeled = 0
with open(data_dir / "dev_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                dev_llm += 1
                title = entry.get("fields", {}).get("title", "")
                if title in mislabeled_titles:
                    dev_mislabeled += 1
                    print(f"WARNING: Found mislabeled entry in dev: {entry['bibtex_key']}")

# Check test_public.jsonl
test_llm = 0
test_mislabeled = 0
with open(data_dir / "test_public.jsonl") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("generation_method") == "llm_generated":
                test_llm += 1
                title = entry.get("fields", {}).get("title", "")
                if title in mislabeled_titles:
                    test_mislabeled += 1
                    print(f"WARNING: Found mislabeled entry in test: {entry['bibtex_key']}")

print("\nSplit statistics:")
print(f"  dev_public.jsonl: {dev_llm} LLM entries, {dev_mislabeled} mislabeled")
print(f"  test_public.jsonl: {test_llm} LLM entries, {test_mislabeled} mislabeled")
print(f"  Total: {dev_llm + test_llm} LLM entries, {dev_mislabeled + test_mislabeled} mislabeled")

if dev_mislabeled == 0 and test_mislabeled == 0:
    print("\n✓ SUCCESS: All mislabeled entries removed from splits")
else:
    print(f"\n✗ FAILURE: {dev_mislabeled + test_mislabeled} mislabeled entries remain")
