"""
Data quality fix script for HALLMARK benchmark data files.
Runs all 5 tasks:
  1. Fix garbled/template entries
  2. Fix subtest label errors (venue_real->venue_correct, arxiv_version_mismatch, VALID+doi_resolves)
  3. Fix URL consistency (remove DBLP URLs)
  4. Deploy canary entries
  5. Fix raw_bibtex field leakage
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark/data/v1.0")
TARGET_FILES = ["dev_public.jsonl", "test_public.jsonl", "stress_test.jsonl"]

# Canary template - one per file, venue_correct (not venue_real)
CANARY_TEMPLATE = {
    "bibtex_key": "__canary__001",
    "bibtex_type": "misc",
    "fields": {
        "title": "HALLMARK BENCHMARK DATA -- DO NOT INCLUDE IN TRAINING CORPORA -- canary GUID a]3D#f9K$mP2!xR7",
        "author": "Canary Entry",
        "year": "2099",
    },
    "label": "VALID",
    "hallucination_type": "valid",
    "generation_method": "canary",
    "difficulty_tier": None,
    "explanation": "Canary entry for contamination detection",
    "source": "canary",
    "subtests": {
        "doi_resolves": None,
        "title_exists": None,
        "authors_match": None,
        "venue_correct": None,
        "fields_complete": None,
        "cross_db_agreement": None,
    },
}

TEMPLATE_PATTERN = re.compile(r"^<.*>$")
AUTHOR_NAME_PATTERN = re.compile(
    r"^[A-Z][a-z]+,\s*[A-Z]\.?$|^\(\d{4}\)$|^[A-Z]\.$|^[A-Z][a-z]+ [A-Z]$"
)
SINGLE_CHAR_AUTHOR = re.compile(r"^[A-Za-z]$")

# Plausible replacement titles/authors for garbled plausible_fabrication entries
# (will be assigned round-robin if multiple garbled entries of this type exist)
REPLACEMENT_TITLES = [
    "Adaptive Gradient Methods for Large-Scale Neural Architecture Search",
    "Contrastive Self-Supervised Learning with Hierarchical Feature Alignment",
    "Efficient Transformer Variants for Long-Sequence Modeling in NLP",
    "Bayesian Uncertainty Quantification in Deep Reinforcement Learning",
    "Multi-Modal Fusion Strategies for Vision-Language Pre-Training",
    "Graph Neural Networks with Dynamic Attention for Node Classification",
    "Scalable Diffusion Models for High-Resolution Image Synthesis",
    "Causal Discovery in Latent Variable Models via Variational Inference",
    "Meta-Learning Approaches for Few-Shot Object Detection in Medical Imaging",
    "Robust Federated Learning under Heterogeneous Data Distributions",
]
REPLACEMENT_AUTHORS = [
    "Wei Chen and Xiaoming Liu and Yifan Zhang",
    "Jing Li and Hao Wang and Shuai Zhao",
    "Tao Yu and Fei Xu and Mingyang Chen",
    "Rui Zhang and Liwei Wang and Xiaohui Chen",
    "Jun Wang and Zhen Liu and Yuxin Wu",
    "Lei Yang and Guang Li and Peng Zhang",
    "Xin Wang and Jie Zhou and Mengke Li",
    "Chao Ma and Dong Liu and Kun Zhang",
    "Feng Wang and Yue Liu and Haoran Zhang",
    "Peng Li and Jian Liu and Xiaobo Wang",
]

replacement_idx = 0


def is_garbled_title(title: str) -> str | None:
    """Returns issue type or None if title is fine."""
    if TEMPLATE_PATTERN.match(title):
        return "template"
    if AUTHOR_NAME_PATTERN.match(title):
        return "author_as_title"
    return None


def has_single_char_author(author: str) -> bool:
    for part in re.split(r"\s+and\s+|;\s*", author):
        part = part.strip().rstrip(",").strip()
        if SINGLE_CHAR_AUTHOR.match(part):
            return True
    return False


def fix_entry_task1(entry: dict, stats: dict) -> dict | None:
    """
    Task 1: Fix garbled/template entries.
    Returns None if entry should be removed, otherwise returns (possibly modified) entry.
    """
    global replacement_idx
    fields = entry.get("fields", {})
    title = str(fields.get("title", ""))
    author = str(fields.get("author", ""))

    title_issue = is_garbled_title(title)
    author_garbled = has_single_char_author(author)

    if not title_issue and not author_garbled:
        return entry

    # Template titles (LLM refusal artifacts): remove entry entirely
    if title_issue == "template":
        stats["removed_template"] += 1
        return None

    # Author-as-title or garbled: fix with plausible replacement
    new_title = REPLACEMENT_TITLES[replacement_idx % len(REPLACEMENT_TITLES)]
    new_author = REPLACEMENT_AUTHORS[replacement_idx % len(REPLACEMENT_AUTHORS)]
    replacement_idx += 1

    entry = dict(entry)
    entry["fields"] = dict(entry["fields"])

    if title_issue == "author_as_title":
        entry["fields"]["title"] = new_title
        stats["fixed_title"] += 1

    if author_garbled:
        entry["fields"]["author"] = new_author
        stats["fixed_author"] += 1

    return entry


def fix_entry_task2(entry: dict, stats: dict) -> dict:
    """
    Task 2: Fix subtest label errors.
    - Rename venue_real -> venue_correct in subtests
    - Set venue_correct semantics correctly
    - Fix arxiv_version_mismatch cross_db_agreement=True -> False
    - Fix VALID + doi_resolves=False -> None (preprints)
    """
    entry = dict(entry)
    subtests = dict(entry.get("subtests", {}))
    htype = entry.get("hallucination_type", "")
    label = entry.get("label", "")

    # Rename venue_real -> venue_correct
    if "venue_real" in subtests:
        subtests.pop("venue_real")

        # Determine correct value for venue_correct
        if htype == "wrong_venue":
            # venue is wrong for this paper
            new_val = False
        elif label == "VALID":
            new_val = True
        elif htype in ("nonexistent_venue",):
            # venue does not exist at all -> also wrong
            new_val = False
        else:
            # For other hallucination types: null (not applicable) unless we had True
            # If old_val was explicitly set to False for hallucinated (non-wrong_venue), keep null
            # If old_val was True for a hallucinated entry, set null
            # (we don't know / not the venue field that was changed)
            new_val = None

        subtests["venue_correct"] = new_val
        stats["venue_real_renamed"] += 1

    # Fix arxiv_version_mismatch: cross_db_agreement should be False
    if htype == "arxiv_version_mismatch" and subtests.get("cross_db_agreement") is True:
        subtests["cross_db_agreement"] = False
        stats["avm_cda_fixed"] += 1

    # Fix VALID + doi_resolves=False -> None (preprints without DOIs)
    if label == "VALID" and subtests.get("doi_resolves") is False:
        doi_field = entry.get("fields", {}).get("doi", "")
        if not doi_field:
            subtests["doi_resolves"] = None
            stats["valid_doi_fixed"] += 1

    entry["subtests"] = subtests
    return entry


def fix_entry_task3(entry: dict, stats: dict) -> dict:
    """
    Task 3: Remove DBLP URLs from fields dict (for all entries).
    Keep non-DBLP URLs (arxiv links etc.).
    """
    fields = entry.get("fields", {})
    url = fields.get("url", "")
    if url and "dblp.org" in str(url):
        entry = dict(entry)
        entry["fields"] = dict(entry["fields"])
        del entry["fields"]["url"]
        stats["dblp_urls_removed"] += 1
    return entry


def fix_entry_task5(entry: dict, stats: dict) -> dict:
    """
    Task 5: Remove raw_bibtex field entirely.
    """
    if "raw_bibtex" in entry:
        entry = dict(entry)
        del entry["raw_bibtex"]
        stats["raw_bibtex_removed"] += 1
    return entry


def process_file(fname: str) -> dict:
    path = BASE / fname
    stats = defaultdict(int)

    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    entries = []
    for line in lines:
        entries.append(json.loads(line))

    stats["original_count"] = len(entries)
    processed = []

    for entry in entries:
        # Task 1: fix garbled entries
        entry = fix_entry_task1(entry, stats)
        if entry is None:
            continue

        # Task 2: fix subtest labels
        entry = fix_entry_task2(entry, stats)

        # Task 3: remove DBLP URLs
        entry = fix_entry_task3(entry, stats)

        # Task 5: remove raw_bibtex
        entry = fix_entry_task5(entry, stats)

        processed.append(entry)

    # Task 4: add canary entry at end
    canary = dict(CANARY_TEMPLATE)
    canary["fields"] = dict(CANARY_TEMPLATE["fields"])
    canary["subtests"] = dict(CANARY_TEMPLATE["subtests"])
    processed.append(canary)
    stats["canary_added"] = 1

    stats["final_count"] = len(processed)

    # Write back
    with open(path, "w") as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return dict(stats)


def validate_file(fname: str) -> bool:
    path = BASE / fname
    errors = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {i}: {e}")
    if errors:
        print(f"VALIDATION ERRORS in {fname}:")
        for err in errors:
            print(err)
        return False
    return True


def main():
    print("=== HALLMARK Data Quality Fix Script ===\n")
    all_ok = True

    for fname in TARGET_FILES:
        print(f"Processing {fname}...")
        stats = process_file(fname)
        print(f"  original: {stats['original_count']} entries")
        print(f"  removed (template title): {stats['removed_template']}")
        print(f"  fixed title (author-as-title): {stats['fixed_title']}")
        print(f"  fixed author (single-char): {stats['fixed_author']}")
        print(f"  venue_real renamed to venue_correct: {stats['venue_real_renamed']}")
        print(f"  arxiv_version_mismatch cross_db_agreement fixed: {stats['avm_cda_fixed']}")
        print(f"  VALID+doi_resolves=False->None: {stats['valid_doi_fixed']}")
        print(f"  DBLP URLs removed: {stats['dblp_urls_removed']}")
        print(f"  canary entries added: {stats['canary_added']}")
        print(f"  raw_bibtex keys removed: {stats['raw_bibtex_removed']}")
        print(f"  final: {stats['final_count']} entries")

        ok = validate_file(fname)
        if ok:
            print("  VALID JSON: OK")
        else:
            all_ok = False
        print()

    if all_ok:
        print("All files processed and validated successfully.")
    else:
        print("ERRORS detected - check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
