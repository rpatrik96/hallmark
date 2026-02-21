#!/usr/bin/env python3
"""Verify sub-test labels against entry metadata for logical consistency.  [analysis]

Reports agreement between assigned sub-tests and what can be inferred from
entry fields without API calls. For full live verification (DOI resolution,
title search), use `hallmark evaluate --verify-subtests`.

Usage:
    python scripts/verify_subtests.py [--fix]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def verify_entry_subtests(entry: dict) -> dict[str, dict]:
    """Check logical consistency of sub-tests for a single entry.

    Returns dict of {subtest_name: {assigned, expected, consistent, reason}}.
    """
    fields = entry.get("fields", {})
    subtests = entry.get("subtests", {})
    label = entry.get("label", "")
    h_type = entry.get("hallucination_type")
    results = {}

    # doi_resolves: check against DOI field presence
    has_doi = bool(fields.get("doi"))
    assigned_dr = subtests.get("doi_resolves")
    if label == "VALID":
        expected_dr = True if has_doi else None
    else:
        # For hallucinated: doi_resolves depends on type
        expected_dr = assigned_dr  # trust generator unless we can infer
        if h_type == "fabricated_doi":
            expected_dr = False
        elif h_type in ("hybrid_fabrication", "partial_author_list", "merged_citation"):
            expected_dr = bool(has_doi)

    results["doi_resolves"] = {
        "assigned": assigned_dr,
        "expected": expected_dr,
        "consistent": assigned_dr == expected_dr,
        "reason": f"has_doi={has_doi}, label={label}, type={h_type}",
    }

    # venue_real: valid entries and most hallucinated types use real venues
    assigned_vr = subtests.get("venue_real")
    if h_type == "nonexistent_venue":
        expected_vr = False
    elif label == "VALID":
        expected_vr = True
    else:
        expected_vr = assigned_vr  # trust generator for other types

    results["venue_real"] = {
        "assigned": assigned_vr,
        "expected": expected_vr,
        "consistent": assigned_vr == expected_vr,
    }

    # fields_complete: check that required BibTeX fields exist
    required = {"title", "author", "year"}
    has_all = required.issubset(set(fields.keys()))
    assigned_fc = subtests.get("fields_complete")
    results["fields_complete"] = {
        "assigned": assigned_fc,
        "expected": bool(has_all),
        "consistent": assigned_fc == bool(has_all),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify sub-test label consistency")
    parser.add_argument("--fix", action="store_true", help="Fix inconsistencies in-place")
    args = parser.parse_args()

    splits = {
        "dev_public": Path("data/v1.0/dev_public.jsonl"),
        "test_public": Path("data/v1.0/test_public.jsonl"),
        "test_hidden": Path("data/hidden/test_hidden.jsonl"),
    }

    overall_stats: dict[str, Counter] = defaultdict(Counter)

    for split_name, path in splits.items():
        entries = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))

        inconsistencies = 0
        total_checks = 0

        for entry in entries:
            results = verify_entry_subtests(entry)
            for subtest_name, result in results.items():
                total_checks += 1
                if result["consistent"]:
                    overall_stats[subtest_name]["consistent"] += 1
                else:
                    overall_stats[subtest_name]["inconsistent"] += 1
                    inconsistencies += 1

                    if args.fix and result["expected"] is not None:
                        entry["subtests"][subtest_name] = result["expected"]

        agreement = (total_checks - inconsistencies) / total_checks * 100
        print(
            f"{split_name}: {agreement:.1f}% agreement ({inconsistencies}/{total_checks} inconsistent)"
        )

        if args.fix:
            with open(path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"  Fixed {inconsistencies} inconsistencies")

    print("\nPer-subtest agreement:")
    for subtest_name, counts in sorted(overall_stats.items()):
        total = counts["consistent"] + counts["inconsistent"]
        pct = counts["consistent"] / total * 100
        print(f"  {subtest_name}: {pct:.1f}% ({counts['consistent']}/{total})")


if __name__ == "__main__":
    main()
