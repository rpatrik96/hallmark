#!/usr/bin/env python3
"""Audit the released sub-test ground truth against the shipped checkers.

Every benchmark entry ships six declared sub-test outcomes. Those values are the
benchmark's diagnostic ground truth, so they have to agree with what the reference
checkers in ``hallmark.evaluation.subtests`` compute on the same metadata. This
script reports every disagreement, grouped by cause, and exits non-zero if any
disagreement is not on the known-issue list.

Only ``fields_complete`` is auditable offline: the other five checkers need live
database lookups, so they are out of scope here by design.

Run:
    python scripts/verify_subtest_ground_truth.py
    python scripts/verify_subtest_ground_truth.py --data-dir data/v1.0 --json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

from hallmark.evaluation.subtests import check_fields_complete

# Splits carrying per-entry labels in the public release. ``test_hidden`` withholds
# labels, so it cannot be audited from the public artifact.
PUBLIC_SPLITS = ("dev_public", "test_public", "stress_test")

# Disagreements we have diagnosed and are tracking. Each key is
# (label, hallucination_type, direction) where direction is "stale_false" when the
# stored value says the entry is incomplete and the checker says it is complete.
KNOWN_ISSUES: dict[tuple[str, str, str], str] = {
    (
        "VALID",
        "-",
        "stale_false",
    ): "arXiv preprints typed @article: stored value predates the entry-type-aware "
    "spec, which no longer requires `journal` of an arXiv-identified @article",
    (
        "HALLUCINATED",
        "future_date",
        "stale_false",
    ): "the taxonomy declares fields_complete as expected-fail for future_date, but "
    "an implausible year is valid 4-digit syntax; year bounds live in the "
    "pre-screening layer, not in the completeness check",
    (
        "HALLUCINATED",
        "plausible_fabrication",
        "stale_false",
    ): "the declared signature for this type is `varies by entry`, so a per-entry "
    "value can legitimately differ from the type-level expectation",
    (
        "HALLUCINATED",
        "fabricated_doi",
        "stale_true",
    ): "generator artifact: the perturbation replaced an arXiv preprint's DOI and "
    "dropped its url, so the entry no longer identifies as a preprint and "
    "`journal` becomes required. The perturbation introduced a second detectable "
    "defect beyond the intended one",
    (
        "HALLUCINATED",
        "swapped_authors",
        "stale_true",
    ): "generator artifact: @inproceedings entries carrying no booktitle, so they "
    "are incomplete independently of the author perturbation",
}


def audit(data_dir: pathlib.Path, splits: tuple[str, ...]) -> dict:
    per_cause: collections.Counter = collections.Counter()
    per_split: collections.Counter = collections.Counter()
    scored: collections.Counter = collections.Counter()
    samples: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)

    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        for line in path.open():
            entry = json.loads(line)
            stored = entry.get("subtests", {}).get("fields_complete")
            if stored is None:
                continue
            scored[split] += 1
            computed = check_fields_complete(entry["bibtex_type"], entry["fields"]).passed
            if computed == stored:
                continue
            direction = "stale_false" if stored is False else "stale_true"
            key = (
                entry["label"],
                entry.get("hallucination_type") or "-",
                direction,
            )
            per_cause[key] += 1
            per_split[split] += 1
            if len(samples[key]) < 3:
                samples[key].append(entry["bibtex_key"])

    return {
        "scored_per_split": dict(scored),
        "disagreements_per_split": dict(per_split),
        "disagreements_by_cause": {"|".join(k): v for k, v in sorted(per_cause.items())},
        "total_disagreements": sum(per_cause.values()),
        "samples": {"|".join(k): v for k, v in sorted(samples.items())},
        "unknown_causes": {
            "|".join(k): v for k, v in sorted(per_cause.items()) if k not in KNOWN_ISSUES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/v1.0", type=pathlib.Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    report = audit(args.data_dir, PUBLIC_SPLITS)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"fields_complete audit over {args.data_dir}")
        print(f"  scored entries: {report['scored_per_split']}")
        print(f"  disagreements:  {report['disagreements_per_split']}")
        print(f"  total:          {report['total_disagreements']}")
        print()
        for cause, count in report["disagreements_by_cause"].items():
            label, htype, direction = cause.split("|")
            note = KNOWN_ISSUES.get((label, htype, direction), "UNDIAGNOSED")
            print(f"  {count:4d}  {label}/{htype} ({direction})")
            print(f"        {note}")
            print(f"        e.g. {', '.join(report['samples'][cause])}")
        if report["unknown_causes"]:
            print()
            print("UNDIAGNOSED disagreements present:")
            for cause, count in report["unknown_causes"].items():
                print(f"  {count:4d}  {cause}")

    return 1 if report["unknown_causes"] else 0


if __name__ == "__main__":
    sys.exit(main())
