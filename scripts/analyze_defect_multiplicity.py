#!/usr/bin/env python3
"""Measure how many *fields* are defective per hallucinated entry.

Motivation
----------
Each hallucinated ``BenchmarkEntry`` carries a single ``hallucination_type``
(the perturbation the generator injected) but a six-valued ``subtests`` dict
recording which per-field checks fail. The type label is therefore a statement
about *cause*, not about how many fields are wrong. This script quantifies the
gap: the distribution of defective-field counts per hallucinated entry.

``cross_db_agreement`` exclusion
--------------------------------
``cross_db_agreement`` is False for **100%** of hallucinated entries in every
split (verified by this script — see the "constant-field audit" section of the
output). A field that never varies within the hallucinated class carries zero
information about defect multiplicity: including it would add exactly 1 to
every entry's count and shift the whole distribution right by one, making
single-defect entries look like two-defect entries. It is a *label restatement*
("this entry is hallucinated"), not an independent observable defect, so the
headline distribution excludes it. Both counts are reported so the choice is
auditable.

Usage
-----
    uv run python scripts/analyze_defect_multiplicity.py
    uv run python scripts/analyze_defect_multiplicity.py --splits dev_public
    uv run python scripts/analyze_defect_multiplicity.py --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data/v1.2")
DEFAULT_SPLITS = ["dev_public", "test_public", "stress_test"]

#: Excluded from the headline defect count — see module docstring.
CONSTANT_FIELD = "cross_db_agreement"


def load_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Load a split's JSONL, dropping canary/watermark rows."""
    path = data_dir / f"{split}.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in entries if not e["bibtex_key"].startswith("__canary__")]


def defective_fields(entry: dict[str, Any], *, exclude_constant: bool = True) -> list[str]:
    """Return the subtest names that are explicitly False for *entry*.

    ``None`` means "not applicable" (e.g. ``doi_resolves`` on an entry with no
    DOI) and is *not* a defect, so only ``is False`` counts.
    """
    subtests = entry.get("subtests") or {}
    return sorted(
        name
        for name, value in subtests.items()
        if value is False and not (exclude_constant and name == CONSTANT_FIELD)
    )


def audit_constant_field(hallucinated: list[dict[str, Any]]) -> dict[str, int]:
    """Tally ``cross_db_agreement`` values, to justify excluding it."""
    return dict(Counter(e.get("subtests", {}).get(CONSTANT_FIELD, "MISSING") for e in hallucinated))


def analyze_split(entries: list[dict[str, Any]]) -> dict[str, Any]:
    hallucinated = [e for e in entries if e["label"] == "HALLUCINATED"]

    dist: Counter[int] = Counter()
    dist_with_constant: Counter[int] = Counter()
    by_type: defaultdict[str, Counter[int]] = defaultdict(Counter)
    by_method: defaultdict[str, Counter[int]] = defaultdict(Counter)

    for entry in hallucinated:
        n = len(defective_fields(entry))
        dist[n] += 1
        dist_with_constant[len(defective_fields(entry, exclude_constant=False))] += 1
        by_type[entry.get("hallucination_type") or "unknown"][n] += 1
        by_method[entry.get("generation_method") or "unknown"][n] += 1

    multi = sum(count for n, count in dist.items() if n >= 2)
    return {
        "num_entries": len(entries),
        "num_hallucinated": len(hallucinated),
        "num_valid": sum(1 for e in entries if e["label"] == "VALID"),
        "distribution": dict(sorted(dist.items())),
        "distribution_including_constant_field": dict(sorted(dist_with_constant.items())),
        "num_multi_defect": multi,
        "share_multi_defect": multi / len(hallucinated) if hallucinated else 0.0,
        "constant_field_audit": audit_constant_field(hallucinated),
        "by_type": {t: dict(sorted(c.items())) for t, c in sorted(by_type.items())},
        "by_generation_method": {m: dict(sorted(c.items())) for m, c in sorted(by_method.items())},
    }


def print_report(split: str, r: dict[str, Any]) -> None:
    print(
        f"\n{'=' * 78}\n{split}  (n={r['num_entries']}, "
        f"hallucinated={r['num_hallucinated']}, valid={r['num_valid']})\n{'=' * 78}"
    )

    audit = r["constant_field_audit"]
    print(f"\n  Constant-field audit — '{CONSTANT_FIELD}' on hallucinated entries: {audit}")
    if set(audit) == {False}:
        print("    -> constant (always False): carries no within-class information, excluded.")
    else:
        print("    -> NOT constant; exclusion assumption does not hold for this split!")

    print("\n  Defective fields per hallucinated entry (excluding the constant field):")
    total = r["num_hallucinated"]
    for n, count in r["distribution"].items():
        bar = "#" * round(50 * count / total) if total else ""
        print(f"    {n} defect(s): {count:>4}  ({count / total:5.1%})  {bar}")
    print(f"    >=2 defects: {r['num_multi_defect']:>4}  ({r['share_multi_defect']:5.1%})")
    print(
        f"\n  Same distribution INCLUDING '{CONSTANT_FIELD}' (shifted right by exactly 1): "
        f"{r['distribution_including_constant_field']}"
    )

    print("\n  By hallucination type (types with any multi-defect entries):")
    for t, c in r["by_type"].items():
        if any(n >= 2 for n in c):
            print(f"    {t:<28} {c}")

    print("\n  By generation method:")
    for m, c in r["by_generation_method"].items():
        n_total = sum(c.values())
        n_multi = sum(v for k, v in c.items() if k >= 2)
        share = n_multi / n_total if n_total else 0.0
        print(f"    {m:<20} n={n_total:>4}  >=2 defects: {n_multi:>4} ({share:5.1%})   {c}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    results = {}
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        results[split] = analyze_split(load_split(args.data_dir, split))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for split, r in results.items():
            print_report(split, r)


if __name__ == "__main__":
    main()
