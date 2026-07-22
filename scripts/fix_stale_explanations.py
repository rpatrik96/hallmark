#!/usr/bin/env python3
"""Replace stale '[Re-classified]' explanation strings in the public splits.

Context (v1.2.1): 133 public-split entries carried explanation strings that
were generation-pipeline re-classifier output rather than ground-truth
diagnoses, in several cases contradicting the released fields and sub-tests
(e.g. claiming a DOI does not resolve while subtests.doi_resolves is true).
Labels, types, tiers, sub-tests, and every other field are untouched — the
2026-05 ground-truth audit and a corpus-wide consistency check (every
HALLUCINATED entry fails at least one sub-test) confirm they are correct.

- HALLUCINATED entries (97): corrected explanations come from the provenance
  log (results/reviewer_experiments/explanation_fixes_v121.json), drafted and
  adversarially verified against CrossRef/arXiv records.
- VALID entries (36): all carry relabel provenance from the ground-truth
  audit; their explanation becomes the audit's relabel_reason verbatim.

The rewrite is byte-minimal: only the explanation value of affected lines
changes; every other line round-trips byte-identically (asserted).

Usage: python scripts/fix_stale_explanations.py [--check]
  --check  verify the fix log against the corpus without writing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["dev_public", "test_public"]
FIX_LOG = ROOT / "results" / "reviewer_experiments" / "explanation_fixes_v121.json"
MARKER = "[Re-classified]"


def load_fixes() -> dict[str, dict]:
    with open(FIX_LOG, encoding="utf-8") as f:
        log = json.load(f)
    return {r["key"]: r for r in log["fixes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    fixes = load_fixes()
    seen: set[str] = set()
    n_h = n_v = 0

    for split in SPLITS:
        path = ROOT / "data" / "v1.0" / f"{split}.jsonl"
        out_lines: list[str] = []
        changed = 0
        with open(path, encoding="utf-8") as fin:
            lines = fin.readlines()
        for raw in lines:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            entry = json.loads(raw, object_pairs_hook=OrderedDict)
            roundtrip = json.dumps(entry, ensure_ascii=False, separators=(", ", ": "))
            if roundtrip != raw:
                sys.exit(f"round-trip mismatch for {entry['bibtex_key']} in {split}")
            explanation = entry.get("explanation") or ""
            if MARKER in explanation:
                key = entry["bibtex_key"]
                if entry["label"] == "HALLUCINATED":
                    if key not in fixes:
                        sys.exit(f"no fix for HALLUCINATED entry {key} ({split})")
                    entry["explanation"] = fixes[key]["explanation"]
                    seen.add(key)
                    n_h += 1
                else:
                    reason = entry.get("relabel_reason")
                    if not entry.get("relabeled_from") or not reason:
                        sys.exit(f"VALID entry {key} lacks relabel provenance")
                    entry["explanation"] = reason
                    n_v += 1
                changed += 1
            out_lines.append(json.dumps(entry, ensure_ascii=False, separators=(", ", ": ")))
        if not args.check:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
        print(f"{split}: {changed} explanations {'would be ' if args.check else ''}replaced")

    unused = set(fixes) - seen
    if unused:
        sys.exit(f"fix log contains {len(unused)} unused keys: {sorted(unused)[:5]}")
    print(f"total: {n_h} HALLUCINATED (from fix log) + {n_v} VALID (relabel_reason)")


if __name__ == "__main__":
    main()
