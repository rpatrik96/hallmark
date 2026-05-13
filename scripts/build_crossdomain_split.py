#!/usr/bin/env python3
"""Build the cross-domain test split from raw valid entries.

Reads ``data/v1.0/raw_crossdomain_valid.jsonl`` (produced by
``scrape_crossdomain.py``), generates hallucinated perturbations using the
existing tier1/tier2/tier3 batch generators, and writes a single split file
``data/v1.0/test_crossdomain.jsonl`` containing both VALID and HALLUCINATED
entries with bibtex_key namespaced as ``hallmark_xd_NNNN``.

This script intentionally:
- Does **not** modify any v1.0 split.
- Reuses the same hallucination generators as the v1.0 splits, so the
  cross-domain entries are directly comparable in evaluation.
- Tags ``source`` with the upstream sub-corpus (biorxiv/pubmed/dblp_cs_non_ml)
  so downstream analysis can compute per-domain breakdowns.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from hallmark.dataset.generators.batch import (
    generate_tier1_batch,
    generate_tier2_batch,
    generate_tier3_batch,
)
from hallmark.dataset.schema import (
    HALLUCINATION_TIER_MAP,
    BenchmarkEntry,
    HallucinationType,
    save_entries,
)

logger = logging.getLogger("build_crossdomain_split")


def load_valid(path: Path) -> list[BenchmarkEntry]:
    out: list[BenchmarkEntry] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(BenchmarkEntry.from_dict(d))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/v1.0/raw_crossdomain_valid.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/v1.0/test_crossdomain.jsonl"))
    p.add_argument(
        "--keep-valid", type=int, default=200, help="How many valid entries to keep in the split."
    )
    p.add_argument("--tier1", type=int, default=80)
    p.add_argument("--tier2", type=int, default=140)
    p.add_argument("--tier3", type=int, default=80)
    p.add_argument("--seed", type=int, default=8042)
    p.add_argument(
        "--key-prefix",
        default="hallmark_xd_",
        help="Prefix for the namespaced bibtex_key (use 'hallmark_t_' for temporal split).",
    )
    p.add_argument(
        "--source-tag",
        default="perturbation_crossdomain",
        help="Source tag stamped onto perturbation entries that don't already have one.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    valid = load_valid(args.input)
    if not valid:
        logger.error("No valid entries in %s", args.input)
        return 1
    logger.info("Loaded %d valid entries", len(valid))

    rng = random.Random(args.seed)
    rng.shuffle(valid)
    kept_valid = valid[: args.keep_valid]

    # Generate hallucinations from the full pool (more donor diversity)
    halls: list[BenchmarkEntry] = []
    halls.extend(generate_tier1_batch(valid, args.tier1, seed=args.seed))
    halls.extend(generate_tier2_batch(valid, args.tier2, seed=args.seed + 1))
    halls.extend(generate_tier3_batch(valid, args.tier3, seed=args.seed + 2))

    # Mark generation method and stamp difficulty_tier from registry
    for h in halls:
        h.generation_method = "perturbation"
        try:
            ht = HallucinationType(h.hallucination_type)
            h.difficulty_tier = HALLUCINATION_TIER_MAP[ht].value
        except (ValueError, KeyError):
            pass
        # Carry sub-corpus origin via source_conference if not already set by generator
        if not h.source:
            h.source = args.source_tag

    # Concatenate, namespace bibtex_keys to avoid collisions with v1.0 splits
    all_entries: list[BenchmarkEntry] = []
    rng.shuffle(halls)
    combined = kept_valid + halls
    rng.shuffle(combined)

    for i, e in enumerate(combined):
        e.bibtex_key = f"{args.key_prefix}{i:04d}"
        all_entries.append(e)

    save_entries(all_entries, args.output)

    # Summary
    from collections import Counter

    label_count = Counter(e.label for e in all_entries)
    src_count = Counter(e.source or "(none)" for e in all_entries)
    type_count = Counter(e.hallucination_type or "valid" for e in all_entries)
    tier_count = Counter(e.difficulty_tier for e in all_entries if e.label == "HALLUCINATED")

    logger.info("Wrote %d entries to %s", len(all_entries), args.output)
    logger.info("Labels: %s", dict(label_count))
    logger.info("Sources: %s", dict(src_count))
    logger.info("Hallucination types:")
    for k, v in type_count.most_common():
        logger.info("  %s: %d", k, v)
    logger.info("Tiers (hallucinated only): %s", dict(tier_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
