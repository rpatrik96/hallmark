#!/usr/bin/env python3
"""Generate hallucinated BibTeX entries from valid entries.

Usage:
    python scripts/generate_hallucinations.py \
        --valid-entries data/v1.0/valid_entries.jsonl \
        --output data/v1.0/hallucinated_entries.jsonl \
        --tier1 40 --tier2 35 --tier3 25
"""

from __future__ import annotations

import argparse
import logging

from hallmark.dataset.generator import (
    generate_tier1_batch,
    generate_tier2_batch,
    generate_tier3_batch,
)
from hallmark.dataset.schema import load_entries, save_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hallucinated entries")
    parser.add_argument("--valid-entries", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tier1", type=int, default=40, help="Number of Tier 1 entries")
    parser.add_argument("--tier2", type=int, default=35, help="Number of Tier 2 entries")
    parser.add_argument("--tier3", type=int, default=25, help="Number of Tier 3 entries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    valid_entries = load_entries(args.valid_entries)
    logging.info(f"Loaded {len(valid_entries)} valid entries")

    hallucinated = []
    hallucinated.extend(generate_tier1_batch(valid_entries, args.tier1, args.seed))
    hallucinated.extend(generate_tier2_batch(valid_entries, args.tier2, args.seed + 1))
    hallucinated.extend(generate_tier3_batch(valid_entries, args.tier3, args.seed + 2))

    save_entries(hallucinated, args.output)
    logging.info(
        f"Generated {len(hallucinated)} hallucinated entries "
        f"(T1={args.tier1}, T2={args.tier2}, T3={args.tier3})"
    )


if __name__ == "__main__":
    main()
