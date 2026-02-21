#!/usr/bin/env python3
"""Power analysis for per-type sample sizes in HALLMARK benchmark.

Computes the Minimum Detectable Effect (MDE) for each hallucination type
and split, given the sample sizes and a two-sided z-test at 80% power and
alpha=0.05.

Usage:
    python scripts/power_analysis.py
    python scripts/power_analysis.py --split dev_public --alpha 0.05 --power 0.80
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

from hallmark.dataset.loader import load_split

# Standard normal z-values (no scipy needed)
Z_ALPHA_TWO_SIDED = {0.01: 2.576, 0.05: 1.960, 0.10: 1.645}
Z_POWER = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}


def mde_two_proportion(
    n: int,
    p0: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum Detectable Effect for a two-sided two-proportion z-test.

    Given n observations per group, baseline proportion p0, significance
    level alpha, and desired power, returns the smallest absolute
    difference |p1 - p0| that can be detected.

    Formula: MDE = (z_alpha + z_power) * sqrt(2 * p0 * (1 - p0) / n)
    """
    if n <= 0:
        return float("inf")
    z_a = Z_ALPHA_TWO_SIDED.get(alpha, 1.960)
    z_p = Z_POWER.get(power, 0.842)
    return (z_a + z_p) * math.sqrt(2 * p0 * (1 - p0) / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Power analysis for HALLMARK per-type sample sizes"
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["dev_public", "test_public"],
        help="Splits to analyze",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    for split in args.split:
        try:
            entries = load_split(split=split, version=args.version, data_dir=args.data_dir)
        except FileNotFoundError:
            print(f"Split {split} not found, skipping.")
            continue

        # Count per hallucination type
        type_counts: Counter[str] = Counter()
        for e in entries:
            if e.label == "HALLUCINATED" and e.hallucination_type:
                type_counts[e.hallucination_type] += 1

        print(f"\n{'=' * 72}")
        print(f"  Power Analysis: {split} (alpha={args.alpha}, power={args.power})")
        print(f"{'=' * 72}")
        print(f"  {'Type':<30} {'n':>5} {'MDE (pp)':>10} {'Detectable?':>12}")
        print(f"  {'-' * 68}")

        for h_type, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            mde = mde_two_proportion(n, p0=0.5, alpha=args.alpha, power=args.power)
            detectable = "yes" if mde < 0.20 else "marginal" if mde < 0.30 else "NO"
            print(f"  {h_type:<30} {n:>5} {mde * 100:>9.1f}% {detectable:>12}")

        # Aggregate
        total_h = sum(type_counts.values())
        agg_mde = mde_two_proportion(total_h, p0=0.5, alpha=args.alpha, power=args.power)
        print(f"  {'-' * 68}")
        print(f"  {'AGGREGATE':<30} {total_h:>5} {agg_mde * 100:>9.1f}%")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
