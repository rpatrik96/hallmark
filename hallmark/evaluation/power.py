"""Statistical power analysis for HALLMARK benchmark subtypes.

Implements Hardt's observation that sample requirements grow quadratically
with the inverse of the effect size to detect. Audits each hallucination
subtype to determine whether rankings are reliable at current sample sizes.

Reference: Hardt, M. (2025). The Emerging Science of Machine Learning Benchmarks, Ch. 3.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field

from hallmark.dataset.schema import HALLUCINATION_TIER_MAP, BenchmarkEntry

logger = logging.getLogger(__name__)


def z_score(p: float) -> float:
    """Approximate inverse normal CDF (probit) using Abramowitz & Stegun 26.2.23.

    Args:
        p: cumulative probability in (0, 1). For two-sided alpha, pass alpha/2.

    Returns:
        z such that P(Z <= z) ≈ p for standard normal Z.

    Raises:
        ValueError: if p is not in (0, 1).
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Use symmetry: for p > 0.5, z(p) = -z(1-p)
    if p > 0.5:
        return -z_score(1.0 - p)

    # Rational approximation constants
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    t = math.sqrt(-2.0 * math.log(p))
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return z


def mde_two_proportion(
    n: int,
    p0: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum Detectable Effect for comparing two proportions.

    Given n samples, what is the smallest detection rate difference
    between two tools that we can reliably detect?

    Formula: MDE = (z_{alpha/2} + z_{power}) * sqrt(2 * p0 * (1-p0) / n)

    Args:
        n: sample size (must be > 0)
        p0: baseline proportion (default 0.5 = maximum variance)
        alpha: significance level (two-sided)
        power: statistical power

    Returns:
        Minimum detectable effect size (absolute proportion difference).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z_alpha = z_score(alpha / 2)
    z_power = z_score(1.0 - power)
    return (z_alpha + z_power) * math.sqrt(2.0 * p0 * (1.0 - p0) / n)


def required_n(
    delta: float,
    p0: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Required sample size to detect an effect of size delta.

    Inverse of mde_two_proportion.

    Args:
        delta: target effect size (absolute proportion difference, must be > 0)
        p0: baseline proportion
        alpha: significance level (two-sided)
        power: statistical power

    Returns:
        Required sample size (ceiling).
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    z_alpha = z_score(alpha / 2)
    z_power = z_score(1.0 - power)
    return math.ceil(2.0 * p0 * (1.0 - p0) * ((z_alpha + z_power) / delta) ** 2)


@dataclass
class SubtypePowerResult:
    """Power analysis result for one hallucination subtype."""

    subtype: str
    tier: int
    n: int  # current sample count
    mde: float  # minimum detectable effect at current n
    required_n_by_delta: dict[float, int] = field(default_factory=dict)
    underpowered: bool = False  # True if MDE > 0.20


def subtype_power_audit(
    entries: list[BenchmarkEntry],
    p0: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    target_deltas: list[float] | None = None,
) -> dict[str, SubtypePowerResult]:
    """Audit per-subtype sample sizes for statistical power.

    For each hallucination subtype, computes MDE at current n and
    required n for common effect sizes.

    Args:
        entries: benchmark entries (counts hallucinated entries per type)
        p0: baseline proportion
        alpha: significance level
        power: statistical power
        target_deltas: effect sizes for required_n (default [0.05, 0.10, 0.15, 0.20])

    Returns:
        dict mapping subtype string -> SubtypePowerResult
    """
    if target_deltas is None:
        target_deltas = [0.05, 0.10, 0.15, 0.20]

    # Build tier lookup
    tier_map: dict[str, int] = {}
    for ht, dt in HALLUCINATION_TIER_MAP.items():
        tier_map[ht.value] = dt.value

    # Count hallucinated entries per type
    type_counts: Counter[str] = Counter()
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            type_counts[e.hallucination_type] += 1

    results: dict[str, SubtypePowerResult] = {}
    for subtype, n in sorted(type_counts.items()):
        tier = tier_map.get(subtype, 0)
        effect = mde_two_proportion(n, p0=p0, alpha=alpha, power=power)
        req_n = {d: required_n(d, p0=p0, alpha=alpha, power=power) for d in target_deltas}
        results[subtype] = SubtypePowerResult(
            subtype=subtype,
            tier=tier,
            n=n,
            mde=effect,
            required_n_by_delta=req_n,
            underpowered=effect > 0.20,
        )

    return results
