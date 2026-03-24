"""Water-filling analysis for HALLMARK benchmark tools.

Detects whether tools concentrate their detection capability on easy
hallucination types (Tier 1) while neglecting harder ones (Tier 3),
following Hardt's observation that rational agents will allocate effort
to tasks with the highest marginal returns (Gibbard-Satterthwaite).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from hallmark.dataset.schema import (
    HALLUCINATION_TIER_MAP,
    BenchmarkEntry,
    DifficultyTier,
    HallucinationType,
    Prediction,
)

logger = logging.getLogger(__name__)


@dataclass
class WaterFillingProfile:
    """Water-filling analysis for a single tool."""

    tool_name: str
    tier_detection_rates: dict[int, float]  # tier int -> DR
    gini_coefficient: float  # 0 = uniform across tiers, 1 = all in one tier
    tier_ratio: float  # T1_DR / T3_DR (float('inf') if T3 DR is 0)
    normalized_entropy: float  # Shannon entropy of tier DRs normalized to [0,1]
    is_water_filling: bool  # True if gini > threshold or ratio > threshold


def compute_tier_detection_rates(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
) -> dict[int, float]:
    """Compute detection rate per difficulty tier.

    Only counts HALLUCINATED entries. VALID entries are excluded.
    Predictions labeled UNCERTAIN are excluded.

    Returns:
        dict mapping tier (1, 2, 3) -> detection rate
    """
    pred_map: dict[str, Prediction] = {p.bibtex_key: p for p in predictions}

    # Group hallucinated entries by tier
    tier_totals: dict[int, int] = {}
    tier_correct: dict[int, int] = {}

    for entry in entries:
        if entry.label != "HALLUCINATED":
            continue
        if entry.hallucination_type is None:
            continue

        # Resolve tier from HALLUCINATION_TIER_MAP
        try:
            h_type = HallucinationType(entry.hallucination_type)
            tier_enum = HALLUCINATION_TIER_MAP[h_type]
            tier = tier_enum.value
        except (ValueError, KeyError):
            # Unknown type — use difficulty_tier field as fallback
            tier = entry.difficulty_tier or 1

        tier_totals[tier] = tier_totals.get(tier, 0) + 1

        pred = pred_map.get(entry.bibtex_key)
        if pred is None or pred.label == "UNCERTAIN":
            # Missing or uncertain prediction: not counted as correct
            tier_correct[tier] = tier_correct.get(tier, 0)
            continue

        if pred.label == "HALLUCINATED":
            tier_correct[tier] = tier_correct.get(tier, 0) + 1
        else:
            tier_correct[tier] = tier_correct.get(tier, 0)

    result: dict[int, float] = {}
    for tier, total in tier_totals.items():
        correct = tier_correct.get(tier, 0)
        result[tier] = correct / total if total > 0 else 0.0

    return result


def water_filling_gini(tier_drs: dict[int, float]) -> float:
    """Compute Gini coefficient of detection rates across tiers.

    0 = perfectly uniform performance across tiers.
    1 = all detections concentrated in one tier.

    Uses the standard Gini formula for a small discrete distribution.
    Handles zero values gracefully.
    """
    values = list(tier_drs.values())
    n = len(values)
    if n == 0:
        return 0.0

    total = sum(values)
    if total == 0.0:
        return 0.0

    # G = (Σ_i Σ_j |x_i - x_j|) / (2 * n * Σ_i x_i)
    pairwise_sum = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))
    return pairwise_sum / (2 * n * total)


def normalized_shannon_entropy(tier_drs: dict[int, float]) -> float:
    """Shannon entropy of tier detection rates, normalized to [0, 1].

    Uses tier DRs as a probability-like distribution (normalized to sum to 1).
    Maximum entropy (=1.0) when all tiers have equal DR.
    Minimum entropy (=0.0) when all detection is in one tier.

    Returns 0.0 if all DRs are zero.
    """
    values = list(tier_drs.values())
    n = len(values)
    if n <= 1:
        return 1.0 if n == 1 else 0.0

    total = sum(values)
    if total == 0.0:
        return 0.0

    # Normalize to get probability-like distribution
    probs = [v / total for v in values]

    # Shannon entropy, treating 0*log(0) = 0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)

    # Normalize by log(n) to put in [0, 1]
    max_entropy = math.log(n)
    return entropy / max_entropy if max_entropy > 0.0 else 0.0


def water_filling_analysis(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    ratio_threshold: float = 3.0,
    gini_threshold: float = 0.3,
) -> dict[str, WaterFillingProfile]:
    """Compute water-filling metrics for all tools.

    A tool exhibits water-filling if its Tier 1 detection rate is
    disproportionately higher than Tier 3, suggesting it only catches
    easy hallucinations.

    Args:
        entries: benchmark entries
        tool_predictions: mapping from tool name to list of predictions
        ratio_threshold: T1/T3 DR ratio above which is_water_filling=True
        gini_threshold: Gini coefficient above which is_water_filling=True

    Returns:
        dict mapping tool_name -> WaterFillingProfile
    """
    profiles: dict[str, WaterFillingProfile] = {}

    for tool_name, predictions in tool_predictions.items():
        tier_drs = compute_tier_detection_rates(entries, predictions)

        gini = water_filling_gini(tier_drs)
        entropy = normalized_shannon_entropy(tier_drs)

        t1_dr = tier_drs.get(DifficultyTier.EASY.value, 0.0)
        t3_dr = tier_drs.get(DifficultyTier.HARD.value, 0.0)

        if t3_dr == 0.0 and t1_dr > 0.0:
            tier_ratio = float("inf")
        elif t3_dr == 0.0 and t1_dr == 0.0:
            tier_ratio = 1.0
        else:
            tier_ratio = t1_dr / t3_dr

        is_water_filling = math.isinf(tier_ratio) or (
            gini > gini_threshold or tier_ratio > ratio_threshold
        )

        profiles[tool_name] = WaterFillingProfile(
            tool_name=tool_name,
            tier_detection_rates=tier_drs,
            gini_coefficient=gini,
            tier_ratio=tier_ratio,
            normalized_entropy=entropy,
            is_water_filling=is_water_filling,
        )

    return profiles
