"""Ranking stability analysis for HALLMARK benchmark subtypes.

Assesses whether tool rankings are stable at current per-subtype sample
sizes using bootstrap resampling, and tests sensitivity to tier weight
choice via Dirichlet sweep.

Reference: Hardt, M. (2025). The Emerging Science of Machine Learning Benchmarks.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from hallmark.dataset.schema import (
    BenchmarkEntry,
    Prediction,
)

logger = logging.getLogger(__name__)


@dataclass
class SubtypeRankingResult:
    """Ranking stability for one hallucination subtype."""

    subtype: str
    n_entries: int
    tool_rankings: list[tuple[str, float]]  # [(tool_name, DR)] sorted best-first
    rank_ci_per_tool: dict[str, tuple[int, int]]  # tool -> (min_rank, max_rank)
    is_stable: bool  # True if no rank inversions in >= 95% of resamples
    pairwise_distinguishable: dict[str, bool]  # "toolA_vs_toolB" -> True if CIs don't overlap


@dataclass
class RankingSensitivityResult:
    """Result of tier-weight sensitivity sweep."""

    rankings_stable: bool
    n_inversions: int
    concordance_fraction: float  # fraction preserving default ranking
    per_tool_range: dict[str, tuple[float, float]]  # tool -> (min, max) TW-F1
    kendall_tau_min: float
    inversions: list[dict] = field(default_factory=list)


def _kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    """Kendall's tau between two rankings of the same items.

    Items in rank_a not in rank_b (and vice versa) are ignored.
    Returns 0.0 if fewer than 2 common items.
    """
    common = [x for x in rank_a if x in set(rank_b)]
    if len(common) < 2:
        return 0.0

    idx_b = {name: i for i, name in enumerate(rank_b)}
    order_b = [idx_b[x] for x in common]

    n = len(order_b)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if order_b[i] < order_b[j]:
                concordant += 1
            else:
                discordant += 1

    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0


def per_subtype_ranking_stability(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, SubtypeRankingResult]:
    """For each hallucination subtype, compute bootstrap CIs on tool rankings.

    Args:
        entries: full benchmark entries
        tool_predictions: {tool_name: [predictions]}
        n_bootstrap: number of bootstrap resamples per subtype
        seed: random seed

    Returns:
        dict mapping subtype -> SubtypeRankingResult
    """
    # Group hallucinated entries by type
    entries_by_type: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for e in entries:
        if e.label == "HALLUCINATED" and e.hallucination_type:
            entries_by_type[e.hallucination_type].append(e)

    # Build prediction maps per tool
    tool_pred_maps: dict[str, dict[str, Prediction]] = {}
    for tool, preds in tool_predictions.items():
        tool_pred_maps[tool] = {p.bibtex_key: p for p in preds}

    tool_names = sorted(tool_predictions.keys())
    rng = random.Random(seed)
    results: dict[str, SubtypeRankingResult] = {}

    for subtype, type_entries in sorted(entries_by_type.items()):
        n_entries = len(type_entries)
        if n_entries == 0:
            continue

        # Point estimate: DR per tool
        point_drs: dict[str, float] = {}
        for tool in tool_names:
            pm = tool_pred_maps[tool]
            correct = sum(
                1
                for e in type_entries
                if e.bibtex_key in pm and pm[e.bibtex_key].label == "HALLUCINATED"
            )
            point_drs[tool] = correct / n_entries

        point_ranking = sorted(point_drs.items(), key=lambda x: -x[1])
        point_order = [t for t, _ in point_ranking]

        # Bootstrap: track rank of each tool per iteration
        rank_tracker: dict[str, list[int]] = {t: [] for t in tool_names}
        concordant_count = 0

        for _ in range(n_bootstrap):
            sample = rng.choices(type_entries, k=n_entries)
            sample_drs: dict[str, float] = {}
            for tool in tool_names:
                pm = tool_pred_maps[tool]
                correct = sum(
                    1
                    for e in sample
                    if e.bibtex_key in pm and pm[e.bibtex_key].label == "HALLUCINATED"
                )
                sample_drs[tool] = correct / len(sample)

            sample_ranking = sorted(sample_drs.items(), key=lambda x: -x[1])
            sample_order = [t for t, _ in sample_ranking]

            for rank_idx, (tool, _) in enumerate(sample_ranking):
                rank_tracker[tool].append(rank_idx + 1)  # 1-indexed

            if sample_order == point_order:
                concordant_count += 1

        # Compute rank CIs (2.5th, 97.5th percentile)
        rank_ci: dict[str, tuple[int, int]] = {}
        for tool in tool_names:
            ranks = sorted(rank_tracker[tool])
            lo_idx = max(0, int(0.025 * len(ranks)))
            hi_idx = min(len(ranks) - 1, int(0.975 * len(ranks)))
            rank_ci[tool] = (ranks[lo_idx], ranks[hi_idx])

        # Pairwise distinguishability
        pairwise: dict[str, bool] = {}
        for i, t1 in enumerate(tool_names):
            for t2 in tool_names[i + 1 :]:
                ci1 = rank_ci[t1]
                ci2 = rank_ci[t2]
                # Distinguishable if rank CIs don't overlap
                distinguishable = ci1[1] < ci2[0] or ci2[1] < ci1[0]
                pairwise[f"{t1}_vs_{t2}"] = distinguishable

        is_stable = concordant_count / n_bootstrap >= 0.95

        results[subtype] = SubtypeRankingResult(
            subtype=subtype,
            n_entries=n_entries,
            tool_rankings=point_ranking,
            rank_ci_per_tool=rank_ci,
            is_stable=is_stable,
            pairwise_distinguishable=pairwise,
        )

    return results


def ranking_sensitivity_analysis(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    n_samples: int = 1000,
    seed: int = 42,
) -> RankingSensitivityResult:
    """Sweep tier weight vectors and check if tool rankings change.

    Samples from Dirichlet(1,1,1) over the 3-tier simplex.
    Requires numpy.

    Args:
        entries: benchmark entries
        tool_predictions: {tool_name: [predictions]}
        n_samples: number of weight vectors to sample
        seed: random seed
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for ranking sensitivity analysis. "
            "Install with: pip install hallmark[ranking]"
        ) from exc

    from hallmark.evaluation.metrics import per_tier_metrics

    # Compute per-tier metrics for each tool
    tool_tier_data: dict[str, dict[int, dict[str, float]]] = {}
    for tool_name, preds in tool_predictions.items():
        pred_map = {p.bibtex_key: p for p in preds}
        tier_data = per_tier_metrics(entries, pred_map)
        tool_tier_data[tool_name] = tier_data

    tool_names = sorted(tool_predictions.keys())

    def _compute_twf1(weights: dict[int, float]) -> dict[str, float]:
        """Compute tier-weighted F1 for each tool given weights."""
        results: dict[str, float] = {}
        for tool in tool_names:
            td = tool_tier_data[tool]
            weighted_tp = 0.0
            weighted_fn = 0.0
            total_fp = 0.0
            for tier in (1, 2, 3):
                w = weights.get(tier, 1.0)
                tm = td.get(tier, {})
                tp = tm.get("num_hallucinated", 0) * tm.get("detection_rate", 0.0)
                fn = tm.get("num_hallucinated", 0) * (1 - tm.get("detection_rate", 0.0))
                fp = tm.get("num_valid", 0) * tm.get("fpr", 0.0)
                weighted_tp += w * tp
                weighted_fn += w * fn
                total_fp += fp  # FPs carry uniform weight
            precision = (
                weighted_tp / (weighted_tp + total_fp) if (weighted_tp + total_fp) > 0 else 0.0
            )
            recall = (
                weighted_tp / (weighted_tp + weighted_fn)
                if (weighted_tp + weighted_fn) > 0
                else 0.0
            )
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            results[tool] = f1
        return results

    # Default ranking (weights 1:1, 2:2, 3:3)
    default_scores = _compute_twf1({1: 1.0, 2: 2.0, 3: 3.0})
    default_ranking = sorted(default_scores.items(), key=lambda x: -x[1])
    default_order = [t for t, _ in default_ranking]

    # Sample weight vectors
    rng = np.random.default_rng(seed)
    weight_samples = rng.dirichlet([1, 1, 1], size=n_samples)

    per_tool_min: dict[str, float] = {t: float("inf") for t in tool_names}
    per_tool_max: dict[str, float] = {t: float("-inf") for t in tool_names}
    concordant_count = 0
    total_inversions = 0
    inversion_list: list[dict] = []
    min_tau = 1.0

    for w in weight_samples:
        # Scale to sum to 6 (like default 1+2+3=6) for comparable magnitudes
        weights = {1: float(w[0]) * 3, 2: float(w[1]) * 3, 3: float(w[2]) * 3}
        scores = _compute_twf1(weights)
        ranking = sorted(scores.items(), key=lambda x: -x[1])
        order = [t for t, _ in ranking]

        for tool, score in scores.items():
            per_tool_min[tool] = min(per_tool_min[tool], score)
            per_tool_max[tool] = max(per_tool_max[tool], score)

        tau = _kendall_tau(default_order, order)
        min_tau = min(min_tau, tau)

        if order == default_order:
            concordant_count += 1
        else:
            # Count pairwise inversions vs default
            for i, t1 in enumerate(default_order):
                for t2 in default_order[i + 1 :]:
                    pos1 = order.index(t1)
                    pos2 = order.index(t2)
                    if pos1 > pos2:
                        total_inversions += 1
                        inversion_list.append(
                            {
                                "tools": [t1, t2],
                                "weights": weights,
                                "gap": abs(scores[t1] - scores[t2]),
                            }
                        )

    concordance = concordant_count / n_samples if n_samples > 0 else 1.0
    per_tool_range = {t: (per_tool_min[t], per_tool_max[t]) for t in tool_names}

    return RankingSensitivityResult(
        rankings_stable=concordance >= 0.95,
        n_inversions=total_inversions,
        concordance_fraction=concordance,
        per_tool_range=per_tool_range,
        kendall_tau_min=min_tau,
        inversions=inversion_list[:50],  # cap stored inversions
    )


def iia_violation_check(
    entries: list[BenchmarkEntry],
    tool_predictions: dict[str, list[Prediction]],
    metric: str = "tier_weighted_f1",
) -> dict:
    """Test Independence of Irrelevant Alternatives for score-based ranking.

    For independent score metrics (TW-F1, F1, DR), each tool's score is
    computed independently, so IIA is trivially satisfied.

    This verifies that property and documents it.

    Returns:
        dict with: iia_satisfied, n_tools, verification_details
    """
    from hallmark.evaluation.metrics import evaluate

    # Compute scores for all tools
    all_scores: dict[str, float] = {}
    for tool_name, preds in tool_predictions.items():
        result = evaluate(entries, preds, tool_name=tool_name, split_name="iia_check")
        all_scores[tool_name] = getattr(result, metric, 0.0)

    # Leave-one-out: remove each tool and verify remaining ordering unchanged
    tool_names = sorted(all_scores.keys())
    full_ranking = sorted(tool_names, key=lambda t: -all_scores[t])

    violations: list[str] = []
    for removed in tool_names:
        reduced = [t for t in full_ranking if t != removed]
        # Scores are independent, so order should be preserved
        reduced_by_score = sorted(reduced, key=lambda t: -all_scores[t])
        if reduced != reduced_by_score:
            violations.append(removed)

    return {
        "iia_satisfied": len(violations) == 0,
        "n_tools": len(tool_names),
        "violations": violations,
        "verification_details": (
            f"Score-based ranking with metric={metric} satisfies IIA by construction: "
            f"each tool's score is computed independently. "
            f"Verified via leave-one-out over {len(tool_names)} tools with 0 violations."
            if not violations
            else f"Unexpected violations found when removing: {violations}"
        ),
    }
