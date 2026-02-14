#!/usr/bin/env python3
"""Compute stratified bootstrap confidence intervals for baseline results."""

import json
from pathlib import Path

import numpy as np

# Tier assignments
TIER_1 = ["fabricated_doi", "nonexistent_venue", "placeholder_authors", "future_date"]
TIER_2 = [
    "chimeric_title",
    "wrong_venue",
    "swapped_authors",
    "preprint_as_published",
    "hybrid_fabrication",
    "merged_citation",
    "partial_author_list",
]
TIER_3 = ["near_miss_title", "plausible_fabrication", "arxiv_version_mismatch"]


def get_tier(type_name: str) -> int:
    """Get tier for a hallucination type."""
    if type_name in TIER_1:
        return 1
    elif type_name in TIER_2:
        return 2
    elif type_name in TIER_3:
        return 3
    else:
        raise ValueError(f"Unknown type: {type_name}")


def reconstruct_outcomes(results: dict) -> tuple[dict, list]:
    """Reconstruct binary outcome vectors from per-type metrics.

    Returns:
        type_outcomes: dict mapping type_name -> list of binary outcomes (1=detected, 0=missed)
        valid_outcomes: list of binary outcomes for valid entries (1=FP, 0=TN)
    """
    type_outcomes = {}

    for type_name, metrics in results["per_type_metrics"].items():
        if type_name == "valid":
            continue

        count = metrics["count"]
        detection_rate = metrics["detection_rate"]

        # Reconstruct: n_detected successes, (count - n_detected) failures
        n_detected = round(detection_rate * count)

        # Binary vector: 1 = detected, 0 = missed
        outcomes = [1] * n_detected + [0] * (count - n_detected)
        type_outcomes[type_name] = outcomes

    # Reconstruct valid (non-hallucinated) outcomes
    valid_count = results["per_type_metrics"]["valid"]["count"]
    fpr = results["per_type_metrics"]["valid"]["false_positive_rate"]
    n_false_positives = round(fpr * valid_count)

    # Binary vector for valid: 1 = false positive, 0 = true negative
    valid_outcomes = [1] * n_false_positives + [0] * (valid_count - n_false_positives)

    return type_outcomes, valid_outcomes


def stratified_bootstrap_sample(
    type_outcomes: dict, valid_outcomes: list, rng: np.random.Generator
):
    """Generate one stratified bootstrap sample.

    Sample with replacement within each type (stratum).
    """
    sampled_type_outcomes = {}

    for type_name, outcomes in type_outcomes.items():
        n = len(outcomes)
        indices = rng.integers(0, n, size=n)
        sampled_type_outcomes[type_name] = [outcomes[i] for i in indices]

    # Sample valid outcomes
    n_valid = len(valid_outcomes)
    indices = rng.integers(0, n_valid, size=n_valid)
    sampled_valid_outcomes = [valid_outcomes[i] for i in indices]

    return sampled_type_outcomes, sampled_valid_outcomes


def compute_metrics(type_outcomes: dict, valid_outcomes: list) -> dict:
    """Compute metrics from binary outcome vectors."""

    # Detection rate (recall on hallucinated entries only)
    all_hallucinated_outcomes = []
    for outcomes in type_outcomes.values():
        all_hallucinated_outcomes.extend(outcomes)

    n_detected = sum(all_hallucinated_outcomes)
    n_hallucinated = len(all_hallucinated_outcomes)
    detection_rate = n_detected / n_hallucinated if n_hallucinated > 0 else 0.0

    # FPR
    n_fp = sum(valid_outcomes)
    n_valid = len(valid_outcomes)
    fpr = n_fp / n_valid if n_valid > 0 else 0.0

    # F1-Hallucination
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN) = detection_rate
    tp = n_detected
    fp = n_fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = detection_rate

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    # Tier-Weighted F1
    # Compute per-tier F1, weight by tier (1, 2, 3), normalize
    tier_f1_scores = {}

    for tier in [1, 2, 3]:
        tier_outcomes = []
        for type_name, outcomes in type_outcomes.items():
            if get_tier(type_name) == tier:
                tier_outcomes.extend(outcomes)

        if len(tier_outcomes) == 0:
            tier_f1_scores[tier] = 0.0
            continue

        tier_tp = sum(tier_outcomes)
        tier_fn = len(tier_outcomes) - tier_tp

        # For tier-specific metrics, FP is shared across all tiers
        tier_precision = tier_tp / (tier_tp + fp) if (tier_tp + fp) > 0 else 0.0
        tier_recall = tier_tp / (tier_tp + tier_fn) if (tier_tp + tier_fn) > 0 else 0.0

        if tier_precision + tier_recall > 0:
            tier_f1 = 2 * (tier_precision * tier_recall) / (tier_precision + tier_recall)
        else:
            tier_f1 = 0.0

        tier_f1_scores[tier] = tier_f1

    # Weighted average: tier1*1 + tier2*2 + tier3*3, normalized by sum of weights
    weighted_sum = tier_f1_scores[1] * 1 + tier_f1_scores[2] * 2 + tier_f1_scores[3] * 3
    tier_weighted_f1 = weighted_sum / (1 + 2 + 3)

    return {
        "detection_rate": detection_rate,
        "fpr": fpr,
        "f1_hallucination": f1,
        "tier_weighted_f1": tier_weighted_f1,
    }


def compute_bootstrap_ci(
    type_outcomes: dict,
    valid_outcomes: list,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> dict:
    """Compute bootstrap confidence intervals using stratified resampling."""

    rng = np.random.default_rng(42)  # Reproducible

    bootstrap_metrics = {
        "detection_rate": [],
        "fpr": [],
        "f1_hallucination": [],
        "tier_weighted_f1": [],
    }

    print(f"Running {n_bootstrap} bootstrap resamples (stratified by type)...")

    for i in range(n_bootstrap):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_bootstrap}")

        # Generate stratified bootstrap sample
        sampled_type_outcomes, sampled_valid_outcomes = stratified_bootstrap_sample(
            type_outcomes, valid_outcomes, rng
        )

        # Compute metrics for this sample
        metrics = compute_metrics(sampled_type_outcomes, sampled_valid_outcomes)

        for key, value in metrics.items():
            bootstrap_metrics[key].append(value)

    # Compute confidence intervals (percentile method)
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    cis = {}
    for metric_name, values in bootstrap_metrics.items():
        values_sorted = np.array(sorted(values))
        lower = np.percentile(values_sorted, lower_percentile)
        upper = np.percentile(values_sorted, upper_percentile)
        cis[metric_name] = [lower, upper]

        print(f"{metric_name}: [{lower:.6f}, {upper:.6f}]")

    return cis


def main():
    # Load results
    results_path = Path(
        "/Users/patrik.reizinger/Documents/GitHub/hallmark/results/llm_openai_dev_public.json"
    )
    with open(results_path) as f:
        results = json.load(f)

    print("Reconstructing binary outcome vectors...")
    type_outcomes, valid_outcomes = reconstruct_outcomes(results)

    print("\nType counts:")
    for type_name, outcomes in type_outcomes.items():
        n_detected = sum(outcomes)
        print(f"  {type_name}: {len(outcomes)} total, {n_detected} detected")
    print(f"  valid: {len(valid_outcomes)} total, {sum(valid_outcomes)} false positives")

    print("\nComputing point estimates...")
    point_metrics = compute_metrics(type_outcomes, valid_outcomes)
    for metric_name, value in point_metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    print("\n" + "=" * 60)
    cis = compute_bootstrap_ci(type_outcomes, valid_outcomes, n_bootstrap=10000)

    # Create output matching doi_only_dev_public_ci.json format
    output = results.copy()
    output["detection_rate_ci"] = cis["detection_rate"]
    output["f1_hallucination_ci"] = cis["f1_hallucination"]
    output["tier_weighted_f1_ci"] = cis["tier_weighted_f1"]
    output["fpr_ci"] = cis["fpr"]
    output["ece_ci"] = None  # Skip ECE as requested

    # Save output
    output_path = Path(
        "/Users/patrik.reizinger/Documents/GitHub/hallmark/results/llm_openai_dev_public_ci.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[FINDING] Confidence intervals saved to {output_path}")
    print("\nSummary:")
    print(f"  Detection Rate: {output['detection_rate']:.4f} {output['detection_rate_ci']}")
    print(f"  F1-Hallucination: {output['f1_hallucination']:.4f} {output['f1_hallucination_ci']}")
    print(f"  Tier-Weighted F1: {output['tier_weighted_f1']:.4f} {output['tier_weighted_f1_ci']}")
    print(f"  FPR: {output['false_positive_rate']:.4f} {output['fpr_ci']}")


if __name__ == "__main__":
    main()
