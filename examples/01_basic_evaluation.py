#!/usr/bin/env python3
"""Example 1: Basic evaluation of a custom tool on HALLMARK.

This example shows how to:
1. Load a benchmark split
2. Create predictions from your tool
3. Evaluate and print results
"""

from hallmark.dataset.loader import get_statistics, load_split
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import evaluate


def main():
    # 1. Load the dev split
    entries = load_split("dev_public")
    stats = get_statistics(entries)
    print(
        f"Loaded {stats['total']} entries ({stats['valid']} valid, "
        f"{stats['hallucinated']} hallucinated)"
    )

    # 2. Create predictions - here we use a simple heuristic:
    #    Flag entries without DOIs as potentially hallucinated
    predictions = []
    for entry in entries:
        has_doi = bool(entry.fields.get("doi"))
        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label="VALID" if has_doi else "HALLUCINATED",
                confidence=0.8 if has_doi else 0.6,
                reason="Has DOI" if has_doi else "Missing DOI - suspicious",
            )
        )

    # 3. Evaluate
    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name="no-doi-heuristic",
        split_name="dev_public",
    )

    # 4. Print results
    print(f"\nResults for '{result.tool_name}':")
    print(f"  Detection Rate:   {result.detection_rate:.3f}")
    print(f"  False Pos. Rate:  {result.false_positive_rate:.3f}")
    print(f"  F1 (Halluc.):     {result.f1_hallucination:.3f}")
    print(f"  Tier-weighted F1: {result.tier_weighted_f1:.3f}")

    # 5. Per-tier breakdown
    if result.per_tier_metrics:
        print("\n  Per-tier breakdown:")
        for tier, metrics in sorted(result.per_tier_metrics.items()):
            if tier == 0:
                continue
            print(
                f"    Tier {tier}: DR={metrics['detection_rate']:.3f} "
                f"F1={metrics['f1']:.3f} (n={metrics['count']:.0f})"
            )


if __name__ == "__main__":
    main()
