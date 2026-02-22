#!/usr/bin/env python3
"""Example 1: Basic evaluation of a custom tool on HALLMARK.

This example shows how to:
1. Load a benchmark split
2. Convert entries to BlindEntry (hiding ground-truth labels)
3. Create predictions from your tool using only BlindEntry fields
4. Evaluate and print results

IMPORTANT: Always use entry.to_blind() before passing entries to your tool.
BenchmarkEntry contains ground-truth labels that must not influence predictions.
"""

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import evaluate


def main():
    # 1. Load the dev split (BenchmarkEntry objects with ground truth)
    entries = load_split("dev_public")
    print(f"Loaded {len(entries)} entries")

    # 2. Convert to BlindEntry — this hides labels, tier, hallucination_type
    blind_entries = [e.to_blind() for e in entries]

    # 3. Create predictions using ONLY BlindEntry fields
    #    BlindEntry has: bibtex_key, bibtex_type, fields, raw_bibtex
    #    It does NOT have: label, hallucination_type, difficulty_tier, subtests
    predictions = []
    for blind in blind_entries:
        has_doi = bool(blind.fields.get("doi"))
        predictions.append(
            Prediction(
                bibtex_key=blind.bibtex_key,
                label="VALID" if has_doi else "HALLUCINATED",
                confidence=0.8 if has_doi else 0.6,
                reason="Has DOI" if has_doi else "Missing DOI - suspicious",
            )
        )

    # 4. Evaluate (pass original entries for ground truth comparison)
    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name="no-doi-heuristic",
        split_name="dev_public",
    )

    # 5. Print results
    print(f"\nResults for '{result.tool_name}':")
    print(f"  Detection Rate:   {result.detection_rate:.3f}")
    print(f"  False Pos. Rate:  {result.false_positive_rate:.3f}")
    print(f"  F1 (Halluc.):     {result.f1_hallucination:.3f}")
    print(f"  Tier-weighted F1: {result.tier_weighted_f1:.3f}")

    if result.per_tier_metrics:
        print("\n  Per-tier breakdown:")
        for tier, metrics in sorted(result.per_tier_metrics.items()):
            if tier == 0:
                continue
            print(
                f"    Tier {tier}: DR={metrics['detection_rate']:.3f} "
                f"F1={metrics['f1']:.3f} (n={metrics['num_hallucinated']:.0f})"
            )


if __name__ == "__main__":
    main()
