#!/usr/bin/env python3
"""Example 2: Analyzing detection performance by hallucination type.

This example shows how to:
1. Load entries and convert to BlindEntry for prediction
2. Build a simple DOI-based detector using only BlindEntry fields
3. Evaluate per-type and per-tier performance
4. Identify which hallucination types are hardest to detect

NOTE: The evaluate() function receives the original BenchmarkEntry objects
(with ground truth) alongside your predictions for metrics computation.
Your prediction logic must only use BlindEntry fields.
"""

from hallmark.dataset.loader import filter_by_tier, load_split
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import (
    build_confusion_matrix,
    per_tier_metrics,
    per_type_metrics,
)


def doi_presence_baseline(blind_entries):
    """Simple DOI-presence baseline using only BlindEntry fields.

    Predicts HALLUCINATED when no DOI is present. This is a trivial
    heuristic — real tools would resolve the DOI and cross-check metadata.
    """
    predictions = {}
    for entry in blind_entries:
        doi = entry.fields.get("doi")
        if not doi:
            predictions[entry.bibtex_key] = Prediction(
                bibtex_key=entry.bibtex_key,
                label="HALLUCINATED",
                confidence=0.6,
                reason="No DOI field present",
            )
        else:
            predictions[entry.bibtex_key] = Prediction(
                bibtex_key=entry.bibtex_key,
                label="VALID",
                confidence=0.8,
                reason="DOI present (not verified)",
            )
    return predictions


def main():
    entries = load_split("dev_public")

    # Convert to blind entries for prediction (hides ground truth)
    blind_entries = [e.to_blind() for e in entries]
    pred_map = doi_presence_baseline(blind_entries)

    # Per-tier analysis (uses original entries for ground truth)
    print("=== Per-Tier Detection Rates (DOI-presence baseline) ===\n")
    tier_m = per_tier_metrics(entries, pred_map)
    for tier in [1, 2, 3]:
        m = tier_m.get(tier, {})
        dr = m.get("detection_rate", 0)
        f1 = m.get("f1", 0)
        n = m.get("count", 0)
        print(f"  Tier {tier}: DR={dr:.3f}  F1={f1:.3f}  (n={n:.0f})")

    # Per-type analysis
    print("\n=== Per-Type Detection Rates ===\n")
    type_m = per_type_metrics(entries, pred_map)
    for h_type, m in sorted(type_m.items()):
        if h_type == "valid":
            continue
        dr = m.get("detection_rate", 0)
        n = m.get("count", 0)
        status = "DETECTED" if dr > 0.5 else "MISSED"
        print(f"  {h_type:<25s} DR={dr:.3f} (n={n:.0f}) [{status}]")

    # Filter to just Tier 3 entries
    print("\n=== Tier 3 Only (Hardest) ===\n")
    tier3 = filter_by_tier(entries, 3)
    tier3_hall = [e for e in tier3 if e.label == "HALLUCINATED"]
    print(f"  {len(tier3_hall)} Tier 3 hallucinated entries")
    cm = build_confusion_matrix(tier3, pred_map)
    print(f"  Detection Rate: {cm.detection_rate:.3f}")
    print(f"  F1: {cm.f1:.3f}")


if __name__ == "__main__":
    main()
