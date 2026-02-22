#!/usr/bin/env python3
"""Example 4: Temporal analysis for contamination detection.

This example shows how to:
1. Segment entries by publication date
2. Compare tool performance across temporal segments
3. Detect potential contamination (memorization vs. genuine verification)

If a tool performs much better on older entries (which may be in its training data)
than on newer entries, it may be relying on memorization rather than verification.
"""

from hallmark.dataset.loader import filter_by_date_range, load_split
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import build_confusion_matrix
from hallmark.evaluation.temporal import default_segments, temporal_analysis


def simulate_baseline(entries):
    """Simulate a DOI-presence heuristic baseline (no ground truth access).

    This is a blind baseline: it calls entry.to_blind() and inspects only the
    fields visible to a real tool. It does NOT access labels, subtests, or any
    other ground-truth attributes.
    """
    pred_map = {}
    for e in entries:
        blind = e.to_blind()
        has_doi = bool(blind.fields.get("doi", "").strip())
        label = "VALID" if has_doi else "HALLUCINATED"
        confidence = 0.8 if has_doi else 0.6
        pred_map[blind.bibtex_key] = Prediction(
            bibtex_key=blind.bibtex_key,
            label=label,
            confidence=confidence,
        )
    return pred_map


def main():
    entries = load_split("dev_public")
    pred_map = simulate_baseline(entries)

    # Show temporal segments
    print("=== Temporal Segments ===\n")
    for seg in default_segments():
        print(f"  {seg.name}: {seg.start} to {seg.end}")

    # Run temporal analysis
    print("\n=== Temporal Analysis ===\n")
    result = temporal_analysis(entries, pred_map)

    for seg_name, metrics in result.segment_metrics.items():
        dr = metrics.get("detection_rate", 0)
        f1 = metrics.get("f1", 0)
        n = metrics.get("num_entries", 0)
        print(f"  {seg_name:<15s} DR={dr:.3f}  F1={f1:.3f}  (n={n})")

    # Contamination score
    if result.contamination_score is not None:
        print(f"\n  Contamination score: {result.contamination_score:.3f}")
        if result.contamination_score > 0.2:
            print("  WARNING: Large performance gap between historical and recent entries.")
            print("  This tool may be relying on memorization.")
        else:
            print("  Performance is stable across time periods - good sign.")

    # Manual date range filtering
    print("\n=== Custom Date Range ===\n")
    recent = filter_by_date_range(entries, start_date="2023-01-01")
    older = filter_by_date_range(entries, end_date="2022-12-31")
    print(f"  Recent entries (2023+): {len(recent)}")
    print(f"  Older entries (<2023):  {len(older)}")

    if recent:
        cm = build_confusion_matrix(recent, pred_map)
        print(f"  Recent DR: {cm.detection_rate:.3f}")
    if older:
        cm = build_confusion_matrix(older, pred_map)
        print(f"  Older DR:  {cm.detection_rate:.3f}")


if __name__ == "__main__":
    main()
