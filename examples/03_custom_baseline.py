#!/usr/bin/env python3
"""Example 3: Writing a custom baseline for HALLMARK.

This example shows how to:
1. Implement a simple rule-based verification tool
2. Produce predictions in the HALLMARK format
3. Evaluate and save results
"""

import re
from pathlib import Path

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import Prediction
from hallmark.evaluation.metrics import evaluate


def rule_based_verifier(entries):
    """A custom rule-based citation verifier.

    Rules:
    1. Flag entries where year > 2025 (future date)
    2. Flag entries where author contains "Doe", "Smith" (placeholder)
    3. Flag entries where DOI starts with 10.9999 (fabricated prefix)
    4. Flag entries where venue contains "Advanced AI" (common fake venue)
    """
    predictions = []

    for entry in entries:
        flags = []
        confidence = 0.3  # default: probably valid

        # Rule 1: Future date
        year = entry.fields.get("year", "")
        if year and year.isdigit() and int(year) > 2025:
            flags.append(f"Future year: {year}")
            confidence = max(confidence, 0.9)

        # Rule 2: Placeholder authors
        authors = entry.fields.get("author", "")
        placeholder_names = ["John Doe", "Jane Smith", "Test Author", "A. Researcher"]
        for name in placeholder_names:
            if name.lower() in authors.lower():
                flags.append(f"Placeholder author: {name}")
                confidence = max(confidence, 0.85)

        # Rule 3: Fabricated DOI prefix
        doi = entry.fields.get("doi", "")
        if doi and re.match(r"^10\.(9999|8888|7777)", doi):
            flags.append(f"Suspicious DOI prefix: {doi[:10]}")
            confidence = max(confidence, 0.9)

        # Rule 4: Fake venue keywords
        venue = entry.fields.get("booktitle", "") or entry.fields.get("journal", "")
        fake_keywords = ["Advanced AI Systems", "Emerging Methods", "Frontier Artificial"]
        for kw in fake_keywords:
            if kw.lower() in venue.lower():
                flags.append(f"Suspicious venue: {kw}")
                confidence = max(confidence, 0.8)

        label = "HALLUCINATED" if flags else "VALID"
        reason = "; ".join(flags) if flags else "No issues found"

        predictions.append(
            Prediction(
                bibtex_key=entry.bibtex_key,
                label=label,
                confidence=confidence,
                reason=reason,
            )
        )

    return predictions


def main():
    # Load and evaluate
    entries = load_split("dev_public")
    predictions = rule_based_verifier(entries)

    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name="rule-based-verifier",
        split_name="dev_public",
    )

    # Print results
    print("Rule-Based Verifier Results:")
    print(f"  Detection Rate:   {result.detection_rate:.3f}")
    print(f"  False Pos. Rate:  {result.false_positive_rate:.3f}")
    print(f"  F1 (Halluc.):     {result.f1_hallucination:.3f}")
    print(f"  Tier-weighted F1: {result.tier_weighted_f1:.3f}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "rule_based_dev.json"
    output_path.write_text(result.to_json())
    print(f"\nResults saved to {output_path}")

    # Show what was caught
    pred_map = {p.bibtex_key: p for p in predictions}
    caught = [
        e
        for e in entries
        if e.label == "HALLUCINATED" and pred_map[e.bibtex_key].label == "HALLUCINATED"
    ]
    missed = [
        e for e in entries if e.label == "HALLUCINATED" and pred_map[e.bibtex_key].label == "VALID"
    ]

    print(f"\nCaught {len(caught)} / {len(caught) + len(missed)} hallucinations:")
    for e in caught[:5]:
        p = pred_map[e.bibtex_key]
        print(f"  [{e.hallucination_type}] {e.bibtex_key}: {p.reason}")

    if missed:
        print(f"\nMissed {len(missed)} hallucinations:")
        for e in missed[:5]:
            print(f"  [{e.hallucination_type}] {e.bibtex_key}")


if __name__ == "__main__":
    main()
