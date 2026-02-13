#!/usr/bin/env python3
"""
Build source provenance mapping for all entries in dev_public and test_public.

This script analyzes entries based on generation_method, explanation text,
and bibtex_key patterns to determine the correct source for each entry.
"""

import json
from collections import Counter
from pathlib import Path


def determine_source(entry: dict) -> str:
    """
    Determine the source of an entry based on generation_method and explanation.

    Rules:
    1. generation_method="scraped" + label="VALID" -> "dblp"
    2. generation_method="perturbation" -> "perturbation"
    3. generation_method="llm_generated" -> "llm_generated"
    4. generation_method="adversarial" -> "adversarial"
    5. generation_method="real_world" -> analyze explanation:
       - GPTZero/NeurIPS 2025 mentions -> "gptzero_neurips2025"
       - GhostCite mentions -> "ghostcite"
       - HalluCitation mentions -> "hallucitation"
       - Otherwise -> "manual_collection"
    """
    generation_method = entry.get("generation_method")
    label = entry.get("label")
    explanation = entry.get("explanation", "").lower()

    # Rule 1: Scraped valid entries from DBLP
    if generation_method == "scraped" and label == "VALID":
        return "dblp"

    # Rule 2: Perturbation
    if generation_method == "perturbation":
        return "perturbation"

    # Rule 3: LLM-generated
    if generation_method == "llm_generated":
        return "llm_generated"

    # Rule 4: Adversarial
    if generation_method == "adversarial":
        return "adversarial"

    # Rule 5: Real-world - need deeper analysis
    if generation_method == "real_world":
        # Check for study references in explanation
        if "gptzero" in explanation or "neurips 2025" in explanation:
            return "gptzero_neurips2025"
        elif "ghostcite" in explanation:
            return "ghostcite"
        elif "hallucitation" in explanation:
            return "hallucitation"
        else:
            return "manual_collection"

    # Unknown/unexpected
    return "unknown"


def build_source_mapping(dev_file: Path, test_file: Path, output_file: Path) -> None:
    """Build and save source mapping for all entries."""

    source_mapping = {}
    source_distribution = Counter()

    # Process both files
    for split_name, filepath in [("dev", dev_file), ("test", test_file)]:
        print(f"\nProcessing {split_name} split: {filepath}")

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                entry = json.loads(line)
                bibtex_key = entry["bibtex_key"]

                # Determine source
                source = determine_source(entry)

                # Store in mapping
                source_mapping[bibtex_key] = source
                source_distribution[source] += 1

                # Warn if unknown
                if source == "unknown":
                    print(
                        f"  WARNING: Unknown source for {bibtex_key} "
                        f"(line {line_num}, method={entry.get('generation_method')})"
                    )

    # Save mapping
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(source_mapping, f, indent=2, sort_keys=True)

    print(f"\n✓ Saved source mapping to {output_file}")
    print(f"  Total entries: {len(source_mapping)}")

    # Print distribution summary
    print("\n" + "=" * 60)
    print("Source Distribution Summary")
    print("=" * 60)

    total = sum(source_distribution.values())
    for source in sorted(source_distribution.keys()):
        count = source_distribution[source]
        pct = 100 * count / total
        print(f"{source:25s} {count:5d}  ({pct:5.1f}%)")

    print("-" * 60)
    print(f"{'TOTAL':25s} {total:5d}  (100.0%)")
    print("=" * 60)

    # Verify completeness
    unknown_count = source_distribution.get("unknown", 0)
    if unknown_count > 0:
        print(f"\n⚠ WARNING: {unknown_count} entries have unknown source!")
    else:
        print("\n✓ All entries have a known source")


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "v1.0"

    dev_file = data_dir / "dev_public.jsonl"
    test_file = data_dir / "test_public.jsonl"
    output_file = data_dir / "source_mapping.json"

    # Verify input files exist
    if not dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Build mapping
    build_source_mapping(dev_file, test_file, output_file)


if __name__ == "__main__":
    main()
