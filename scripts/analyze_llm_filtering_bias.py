"""P2.6: Document LLM filtering selection bias.  [analysis]

Analyzes llm_generated.jsonl to identify type distribution gaps and filtering bias.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_llm_entries(path: Path) -> list[dict[str, Any]]:
    """Load LLM-generated entries."""
    entries = []
    with path.open("r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_perturbation_entries(path: Path) -> list[dict[str, Any]]:
    """Load perturbation entries for comparison."""
    entries = []
    with path.open("r") as f:
        for line in f:
            data = json.loads(line)
            if (
                data.get("generation_method") == "perturbation"
                and data.get("label") == "HALLUCINATED"
            ):
                entries.append(data)
    return entries


def analyze_type_distribution(entries: list[dict[str, Any]]) -> None:
    """Analyze hallucination type distribution."""
    print("--- Type Distribution ---")
    type_counts = Counter(e.get("hallucination_type") for e in entries)

    total = len(entries)
    print(f"Total entries: {total}")
    print("\nBreakdown by type:")

    for h_type, count in type_counts.most_common():
        pct = (count / total) * 100
        print(f"  {h_type:30s}: {count:4d} ({pct:5.1f}%)")

    # Check for catch-all dominance
    plausible_count = type_counts.get("plausible_fabrication", 0)
    plausible_pct = (plausible_count / total) * 100 if total > 0 else 0

    print(f"\n⚠ Plausible fabrication (catch-all): {plausible_pct:.1f}%")
    if plausible_pct > 50:
        print("  → High reliance on catch-all category; LLMs struggle with specificity")


def analyze_features(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute feature statistics."""
    print("\n--- Feature Statistics ---")

    doi_count = sum(1 for e in entries if e.get("fields", {}).get("doi"))
    doi_pct = (doi_count / len(entries)) * 100 if entries else 0
    print(f"DOI presence: {doi_count}/{len(entries)} ({doi_pct:.1f}%)")

    # Title lengths
    title_lengths = []
    for e in entries:
        title = e.get("fields", {}).get("title", "")
        title_lengths.append(len(title.split()))

    print(
        f"Title length: mean={np.mean(title_lengths):.1f}, "
        f"median={np.median(title_lengths):.1f}, "
        f"std={np.std(title_lengths):.1f}"
    )

    # Author counts
    author_counts = []
    for e in entries:
        author = e.get("fields", {}).get("author", "")
        # BibTeX uses " and " as separator
        count = len([a.strip() for a in author.split(" and ") if a.strip()])
        author_counts.append(count)

    print(
        f"Author count: mean={np.mean(author_counts):.1f}, "
        f"median={np.median(author_counts):.1f}, "
        f"std={np.std(author_counts):.1f}"
    )

    # Year distribution
    years = []
    for e in entries:
        year_str = e.get("fields", {}).get("year", "")
        if year_str.isdigit():
            years.append(int(year_str))

    if years:
        print(
            f"Year: mean={np.mean(years):.1f}, "
            f"median={np.median(years):.1f}, "
            f"std={np.std(years):.1f}, "
            f"range=[{min(years)}, {max(years)}]"
        )

    return {
        "doi_pct": doi_pct,
        "mean_title_length": np.mean(title_lengths),
        "mean_author_count": np.mean(author_counts),
        "mean_year": np.mean(years) if years else 0,
    }


def compare_with_perturbation(
    llm_entries: list[dict[str, Any]], pert_entries: list[dict[str, Any]]
) -> None:
    """Compare LLM distribution with perturbation entries."""
    print("\n--- Comparison with Perturbation Entries ---")

    llm_types = Counter(e.get("hallucination_type") for e in llm_entries)
    pert_types = Counter(e.get("hallucination_type") for e in pert_entries)

    print(f"LLM types: {len(llm_types)}")
    print(f"Perturbation types: {len(pert_types)}")

    # Find gaps: types present in perturbation but absent/rare in LLM
    all_types = set(llm_types.keys()) | set(pert_types.keys())

    print("\nType coverage gaps:")
    gaps = []
    for h_type in sorted(all_types, key=lambda x: x or ""):
        llm_count = llm_types.get(h_type, 0)
        pert_count = pert_types.get(h_type, 0)

        llm_pct = (llm_count / len(llm_entries)) * 100 if llm_entries else 0
        pert_pct = (pert_count / len(pert_entries)) * 100 if pert_entries else 0

        gap = pert_pct - llm_pct
        if gap > 5:  # >5pp underrepresented in LLM
            gaps.append((h_type, gap, llm_pct, pert_pct))
            print(f"  {h_type:30s}: LLM={llm_pct:5.1f}%, Pert={pert_pct:5.1f}%, gap={gap:+5.1f}pp")

    if not gaps:
        print("  No significant gaps (all types within 5pp)")


def identify_recommendations(
    llm_entries: list[dict[str, Any]], pert_entries: list[dict[str, Any]]
) -> None:
    """Generate recommendations for expanding LLM generation."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    llm_types = Counter(e.get("hallucination_type") for e in llm_entries)
    pert_types = Counter(e.get("hallucination_type") for e in pert_entries)

    underrepresented = []
    for h_type in pert_types:
        llm_count = llm_types.get(h_type, 0)
        pert_count = pert_types[h_type]

        llm_pct = (llm_count / len(llm_entries)) * 100 if llm_entries else 0
        pert_pct = (pert_count / len(pert_entries)) * 100 if pert_entries else 0

        if pert_pct - llm_pct > 5:
            underrepresented.append((h_type, llm_count, pert_count))

    if underrepresented:
        print("\n1. Expand LLM generation for underrepresented types:")
        for h_type, llm_count, pert_count in underrepresented:
            print(f"   - {h_type}: add ~{pert_count - llm_count} entries")

    plausible_pct = (
        (llm_types.get("plausible_fabrication", 0) / len(llm_entries)) * 100 if llm_entries else 0
    )
    if plausible_pct > 40:
        print("\n2. Reduce catch-all reliance: prompt LLMs with specific type constraints")

    print("\n3. Increase structured prompting:")
    print("   - Provide type-specific templates (e.g., 'cite a paper with wrong venue')")
    print("   - Use few-shot examples for each hallucination type")

    print("\n4. Validate LLM outputs with pre-screening:")
    print("   - DOI resolution checks")
    print("   - Author/title existence verification via external APIs")


def main() -> None:
    """Run LLM filtering bias analysis."""
    data_dir = Path(__file__).parent.parent / "data" / "v1.0"
    llm_path = data_dir / "llm_generated.jsonl"
    dev_path = data_dir / "dev_public.jsonl"

    print("=" * 80)
    print("LLM FILTERING BIAS ANALYSIS")
    print("=" * 80)

    print("\nLoading llm_generated.jsonl...")
    llm_entries = load_llm_entries(llm_path)
    print(f"Loaded {len(llm_entries)} LLM-generated entries")

    analyze_type_distribution(llm_entries)
    llm_features = analyze_features(llm_entries)

    print("\nLoading perturbation entries for comparison...")
    pert_entries = load_perturbation_entries(dev_path)
    print(f"Loaded {len(pert_entries)} perturbation entries")

    compare_with_perturbation(llm_entries, pert_entries)
    identify_recommendations(llm_entries, pert_entries)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"LLM-generated entries: {len(llm_entries)}")
    print(f"Mean title length: {llm_features['mean_title_length']:.1f} words")
    print(f"Mean author count: {llm_features['mean_author_count']:.1f}")
    print(f"DOI presence rate: {llm_features['doi_pct']:.1f}%")
    print("\nKey finding: LLM generation shows selection bias toward certain types;")
    print("structured prompting and type-specific generation needed for balance.")


if __name__ == "__main__":
    main()
