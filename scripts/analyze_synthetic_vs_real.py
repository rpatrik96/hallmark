"""P1.3: Compare synthetic vs real-world hallucination distributions.  [analysis]

Analyzes whether synthetic (perturbation + adversarial) hallucinations
match real-world hallucination patterns across sub-test failures and features.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def load_entries(path: Path) -> list[dict[str, Any]]:
    """Load JSONL entries."""
    entries = []
    with path.open("r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def extract_subtest_pattern(entry: dict[str, Any]) -> tuple[bool, ...]:
    """Extract ordered tuple of subtest results."""
    subtests = entry.get("subtests", {})
    # Fixed order for consistency
    keys = [
        "doi_resolves",
        "title_exists",
        "authors_match",
        "venue_real",
        "fields_complete",
        "cross_db_agreement",
    ]
    return tuple(subtests.get(k, False) for k in keys)


def extract_features(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract numeric features for comparison."""
    fields = entry.get("fields", {})
    title = fields.get("title", "")
    author = fields.get("author", "")
    year = fields.get("year", "")
    doi = fields.get("doi")

    # Count authors (split by "and")
    author_count = len([a.strip() for a in author.split(" and ") if a.strip()])

    return {
        "has_doi": doi is not None,
        "author_count": author_count,
        "title_word_count": len(title.split()),
        "year": int(year) if year.isdigit() else 0,
    }


def compare_distributions(
    real_entries: list[dict[str, Any]],
    synthetic_entries: list[dict[str, Any]],
    hallucination_type: str,
) -> dict[str, Any]:
    """Compare real vs synthetic for a specific hallucination type."""
    print(f"\n{'=' * 80}")
    print(f"Hallucination Type: {hallucination_type}")
    print(f"{'=' * 80}")
    print(f"Real-world entries: {len(real_entries)}")
    print(f"Synthetic entries: {len(synthetic_entries)}")

    results = {}

    # 1. Compare subtest failure patterns
    real_patterns = [extract_subtest_pattern(e) for e in real_entries]
    synth_patterns = [extract_subtest_pattern(e) for e in synthetic_entries]

    # Count pattern frequencies
    real_pattern_counts: dict[tuple[bool, ...], int] = defaultdict(int)
    synth_pattern_counts: dict[tuple[bool, ...], int] = defaultdict(int)

    for p in real_patterns:
        real_pattern_counts[p] += 1
    for p in synth_patterns:
        synth_pattern_counts[p] += 1

    print("\n--- Sub-test Failure Patterns ---")
    print(f"Unique patterns (real): {len(real_pattern_counts)}")
    print(f"Unique patterns (synthetic): {len(synth_pattern_counts)}")

    # Chi-squared test on pattern frequencies
    all_patterns = set(real_pattern_counts.keys()) | set(synth_pattern_counts.keys())
    real_counts = [real_pattern_counts[p] for p in all_patterns]
    synth_counts = [synth_pattern_counts[p] for p in all_patterns]

    # Chi-squared requires expected frequencies >= 5; use Fisher's exact if too sparse
    if len(all_patterns) > 1:
        try:
            _chi2, p_val = stats.chisquare(synth_counts, f_exp=real_counts)
            results["pattern_chi2_p"] = p_val
            print(f"Chi-squared test p-value: {p_val:.4f}")
            if p_val > 0.05:
                print("✓ Patterns match (p > 0.05)")
            else:
                print("✗ Patterns differ significantly (p ≤ 0.05)")
        except Exception as e:
            print(f"Chi-squared test failed: {e}")
            results["pattern_chi2_p"] = None
    else:
        print("Insufficient pattern diversity for chi-squared test")
        results["pattern_chi2_p"] = None

    # 2. Compare continuous features
    real_features = [extract_features(e) for e in real_entries]
    synth_features = [extract_features(e) for e in synthetic_entries]

    print("\n--- Feature Distributions ---")

    # DOI presence rate
    real_doi_rate = sum(f["has_doi"] for f in real_features) / len(real_features)
    synth_doi_rate = sum(f["has_doi"] for f in synth_features) / len(synth_features)
    print(f"DOI presence: real={real_doi_rate:.2%}, synthetic={synth_doi_rate:.2%}")
    results["doi_rate_diff"] = abs(real_doi_rate - synth_doi_rate)

    # Author count
    real_authors = [f["author_count"] for f in real_features]
    synth_authors = [f["author_count"] for f in synth_features]
    print(
        f"Author count: real={np.mean(real_authors):.1f}±{np.std(real_authors):.1f}, "
        f"synthetic={np.mean(synth_authors):.1f}±{np.std(synth_authors):.1f}"
    )

    # KS test for author count
    _ks_authors, p_authors = stats.ks_2samp(real_authors, synth_authors)
    results["author_count_ks_p"] = p_authors
    print(f"KS test (author count) p-value: {p_authors:.4f}")

    # Title word count
    real_titles = [f["title_word_count"] for f in real_features]
    synth_titles = [f["title_word_count"] for f in synth_features]
    print(
        f"Title length: real={np.mean(real_titles):.1f}±{np.std(real_titles):.1f}, "
        f"synthetic={np.mean(synth_titles):.1f}±{np.std(synth_titles):.1f}"
    )

    _ks_titles, p_titles = stats.ks_2samp(real_titles, synth_titles)
    results["title_length_ks_p"] = p_titles
    print(f"KS test (title length) p-value: {p_titles:.4f}")

    # Year distribution
    real_years = [f["year"] for f in real_features if f["year"] > 0]
    synth_years = [f["year"] for f in synth_features if f["year"] > 0]
    if real_years and synth_years:
        print(
            f"Year: real={np.mean(real_years):.1f}±{np.std(real_years):.1f}, "
            f"synthetic={np.mean(synth_years):.1f}±{np.std(synth_years):.1f}"
        )
        _ks_years, p_years = stats.ks_2samp(real_years, synth_years)
        results["year_ks_p"] = p_years
        print(f"KS test (year) p-value: {p_years:.4f}")

    return results


def main() -> None:
    """Run synthetic vs real-world comparison analysis."""
    data_dir = Path(__file__).parent.parent / "data" / "v1.0"
    dev_path = data_dir / "dev_public.jsonl"

    print("Loading dev_public.jsonl...")
    all_entries = load_entries(dev_path)

    # Filter hallucinated entries
    hallucinated = [e for e in all_entries if e.get("label") == "HALLUCINATED"]

    # Group by hallucination type and generation method
    by_type_method: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for entry in hallucinated:
        h_type = entry.get("hallucination_type")
        gen_method = entry.get("generation_method")
        if h_type and gen_method:
            by_type_method[h_type][gen_method].append(entry)

    # Find types with both real_world and synthetic entries
    summary_results = {}
    types_with_both = []

    for h_type, methods in by_type_method.items():
        has_real = "real_world" in methods and len(methods["real_world"]) > 0
        has_synthetic = ("perturbation" in methods or "adversarial" in methods) and (
            len(methods.get("perturbation", [])) + len(methods.get("adversarial", [])) > 0
        )

        if has_real and has_synthetic:
            types_with_both.append(h_type)
            real_entries = methods["real_world"]
            synthetic_entries = methods.get("perturbation", []) + methods.get("adversarial", [])

            results = compare_distributions(real_entries, synthetic_entries, h_type)
            summary_results[h_type] = results

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Types with both real and synthetic: {len(types_with_both)}")
    print(f"Types analyzed: {', '.join(types_with_both)}")

    print("\n--- Representativeness Assessment ---")
    for h_type, results in summary_results.items():
        print(f"\n{h_type}:")
        matches = []

        if results.get("pattern_chi2_p"):
            if results["pattern_chi2_p"] > 0.05:
                matches.append("patterns match")
            else:
                matches.append("patterns differ")

        if results.get("author_count_ks_p"):
            if results["author_count_ks_p"] > 0.05:
                matches.append("author counts match")
            else:
                matches.append("author counts differ")

        if results.get("title_length_ks_p"):
            if results["title_length_ks_p"] > 0.05:
                matches.append("title lengths match")
            else:
                matches.append("title lengths differ")

        if all("match" in m for m in matches):
            print("  ✓ Synthetic entries are representative (all p > 0.05)")
        else:
            print(f"  ⚠ Synthetic entries differ from real-world: {'; '.join(matches)}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
