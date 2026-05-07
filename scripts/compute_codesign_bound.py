"""Compute construct-overfitting bound for bibtex-updater on dev_public.

Uses per_type_metrics already stored in the reference results JSON (DR + count
per hallucination type). Bootstrap CIs use a binomial resampling approach:
we treat each type as n Bernoulli trials with p = observed DR and resample
with replacement (1000 samples, percentile method, no bias correction).

Stress-test types (merged_citation, partial_author_list, arxiv_version_mismatch)
appear in dev_public (30 entries each) per the benchmark design; no separate
BTU stress_test predictions file exists, so all analysis uses dev_public.

Output: prints per-type table and gap summary; saves tables/codesign_bound.csv.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "data" / "v1.0" / "baseline_results" / "bibtexupdater_dev_public.json"
TABLES_DIR = REPO_ROOT / "tables"
OUTPUT_CSV = TABLES_DIR / "codesign_bound.csv"

# Hallucination types that are theoretically orthogonal to BTU's design lineage
# (stress-test / theoretically-motivated types added independently of the tool)
STRESS_TEST_TYPES = {"merged_citation", "partial_author_list", "arxiv_version_mismatch"}

# Design-aligned: Tier 1-2 perturbation types BTU was originally targeted at
DESIGN_ALIGNED_TYPES = {
    "fabricated_doi",
    "nonexistent_venue",
    "placeholder_authors",
    "future_date",
    "chimeric_title",
    "wrong_venue",
    "swapped_authors",  # enum value for author_mismatch
    "preprint_as_published",
    "hybrid_fabrication",
    # Tier-3 main types are included in the gap calculation (they are part of
    # the design-aligned aggregate reported in the paper).
    "near_miss_title",
    "plausible_fabrication",
}

N_BOOTSTRAP = 1000
SEED = 42


def bootstrap_dr_ci(
    n: int, dr: float, n_samples: int = N_BOOTSTRAP, rng: random.Random | None = None
) -> tuple[float, float]:
    """Binomial bootstrap 95% CI on detection rate.

    Treats the observed (n * dr) correct detections out of n as the empirical
    distribution and resamples with replacement.
    """
    if rng is None:
        rng = random.Random(SEED)
    n_correct = round(n * dr)
    # Build population: n_correct 1s and (n - n_correct) 0s
    population = [1] * n_correct + [0] * (n - n_correct)
    boot_drs = []
    for _ in range(n_samples):
        sample = rng.choices(population, k=n)
        boot_drs.append(sum(sample) / n)
    boot_drs.sort()
    lo = boot_drs[int(0.025 * n_samples)]
    hi = boot_drs[int(0.975 * n_samples) - 1]
    return lo, hi


def aggregate_dr(types: list[str], per_type: dict[str, dict]) -> tuple[float, int, float, float]:
    """Pool entries across types and compute aggregate DR + bootstrap CI."""
    total_correct = 0
    total_n = 0
    for t in types:
        if t not in per_type:
            continue
        n = per_type[t]["count"]
        dr = per_type[t]["detection_rate"]
        total_correct += round(n * dr)
        total_n += n
    if total_n == 0:
        return 0.0, 0, 0.0, 0.0
    agg_dr = total_correct / total_n
    rng = random.Random(SEED)
    lo, hi = bootstrap_dr_ci(total_n, agg_dr, rng=rng)
    return agg_dr, total_n, lo, hi


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"BTU results not found: {RESULTS_FILE}\n"
            "Available files in baseline_results/:\n"
            + "\n".join(
                os.listdir(RESULTS_FILE.parent)
                if RESULTS_FILE.parent.exists()
                else ["<dir missing>"]
            )
        )

    data = json.loads(RESULTS_FILE.read_text())
    per_type: dict[str, dict] = data["per_type_metrics"]

    # Filter to hallucination types only (exclude 'valid')
    hall_types = {k: v for k, v in per_type.items() if k != "valid"}

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    rows: list[dict] = []
    for htype, metrics in sorted(hall_types.items()):
        n = metrics["count"]
        dr = metrics["detection_rate"]
        lo, hi = bootstrap_dr_ci(n, dr, rng=rng)
        is_stress = htype in STRESS_TEST_TYPES
        rows.append(
            {
                "type": htype,
                "n": n,
                "dr": dr,
                "ci_low": lo,
                "ci_high": hi,
                "is_stress_test": is_stress,
            }
        )

    # Print per-type table
    print(f"\n{'Type':<30} {'n':>4} {'DR':>6} {'95% CI':>20} {'Stress?':>8}")
    print("-" * 72)
    for r in rows:
        ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        print(
            f"{r['type']:<30} {r['n']:>4} {r['dr']:>6.3f} {ci_str:>20} {r['is_stress_test']!s:>8}"
        )

    # Aggregate: design-aligned vs stress-test
    stress_types_present = [t for t in STRESS_TEST_TYPES if t in hall_types]
    aligned_types_present = [t for t in DESIGN_ALIGNED_TYPES if t in hall_types]

    agg_aligned_dr, agg_aligned_n, agg_aligned_lo, agg_aligned_hi = aggregate_dr(
        aligned_types_present, per_type
    )
    agg_stress_dr, agg_stress_n, agg_stress_lo, agg_stress_hi = aggregate_dr(
        stress_types_present, per_type
    )

    gap_pp = (agg_aligned_dr - agg_stress_dr) * 100

    print("\n" + "=" * 72)
    print(f"\nStress-test types with BTU predictions: {stress_types_present}")
    print(
        f"\nDesign-aligned aggregate: DR={agg_aligned_dr:.4f} "
        f"(95% CI [{agg_aligned_lo:.4f}, {agg_aligned_hi:.4f}], n={agg_aligned_n})"
    )
    print(
        f"Stress-test aggregate:    DR={agg_stress_dr:.4f} "
        f"(95% CI [{agg_stress_lo:.4f}, {agg_stress_hi:.4f}], n={agg_stress_n})"
    )
    print(f"\nDR gap (aligned - stress): {gap_pp:.1f} pp")
    print(
        "\nNOTE: All results from dev_public split. No separate BTU stress_test "
        "predictions exist (requires Semantic Scholar API). Stress-test types "
        "(merged_citation, partial_author_list, arxiv_version_mismatch) appear "
        "in dev_public with 30 entries each per benchmark design."
    )

    # Save CSV
    fieldnames = ["type", "n", "dr", "ci_low", "ci_high", "is_stress_test"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}")

    # Print LaTeX-ready numbers for paper
    print("\n--- LaTeX numbers ---")
    print(
        f"Design-aligned: DR={agg_aligned_dr:.3f}, "
        f"CI=[{agg_aligned_lo:.3f}, {agg_aligned_hi:.3f}], n={agg_aligned_n}"
    )
    print(
        f"Stress-test:    DR={agg_stress_dr:.3f}, "
        f"CI=[{agg_stress_lo:.3f}, {agg_stress_hi:.3f}], n={agg_stress_n}"
    )
    print(f"Gap: {gap_pp:.1f} pp")


if __name__ == "__main__":
    main()
