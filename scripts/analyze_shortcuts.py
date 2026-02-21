#!/usr/bin/env python3
"""Shortcut characterization: can metadata features predict labels without citation content?

Extracts simple metadata features (has_doi, field_count, author_count, title_length,
bibtex_type) and trains a logistic regression classifier to predict HALLUCINATED vs VALID.
If accuracy significantly exceeds the prevalence baseline, shortcut features leak label info.

Requires: scikit-learn (optional dependency, install with `pip install hallmark[analysis]`)

Usage:
    python scripts/analyze_shortcuts.py
    python scripts/analyze_shortcuts.py --split dev_public test_public --cv-folds 10
"""

from __future__ import annotations

import argparse
import sys

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import BenchmarkEntry


def extract_features(entry: BenchmarkEntry) -> list[float]:
    """Extract metadata features from a benchmark entry."""
    fields = entry.fields
    return [
        float(bool(fields.get("doi"))),  # has_doi
        float(len(fields)),  # field_count
        float(len(fields.get("author", "").split(" and "))),  # author_count
        float(len(fields.get("title", ""))),  # title_length
        float(hash(entry.bibtex_type) % 10),  # bibtex_type (ordinal encoding)
        float(len(fields.get("year", ""))),  # year_length
        float(bool(fields.get("url"))),  # has_url
        float(bool(fields.get("booktitle") or fields.get("journal"))),  # has_venue
    ]


FEATURE_NAMES = [
    "has_doi",
    "field_count",
    "author_count",
    "title_length",
    "bibtex_type",
    "year_length",
    "has_url",
    "has_venue",
]


def main() -> None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print(
            "scikit-learn is required for shortcut analysis.\n"
            "Install with: pip install scikit-learn>=1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    import numpy as np

    parser = argparse.ArgumentParser(description="Shortcut characterization for HALLMARK")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["dev_public", "test_public"],
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    for split in args.split:
        try:
            entries = load_split(split=split, version=args.version, data_dir=args.data_dir)
        except FileNotFoundError:
            print(f"Split {split} not found, skipping.")
            continue

        X = np.array([extract_features(e) for e in entries])
        y = np.array([1 if e.label == "HALLUCINATED" else 0 for e in entries])

        prevalence = y.mean()
        majority_baseline = max(prevalence, 1 - prevalence)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=args.cv_folds, scoring="accuracy")

        # Feature importances from full-data fit
        clf.fit(X, y)
        coefs = clf.coef_[0]

        print(f"\n{'=' * 60}")
        print(f"  Shortcut Analysis: {split}")
        print(f"{'=' * 60}")
        print(f"  Entries:             {len(entries)}")
        print(f"  Prevalence:          {prevalence:.1%} hallucinated")
        print(f"  Majority baseline:   {majority_baseline:.1%}")
        print(
            f"  Logistic regression: {scores.mean():.1%} +/- {scores.std():.1%} ({args.cv_folds}-fold CV)"
        )
        print(
            f"  Shortcut leakage:    {scores.mean() - majority_baseline:+.1%} over majority baseline"
        )
        print(f"{'─' * 60}")
        print("  Feature importances (logistic regression coefficients):")
        for name, coef in sorted(zip(FEATURE_NAMES, coefs, strict=True), key=lambda x: -abs(x[1])):
            print(f"    {name:<20} {coef:+.3f}")
        print(f"{'=' * 60}")

        if scores.mean() - majority_baseline > 0.05:
            print("  WARNING: Metadata features predict labels >5pp above baseline.")
            print("  This suggests potential shortcut learning risk.")
        else:
            print("  OK: No significant shortcut leakage detected.")


if __name__ == "__main__":
    main()
