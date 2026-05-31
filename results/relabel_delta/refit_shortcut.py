#!/usr/bin/env python3
"""Re-fit the app:shortcuts 5-fold CV logistic regression on the NEW dev_public labels.

The paper (app:shortcuts, appendix.tex L550) describes EIGHT entry-level metadata
features: presence of DOI, field count, author count, title length (characters),
title word count, year (numeric), BibTeX entry type, presence of a booktitle field.
We reproduce that exact feature set (the shipped scripts/analyze_shortcuts.py uses a
slightly different 8-feature set: year_length/has_url/has_venue instead of
title_word_count/year_numeric/has_booktitle) so the refreshed numbers replace the
paper's printed 58.5% / 56.6% / 1.9pp faithfully.

5-fold CV, LogisticRegression(max_iter=1000, random_state=42), accuracy scoring,
same as the shipped script. No API calls; pure offline re-fit on the relabeled split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
DEV = REPO / "data/v1.0/dev_public.jsonl"

# Paper's 8 features (appendix.tex L550), in caption order.
FEATURE_NAMES = [
    "has_doi",
    "field_count",
    "author_count",
    "title_length_chars",
    "title_word_count",
    "year_numeric",
    "bibtex_type",
    "has_booktitle",
]


def _year_numeric(year_str: str) -> float:
    s = "".join(ch for ch in str(year_str) if ch.isdigit())
    try:
        return float(int(s[:4])) if s else 0.0
    except ValueError:
        return 0.0


def extract_features(entry: dict, type_index: dict[str, int]) -> list[float]:
    fields = entry.get("fields", {}) or {}
    title = fields.get("title", "") or ""
    author = fields.get("author", "") or ""
    return [
        float(bool(fields.get("doi"))),
        float(len(fields)),
        float(len(author.split(" and ")) if author.strip() else 0),
        float(len(title)),
        float(len(title.split())),
        _year_numeric(fields.get("year", "")),
        # DETERMINISTIC ordinal encoding (sorted-vocab index). The shipped
        # scripts/analyze_shortcuts.py used hash(type)%10, which is non-reproducible
        # under PYTHONHASHSEED randomization; we replace it with a stable index so
        # the refreshed number is exactly reproducible.
        float(type_index.get(entry.get("bibtex_type", ""), 0)),
        float(bool(fields.get("booktitle"))),
    ]


def main() -> dict:
    entries = [json.loads(line) for line in DEV.read_text().splitlines() if line.strip()]
    type_vocab = sorted({e.get("bibtex_type", "") for e in entries})
    type_index = {t: i for i, t in enumerate(type_vocab)}
    X = np.array([extract_features(e, type_index) for e in entries])
    y = np.array([1 if e["label"] == "HALLUCINATED" else 0 for e in entries])

    n = len(entries)
    n_hall = int(y.sum())
    n_valid = n - n_hall
    prevalence = float(y.mean())  # fraction hallucinated
    majority_baseline = float(max(prevalence, 1 - prevalence))

    # --- Variant A: faithful to shipped scripts/analyze_shortcuts.py ---
    # plain cv=5 (unshuffled), raw (unscaled) features. The dev file has a trailing
    # all-HALLUCINATED block (scaled-up additions appended at the end), so unshuffled
    # KFold puts that block in the last fold -> a pathological fold-5 accuracy. Reported
    # for exact-method continuity with the shipped script only.
    clf_raw = LogisticRegression(max_iter=1000, random_state=42)
    scores_raw = cross_val_score(clf_raw, X, y, cv=5, scoring="accuracy")

    # --- Variant B: statistically correct refresh (standardized + shuffled stratified) ---
    # Standardize features (the 8 features span vastly different scales: year~2022 vs
    # has_doi in {0,1}); shuffle+stratify the 5 folds so each fold mirrors the 54/46
    # class balance. This is the only setup under which logistic-regression coefficients
    # are comparable across features and can match the paper's reported magnitudes
    # ("has_doi most informative, all others small"). Use this as the refreshed number.
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_std = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")

    cv_acc = float(scores_std.mean())
    cv_std = float(scores_std.std())
    margin = cv_acc - majority_baseline

    # Coefficients from the standardized full-data fit (interpretable across features).
    scaler = StandardScaler().fit(X)
    clf_std = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(X), y)
    coefs = clf_std.coef_[0]
    coef_map = {name: float(c) for name, c in zip(FEATURE_NAMES, coefs, strict=True)}

    result = {
        "n": n,
        "n_hallucinated": n_hall,
        "n_valid": n_valid,
        "prevalence_hallucinated": round(prevalence, 4),
        "majority_class": "HALLUCINATED" if n_hall >= n_valid else "VALID",
        "majority_baseline_acc": round(majority_baseline, 4),
        "cv_folds": 5,
        # --- refreshed faithful numbers (Variant B) ---
        "cv_accuracy_mean": round(cv_acc, 4),
        "cv_accuracy_std": round(cv_std, 4),
        "margin_over_baseline": round(margin, 4),
        "fold_accuracies": [round(float(s), 4) for s in scores_std],
        "coefficients": {k: round(v, 4) for k, v in coef_map.items()},
        # --- shipped-script-exact variant (raw features, unshuffled cv=5) ---
        "variant_shipped_script_unshuffled_raw": {
            "cv_accuracy_mean": round(float(scores_raw.mean()), 4),
            "cv_accuracy_std": round(float(scores_raw.std()), 4),
            "margin_over_baseline": round(float(scores_raw.mean()) - majority_baseline, 4),
            "fold_accuracies": [round(float(s), 4) for s in scores_raw],
            "note": (
                "Exact reproduction of scripts/analyze_shortcuts.py (cv=5 unshuffled, "
                "unscaled features). Fold 5 collapses because the dev file ends with an "
                "all-HALLUCINATED block; not a faithful estimate of metadata signal."
            ),
        },
        "features": FEATURE_NAMES,
        "method": (
            "Variant B (headline): StandardScaler + StratifiedKFold(5, shuffle, seed=42); "
            "LogisticRegression(max_iter=1000, seed=42); accuracy scoring."
        ),
        "labels_source": "data/v1.0/dev_public.jsonl (v1.1.1 relabeled, dev 513 valid / 606 hall)",
    }

    (REPO / "results/relabel_delta/shortcut_refit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
