"""Can a classifier that never looks anything up separate the four real-paper modes?

`cascade_db_diagnosis` scores detection rate 1.000 on `wrong_venue`,
`preprint_as_published`, `partial_author_list` and `arxiv_version_mismatch`.
That is either real detection or the cascade keying on a surface property of how
those entries were constructed.

This is the control. It uses ONLY features readable off the BibTeX entry with no
network access and no semantics: which fields are present, how long the strings
are, how many authors, whether a venue string mentions arXiv, the entry type, the
year. If a model on those features alone approaches the same detection rate, the
separation is in the data rather than in the checking, and a detector could score
well by reading the generator.

Run: python scripts/shortcut_control_four_modes.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FOUR = {"wrong_venue", "preprint_as_published", "partial_author_list", "arxiv_version_mismatch"}


def features(rec: dict) -> dict[str, float]:
    """Surface only. Nothing here requires knowing whether the paper exists."""
    f = {k.lower(): str(v) for k, v in (rec.get("fields") or {}).items()}
    author = f.get("author", "")
    title = f.get("title", "")
    venue = f.get("journal", "") + " " + f.get("booktitle", "")
    year = f.get("year", "")
    return {
        "n_fields": len(f),
        "has_doi": float("doi" in f),
        "has_eprint": float("eprint" in f),
        "has_url": float("url" in f),
        "has_pages": float("pages" in f),
        "has_volume": float("volume" in f),
        "has_publisher": float("publisher" in f),
        "has_journal": float("journal" in f),
        "has_booktitle": float("booktitle" in f),
        "title_chars": len(title),
        "title_words": len(title.split()),
        "title_has_colon": float(":" in title),
        "title_has_braces": float("{" in title),
        "author_chars": len(author),
        "n_authors": float(author.count(" and ") + 1 if author else 0),
        "author_has_others": float("others" in author.lower()),
        "venue_chars": len(venue.strip()),
        "venue_mentions_arxiv": float("arxiv" in venue.lower()),
        "venue_mentions_preprint": float("preprint" in venue.lower()),
        "is_article": float(rec.get("bibtex_type") == "article"),
        "is_inproceedings": float(rec.get("bibtex_type") == "inproceedings"),
        "is_misc": float(rec.get("bibtex_type") == "misc"),
        "year_val": float(year) if re.fullmatch(r"\d{4}", year) else 0.0,
        "doi_is_arxiv": float("10.48550" in f.get("doi", "")),
    }


def load(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(ROOT / f"data/v1.2/{split}.jsonl") as fh:
        rows = [json.loads(line) for line in fh]
    keep = [r for r in rows if r["label"] == "VALID" or r.get("hallucination_type") in FOUR]
    y = np.array([1 if r["label"] == "HALLUCINATED" else 0 for r in keep])
    feats = [features(r) for r in keep]
    names = sorted(feats[0])
    X = np.array([[fe[n] for n in names] for fe in feats])
    return X, y, names


def detection_rate(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == 1
    return float((pred[pos] == 1).mean())


def fpr(y: np.ndarray, pred: np.ndarray) -> float:
    neg = y == 0
    return float((pred[neg] == 1).mean())


def main() -> int:
    print("Surface-feature shortcut control")
    print("Positives: the four real-paper modes. Negatives: VALID entries.")
    print("No lookups, no semantics -- only what is readable off the entry.\n")

    for split in ("dev_public", "test_public"):
        path = ROOT / f"data/v1.2/{split}.jsonl"
        if not path.exists():
            print(f"{split}: not present, skipped")
            continue
        X, y, names = load(split)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        print(f"--- {split}: {int(y.sum())} four-mode positives, {int((y == 0).sum())} valid ---")
        for label, model in (
            ("majority-class baseline", DummyClassifier(strategy="most_frequent")),
            (
                "logistic regression",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            ),
            ("gradient boosting", GradientBoostingClassifier(random_state=0)),
        ):
            pred = cross_val_predict(model, X, y, cv=cv)
            print(f"  {label:24s} DR {detection_rate(y, pred):.3f}   FPR {fpr(y, pred):.3f}")

        gb = GradientBoostingClassifier(random_state=0).fit(X, y)
        top = sorted(zip(names, gb.feature_importances_, strict=True), key=lambda t: -t[1])[:6]
        print("  most informative surface features:")
        for n, imp in top:
            print(f"      {imp:.3f}  {n}")
        print()

    print("Read: cascade_db_diagnosis scores DR 1.000 on these modes. The closer a")
    print("surface-only model gets, the more of that number is separability in the")
    print("data rather than checking. A low number here does not prove the cascade")
    print("is sound, but a high one would show the benchmark can be scored without")
    print("verifying anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
