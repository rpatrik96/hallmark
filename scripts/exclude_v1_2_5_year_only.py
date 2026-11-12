#!/usr/bin/env python3
"""Exclude six test_public entries whose only defect is a year no version carries.

Each cites a real paper with the right title, authors and venue, but a year
(2021) that neither any arXiv version nor the proceedings carries. They were
VALID only because hand relabels (``mislabel-audit-2026-05-30-leak-followup``)
named the correct year in their notes but left the cited year as it was.

The taxonomy has no type for a year-only error on a real work:
``arxiv_version_mismatch`` requires a wrong venue as well. The Walters-Wilder
supplement excludes the same case (34 entries, paper App. on Walters-Wilder),
so these six are removed from the split rather than relabeled. The removed
records are kept verbatim in the exclusion log.

The script also refreshes ``metadata.json`` for v1.2.5: split statistics for
dev_public, test_public and stress_test are recomputed from the split files, so
they include the label corrections from ``scripts/fix_v1_2_5_labels.py``.

Usage:
    uv run python scripts/exclude_v1_2_5_year_only.py --check   # report, write nothing
    uv run python scripts/exclude_v1_2_5_year_only.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data/v1.2"
SPLIT = DATA / "test_public.jsonl"
BLIND = DATA / "test_public_blind.jsonl"
METADATA = DATA / "metadata.json"
LOG = REPO / "results/relabel_recheck/v1_2_5_exclusion_log.json"
VERSION = "1.2.5"
DATE = "2026-11-11"
GUARD_BY = "mislabel-audit-2026-05-30-leak-followup"

# key -> (paper, cited year, evidence)
EXCLUDE: dict[str, tuple[str, str, str]] = {
    "c437b8d2fcd4": (
        "Denoising Diffusion Probabilistic Models",
        "2021",
        "arXiv 2006.11239 v1 2020-06, v2 2020-12; NeurIPS 2020 proceedings",
    ),
    "bb59c4bd80e7": (
        "A Simple Framework for Contrastive Learning of Visual Representations",
        "2021",
        "arXiv 2002.05709 v1-v3 2020; PMLR v119 (ICML 2020) chen20j",
    ),
    "bbd1bcc55e09": (
        "Geom-GCN: Geometric Graph Convolutional Networks",
        "2021",
        "arXiv 2002.05287 v1-v2 2020-02, 'Published as a conference paper at ICLR 2020'",
    ),
    "f34f26c83153": (
        "NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search",
        "2021",
        "arXiv 2001.00326 v1-v2 2020-01, 'Published at ICLR 2020'; OpenReview HJxyZkBKDr",
    ),
    "b97858a4638f": (
        "Dark Experience for General Continual Learning: a Strong, Simple Baseline",
        "2021",
        "arXiv 2004.07211 v1-v2 2020; NeurIPS 2020 proceedings",
    ),
    "b6a26b30b14a": (
        "Time Series Deconfounder",
        "2021",
        "arXiv 1902.00450 v1 2019-02 to v4 2020-09, journal-ref ICML 2020; PMLR v119 bica20a",
    ),
}

NOTE = (
    "v1.2.5 contains two ground-truth corrections from the OpenCite errata (2026-11-09), each "
    "re-checked by hand on 2026-11-11 against Crossref, the arXiv API (all versions), the NeurIPS "
    "proceedings, PMLR, OpenReview and DBLP. (1) Labels: 5 VALID entries are HALLUCINATED "
    "(dev_public 3, test_public 2: a DOI of a different paper, a duplicated author, a skipped "
    "author before 'et al.', a venue the paper never appeared at, and a preprint's title paired with a later paper's venue), and 2 stress_test "
    "partial_author_list entries whose perturbation removed no author are VALID. (2) Exclusion: 6 "
    "test_public entries whose only defect is a year no version of the paper carries are removed, "
    "since the taxonomy has no type for it (the Walters-Wilder supplement excludes the same case). "
    "Net effect: dev_public 501 valid / 618 hallucinated (was 504/615), test_public 299/526 over "
    "825 entries (was 307/524 over 831), stress_test 3/119 (was 1/121). Every reported metric on "
    "these splits must be rescored against v1.2.5. "
)


def split_stats(path: Path) -> dict:
    rs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    h = [r for r in rs if r["label"] == "HALLUCINATED"]
    count = collections.Counter
    return {
        "total": len(rs),
        "valid": len(rs) - len(h),
        "hallucinated": len(h),
        "tier_distribution": dict(sorted(count(str(r["difficulty_tier"]) for r in h).items())),
        "type_distribution": dict(sorted(count(r["hallucination_type"] for r in h).items())),
        "generation_methods": dict(sorted(count(r.get("generation_method") for r in rs).items())),
        "count": len(rs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check", action="store_true", help="report the planned changes, write nothing"
    )
    args = ap.parse_args()

    lines = [line for line in SPLIT.read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    removed = [r for r in records if r["bibtex_key"] in EXCLUDE]
    if {r["bibtex_key"] for r in removed} != EXCLUDE.keys():
        raise SystemExit(
            f"keys not in test_public: {sorted(EXCLUDE.keys() - {r['bibtex_key'] for r in removed})}"
        )
    for r in removed:
        if r["label"] != "VALID" or r.get("relabeled_by") != GUARD_BY:
            raise SystemExit(f"{r['bibtex_key']}: expected VALID relabeled_by={GUARD_BY}; refusing")
        if r["fields"].get("year") != EXCLUDE[r["bibtex_key"]][1]:
            raise SystemExit(f"{r['bibtex_key']}: cited year changed; refusing")

    blind_lines = [line for line in BLIND.read_text().splitlines() if line.strip()]
    kept_blind = [line for line in blind_lines if json.loads(line)["bibtex_key"] not in EXCLUDE]
    if len(blind_lines) - len(kept_blind) != len(EXCLUDE):
        raise SystemExit("blind file does not carry exactly the excluded keys")
    kept = [line for line, r in zip(lines, records, strict=True) if r["bibtex_key"] not in EXCLUDE]
    print(
        f"test_public: {len(lines)} -> {len(kept)} entries; blind {len(blind_lines)} -> {len(kept_blind)}"
    )
    if args.check:
        return

    SPLIT.write_text("\n".join(kept) + "\n")
    BLIND.write_text("\n".join(kept_blind) + "\n")
    LOG.write_text(
        json.dumps(
            {
                "reason": "year no version of the paper carries; no matching taxonomy type",
                "excluded_by": "opencite-errata-recheck-2026-11-11",
                "entries": [
                    {
                        "bibtex_key": r["bibtex_key"],
                        "evidence": EXCLUDE[r["bibtex_key"]][2],
                        "record": r,
                    }
                    for r in removed
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    meta = json.loads(METADATA.read_text())
    meta["version"] = VERSION
    meta["build_date"] = DATE
    meta["release_date"] = DATE
    meta["release_notes"] = NOTE + meta["release_notes"]
    meta["total_entries"] -= len(EXCLUDE)
    for split in ("dev_public", "test_public", "stress_test"):
        meta["splits"][split].update(split_stats(DATA / f"{split}.jsonl"))
    meta["opencite_errata_fix"] = {
        "description": (
            "v1.2.5: 7 labels corrected (5 VALID -> HALLUCINATED in dev_public/test_public, 2 "
            "stress_test no-op perturbations -> VALID) and 6 test_public year-only entries excluded."
        ),
        "scripts": ["scripts/fix_v1_2_5_labels.py", "scripts/exclude_v1_2_5_year_only.py"],
        "fix_log": "results/relabel_recheck/v1_2_5_label_log.json",
        "exclusion_log": "results/relabel_recheck/v1_2_5_exclusion_log.json",
        "entries_changed": 7,
        "entries_excluded": len(EXCLUDE),
        "by_split": {"dev_public": 3, "test_public": 2 + len(EXCLUDE), "stress_test": 2},
    }
    METADATA.write_text(json.dumps(meta, indent=4, ensure_ascii=False) + "\n")
    print(f"written; log at {LOG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
