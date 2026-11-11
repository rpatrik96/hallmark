#!/usr/bin/env python3
"""Apply the v1.2.5 label corrections that the OpenCite errata (2026-11-09) raised.

Every row below was re-checked by hand on 2026-11-11 against Crossref, the arXiv
API (all versions), the NeurIPS proceedings index, PMLR, OpenReview and DBLP.

Five dev/test entries go VALID -> HALLUCINATED. They were never covered by
``recheck_relabel_flips.py`` (v1.2.4): four came from hand relabels
(``mislabel-audit-*``), and the recheck passed the other two on checks that
cannot see their defect (PaLM-E: membership-only author check; Fixing Data
Augmentation: left at CHECK_BY_HAND).

Two stress_test entries go HALLUCINATED -> VALID. Their perturbation was a no-op
("3 authors reduced to 3"), the same defect as the three dev entries the
2026-05-29 audit relabeled (``a4d318041956`` and others); they get the same
record shape.

Six test_public entries with a year that no version of the paper carries are
NOT handled here: they are excluded from scoring in a separate step.

Only labels change; the cited fields are untouched, so the blind files are too.

Usage:
    uv run python scripts/fix_v1_2_5_labels.py --check   # report, write nothing
    uv run python scripts/fix_v1_2_5_labels.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hallmark.dataset.schema import (  # noqa: E402
    EXPECTED_SUBTESTS,
    HALLUCINATION_TIER_MAP,
    HallucinationType,
)

RELABELED_BY = "opencite-errata-recheck-2026-11-11"
SPLIT_FILES = {
    "dev_public": REPO / "data/v1.2/dev_public.jsonl",
    "test_public": REPO / "data/v1.2/test_public.jsonl",
    "stress_test": REPO / "data/v1.2/stress_test.jsonl",
}
LOG = REPO / "results/relabel_recheck/v1_2_5_label_log.json"

WRONG_DOI = "hybrid_fabrication"
AUTHORS = "swapped_authors"
VENUE = "wrong_venue"
PREPRINT = "preprint_as_published"
PARTIAL = "partial_author_list"
VERSION_MIX = "arxiv_version_mismatch"

DEFECT = {
    WRONG_DOI: "identifier belongs to a different paper",
    AUTHORS: "author list matches no version of the paper",
    VENUE: "venue is not where the paper was published",
    PREPRINT: "venue is claimed for a paper that exists only as a preprint",
    PARTIAL: "author list leaves out a real author",
    VERSION_MIX: "metadata mixes a preprint with a different published version",
}

# split, key, expected current relabeled_by (guard), new type, explanation
TO_HALLUCINATED: list[tuple[str, str, str, str, str]] = [
    (
        "dev_public",
        "ed071a6dfa34",
        "mislabel-audit-2026-05-29",
        WRONG_DOI,
        "Improving Robustness using Generated Data (Gowal et al., NeurIPS 2021); cited DOI "
        "10.48550/arXiv.2104.09425 is Sehwag et al., 'Robust Learning Meets Generative Models' "
        "(the paper is arXiv:2110.09468).",
    ),
    (
        "dev_public",
        "a0478afc6fb9",
        "mislabel-audit-2026-05-30-cnv-followup",
        VENUE,
        "Classifier-Free Diffusion Guidance cited as NeurIPS 2022; it appeared at the NeurIPS 2021 "
        "Workshop on Deep Generative Models and Downstream Applications (OpenReview qw8AKxfYbI) and "
        "on arXiv (2207.12598). It is not in the NeurIPS 2022 proceedings or the 2022 Score-Based "
        "Methods workshop's accepted list.",
    ),
    (
        "dev_public",
        "fa77166308b0",
        "mislabel-audit-2026-05-30-postfix-followup",
        AUTHORS,
        "Imagen (NeurIPS 2022); 15 cited names for a paper with 14 authors on arXiv and 13 in the "
        "NeurIPS proceedings: Raphael Gontijo Lopes is listed twice ('Raphael Gontijo-Lopes' and "
        "'Raphael Lopes'). The 2026-05-29 adversarial review had already kept it HALLUCINATED.",
    ),
    (
        "test_public",
        "d632f7211d63",
        "systematic-relabel-2026-05-30",
        VERSION_MIX,
        "Fixing Data Augmentation to Improve Adversarial Robustness cited as NeurIPS 2021 with "
        "arXiv 2103.01946. That preprint's own comment says it was accepted to NeurIPS 2021 as two "
        "separate, updated papers with their own arXiv records; the one with these six authors in "
        "this order is 'Data Augmentation Can Improve Robustness' (arXiv 2111.05328). The citation "
        "pairs the earlier preprint's title and ID with the later paper's venue.",
    ),
    (
        "test_public",
        "d27ac16ca85a",
        "systematic-relabel-2026-05-30",
        PARTIAL,
        "PaLM-E (ICML 2023); 'Driess ... Ichter, Tompson, et al.' skips the 7th author, Ayzaan "
        "Wahid, who is 7th on every version (PMLR, OpenReview, arXiv v1, DBLP), so the leading list "
        "matches no version. Every cited name is a real author; one is left out before 'et al.', "
        "so the type is partial_author_list (its usual form drops authors without 'et al.'). The "
        "v1.2.4 recheck passed it because it checked membership, not order.",
    ),
]

# split, key, relabel_reason
TO_VALID: list[tuple[str, str, str]] = [
    (
        "stress_test",
        "bcc2d959b101",
        "Real paper (Agarwal, Yurochkin, Sun, 'On sensitivity of meta-learning to support data', "
        "NeurIPS 2021); no-op author injection ('3 authors reduced to 3' — none removed).",
    ),
    (
        "stress_test",
        "acbaa1cad903",
        "Real paper (Agarwal, Vatsa, Singh, AAAI 2021, Crossref 10.1609/AAAI.V35I18.17869); no-op "
        "author injection ('3 authors reduced to 3' — none removed).",
    ),
]

REASON = (
    "Relabeled after the OpenCite errata (2026-11-09), re-checked by hand on 2026-11-11 against "
    "Crossref/arXiv/NeurIPS proceedings/PMLR/OpenReview/DBLP; the citation's {defect}."
)


def _doi_resolves(r: dict) -> bool | None:
    # Every DOI in these rows resolves (some to another paper); no DOI -> not applicable.
    return True if r.get("fields", {}).get("doi") else None


def to_hallucinated(r: dict, htype: str, explanation: str) -> dict:
    t = HallucinationType(htype)
    subtests = dict(EXPECTED_SUBTESTS[t])
    subtests["doi_resolves"] = _doi_resolves(r)
    out = {k: v for k, v in r.items() if k not in ("matched_source", "matched_id", "match_scores")}
    out.update(
        label="HALLUCINATED",
        hallucination_type=htype,
        difficulty_tier=int(HALLUCINATION_TIER_MAP[t]),
        subtests=subtests,
        explanation=explanation,
        relabeled_from="VALID",
        relabel_reason=REASON.format(defect=DEFECT[htype]),
        relabeled_by=RELABELED_BY,
    )
    return out


def to_valid(r: dict, reason: str) -> dict:
    # Same shape as the 2026-05-29 no-op relabels (e.g. dev_public a4d318041956).
    out = {k: v for k, v in r.items() if k not in ("hallucination_type", "difficulty_tier")}
    out.update(
        label="VALID",
        subtests={
            "doi_resolves": _doi_resolves(r),
            "title_exists": True,
            "authors_match": True,
            "fields_complete": True,
            "cross_db_agreement": True,
            "venue_correct": True,
        },
        relabeled_from="HALLUCINATED",
        relabel_reason=reason,
        relabeled_by=RELABELED_BY,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check", action="store_true", help="report the planned changes, write nothing"
    )
    args = ap.parse_args()

    logs: dict[str, list[dict]] = {}
    for split, path in SPLIT_FILES.items():
        to_h = {k: (by, t, e) for s, k, by, t, e in TO_HALLUCINATED if s == split}
        to_v = {k: reason for s, k, reason in TO_VALID if s == split}
        todo = to_h.keys() | to_v.keys()
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        found = {r["bibtex_key"] for r in records} & todo
        if found != todo:
            raise SystemExit(f"[{split}] keys not in the split: {sorted(todo - found)}")
        log = []
        for i, r in enumerate(records):
            key = r["bibtex_key"]
            if key in to_h:
                by, htype, expl = to_h[key]
                if r["label"] != "VALID" or r.get("relabeled_by") != by:
                    raise SystemExit(
                        f"[{split}] {key}: expected VALID relabeled_by={by}, found "
                        f"{r['label']} relabeled_by={r.get('relabeled_by')}; refusing"
                    )
                records[i] = to_hallucinated(r, htype, expl)
            elif key in to_v:
                if r["label"] != "HALLUCINATED" or r.get("hallucination_type") != (
                    "partial_author_list"
                ):
                    raise SystemExit(f"[{split}] {key}: not a partial_author_list entry; refusing")
                records[i] = to_valid(r, to_v[key])
            else:
                continue
            log.append(
                {
                    "bibtex_key": key,
                    "before": {
                        k: r.get(k) for k in ("label", "hallucination_type", "relabeled_by")
                    },
                    "after": {
                        k: records[i].get(k)
                        for k in ("label", "hallucination_type", "difficulty_tier")
                    },
                }
            )
        n_h = sum(1 for e in log if e["after"]["label"] == "HALLUCINATED")
        print(f"[{split}] {n_h} VALID -> HALLUCINATED, {len(log) - n_h} HALLUCINATED -> VALID")
        logs[split] = log
        if not args.check:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    if args.check:
        return
    LOG.write_text(json.dumps(logs, indent=2, ensure_ascii=False) + "\n")
    print(f"written; log at {LOG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
