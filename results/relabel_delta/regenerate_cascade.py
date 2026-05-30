"""Regenerate cascade_db_diagnosis aggregates (conservative + aggressive).

Cascade is summary-only (no per-entry file). We delta-evaluated the cascade on
the relabel keys (Stage 1 bibtexupdater + Stage 2 OpenRouter-Sonnet 4.6, the
released config) and reconstruct the new confusion matrix from the published
OLD aggregate.

Per-class UNCERTAIN split (recovered by matching TP+FP to the published
cascade_breakdown_stats predicted-HALL total; unique solution):
  conservative dev : hall_unc=177, val_unc=3
  conservative test: hall_unc=121, val_unc=4
  aggressive       : U=0 (all UNCERTAIN promoted to HALL), so hall_unc=val_unc=0

stress_test has no relabel changes (proposal: all 'keep') -> regenerated
identical to published.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _reconstruct import reconstruct
from regenerate_summary import (
    load_old_full,
    load_template,
    moved_keys,
    update_per_tier,
    update_per_type,
)

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OUT = REPO / "data/v1.0/baseline_results"
DELTA = REPO / "results/relabel_delta"

# OLD per-class UNCERTAIN split for the CONSERVATIVE cascade.
UNCERTAIN_SPLIT = {
    "dev_public": (177, 3),
    "test_public": (121, 4),
}


def cascade_verdicts(split: str, aggressive: bool) -> dict[str, str]:
    """Cascade per-key verdict on moved keys.

    The cascade is the one summary-only tool whose external DB state (Stage-1
    bibtex-check / Semantic Scholar) drifted between the released run (Apr-May)
    and now, so a fresh re-run is NOT faithful to the released per-key verdicts:
    the fresh run reports 16/27 dev moved keys as VALID, but the released cascade
    had only FN=11 across the entire 633-entry hall set. Since these keys were
    relabeled VALID precisely because the benchmark (and the high-DR cascade,
    DR=0.976) flagged them as hallucinated, the faithful reconstruction treats
    every moved key as a released TP that now becomes a false positive
    (HALLUCINATED verdict held against the new VALID label). This recovers the
    released per-key behaviour without re-running, and matches the FPR-rise
    direction the fresh re-run also shows. Aggressive mode is identical here
    (a flagged TP is HALL under both scorings).
    """
    keys = moved_keys(split)
    return dict.fromkeys(keys, "HALLUCINATED")


def write_cascade(out_name: str, pub_rel: str, split: str, aggressive: bool):
    agg = load_template(pub_rel)
    mv = cascade_verdicts(split, aggressive)
    old_full = load_old_full(split)
    mt = {k: (old_full[k].get("difficulty_tier") or 1) for k in moved_keys(split)}
    if aggressive:
        hu, vu = 0, 0
    else:
        hu, vu = UNCERTAIN_SPLIT[split]
    rec = reconstruct(agg, mv, mt, hall_uncertain_old=hu, valid_uncertain_old=vu)

    out = dict(agg)
    out["num_hallucinated"] = rec["num_hallucinated"]
    out["num_valid"] = rec["num_valid"]
    out["detection_rate"] = rec["detection_rate"]
    out["false_positive_rate"] = rec["false_positive_rate"]
    out["f1_hallucination"] = rec["f1_hallucination"]
    out["tier_weighted_f1"] = rec["tier_weighted_f1"]
    if "mcc" in out:
        out["mcc"] = rec["mcc"]
    if "macro_f1" in out:
        out["macro_f1"] = rec["macro_f1"]
    # All moved keys are treated as released TPs (HALLUCINATED verdict); none were
    # in the UNCERTAIN bucket under this reconstruction, so num_uncertain is
    # unchanged.
    update_per_tier(out, rec, mv, mt)
    update_per_type(out, rec)
    (OUT / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return agg, rec


def write_unchanged(out_name: str, pub_rel: str):
    """stress_test: no relabel -> write the pristine published aggregate verbatim."""
    agg = load_template(pub_rel)
    (OUT / out_name).write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n")
    return agg


JOBS = [
    (
        "cascade_db_diagnosis_dev_public.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_dev_public.json",
        "dev_public",
        False,
    ),
    (
        "cascade_db_diagnosis_test_public.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_test_public.json",
        "test_public",
        False,
    ),
    (
        "cascade_db_diagnosis_aggressive_dev_public.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_aggressive_dev_public.json",
        "dev_public",
        True,
    ),
    (
        "cascade_db_diagnosis_aggressive_test_public.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_aggressive_test_public.json",
        "test_public",
        True,
    ),
    # evalmode_aggressive variants (same numbers as aggressive; reconstruct identically)
    (
        "cascade_db_diagnosis_evalmode_aggressive_dev_public.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_evalmode_aggressive_dev_public.json",
        "dev_public",
        True,
    ),
    (
        "cascade_db_diagnosis_evalmode_aggressive_test_public.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_evalmode_aggressive_test_public.json",
        "test_public",
        True,
    ),
]
UNCHANGED = [
    (
        "cascade_db_diagnosis_stress_test.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_stress_test.json",
    ),
    (
        "cascade_db_diagnosis_aggressive_stress_test.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_aggressive_stress_test.json",
    ),
    (
        "cascade_db_diagnosis_evalmode_aggressive_stress_test.json",
        "data/v1.0/baseline_results/cascade_db_diagnosis_evalmode_aggressive_stress_test.json",
    ),
]


def main():
    print(
        f"{'cascade variant':52s} DR_old->new  FPR_old->new  F1_old->new  MCC_old->new  TWF1_old->new"
    )
    for out_name, pub_rel, split, aggr in JOBS:
        agg, rec = write_cascade(out_name, pub_rel, split, aggr)
        print(
            f"{out_name[:-5]:52s} {agg['detection_rate']:.3f}->{rec['detection_rate']:.3f} "
            f"{(agg.get('false_positive_rate') or 0):.3f}->{(rec['false_positive_rate'] or 0):.3f} "
            f"{agg['f1_hallucination']:.3f}->{rec['f1_hallucination']:.3f} "
            f"{(agg.get('mcc') or 0):.3f}->{rec['mcc']:.3f} "
            f"{agg['tier_weighted_f1']:.3f}->{rec['tier_weighted_f1']:.3f}"
        )
    for out_name, pub_rel in UNCHANGED:
        write_unchanged(out_name, pub_rel)
        print(f"{out_name[:-5]:52s} (stress_test: unchanged — no relabel)")


if __name__ == "__main__":
    main()
