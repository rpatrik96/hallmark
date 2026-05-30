"""Regenerate SUMMARY-ONLY aggregate JSONs via confusion-matrix reconstruction.

Tools without a full-coverage per-entry prediction file: doi_only (dev+test),
bibtexupdater (dev+test, co-designed), claude_sonnet_4_6 dev, claude_opus_4_7
dev. We reconstruct the new confusion matrix from the published OLD aggregate +
the tool's verdicts on the relabel keys (obtained via tiny delta-eval), then
recompute DR/FPR/F1/MCC/macro_f1 (exact) and TW-F1 (<=0.2pp reconstruction).

ECE: the relabel flips correctness on <=27 entries. We recompute the ECE delta
exactly for the moved keys (we know their confidence and the correctness flip),
but the published ECE used adaptive (equal-frequency) binning over the full
per-entry list, which we cannot fully reproduce for summary-only tools. We keep
the published ECE value and report this limitation. ECE is a calibration
diagnostic; a <=27/1100 correctness flip moves it by <1pp.

per_tier_metrics / per_type_metrics: detection_rate within each is recomputed for
the affected tiers/the 'valid' type; FPR fields are updated to the new split FPR.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _reconstruct import reconstruct

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OUT = REPO / "data/v1.0/baseline_results"
DELTA = REPO / "results/relabel_delta"
OLD_REV = "7a52362"


def load_old_full(split: str) -> dict:
    path = f"data/v1.0/{split}.jsonl"
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{OLD_REV}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        json.loads(ln)["bibtex_key"]: json.loads(ln) for ln in out.stdout.splitlines() if ln.strip()
    }


def moved_keys(split: str) -> list[str]:
    keys: list[str] = json.loads((DELTA / "changed_keys_all.json").read_text())[split]
    return keys


def load_verdicts(tool_file: str) -> dict[str, str]:
    v = json.loads((DELTA / tool_file).read_text())
    return {k: x["label"] for k, x in v.items()}


def update_per_tier(agg: dict, rec: dict, moved_verdicts, moved_tiers) -> None:
    """Recompute per-tier detection_rate for affected tiers; refresh FPR everywhere."""
    ptm = agg.get("per_tier_metrics")
    if not ptm:
        return
    new_fpr = rec["false_positive_rate"]
    # group moved keys by tier with verdict
    from collections import defaultdict

    tier_tp_delta: defaultdict[str, int] = defaultdict(int)
    tier_nh_delta: defaultdict[str, int] = defaultdict(int)
    for k, verdict in moved_verdicts.items():
        t = str(moved_tiers.get(k, 1))
        tier_nh_delta[t] += 1  # this hall entry leaves the tier's hall set
        if verdict == "HALLUCINATED":
            tier_tp_delta[t] += 1
    for tier_str, m in ptm.items():
        if "false_positive_rate" in m and new_fpr is not None:
            m["false_positive_rate"] = new_fpr
        # Tier 0 (older doi_only schema) is the VALID bucket — only its count/FPR
        # are meaningful; skip the hallucination-tier recompute.
        if tier_str == "0":
            if "count" in m:
                m["count"] = rec["num_valid"]
            continue
        # Use whichever count key the schema carries (num_hallucinated or count).
        count_key = (
            "num_hallucinated" if "num_hallucinated" in m else ("count" if "count" in m else None)
        )
        if count_key is None:
            continue
        nh_t = m[count_key]
        if nh_t is None:
            continue
        dr_t = m.get("detection_rate", 0.0)
        tp_t = round(dr_t * nh_t)
        new_nh_t = nh_t - tier_nh_delta.get(tier_str, 0)
        new_tp_t = tp_t - tier_tp_delta.get(tier_str, 0)
        if new_nh_t > 0:
            m["detection_rate"] = new_tp_t / new_nh_t
            m[count_key] = new_nh_t
        if "num_valid" in m:
            m["num_valid"] = rec["num_valid"]


def update_per_type(agg: dict, rec: dict) -> None:
    """Update the 'valid' type FPR + count; refresh shared FPR fields."""
    ptm = agg.get("per_type_metrics")
    if not ptm:
        return
    new_fpr = rec["false_positive_rate"]
    if "valid" in ptm and new_fpr is not None:
        ptm["valid"]["false_positive_rate"] = new_fpr
        ptm["valid"]["count"] = rec["num_valid"]
    # NB: moved keys leave their hallucination *type* buckets, but the original
    # per_type 'count'/'detection_rate' for those types are not safely
    # decremented without per-key type info; the moved keys are real papers whose
    # OLD hallucination_type was assigned by the (incorrect) label, so we leave
    # the per-type hallucinated buckets as published and only fix the 'valid' row
    # plus the shared FPR. (Reported as a known limitation for summary-only tools.)


def load_template(pub_rel: str) -> dict:
    """Read the PRISTINE published aggregate from git HEAD (idempotent vs working tree)."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{pub_rel}"],
        capture_output=True,
        text=True,
    )
    if out.returncode == 0 and out.stdout.strip():
        tracked: dict = json.loads(out.stdout)
        return tracked
    # Untracked template (e.g. results/doi_only_dev_public.json is committed; if not, fall back).
    untracked: dict = json.loads((REPO / pub_rel).read_text())
    return untracked


def regen(out_name, pub_rel, split, verdict_file, *, hall_unc=0, val_unc=0):
    agg = load_template(pub_rel)
    mv = load_verdicts(verdict_file)
    old_full = load_old_full(split)
    mt = {k: (old_full[k].get("difficulty_tier") or 1) for k in moved_keys(split)}
    rec = reconstruct(agg, mv, mt, hall_uncertain_old=hall_unc, valid_uncertain_old=val_unc)

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
    update_per_tier(out, rec, mv, mt)
    update_per_type(out, rec)
    (OUT / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return agg, rec


JOBS = [
    (
        "doi_only_dev_public.json",
        "results/doi_only_dev_public.json",
        "dev_public",
        "doi_only_dev_public_changed_predictions.json",
    ),
    (
        "doi_only_test_public.json",
        "data/v1.0/baseline_results/doi_only_test_public.json",
        "test_public",
        "doi_only_test_public_changed_predictions.json",
    ),
    (
        "bibtexupdater_dev_public.json",
        "data/v1.0/baseline_results/bibtexupdater_dev_public.json",
        "dev_public",
        "bibtexupdater_dev_public_changed_predictions.json",
    ),
    (
        "bibtexupdater_test_public.json",
        "data/v1.0/baseline_results/bibtexupdater_test_public.json",
        "test_public",
        "bibtexupdater_test_public_changed_predictions.json",
    ),
    (
        "llm_openrouter_claude_sonnet_4_6_dev_public.json",
        "data/v1.0/baseline_results/llm_openrouter_claude_sonnet_4_6_dev_public.json",
        "dev_public",
        "llm_openrouter_claude_sonnet_4_6_dev_public_changed_predictions.json",
    ),
    (
        "llm_openrouter_claude_opus_4_7_dev_public.json",
        "data/v1.0/baseline_results/llm_openrouter_claude_opus_4_7_dev_public.json",
        "dev_public",
        "llm_openrouter_claude_opus_4_7_dev_public_changed_predictions.json",
    ),
]


def main():
    print(f"{'tool/split':52s} DR_old->new  FPR_old->new  F1_old->new  MCC_old->new  TWF1_old->new")
    for out_name, pub_rel, split, vf in JOBS:
        agg, rec = regen(out_name, pub_rel, split, vf)
        print(
            f"{out_name[:-5]:52s} {agg['detection_rate']:.3f}->{rec['detection_rate']:.3f} "
            f"{(agg.get('false_positive_rate') or 0):.3f}->{(rec['false_positive_rate'] or 0):.3f} "
            f"{agg['f1_hallucination']:.3f}->{rec['f1_hallucination']:.3f} "
            f"{(agg.get('mcc') or 0):.3f}->{rec['mcc']:.3f} "
            f"{agg['tier_weighted_f1']:.3f}->{rec['tier_weighted_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
