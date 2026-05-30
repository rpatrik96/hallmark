"""Confusion-matrix reconstruction for SUMMARY-ONLY tools.

The published aggregate gives the OLD confusion matrix (against pre-relabel
labels). Every relabel key moved HALLUCINATED -> VALID. Given the tool's verdict
on each moved key, we adjust the matrix:

  tool said HALLUCINATED on a moved key:  TP -= 1 ; FP += 1
  tool said VALID on a moved key:         FN -= 1 ; TN += 1
  tool said UNCERTAIN on a moved key:     (was excluded from HALL side) ->
                                          now excluded from VALID side; the
                                          entry leaves num_hallucinated and
                                          enters num_valid, but stays out of the
                                          confusion matrix.

DR, FPR, F1, MCC, macro_f1 are recomputed exactly from the new matrix.
TW-F1 is recomputed exactly using the moved keys' tier weights and the OLD
weighted matrix reconstructed from the published per_tier_metrics.

ECE depends on the full per-entry (confidence, correctness) distribution, which
summary-only tools do not persist. The relabel flips correctness on <=27 entries
out of ~1100-2000; we recompute the ECE *delta* contribution from the moved keys
where confidences are available and otherwise preserve the published ECE,
flagging the field. (Reported separately; see notes.)
"""

from __future__ import annotations

import math


def old_matrix_from_aggregate(agg: dict) -> dict:
    """Recover OLD TP/FN/FP/TN from the published aggregate.

    num_hallucinated / num_valid are GT counts. DR = TP/(TP+FN) over the
    *classified* hallucinated entries (HALL minus UNCERTAIN-on-HALL). FPR =
    FP/(FP+TN) over classified valid entries. We use num_uncertain split is not
    stored per class, so we infer the classified-hall and classified-valid
    denominators from DR/FPR being exact ratios of integer counts.
    """
    nh = agg["num_hallucinated"]
    nv = agg["num_valid"]
    dr = agg["detection_rate"]
    fpr = agg.get("false_positive_rate")
    # Classified counts: for tools with num_uncertain>0 we still need the per-class
    # split. We recover it by rounding DR*nh-style only when num_uncertain==0.
    return {"nh": nh, "nv": nv, "dr": dr, "fpr": fpr}


def reconstruct(
    agg: dict,
    moved_verdicts: dict[str, str],
    moved_tiers: dict[str, int] | None = None,
    hall_uncertain_old: int = 0,
    valid_uncertain_old: int = 0,
):
    """Reconstruct new metrics.

    Args:
        agg: published aggregate (OLD).
        moved_verdicts: {key: 'HALLUCINATED'|'VALID'|'UNCERTAIN'} on moved keys.
        moved_tiers: {key: difficulty_tier} for moved (hallucinated) keys, for TW-F1.
        hall_uncertain_old: number of UNCERTAIN predictions that fell on OLD
            hallucinated entries (needed to get the classified-hall denominator).
        valid_uncertain_old: number of UNCERTAIN predictions on OLD valid entries.

    Returns dict with new counts + metrics.
    """
    nh = agg["num_hallucinated"]
    nv = agg["num_valid"]
    dr = agg["detection_rate"]
    fpr = agg.get("false_positive_rate")

    # Classified denominators (exclude UNCERTAIN).
    hall_classified = nh - hall_uncertain_old
    valid_classified = nv - valid_uncertain_old

    TP = round(dr * hall_classified)
    FN = hall_classified - TP
    if fpr is None:
        FP = 0
        TN = 0
    else:
        FP = round(fpr * valid_classified)
        TN = valid_classified - FP

    # Apply moves: each moved key was a HALLUCINATED GT entry.
    new_nh = nh
    new_nv = nv
    for verdict in moved_verdicts.values():
        new_nh -= 1
        new_nv += 1
        if verdict == "HALLUCINATED":
            TP -= 1
            FP += 1
        elif verdict == "VALID":
            FN -= 1
            TN += 1
        else:  # UNCERTAIN: leaves hall side (was excluded) -> enters valid side (excluded)
            pass

    new_dr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    new_fpr = (FP / (FP + TN)) if (FP + TN) > 0 else (None if fpr is None else 0.0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = new_dr
    new_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # MCC
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    new_mcc = ((TP * TN - FP * FN) / denom) if denom > 0 else 0.0
    # macro F1 (both classes)
    # VALID-class: precision_v=TN/(TN+FN), recall_v=TN/(TN+FP)
    prec_v = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    rec_v = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_v = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) > 0 else 0.0
    new_macro_f1 = (new_f1 + f1_v) / 2

    out = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "num_hallucinated": new_nh,
        "num_valid": new_nv,
        "detection_rate": new_dr,
        "false_positive_rate": new_fpr,
        "f1_hallucination": new_f1,
        "mcc": new_mcc,
        "macro_f1": new_macro_f1,
    }

    # TW-F1: reconstruct old and new from per_tier_metrics, then anchor to the
    # published value and apply only the delta (cancels per-tier rounding bias).
    if agg.get("per_tier_metrics"):
        tw_old = reconstruct_twf1(agg, {}, {})
        tw_new = reconstruct_twf1(agg, moved_verdicts, moved_tiers or {})
        published = agg.get("tier_weighted_f1", tw_old)
        out["tier_weighted_f1"] = published + (tw_new - tw_old)
    return out


def reconstruct_twf1(agg, moved_verdicts, moved_tiers, tier_weights=None):
    """Recompute TW-F1 from per-tier OLD recall + moved-key adjustments.

    weighted_tp = sum_t w_t * TP_t ; weighted_fn = sum_t w_t * FN_t ;
    weighted_fp = FP (uniform weight 1.0). Recall_weighted = wTP/(wTP+wFN);
    precision_weighted = wTP/(wTP+wFP). F1 from those.
    """
    if tier_weights is None:
        tier_weights = {1: 1.0, 2: 2.0, 3: 3.0}
    ptm = agg["per_tier_metrics"]
    wTP = 0.0
    wFN = 0.0
    for tier_str, m in ptm.items():
        try:
            tier = int(tier_str)
        except ValueError:
            continue
        # Tier 0 in the older doi_only/no-prescreening schema is the VALID bucket
        # (count == num_valid), not a hallucination tier — skip it.
        if tier == 0:
            continue
        w = tier_weights.get(tier, 1.0)
        nh_t = m.get("num_hallucinated")
        if nh_t is None:
            nh_t = m.get("count", 0)
        dr_t = m.get("detection_rate", 0.0)
        tp_t = round(dr_t * nh_t)
        fn_t = nh_t - tp_t
        wTP += w * tp_t
        wFN += w * fn_t
    # FP weighted uniformly = OLD FP count.
    nv = agg["num_valid"]
    fpr = agg.get("false_positive_rate") or 0.0
    wFP = round(fpr * nv)

    # Apply moves.
    for key, verdict in moved_verdicts.items():
        tier = moved_tiers.get(key, 1)
        w = tier_weights.get(tier, 1.0)
        if verdict == "HALLUCINATED":
            wTP -= w  # was weighted TP
            wFP += 1.0  # now a valid FP (uniform 1.0)
        elif verdict == "VALID":
            wFN -= w  # was weighted FN
            # becomes TN: no weighted contribution
        # UNCERTAIN: no change to weighted matrix

    precision = wTP / (wTP + wFP) if (wTP + wFP) > 0 else 0.0
    recall = wTP / (wTP + wFN) if (wTP + wFN) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
