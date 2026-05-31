#!/usr/bin/env python3
"""A5 inter-annotator-agreement: compute kappa statistics.

Reads the per-rater predictions (``rater_predictions/rater_*.jsonl``) and the
gold key (``substrate_gold.jsonl``), then computes:

  * pairwise Cohen's kappa between every pair of raters (binary
    VALID-vs-HALLUCINATED collapse),
  * Fleiss' kappa across all raters (multi-rater chance-corrected agreement),
  * each rater vs the benchmark gold label (Cohen's kappa + accuracy),
  * majority-vote-of-raters vs gold,
  * raw percent agreement and a hallucination-type agreement rate on the
    subset where two raters both say HALLUCINATED.

All measures are implemented in pure numpy (no sklearn/scipy dependency).
UNCERTAIN handling is reported under two policies so the choice is transparent:
  * ``binary``  -- map UNCERTAIN -> HALLUCINATED (a flag is a flag; the verifier
    is not asserting the citation is clean). This is the headline policy.
  * ``drop``    -- drop entries where either rater answered UNCERTAIN.

Output: ``kappa_results.json``.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "ablations" / "a5_kappa"
PRED_DIR = OUT / "rater_predictions"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over paired categorical labels."""
    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(a)
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b, strict=True):
        cm[idx[x], idx[y]] += 1
    po = np.trace(cm) / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    pe = float((row * col).sum())
    if pe >= 1.0:
        return 1.0  # degenerate: both raters constant and identical
    return float((po - pe) / (1.0 - pe))


def fleiss_kappa(ratings: list[list[str]], categories: list[str]) -> float:
    """Fleiss' kappa. ``ratings`` is one list of rater labels per subject."""
    n_subjects = len(ratings)
    n_raters = len(ratings[0])
    cat_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((n_subjects, len(categories)), dtype=float)
    for i, subj in enumerate(ratings):
        for lab in subj:
            counts[i, cat_idx[lab]] += 1
    p_j = counts.sum(axis=0) / (n_subjects * n_raters)
    p_i = (np.square(counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = float(np.square(p_j).sum())
    if p_e >= 1.0:
        return 1.0
    return float((p_bar - p_e) / (1.0 - p_e))


def percent_agreement(a: list[str], b: list[str]) -> float:
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


def interpret(kappa: float) -> str:
    """Landis & Koch (1977) benchmark scale."""
    if kappa < 0.0:
        return "poor (worse than chance)"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def _binary(label: str) -> str:
    """Collapse to VALID vs HALLUCINATED; UNCERTAIN -> HALLUCINATED (flag)."""
    return "VALID" if label == "VALID" else "HALLUCINATED"


def main() -> None:
    gold_rows = _load_jsonl(OUT / "substrate_gold.jsonl")
    gold = {g["bibtex_key"]: g for g in gold_rows}

    rater_files = sorted(PRED_DIR.glob("rater_*.jsonl"))
    if not rater_files:
        raise SystemExit(f"no rater predictions in {PRED_DIR}; run a5_run_raters.py")

    raters: dict[str, dict[str, dict]] = {}
    rater_models: dict[str, str] = {}
    rater_run_dates: dict[str, str] = {}
    for f in rater_files:
        rows = _load_jsonl(f)
        rid = rows[0]["rater_id"]
        raters[rid] = {r["bibtex_key"]: r for r in rows}
        rater_models[rid] = rows[0]["model"]
        # Earliest prediction timestamp -> the rater's run date (date only).
        stamps = [r["run_utc"] for r in rows if r.get("run_utc")]
        rater_run_dates[rid] = min(stamps)[:10] if stamps else "unknown"

    rater_ids = sorted(raters)
    # Keys all raters + gold cover.
    common: set[str] = set(gold)
    for r in rater_ids:
        common &= set(raters[r])
    keys = sorted(common)

    # Per-rater raw label distribution + vs-gold.
    per_rater: dict[str, dict] = {}
    for rid in rater_ids:
        raw_labels = [raters[rid][k]["label"] for k in keys]
        bin_labels = [_binary(x) for x in raw_labels]
        gold_labels = [gold[k]["gold_label"] for k in keys]
        dist = {lab: raw_labels.count(lab) for lab in ("VALID", "HALLUCINATED", "UNCERTAIN")}
        kap = cohen_kappa(bin_labels, gold_labels)
        acc = percent_agreement(bin_labels, gold_labels)
        per_rater[rid] = {
            "model": rater_models[rid],
            "raw_label_distribution": dist,
            "n_uncertain": dist["UNCERTAIN"],
            "vs_gold_cohen_kappa": round(kap, 4),
            "vs_gold_interpretation": interpret(kap),
            "vs_gold_accuracy": round(acc, 4),
        }

    # Pairwise Cohen's kappa between raters, under two UNCERTAIN policies.
    def labels_for(rid: str, policy: str) -> dict[str, str]:
        out = {}
        for k in keys:
            lab = raters[rid][k]["label"]
            out[k] = _binary(lab) if policy == "binary" else lab
        return out

    pairwise: dict[str, dict] = {"binary": {}, "drop": {}}
    for ri, rj in itertools.combinations(rater_ids, 2):
        # binary policy: UNCERTAIN -> HALLUCINATED, keep all keys.
        a = [labels_for(ri, "binary")[k] for k in keys]
        b = [labels_for(rj, "binary")[k] for k in keys]
        pairwise["binary"][f"{ri}__vs__{rj}"] = {
            "cohen_kappa": round(cohen_kappa(a, b), 4),
            "percent_agreement": round(percent_agreement(a, b), 4),
            "n": len(keys),
            "interpretation": interpret(cohen_kappa(a, b)),
        }
        # drop policy: keep only keys where neither said UNCERTAIN.
        kk = [
            k
            for k in keys
            if raters[ri][k]["label"] != "UNCERTAIN" and raters[rj][k]["label"] != "UNCERTAIN"
        ]
        a2 = [raters[ri][k]["label"] for k in kk]
        b2 = [raters[rj][k]["label"] for k in kk]
        pairwise["drop"][f"{ri}__vs__{rj}"] = {
            "cohen_kappa": round(cohen_kappa(a2, b2), 4) if kk else None,
            "percent_agreement": round(percent_agreement(a2, b2), 4) if kk else None,
            "n": len(kk),
            "interpretation": interpret(cohen_kappa(a2, b2)) if kk else "n/a",
        }

    # Fleiss' kappa across all raters (binary policy, all keys).
    fleiss_ratings = [[labels_for(r, "binary")[k] for r in rater_ids] for k in keys]
    fk = fleiss_kappa(fleiss_ratings, ["VALID", "HALLUCINATED"])

    # Majority vote of raters vs gold (binary).
    maj_labels = []
    for k in keys:
        votes = [_binary(raters[r][k]["label"]) for r in rater_ids]
        maj_labels.append("HALLUCINATED" if votes.count("HALLUCINATED") >= 2 else "VALID")
    gold_labels = [gold[k]["gold_label"] for k in keys]
    maj_kappa = cohen_kappa(maj_labels, gold_labels)
    maj_acc = percent_agreement(maj_labels, gold_labels)

    # Hallucination-type agreement: where two raters both say HALLUCINATED,
    # do they agree on the type? (diagnostic, on real-world-hallucinated pool).
    type_pairs: dict[str, dict] = {}
    for ri, rj in itertools.combinations(rater_ids, 2):
        both_h = [
            k
            for k in keys
            if raters[ri][k]["label"] == "HALLUCINATED" and raters[rj][k]["label"] == "HALLUCINATED"
        ]
        agree = sum(
            1
            for k in both_h
            if raters[ri][k]["predicted_hallucination_type"]
            == raters[rj][k]["predicted_hallucination_type"]
            and raters[ri][k]["predicted_hallucination_type"] is not None
        )
        type_pairs[f"{ri}__vs__{rj}"] = {
            "n_both_hallucinated": len(both_h),
            "n_type_agree": agree,
            "type_agreement_rate": round(agree / len(both_h), 4) if both_h else None,
        }

    # Per-pool breakdown of majority-vote accuracy.
    by_pool: dict[str, dict] = {}
    for pool in ("real_world_incident", "relabel_recovered"):
        pk = [k for k in keys if gold[k]["pool"] == pool]
        if not pk:
            continue
        votes = []
        for k in pk:
            v = [_binary(raters[r][k]["label"]) for r in rater_ids]
            votes.append("HALLUCINATED" if v.count("HALLUCINATED") >= 2 else "VALID")
        gl = [gold[k]["gold_label"] for k in pk]
        by_pool[pool] = {
            "n": len(pk),
            "gold_label": gl[0],
            "majority_vote_accuracy": round(percent_agreement(votes, gl), 4),
        }

    results = {
        "description": "A5 automated multi-rater agreement (reliability proxy, "
        "NOT human IAA). 3 independent LLM raters label blinded entries.",
        "endpoint": "https://openrouter.ai/api/v1",
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "rater_run_dates": rater_run_dates,
        "snapshot_note": "fresh dated snapshot; OpenRouter endpoint may drift "
        "vs published eval -- these are NEW numbers, not a reproduction.",
        "raters": rater_models,
        "n_entries_scored": len(keys),
        "uncertain_policy_headline": "binary (UNCERTAIN -> HALLUCINATED)",
        "scale": "Landis & Koch (1977)",
        "fleiss_kappa_binary": round(fk, 4),
        "fleiss_interpretation": interpret(fk),
        "pairwise_cohen_kappa": pairwise,
        "per_rater_vs_gold": per_rater,
        "majority_vote_vs_gold": {
            "cohen_kappa": round(maj_kappa, 4),
            "interpretation": interpret(maj_kappa),
            "accuracy": round(maj_acc, 4),
        },
        "majority_vote_vs_gold_by_pool": by_pool,
        "hallucination_type_agreement": type_pairs,
    }

    (OUT / "kappa_results.json").write_text(json.dumps(results, indent=2))

    print(f"Scored {len(keys)} entries across {len(rater_ids)} raters")
    print(f"Fleiss kappa (binary): {fk:.3f} -> {interpret(fk)}")
    for pair, pv in pairwise["binary"].items():
        print(f"  Cohen {pair}: {pv['cohen_kappa']:.3f} ({pv['interpretation']})")
    for rid, rv in per_rater.items():
        print(
            f"  {rid} vs gold: kappa={rv['vs_gold_cohen_kappa']:.3f} "
            f"acc={rv['vs_gold_accuracy']:.3f} uncertain={rv['n_uncertain']}"
        )
    print(
        f"Majority-vote vs gold: kappa={maj_kappa:.3f} acc={maj_acc:.3f} ({interpret(maj_kappa)})"
    )
    print(f"Wrote -> {OUT / 'kappa_results.json'}")


if __name__ == "__main__":
    main()
