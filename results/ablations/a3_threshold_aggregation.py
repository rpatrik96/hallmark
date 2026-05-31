"""Ablation 3: threshold sweep + aggregation (any-miss vs k-of-n) on n=60 pilot.

Purely OFFLINE: re-scores EXISTING stored per-entry predictions/confidences and
the benchmark's own structured subtest signals against the NEW v1.1.1 labels.

(a) Threshold sweep: map each model's stated confidence to P(hallucinated)
    (score = conf if pred==HALL else 1-conf, matching hallmark AUROC convention),
    sweep the decision threshold, trace the DR-FPR operating curve, compute AUROC,
    and locate the paper's de-facto 0.5 operating point.

(b) Aggregation: treat the 4 stored structured DB-style checks
    (doi_resolves, title_exists, authors_match, venue_correct) as n independent
    lookups. Compare ANY-MISS (flag HALL if any check fails -> the agentic
    'any single DB misses' rule) against MAJORITY-VOTE and k-of-n. Report the
    FPR reduction from majority-vote vs any-miss.

No API calls. Writes JSON + a console table.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

ABL = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark/results/ablations")
SAMPLE = ABL / "pilot_sample_dev60.jsonl"
PRED_GLOB = "/Users/patrik.reizinger/Documents/GitHub/hallmark/results/*_predictions.jsonl"
AGENTIC = (
    "/Users/patrik.reizinger/Documents/GitHub/hallmark/results/checkpoints/"
    "llm_agentic_btu_sonnet_4_6_dev_public/"
    "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl"
)
SONNET_ZS = (
    "/Users/patrik.reizinger/Documents/GitHub/hallmark/results/"
    "llm_openrouter_claude_sonnet_4_6_dev_public_predictions.jsonl"
)

# The 4 structured lookups that survive in stored data on every entry.
# cross_db_agreement is held out as the "any-miss proxy" reference signal.
DB_CHECKS = ["doi_resolves", "title_exists", "authors_match", "venue_correct"]


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def load_sample() -> list[dict]:
    return read_jsonl(SAMPLE)


def confusion(pred_labels: dict[str, str], truth: dict[str, str]) -> dict[str, float]:
    """DR, FPR, F1 with UNCERTAIN excluded (hallmark protocol)."""
    tp = fp = tn = fn = unc = 0
    for k, t in truth.items():
        p = pred_labels.get(k)
        if p == "UNCERTAIN" or p is None:
            unc += 1
            continue
        if t == "HALLUCINATED":
            if p == "HALLUCINATED":
                tp += 1
            else:
                fn += 1
        else:  # VALID
            if p == "HALLUCINATED":
                fp += 1
            else:
                tn += 1
    dr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) else 0.0
    return {
        "DR": dr,
        "FPR": fpr,
        "F1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "uncertain": unc,
    }


def auroc(scores: list[tuple[float, int]]) -> float | None:
    """Mann-Whitney AUROC. scores = [(p_hall, true_is_hall)]. Handles ties."""
    pos = [s for s, y in scores if y == 1]
    neg = [s for s, y in scores if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def score_for_entry(row: dict) -> float:
    """Map stated confidence to P(hallucinated), hallmark AUROC convention."""
    c = float(row["confidence"])
    return c if row["label"] == "HALLUCINATED" else 1.0 - c


def main() -> None:
    sample = load_sample()
    truth = {r["bibtex_key"]: r["label"] for r in sample}
    skeys = set(truth)
    n_hall = sum(1 for v in truth.values() if v == "HALLUCINATED")
    n_valid = len(truth) - n_hall

    out: dict[str, Any] = {
        "n": len(truth),
        "n_hall": n_hall,
        "n_valid": n_valid,
        "note": "offline re-scoring on v1.1.1 labels; no API calls",
    }

    # ---------- (a) THRESHOLD SWEEP ----------
    files = [*sorted(glob.glob(PRED_GLOB)), AGENTIC]
    sweep_out: dict[str, Any] = {}
    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95

    for f in files:
        name = Path(f).stem.replace("llm_openrouter_", "").replace("_dev_public_predictions", "")
        if "agentic_btu" in name:
            name = "agentic_btu_sonnet_4_6"
        rows = {r["bibtex_key"]: r for r in read_jsonl(f) if r["bibtex_key"] in skeys}
        # AUROC on definite preds only (exclude UNCERTAIN)
        auroc_pairs = [
            (score_for_entry(r), 1 if truth[k] == "HALLUCINATED" else 0)
            for k, r in rows.items()
            if r["label"] in ("HALLUCINATED", "VALID")
        ]
        au = auroc(auroc_pairs)

        # Default operating point = the model's own emitted labels (its native 0.5-ish decision)
        default_labels = {k: r["label"] for k, r in rows.items()}
        default_cm = confusion(default_labels, truth)

        # Sweep: classify HALL iff p_hall >= thr. UNCERTAIN rows kept as UNCERTAIN
        # (no reliable score) to match the protocol.
        curve = []
        for thr in thresholds:
            tl: dict[str, str] = {}
            for k, r in rows.items():
                if r["label"] == "UNCERTAIN":
                    tl[k] = "UNCERTAIN"
                    continue
                p_hall = score_for_entry(r)
                tl[k] = "HALLUCINATED" if p_hall >= thr else "VALID"
            cm = confusion(tl, truth)
            curve.append({"thr": thr, **{m: round(cm[m], 4) for m in ("DR", "FPR", "F1")}})

        sweep_out[name] = {
            "auroc": round(au, 4) if au is not None else None,
            "default": {m: round(default_cm[m], 4) for m in ("DR", "FPR", "F1")},
            "default_uncertain": default_cm["uncertain"],
            "curve": curve,
            "n_confidence_levels": len({round(r["confidence"], 2) for r in rows.values()}),
        }
    out["threshold_sweep"] = sweep_out

    # ---------- (b) AGGREGATION: any-miss vs k-of-n ----------
    # Use the benchmark's own structured DB-style checks.
    agg_truth = truth  # all 60 have subtests
    n_checks = len(DB_CHECKS)

    def agg_labels(min_fails: int) -> dict[str, str]:
        """Flag HALLUCINATED iff >= min_fails of the n checks return False.
        min_fails=1 -> ANY-MISS (the agentic 'any single DB misses' rule)."""
        labels = {}
        for r in sample:
            sub = r["subtests"]
            fails = sum(1 for c in DB_CHECKS if sub.get(c) is False)
            labels[r["bibtex_key"]] = "HALLUCINATED" if fails >= min_fails else "VALID"
        return labels

    agg_out: dict[str, Any] = {}
    rules = {
        "any_miss (k>=1)": 1,
        "two_of_four (k>=2)": 2,
        "majority (k>=3 of 4)": 3,
        "unanimous (k>=4)": 4,
    }
    for rule_name, mf in rules.items():
        cm = confusion(agg_labels(mf), agg_truth)
        agg_out[rule_name] = {m: round(cm[m], 4) for m in ("DR", "FPR", "F1")} | {
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"],
            "tn": cm["tn"],
        }

    # cross_db_agreement alone = the literal "any DB disagrees" single signal
    cdb_labels = {
        r["bibtex_key"]: (
            "HALLUCINATED" if r["subtests"].get("cross_db_agreement") is False else "VALID"
        )
        for r in sample
    }
    cm_cdb = confusion(cdb_labels, agg_truth)
    agg_out["cross_db_agreement_only"] = {m: round(cm_cdb[m], 4) for m in ("DR", "FPR", "F1")} | {
        "fp": cm_cdb["fp"],
        "fn": cm_cdb["fn"],
    }

    any_fpr = agg_out["any_miss (k>=1)"]["FPR"]
    maj_fpr = agg_out["majority (k>=3 of 4)"]["FPR"]
    out["aggregation"] = {
        "n_checks": n_checks,
        "checks": DB_CHECKS,
        "rules": agg_out,
        "fpr_reduction_majority_vs_anymiss_pp": round((any_fpr - maj_fpr) * 100, 1),
        "dr_change_majority_vs_anymiss_pp": round(
            (agg_out["majority (k>=3 of 4)"]["DR"] - agg_out["any_miss (k>=1)"]["DR"]) * 100, 1
        ),
    }

    # ---------- (b2) AGGREGATION over NOISY voters (real model predictions) ----------
    # The ground-truth subtests are clean on valid entries (FPR=0 by construction),
    # so they cannot model the reviewer's concern: a *real* DB failing to return a
    # *real* paper. The empirically faithful proxy is the 7 zero-shot LLM verifiers
    # treated as 7 independent noisy "lookups". any-miss = flag HALL if ANY voter
    # says HALL (the agentic harness rule); majority = flag HALL if >= ceil(n/2) do.
    voter_files = sorted(glob.glob(PRED_GLOB))
    voter_preds: dict[str, dict[str, str]] = {}
    for f in voter_files:
        name = Path(f).stem.replace("llm_openrouter_", "").replace("_dev_public_predictions", "")
        voter_preds[name] = {
            r["bibtex_key"]: r["label"] for r in read_jsonl(f) if r["bibtex_key"] in skeys
        }
    n_voters = len(voter_preds)

    def voter_labels(min_hall_votes: int) -> dict[str, str]:
        labels = {}
        for k in skeys:
            votes = sum(1 for vp in voter_preds.values() if vp.get(k) == "HALLUCINATED")
            labels[k] = "HALLUCINATED" if votes >= min_hall_votes else "VALID"
        return labels

    voter_out: dict[str, Any] = {}
    voter_rules = {
        f"any_miss (>=1 of {n_voters})": 1,
        f"two_votes (>=2 of {n_voters})": 2,
        f"majority (>={(n_voters // 2) + 1} of {n_voters})": (n_voters // 2) + 1,
        f"unanimous (>={n_voters} of {n_voters})": n_voters,
    }
    for rn, mv in voter_rules.items():
        cm = confusion(voter_labels(mv), truth)
        voter_out[rn] = {m: round(cm[m], 4) for m in ("DR", "FPR", "F1")} | {
            "fp": cm["fp"],
            "fn": cm["fn"],
        }
    v_any = next(v for k, v in voter_out.items() if k.startswith("any_miss"))
    v_maj = next(v for k, v in voter_out.items() if k.startswith("majority"))
    out["aggregation_noisy_voters"] = {
        "n_voters": n_voters,
        "voters": list(voter_preds),
        "rules": voter_out,
        "fpr_reduction_majority_vs_anymiss_pp": round((v_any["FPR"] - v_maj["FPR"]) * 100, 1),
        "dr_change_majority_vs_anymiss_pp": round((v_maj["DR"] - v_any["DR"]) * 100, 1),
    }

    # Agentic-BTU (single tool) vs its zero-shot Sonnet base: the paper's exact claim
    # ("agentic inflates FPR over zero-shot base"). Both fully cover the sample.
    zs_sonnet = {
        r["bibtex_key"]: r["label"] for r in read_jsonl(SONNET_ZS) if r["bibtex_key"] in skeys
    }
    ag_sonnet = {
        r["bibtex_key"]: r["label"] for r in read_jsonl(AGENTIC) if r["bibtex_key"] in skeys
    }
    cm_zs = confusion(zs_sonnet, truth)
    cm_ag = confusion(ag_sonnet, truth)
    out["agentic_vs_zeroshot_sonnet"] = {
        "zeroshot": {m: round(cm_zs[m], 4) for m in ("DR", "FPR", "F1")},
        "agentic_btu": {m: round(cm_ag[m], 4) for m in ("DR", "FPR", "F1")},
        "fpr_inflation_pp": round((cm_ag["FPR"] - cm_zs["FPR"]) * 100, 1),
        "dr_gain_pp": round((cm_ag["DR"] - cm_zs["DR"]) * 100, 1),
    }

    (ABL / "a3_threshold_aggregation_result.json").write_text(json.dumps(out, indent=2))

    # ---------- console summary ----------
    print(f"\n=== A3 OFFLINE: n={len(truth)} ({n_hall} HALL / {n_valid} VALID) ===\n")
    print("(a) THRESHOLD SWEEP -- AUROC + default operating point vs sweep best-F1")
    print(
        f"{'model':30s} {'AUROC':>6s} {'def-DR':>7s} {'def-FPR':>8s} {'def-F1':>7s} {'bestF1@thr':>12s}"
    )
    for name, d in sweep_out.items():
        df = d["default"]  # type: ignore[index]
        best = max(d["curve"], key=lambda c: c["F1"])  # type: ignore[index]
        au = d["auroc"]  # type: ignore[index]
        print(
            f"{name:30s} {(au if au is not None else float('nan')):6.3f} "
            f"{df['DR']:7.3f} {df['FPR']:8.3f} {df['F1']:7.3f} "
            f"{best['F1']:.3f}@{best['thr']:.2f}"
        )

    print("\n(b) AGGREGATION over 4 DB-style checks (FPR reduction = the any-miss critique)")
    print(f"{'rule':24s} {'DR':>6s} {'FPR':>6s} {'F1':>6s} {'fp':>4s} {'fn':>4s}")
    for rn, d in agg_out.items():  # type: ignore[assignment]
        print(
            f"{rn:24s} {d['DR']:6.3f} {d['FPR']:6.3f} {d['F1']:6.3f} {d.get('fp', '-'):>4} {d.get('fn', '-'):>4}"
        )
    print(
        "\n[ground-truth subtests are clean on valid entries -> any-miss FPR already 0; "
        "stricter k-of-n only sheds DR]"
    )

    print("\n(b2) AGGREGATION over NOISY voters (7 real zero-shot LLMs as independent lookups)")
    print(f"{'rule':24s} {'DR':>6s} {'FPR':>6s} {'F1':>6s} {'fp':>4s} {'fn':>4s}")
    for rn, d in voter_out.items():  # type: ignore[assignment]
        print(f"{rn:24s} {d['DR']:6.3f} {d['FPR']:6.3f} {d['F1']:6.3f} {d['fp']:>4} {d['fn']:>4}")
    nv = out["aggregation_noisy_voters"]  # type: ignore[assignment]
    print(
        f"FPR reduction majority vs any-miss: {nv['fpr_reduction_majority_vs_anymiss_pp']} pp "  # type: ignore[index]
        f"(DR change {nv['dr_change_majority_vs_anymiss_pp']} pp)"  # type: ignore[index]
    )

    av = out["agentic_vs_zeroshot_sonnet"]  # type: ignore[assignment]
    zs, ag = av["zeroshot"], av["agentic_btu"]  # type: ignore[index]
    print("\n(b3) AGENTIC-BTU vs ZERO-SHOT Sonnet (the paper's exact +FPR claim)")
    print(f"  zero-shot : DR={zs['DR']:.3f} FPR={zs['FPR']:.3f} F1={zs['F1']:.3f}")
    print(f"  agentic   : DR={ag['DR']:.3f} FPR={ag['FPR']:.3f} F1={ag['F1']:.3f}")
    print(f"  FPR inflation {av['fpr_inflation_pp']} pp ; DR gain {av['dr_gain_pp']} pp")  # type: ignore[index]
    print(f"\nwrote {ABL / 'a3_threshold_aggregation_result.json'}")


if __name__ == "__main__":
    main()
