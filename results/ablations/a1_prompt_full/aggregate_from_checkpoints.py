"""Re-derive the A1 summary purely from persisted checkpoint JSONL files.

The main runner writes summary.json at the end of a live run. This script
reconstructs the identical aggregates from the persisted per-entry predictions
alone (results/ablations/a1_prompt_full/checkpoints/<model>/<variant>/*.jsonl)
plus the fixed sample. It calls NO API, so it is the reproducible path: anyone
can regenerate every reported number from the committed predictions.

Writes summary_from_checkpoints.json (byte-comparable analysis block to
summary.json's, minus live wall-clock). Use to verify the live summary or to
recompute after a metric definition change.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scipy.stats import spearmanr

from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation import evaluate

ROOT = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OUTDIR = ROOT / "results/ablations/a1_prompt_full"
SAMPLE = OUTDIR / "sample_150.jsonl"
CKPT = OUTDIR / "checkpoints"
ENDPOINT = "https://openrouter.ai/api/v1"

MODELS: dict[str, str] = {
    "sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
}
VARIANTS = ["default", "notaxo", "uncertain", "terse"]


def load_entries() -> list[BenchmarkEntry]:
    return [
        BenchmarkEntry.from_dict(json.loads(line))
        for line in SAMPLE.read_text().splitlines()
        if line.strip()
    ]


def load_preds(model: str, variant: str) -> list[Prediction]:
    safe_model = MODELS[model].replace("/", "_")
    path = CKPT / model / variant / f"openrouter_{safe_model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    preds = []
    for line in path.read_text().splitlines():
        if line.strip():
            preds.append(Prediction.from_dict(json.loads(line)))
    return preds


def rank_from_scores(scores: dict[str, float]) -> dict[str, float]:
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    vals = [v for _, v in items]
    names = [k for k, _ in items]
    ranks: dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        while j < len(items) and vals[j] == vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[names[k]] = avg_rank
        i = j
    return ranks


def main() -> None:
    entries = load_entries()
    keys = [e.bibtex_key for e in entries]
    by_key = {e.bibtex_key: e for e in entries}

    results: dict[str, dict[str, dict]] = {}
    preds_by: dict[str, dict[str, dict[str, str]]] = {}
    for mname in MODELS:
        results[mname] = {}
        preds_by[mname] = {}
        for vname in VARIANTS:
            preds = load_preds(mname, vname)
            # align prediction order to the sample order for evaluate()
            pmap = {p.bibtex_key: p for p in preds}
            ordered = [pmap[k] for k in keys if k in pmap]
            aligned_entries = [by_key[p.bibtex_key] for p in ordered]
            res = evaluate(
                aligned_entries, ordered, tool_name=f"{mname}-{vname}", split_name="a1_dev150"
            )
            n_unc = sum(1 for p in ordered if p.label == "UNCERTAIN")
            results[mname][vname] = {
                "model": mname,
                "variant": vname,
                "n": len(ordered),
                "detection_rate": res.detection_rate,
                "false_positive_rate": res.false_positive_rate,
                "f1_hallucination": res.f1_hallucination,
                "ece": res.ece,
                "num_uncertain": n_unc,
                "uncertain_rate": n_unc / len(ordered) if ordered else 0.0,
                "coverage": res.coverage,
            }
            preds_by[mname][vname] = {p.bibtex_key: p.label for p in ordered}

    # verdict-flip
    flip_report: dict[str, dict] = {}
    pf_all = pt_all = pf_strict = pt_strict = 0
    for mname in MODELS:
        base = preds_by[mname]["default"]
        per_variant = {}
        for vname in VARIANTS:
            if vname == "default":
                continue
            cur = preds_by[mname][vname]
            nf_all = nf_strict = ne = 0
            for k in keys:
                b, c = base.get(k), cur.get(k)
                if b != c:
                    nf_all += 1
                if b in ("VALID", "HALLUCINATED") and c in ("VALID", "HALLUCINATED"):
                    ne += 1
                    if b != c:
                        nf_strict += 1
            per_variant[vname] = {
                "flip_rate_all": nf_all / len(keys),
                "flip_rate_strict": nf_strict / ne if ne else 0.0,
            }
            pf_all += nf_all
            pt_all += len(keys)
            pf_strict += nf_strict
            pt_strict += ne
        flip_report[mname] = {
            "per_variant_vs_default": per_variant,
            "mean_flip_rate_all": sum(v["flip_rate_all"] for v in per_variant.values())
            / len(per_variant),
        }
    flip_report["_pooled"] = {
        "flip_rate_all": pf_all / pt_all,
        "flip_rate_strict": pf_strict / pt_strict if pt_strict else 0.0,
    }

    # Spearman ranking stability
    ranking_stability: dict[str, dict] = {}
    for metric in ("f1_hallucination", "detection_rate", "false_positive_rate"):
        pvr: dict[str, dict[str, float]] = {}
        for vname in VARIANTS:
            scores = {}
            for mname in MODELS:
                val = results[mname][vname][metric]
                if val is None:
                    val = 0.0
                scores[mname] = -val if metric == "false_positive_rate" else val
            pvr[vname] = rank_from_scores(scores)
        order = list(MODELS.keys())
        pair_rhos = []
        for i in range(len(VARIANTS)):
            for j in range(i + 1, len(VARIANTS)):
                vi = [pvr[VARIANTS[i]][m] for m in order]
                vj = [pvr[VARIANTS[j]][m] for m in order]
                r, _ = spearmanr(vi, vj)
                if r == r:
                    pair_rhos.append(float(r))
        ranking_stability[metric] = {
            "per_variant_ranks": {v: {m: pvr[v][m] for m in order} for v in VARIANTS},
            "mean_pairwise_rho": sum(pair_rhos) / len(pair_rhos) if pair_rhos else None,
        }

    fpr_decomp: dict[str, dict] = {}
    for mname in MODELS:
        fprs = {
            v: results[mname][v]["false_positive_rate"]
            for v in VARIANTS
            if results[mname][v]["false_positive_rate"] is not None
        }
        if not fprs:
            continue
        dfpr = results[mname]["default"]["false_positive_rate"]
        min_v = min(fprs, key=lambda v: fprs[v])
        fpr_decomp[mname] = {
            "default_fpr": dfpr,
            "min_fpr": fprs[min_v],
            "min_fpr_variant": min_v,
            "prompt_induced_fpr_drop": (dfpr - fprs[min_v] if dfpr is not None else None),
            "all_variant_fpr": fprs,
        }

    summary = {
        "experiment": "A1_prompt_sensitivity_full (recomputed from checkpoints)",
        "recomputed_on": date.today().isoformat(),
        "endpoint": ENDPOINT,
        "models": MODELS,
        "variants": VARIANTS,
        "per_model_per_variant": results,
        "verdict_flip_rate": flip_report,
        "ranking_stability_spearman": ranking_stability,
        "fpr_decomposition": fpr_decomp,
    }
    (OUTDIR / "summary_from_checkpoints.json").write_text(json.dumps(summary, indent=2))
    print("Recomputed from checkpoints -> summary_from_checkpoints.json")
    for mname in MODELS:
        for vname in VARIANTS:
            r = results[mname][vname]
            fpr = r["false_positive_rate"]
            print(
                f"{mname:<18}{vname:<11}DR={r['detection_rate']:.3f} "
                f"FPR={fpr if fpr is not None else float('nan'):.3f} "
                f"UNC%={r['uncertain_rate'] * 100:.1f} F1={r['f1_hallucination']:.3f}"
            )


if __name__ == "__main__":
    main()
