#!/usr/bin/env python
"""E3: GPT-5.1 run-to-run variance on a fixed stratified subsample of dev_public.

Reviewer experiment. GPT-5.1 is forced to temperature=1.0 by the API, so a single
run is one stochastic draw. This script re-runs the OpenAI baseline N=3 times on the
SAME N=150 stratified subsample (each run with a distinct checkpoint_dir so the
checkpoint cache cannot just replay run 1) and reports run-to-run variance of
DR / FPR / F1 / MCC, plus label-flip counts.

Usage (keys must already be in env; do NOT echo them):
    set -a; . /tmp/.or_env; . /tmp/.openai_env; set +a
    cd /Users/patrik.reizinger/Documents/GitHub/hallmark
    .venv/bin/python scripts/reviewer_experiments/e3_gpt51_variance.py

Cost cap: 150 entries x 3 runs = 450 API calls. Do not exceed.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from hallmark.baselines.llm_verifier import verify_with_openai
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate

# --- Config ---
SPLIT = "dev_public"
N = 150
SEED = 42
N_RUNS = 3
MODEL = "gpt-5.1"
OUT_DIR = Path("results/reviewer_experiments/e3_variance")
# Published full-split F1 gaps (for the ranking-stability conclusion).
GAP_SONNET_GPT51 = 0.069  # Sonnet 4.6 - GPT-5.1
GAP_SONNET_OPUS = 0.016  # Sonnet 4.6 - Opus 4.7

METRIC_KEYS = ["detection_rate", "false_positive_rate", "f1_hallucination", "mcc"]


def stratified_subsample(entries, n, seed):
    """Stratified subsample preserving the HALLUCINATED:VALID ratio.

    Sizes are computed by rounding n * (population hallucinated fraction) so the
    subsample fraction tracks the full split as closely as integer counts allow.
    """
    hall = [e for e in entries if e.label == "HALLUCINATED"]
    val = [e for e in entries if e.label == "VALID"]
    frac_hall = len(hall) / len(entries)
    n_hall = round(n * frac_hall)
    n_val = n - n_hall
    rng = random.Random(seed)
    samp_hall = rng.sample(hall, n_hall)
    samp_val = rng.sample(val, n_val)
    sample = samp_hall + samp_val
    rng.shuffle(sample)
    return sample, frac_hall, n_hall, n_val


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = load_split(SPLIT)
    print(
        f"[load] {SPLIT}: {len(entries)} entries "
        f"({sum(e.label == 'HALLUCINATED' for e in entries)} HALLUCINATED + "
        f"{sum(e.label == 'VALID' for e in entries)} VALID)"
    )

    sample, frac_hall, n_hall, n_val = stratified_subsample(entries, N, SEED)
    print(
        f"[sample] N={len(sample)} (seed={SEED}): {n_hall} HALLUCINATED + {n_val} VALID; "
        f"subsample frac_hall={n_hall / N:.4f} vs population {frac_hall:.4f}"
    )

    sample_keys = [e.bibtex_key for e in sample]
    (OUT_DIR / "sample_keys.json").write_text(
        json.dumps(
            {
                "split": SPLIT,
                "n": N,
                "seed": SEED,
                "n_hallucinated": n_hall,
                "n_valid": n_val,
                "population_frac_hallucinated": frac_hall,
                "subsample_frac_hallucinated": n_hall / N,
                "bibtex_keys": sample_keys,
            },
            indent=2,
        )
    )

    blind = [e.to_blind() for e in sample]

    # --- N runs, distinct checkpoint_dir each so caching cannot replay run 1 ---
    runs_preds = []  # list[dict[bibtex_key -> Prediction]]
    runs_results = []  # list[EvaluationResult]
    total_api_calls = 0

    for r in range(1, N_RUNS + 1):
        ckpt = OUT_DIR / f"run{r}"
        ckpt.mkdir(parents=True, exist_ok=True)
        print(f"[run {r}/{N_RUNS}] verify_with_openai(model={MODEL!r}, checkpoint_dir={ckpt})")
        preds = verify_with_openai(blind, model=MODEL, checkpoint_dir=ckpt)
        run_calls = sum(p.api_calls for p in preds)
        total_api_calls += run_calls
        print(f"[run {r}/{N_RUNS}] done: {len(preds)} predictions, {run_calls} api_calls")

        pred_by_key = {p.bibtex_key: p for p in preds}
        runs_preds.append(pred_by_key)

        result = evaluate(sample, preds, tool_name=f"{MODEL}-run{r}", split_name=SPLIT)
        runs_results.append(result)

        rd = result.to_dict()
        (OUT_DIR / f"run{r}" / "metrics.json").write_text(json.dumps(rd, indent=2, default=str))
        (OUT_DIR / f"run{r}" / "predictions.json").write_text(
            json.dumps(
                {
                    k: {"label": p.label, "confidence": p.confidence, "api_calls": p.api_calls}
                    for k, p in pred_by_key.items()
                },
                indent=2,
            )
        )
        print(
            f"[run {r}/{N_RUNS}] DR={result.detection_rate:.4f} "
            f"FPR={result.false_positive_rate:.4f} "
            f"F1={result.f1_hallucination:.4f} MCC={result.mcc:.4f}"
        )

    # --- Sanity: are the 3 prediction sets identical? count label flips ---
    # An entry "flipped" if its predicted label is not identical across all N runs.
    flipped_keys = []
    for key in sample_keys:
        labels = {runs_preds[r].get(key).label for r in range(N_RUNS) if runs_preds[r].get(key)}
        if len(labels) > 1:
            flipped_keys.append(key)
    n_flipped = len(flipped_keys)
    print(f"[sanity] entries with label flips across {N_RUNS} runs: {n_flipped}/{N}")

    # Pairwise disagreement (Hamming) between run pairs for extra context.
    pairwise = {}
    for i in range(N_RUNS):
        for j in range(i + 1, N_RUNS):
            diff = sum(
                1
                for k in sample_keys
                if runs_preds[i].get(k)
                and runs_preds[j].get(k)
                and runs_preds[i][k].label != runs_preds[j][k].label
            )
            pairwise[f"run{i + 1}_vs_run{j + 1}"] = diff
    print(f"[sanity] pairwise label disagreements: {pairwise}")

    # --- Aggregate metrics: mean +/- sample std across runs ---
    per_run = []
    for r, res in enumerate(runs_results, start=1):
        per_run.append(
            {
                "run": r,
                "detection_rate": res.detection_rate,
                "false_positive_rate": res.false_positive_rate,
                "f1_hallucination": res.f1_hallucination,
                "mcc": res.mcc,
            }
        )

    agg = {}
    for m in METRIC_KEYS:
        vals = [pr[m] for pr in per_run]
        agg[m] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,  # sample std (ddof=1)
            "min": min(vals),
            "max": max(vals),
            "values": vals,
        }

    f1_std = agg["f1_hallucination"]["std"]

    summary_data = {
        "config": {
            "split": SPLIT,
            "n": N,
            "seed": SEED,
            "n_runs": N_RUNS,
            "model": MODEL,
            "n_hallucinated": n_hall,
            "n_valid": n_val,
            "subsample_frac_hallucinated": n_hall / N,
            "population_frac_hallucinated": frac_hall,
            "temperature": "1.0 (forced by API)",
        },
        "per_run": per_run,
        "aggregate": agg,
        "label_flips": {
            "n_flipped": n_flipped,
            "flipped_keys": flipped_keys,
            "pairwise_disagreements": pairwise,
            "prediction_sets_identical": n_flipped == 0,
        },
        "total_api_calls": total_api_calls,
        "published_gaps": {
            "sonnet46_minus_gpt51_f1": GAP_SONNET_GPT51,
            "sonnet46_minus_opus47_f1": GAP_SONNET_OPUS,
        },
        "ranking_stability": {
            "f1_run_to_run_std": f1_std,
            "sonnet_gpt51_gap_within_std": f1_std >= GAP_SONNET_GPT51,
            "sonnet_opus_gap_within_std": f1_std >= GAP_SONNET_OPUS,
            "sonnet_gpt51_gap_over_std_ratio": GAP_SONNET_GPT51 / f1_std if f1_std > 0 else None,
            "sonnet_opus_gap_over_std_ratio": GAP_SONNET_OPUS / f1_std if f1_std > 0 else None,
        },
    }

    (OUT_DIR / "summary_data.json").write_text(json.dumps(summary_data, indent=2, default=str))

    write_markdown(summary_data)
    print(f"[done] total_api_calls={total_api_calls} (cap=450)")
    print(f"[done] SUMMARY written to {OUT_DIR / 'SUMMARY.md'}")


def write_markdown(d):
    cfg = d["config"]
    pr = d["per_run"]
    agg = d["aggregate"]
    lf = d["label_flips"]
    rs = d["ranking_stability"]
    f1_std = rs["f1_run_to_run_std"]
    run1 = pr[0]

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

    lines = []
    lines.append("# E3: GPT-5.1 Run-to-Run Variance\n")
    lines.append(
        f"GPT-5.1 runs at **temperature=1.0** (forced by the OpenAI API), so a single "
        f'baseline run is one stochastic draw. We re-ran `verify_with_openai(model="gpt-5.1")` '
        f"**{cfg['n_runs']} times** on the **same** fixed stratified subsample of `{cfg['split']}` "
        f"(N={cfg['n']}, seed={cfg['seed']}) to quantify run-to-run variance. Each run used a "
        f"**distinct `checkpoint_dir`** (run1/run2/run3) so checkpoint caching could not replay "
        f"run 1 — every run actually re-called the API.\n"
    )
    lines.append("## Subsample\n")
    lines.append(
        f"- Stratified: **{cfg['n_hallucinated']} HALLUCINATED + {cfg['n_valid']} VALID** "
        f"= {cfg['n']} entries.\n"
        f"- Subsample hallucinated fraction: **{cfg['subsample_frac_hallucinated']:.4f}** "
        f"(population: {cfg['population_frac_hallucinated']:.4f}).\n"
        f"- Sampled `bibtex_keys` saved to `sample_keys.json`.\n"
    )

    lines.append("## Per-run metrics\n")
    lines.append("| Run | DR | FPR | F1 (hall) | MCC |")
    lines.append("|----:|---:|----:|----------:|----:|")
    for row in pr:
        lines.append(
            f"| {row['run']} | {fmt(row['detection_rate'])} | "
            f"{fmt(row['false_positive_rate'])} | {fmt(row['f1_hallucination'])} | "
            f"{fmt(row['mcc'])} |"
        )
    lines.append("")

    lines.append("## Mean +/- sample std (across 3 runs)\n")
    lines.append("| Metric | Mean | Sample std | Min | Max |")
    lines.append("|--------|-----:|-----------:|----:|----:|")
    label = {
        "detection_rate": "Detection rate (DR)",
        "false_positive_rate": "False positive rate (FPR)",
        "f1_hallucination": "F1 (hallucination)",
        "mcc": "MCC",
    }
    for m in METRIC_KEYS:
        a = agg[m]
        lines.append(
            f"| {label[m]} | {fmt(a['mean'])} | {fmt(a['std'])} | "
            f"{fmt(a['min'])} | {fmt(a['max'])} |"
        )
    lines.append("")

    lines.append("## Sanity check: predictions are not identical across runs\n")
    lines.append(
        f"- Entries whose predicted label **flipped** across the {cfg['n_runs']} runs: "
        f"**{lf['n_flipped']} / {cfg['n']}**.\n"
        f"- Prediction sets identical across runs: **{lf['prediction_sets_identical']}** "
        f"(False confirms each run genuinely re-called the API and drew a fresh sample).\n"
        f"- Pairwise label disagreements: "
        + ", ".join(f"{k}={v}" for k, v in lf["pairwise_disagreements"].items())
        + ".\n"
    )

    lines.append("## Ranking-stability conclusion\n")
    lines.append(
        f"GPT-5.1's **F1 run-to-run sample std is {f1_std:.4f}** (on this N={cfg['n']} subsample, "
        f"{cfg['n_runs']} runs). Comparing against the published full-split F1 gaps:\n"
    )
    gap_sg = d["published_gaps"]["sonnet46_minus_gpt51_f1"]
    gap_so = d["published_gaps"]["sonnet46_minus_opus47_f1"]

    def verdict(gap, within):
        return (
            "**within** GPT-5.1's run-to-run std (threatened — could be noise)"
            if within
            else "**outside** GPT-5.1's run-to-run std (stable)"
        )

    sg_ratio = rs["sonnet_gpt51_gap_over_std_ratio"]
    so_ratio = rs["sonnet_opus_gap_over_std_ratio"]
    lines.append(
        f"- **Sonnet 4.6 - GPT-5.1 gap = {gap_sg:.3f}** ({gap_sg / f1_std:.1f}x the F1 std): "
        f"{verdict(gap_sg, rs['sonnet_gpt51_gap_within_std'])}."
        if f1_std > 0
        else f"- **Sonnet 4.6 - GPT-5.1 gap = {gap_sg:.3f}**: F1 std is 0."
    )
    lines.append(
        f"- **Sonnet 4.6 - Opus 4.7 gap = {gap_so:.3f}** ({gap_so / f1_std:.1f}x the F1 std): "
        f"{verdict(gap_so, rs['sonnet_opus_gap_within_std'])}."
        if f1_std > 0
        else f"- **Sonnet 4.6 - Opus 4.7 gap = {gap_so:.3f}**: F1 std is 0."
    )
    lines.append("")
    _ = (sg_ratio, so_ratio)

    lines.append("## Caveats\n")
    lines.append(
        f"- **N={cfg['n_runs']} runs** is a small variance estimate (sample std on 3 points; "
        f"wide uncertainty on the std itself).\n"
        f"- The **N={cfg['n']}** subsample means absolute metrics differ from the full-split "
        f"paper numbers. For context, run 1 on this subsample gave "
        f"DR={fmt(run1['detection_rate'])}, FPR={fmt(run1['false_positive_rate'])}, "
        f"F1={fmt(run1['f1_hallucination'])}, MCC={fmt(run1['mcc'])}.\n"
        f"- The published gaps (Sonnet-GPT-5.1 {gap_sg:.3f}, Sonnet-Opus {gap_so:.3f}) are "
        f"full-split numbers; this experiment measures only **GPT-5.1's** stochasticity, not "
        f"the other models'. The comparison treats GPT-5.1's std as a proxy for the noise floor "
        f"on temperature=1.0 single-run F1 estimates.\n"
    )

    lines.append("## Provenance\n")
    lines.append(
        f"- Total API calls: **{d['total_api_calls']}** (cap = {cfg['n']} x {cfg['n_runs']} = "
        f"{cfg['n'] * cfg['n_runs']}).\n"
        f"- Raw per-run metrics + predictions: `run1/`, `run2/`, `run3/`.\n"
        f"- Machine-readable aggregate: `summary_data.json`.\n"
    )

    OUT_DIR.joinpath("SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
