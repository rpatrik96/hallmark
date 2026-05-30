#!/usr/bin/env python3
"""E3: GPT-5.1 run-to-run variance at temperature=1.0 (forced by the API).

Runs verify_with_openai(gpt-5.1) N=3 times on a fixed 150-entry stratified
subsample of dev_public (seed 42), each run in its own checkpoint dir so it is
an independent API draw (run1 resumes from a partial 29-entry checkpoint).
Reports per-run DR/FPR/F1/MCC, mean +/- sample std, and label-flip count, then
judges whether the published Sonnet-GPT5.1 (~0.069) and Sonnet-Opus (~0.016) F1
gaps survive the run-to-run noise.
"""

import json
import statistics
from pathlib import Path

from hallmark.baselines.llm_verifier import verify_with_openai
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate

OUT = Path("results/reviewer_experiments/e3_variance")
N_RUNS = 3
keys_meta = json.loads((OUT / "sample_keys.json").read_text())
sample_keys = keys_meta["bibtex_keys"]
key_set = set(sample_keys)

# Load dev_public and select the fixed sample, preserving the saved order.
all_entries = load_split("dev_public")
by_key = {e.bibtex_key: e for e in all_entries}
entries = [by_key[k] for k in sample_keys if k in by_key]
assert len(entries) == len(sample_keys), f"{len(entries)} != {len(sample_keys)}"
blind = [e.to_blind() for e in entries]
print(
    f"[e3] {len(entries)} entries "
    f"({sum(e.label == 'HALLUCINATED' for e in entries)} hall / "
    f"{sum(e.label == 'VALID' for e in entries)} valid)"
)

per_run = []
labels_by_run = []  # list (per run) of {key: label}
for i in range(1, N_RUNS + 1):
    ckpt = OUT / f"run{i}"
    ckpt.mkdir(parents=True, exist_ok=True)
    print(f"[e3] run {i} -> {ckpt}")
    preds = verify_with_openai(blind, model="gpt-5.1", checkpoint_dir=ckpt)
    res = evaluate(entries, preds, tool_name=f"gpt5.1_run{i}", split_name="dev_public_n150")
    row = {
        "run": i,
        "detection_rate": res.detection_rate,
        "false_positive_rate": res.false_positive_rate,
        "f1_hallucination": res.f1_hallucination,
        "mcc": res.mcc,
        "api_calls": sum(getattr(p, "api_calls", 0) or 0 for p in preds),
    }
    per_run.append(row)
    labels_by_run.append({p.bibtex_key: p.label for p in preds})
    print(
        f"[e3] run {i}: DR={row['detection_rate']:.3f} "
        f"FPR={row['false_positive_rate']:.3f} F1={row['f1_hallucination']:.3f}"
    )

# Label-flip analysis across runs.
n_flipped = 0
for k in sample_keys:
    labs = {lr.get(k) for lr in labels_by_run}
    if len(labs) > 1:
        n_flipped += 1


def mstd(field):
    vals = [r[field] for r in per_run]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


agg = {
    f: {"mean": mstd(f)[0], "std": mstd(f)[1]}
    for f in ("detection_rate", "false_positive_rate", "f1_hallucination", "mcc")
}
total_calls = sum(r["api_calls"] for r in per_run)
f1_std = agg["f1_hallucination"]["std"]

(OUT / "per_run.json").write_text(
    json.dumps(
        {
            "per_run": per_run,
            "aggregate": agg,
            "n_flipped": n_flipped,
            "n": len(entries),
            "total_api_calls": total_calls,
        },
        indent=2,
    )
)

lines = []
lines.append("# E3 — GPT-5.1 run-to-run variance (temperature=1.0)\n")
lines.append(
    f"- Sample: {len(entries)} dev_public entries "
    f"({keys_meta['n_hallucinated']} hall / {keys_meta['n_valid']} valid), seed {keys_meta['seed']}."
)
lines.append(
    f"- Runs: {N_RUNS} independent draws (temp=1.0, forced by the OpenAI API). "
    f"Total GPT-5.1 calls: {total_calls}."
)
lines.append(
    f"- Label flips across runs: {n_flipped}/{len(entries)} entries changed verdict in >=1 run "
    f"(confirms the draws are non-identical).\n"
)
lines.append("## Per-run metrics\n")
lines.append("| run | DR | FPR | F1 | MCC |")
lines.append("|----|----|----|----|----|")
for r in per_run:
    lines.append(
        f"| {r['run']} | {r['detection_rate']:.3f} | {r['false_positive_rate']:.3f} "
        f"| {r['f1_hallucination']:.3f} | {r['mcc']:.3f} |"
    )
lines.append("\n## Mean +/- sample std (N=3)\n")
for f, lbl in (
    ("detection_rate", "DR"),
    ("false_positive_rate", "FPR"),
    ("f1_hallucination", "F1"),
    ("mcc", "MCC"),
):
    lines.append(f"- {lbl}: {agg[f]['mean']:.3f} +/- {agg[f]['std']:.3f}")
lines.append("\n## Ranking-stability verdict\n")
lines.append(f"GPT-5.1 F1 run-to-run std = **{f1_std:.3f}** (on this n={len(entries)} subsample).")
lines.append(
    f"- Sonnet 4.6 - GPT-5.1 published F1 gap (~0.069): "
    f"{'SURVIVES' if 2 * f1_std < 0.069 else 'WITHIN ~2 std — at risk'} "
    f"(gap {'>' if 2 * f1_std < 0.069 else '<='} 2*std={2 * f1_std:.3f})."
)
lines.append(
    f"- Sonnet 4.6 - Opus 4.7 published F1 gap (~0.016): "
    f"{'SURVIVES' if 2 * f1_std < 0.016 else 'WITHIN ~2 std — not distinguishable from noise'} "
    f"(gap {'>' if 2 * f1_std < 0.016 else '<='} 2*std={2 * f1_std:.3f})."
)
lines.append(
    "\n*Caveat:* N=3 runs is a small variance estimate; the n=150 subsample's absolute "
    "metrics differ from the full-split paper numbers (run1 values above are the per-subsample point)."
)
(OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print(
    "[e3] DONE. total_api_calls=", total_calls, "f1_std=", round(f1_std, 4), "n_flipped=", n_flipped
)
print((OUT / "SUMMARY.md").read_text())
