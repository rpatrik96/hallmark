"""Offline rescore of the GPT-5.4 zero-shot temporal probe (canonical 448).

Resolves flag F10: the paper's GPT-5.4 supplement FPR 0.413 (app:gpt54-probe,
analysis.tex L48) traced only to the aggregate figures/gpt54_probe_results.json.
The per-entry predictions DO exist
(results/gpt54_probe/llm_openai_gpt54_default_temporal_predictions.jsonl, 448 lines);
this script joins them to the ground-truth labels and reproduces FPR/DR with no
API calls, and also writes the self-contained per-entry export consumed by reviewers.

Run: python results/temporal_supplement/gpt54_448/rescore_gpt54_448.py
Expect: FPR=0.41333  DR=0.89865  (N=448, 300 VALID / 148 HALLUCINATED).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
PREDS = REPO / "results/gpt54_probe/llm_openai_gpt54_default_temporal_predictions.jsonl"
SUPP = REPO / "results/temporal_supplement/temporal_supplement_2024_2025.jsonl"


def main() -> None:
    preds: dict[str, str] = {}
    for line in PREDS.read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            preds[d["bibtex_key"]] = d["label"]

    gt: dict[str, str] = {}
    for line in SUPP.read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            k = d.get("bibtex_key") or d.get("key")
            gt[k] = d["label"]

    tp = fp = fn = tn = 0
    nv = nh = 0
    for k, pl in preds.items():
        gl = gt[k]
        if gl == "VALID":
            nv += 1
            fp += pl == "HALLUCINATED"
            tn += pl == "VALID"
        else:
            nh += 1
            tp += pl == "HALLUCINATED"
            fn += pl == "VALID"

    fpr = fp / nv
    dr = tp / nh
    print(f"N={len(preds)}  VALID={nv}  HALLUCINATED={nh}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"FPR={fpr:.5f}  DR={dr:.5f}")
    assert abs(fpr - 0.41333) < 1e-3, "FPR drifted from published 0.413"
    assert abs(dr - 0.89865) < 1e-3, "DR drifted from published 0.899"
    print("OK: reproduces paper app:gpt54-probe / analysis.tex L48 (FPR 0.413).")


if __name__ == "__main__":
    main()
