#!/usr/bin/env python3
"""Regenerate per-generation-method DR/FPR for tab:llm_gen_method (post-relabel).

Closes the F5 TODO in hallmark-paper/tables/llm_comparison.tex for the two models
that HAVE per-entry predictions on dev_public: Qwen3-235B and DeepSeek-V3.2.
(GPT-5.1 has only evaluation-level predictions on dev_public — no per-entry file
exists — so its per-source column cannot be regenerated and is documented, not
fabricated.)

Faithful by construction:
  * Groups the *relabeled* dev_public split strictly by ``generation_method``.
  * Uses the canonical ``build_confusion_matrix`` (same UNCERTAIN protocol and
    DR/FPR definitions as the main evaluation harness and as the original
    ``scripts/analyze_by_generation_method.py``).
  * DR for hallucinated-source rows = TP/(TP+FN) over that group's HALLUCINATED
    entries only; FPR for the scraped (valid) row = FP/(FP+TN) over valids.

The relabel moved recovered real papers HALL->VALID; some landed inside the
``perturbation`` and ``llm_generated`` generation methods, so those groups now
contain a handful of VALID entries. ``build_confusion_matrix`` handles this
correctly (valids contribute to FP/TN, never to the DR denominator).

Run:
    uv run python results/relabel_delta/regen_gen_method.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import BenchmarkEntry, Prediction
from hallmark.evaluation.metrics import build_confusion_matrix

REPO = Path(__file__).resolve().parents[2]
PRED_FILES = {
    "Qwen3-235B": REPO / "results" / "llm_openrouter_qwen_dev_public_predictions.jsonl",
    "DeepSeek-V3.2": REPO / "results" / "llm_openrouter_deepseek_v3_dev_public_predictions.jsonl",
}
# Display label -> generation_method value
ROWS = [
    ("Adversarial", "adversarial"),
    ("Perturbation", "perturbation"),
    ("Real-world", "real_world"),
    ("LLM-generated", "llm_generated"),
    ("Scraped (FPR)", "scraped"),
]


def load_predictions(path: Path) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec["bibtex_key"]
            # first-wins dedup (matches the rest of the pipeline)
            if key in preds:
                continue
            preds[key] = Prediction(
                bibtex_key=key,
                label=rec["label"],
                confidence=float(rec.get("confidence", 0.5)),
            )
    return preds


def main() -> None:
    entries = load_split("dev_public", version="v1.0")
    by_method: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for e in entries:
        by_method[e.generation_method].append(e)

    # The "Scraped (FPR)" row reports FPR over the FULL valid pool (all VALID
    # entries, n=513), matching the main-results dev_public FPR (manifest §1).
    # This keeps the table's FPR row consistent with tab:results rather than
    # measuring FPR over only generation_method=="scraped" valids (the relabel
    # moved 27 recovered-valid entries into the perturbation/llm_generated groups).
    valid_pool = [e for e in entries if e.label == "VALID"]

    out: dict[str, dict] = {}
    for model, path in PRED_FILES.items():
        preds = load_predictions(path)
        out[model] = {}
        for label, method in ROWS:
            is_valid_row = method == "scraped"
            group = valid_pool if is_valid_row else by_method.get(method, [])
            cm = build_confusion_matrix(group, preds)
            n_hall = cm.tp + cm.fn
            n_valid = cm.fp + cm.tn
            out[model][label] = {
                "generation_method": method,
                "group_basis": "all_valid_entries" if is_valid_row else method,
                "n_group": len(group),
                "n_hallucinated": n_hall,
                "n_valid": n_valid,
                "detection_rate": round(cm.detection_rate, 4),
                "false_positive_rate": round(cm.false_positive_rate, 4),
                "metric_reported": "FPR" if is_valid_row else "DR",
                "value": round(cm.false_positive_rate if is_valid_row else cm.detection_rate, 4),
                "tp": cm.tp,
                "fp": cm.fp,
                "tn": cm.tn,
                "fn": cm.fn,
            }

    print(json.dumps(out, indent=2))

    # Dataset-level n per row (independent of per-model UNCERTAIN exclusion):
    # hallucinated count within the generation method for DR rows; full valid
    # pool size for the FPR row.
    n_dataset = {
        label: (
            len(valid_pool)
            if method == "scraped"
            else sum(1 for e in by_method.get(method, []) if e.label == "HALLUCINATED")
        )
        for label, method in ROWS
    }

    print("\n# Paper-ready (3dp) tab:llm_gen_method values (post-relabel)\n")
    header = f"{'Row':<16}{'n':>6}  {'Qwen3-235B':>12}  {'DeepSeek-V3.2':>14}"
    print(header)
    for label, _method in ROWS:
        q = out["Qwen3-235B"][label]
        d = out["DeepSeek-V3.2"][label]
        print(f"{label:<16}{n_dataset[label]:>6}  {q['value']:>12.3f}  {d['value']:>14.3f}")
        out["Qwen3-235B"][label]["n_dataset"] = n_dataset[label]
        out["DeepSeek-V3.2"][label]["n_dataset"] = n_dataset[label]

    dst = Path(__file__).resolve().parent / "gen_method_postrelabel.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {dst}")


if __name__ == "__main__":
    main()
