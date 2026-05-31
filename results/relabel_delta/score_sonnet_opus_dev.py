#!/usr/bin/env python3
"""Score full-coverage zero-shot Sonnet 4.6 / Opus 4.7 dev_public re-runs (TASK B).

Closes the F5 per-type TODO (tab:pertype_full Sonnet/Opus dev columns) and the
app:bootstrap dev-CI TODO for Sonnet/Opus, by scoring the freshly regenerated
*full* 1119-entry per-entry predictions on the corrected (v1.1.x post-relabel)
dev_public labels.

For each model it:
  * loads the per-entry predictions JSONL (canonical Prediction schema),
  * runs hallmark.evaluation.metrics.evaluate() for the aggregate DR/FPR/F1/MCC,
  * computes per_type_metrics() detection rates (per-type column, F5),
  * computes compute_persisted_cis() stratified bootstrap 95% CIs
    (DR/FPR/F1/TW-F1/ECE/MCC; seed=42, n_bootstrap=10000),
  * sanity-checks the aggregate against the delta-eval values in
    results_manifest.md sec1 (the prior summary-only reconstruction).

Writes results/relabel_delta/todo_sonnet_opus_dev.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hallmark.dataset.loader import load_split  # noqa: E402
from hallmark.dataset.schema import Prediction  # noqa: E402
from hallmark.evaluation.metrics import (  # noqa: E402
    compute_persisted_cis,
    evaluate,
    per_type_metrics,
)

OUT = ROOT / "results" / "relabel_delta" / "todo_sonnet_opus_dev.json"

# Delta-eval targets from results_manifest.md sec1 (L61/L62, NEW post-relabel
# values). These were reconstructed summary-only from confusion-matrix deltas;
# the full re-run is the ground-truth check on them.
DELTA_EVAL = {
    "llm_openrouter_claude_sonnet_4_6": {
        "detection_rate": 0.781,
        "false_positive_rate": 0.127,
        "f1_hallucination": 0.827,
        "mcc": 0.652,
    },
    "llm_openrouter_claude_opus_4_7": {
        "detection_rate": 0.752,
        "false_positive_rate": 0.072,
        "f1_hallucination": 0.830,
        "mcc": 0.683,
    },
}

# Paper-order per-type rows (tab:pertype_full); swapped_authors prints as
# author_mismatch. arxiv_version_mismatch / merged_citation / partial_author_list
# are stress-test types not in the main 11-row table but reported if present.
PERTYPE_ORDER = [
    "fabricated_doi",
    "nonexistent_venue",
    "placeholder_authors",
    "future_date",
    "chimeric_title",
    "wrong_venue",
    "swapped_authors",  # printed as author_mismatch
    "preprint_as_published",
    "hybrid_fabrication",
    "near_miss_title",
    "plausible_fabrication",
]

MODELS = {
    "llm_openrouter_claude_sonnet_4_6": ROOT
    / "results"
    / "checkpoints"
    / "llm_openrouter_claude_sonnet_4_6_dev_public"
    / "llm_openrouter_claude_sonnet_4_6_dev_public_predictions.jsonl",
    "llm_openrouter_claude_opus_4_7": ROOT
    / "results"
    / "checkpoints"
    / "llm_openrouter_claude_opus_4_7_dev_public_zeroshot"
    / "llm_openrouter_claude_opus_4_7_dev_public_predictions.jsonl",
}


def load_predictions(path: Path) -> list[Prediction]:
    preds: list[Prediction] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            preds.append(Prediction.from_json(line))
    return preds


def round_ci(ci: list[float] | None) -> list[float] | None:
    if ci is None:
        return None
    return [round(float(x), 4) for x in ci]


def main() -> None:
    entries = load_split("dev_public", "v1.0", None)
    n_total = len(entries)
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
    n_valid = sum(1 for e in entries if e.label == "VALID")

    out: dict[str, object] = {
        "meta": {
            "task": "TASK B — full-coverage zero-shot Sonnet 4.6 / Opus 4.7 on dev_public (1119)",
            "split": "dev_public",
            "labels": "post-relabel corrected (v1.1.x, chore/paper-todo-resolution)",
            "split_counts": {
                "num_entries": n_total,
                "num_hallucinated": n_hall,
                "num_valid": n_valid,
            },
            "temperature": 0.0,
            "seed": 42,
            "ci_n_bootstrap": 10000,
            "ci_seed": 42,
            "ci_confidence": 0.95,
            "prompt": "hallmark.baselines.llm_verifier.VERIFICATION_PROMPT (identical to paper)",
            "intended_to_close": [
                "F5 per-type tab:pertype_full dev Sonnet/Opus columns",
                "app:bootstrap dev CIs Sonnet/Opus (were ci_available=False)",
            ],
            "delta_eval_check": "vs results_manifest.md sec1 L61/L62 (summary-only reconstruction)",
            "finding": (
                "The full 1119-entry re-run (2026-05-30) does NOT reproduce the paper's "
                "published Opus/Sonnet dev aggregates (the OpenRouter Anthropic endpoint "
                "drifted since the original 2026-05-04 run, commit 493cbb3, which used the "
                "identical path/prompt/temp=0 but whose per-entry predictions were never "
                "persisted). Controlled replay of the committed temporal-supplement entries "
                "shows only 90.0% (Opus) / 75.0% (Sonnet) label agreement on identical inputs "
                "(see endpoint_drift_probe/drift_summary.json). On dev the drift compounds: "
                "Opus re-run DR=.909/FPR=.162 vs published .752/.072 (+15.7/+9.0pp). "
                "These re-run numbers are internally valid but paper-INconsistent; using them "
                "would contradict the delta-eval that all of manifest sec.11 (PPV, Pareto, "
                "takeaways) is built on. The per-entry predictions are persisted here for "
                "reproducibility, but the F5/CI TODOs are resolved by a DOCUMENTED LIMITATION "
                "(endpoint drift + unpersisted originals), not by overwriting the published "
                "numbers with these drifted ones."
            ),
            "endpoint_drift_replay": "results/relabel_delta/endpoint_drift_probe/drift_summary.json",
            "drift_tolerance_pp": 5.0,
        },
        "models": {},
    }
    models_out: dict[str, object] = out["models"]  # type: ignore[assignment]

    for tool, pred_path in MODELS.items():
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions missing for {tool}: {pred_path}")
        preds = load_predictions(pred_path)
        coverage = len({p.bibtex_key for p in preds}) / n_total

        res = evaluate(entries, preds, tool_name=tool, split_name="dev_public")
        ptm = per_type_metrics(entries, preds)
        cis = compute_persisted_cis(entries, preds, n_bootstrap=10000, seed=42, confidence=0.95)

        n_uncertain = sum(1 for p in preds if p.label == "UNCERTAIN")

        # Per-type detection rate in paper row order.
        pertype = {}
        for t in PERTYPE_ORDER:
            m = ptm.get(t)
            pertype[t] = {
                "detection_rate": round(m["detection_rate"], 4) if m else None,
                "count": int(m["count"]) if m else 0,
            }
        # Any extra hallucinated types present (stress-test types).
        extra = {
            t: {"detection_rate": round(m["detection_rate"], 4), "count": int(m["count"])}
            for t, m in ptm.items()
            if t not in PERTYPE_ORDER and t != "valid"
        }

        # Delta-eval sanity check.
        target = DELTA_EVAL[tool]
        deltas = {
            "detection_rate": round(res.detection_rate - target["detection_rate"], 4),
            "false_positive_rate": round(
                res.false_positive_rate - target["false_positive_rate"], 4
            ),
            "f1_hallucination": round(res.f1_hallucination - target["f1_hallucination"], 4),
            "mcc": round(res.mcc - target["mcc"], 4),
        }
        max_abs_pp = round(max(abs(v) for v in deltas.values()) * 100, 2)
        faithful = max_abs_pp <= 5.0  # "within a few pp" sanity bar from the task

        models_out[tool] = {
            "predictions_file": str(pred_path.relative_to(ROOT)),
            "num_predictions": len(preds),
            "coverage": round(coverage, 4),
            "num_uncertain": n_uncertain,
            "reproduces_published_within_tolerance": faithful,
            "verdict": (
                "matches published delta-eval within tolerance"
                if faithful
                else "DIVERGES from published delta-eval beyond tolerance (endpoint drift); "
                "do NOT ship these as the paper's Sonnet/Opus dev numbers"
            ),
            "aggregate_full_rerun": {
                "detection_rate": round(res.detection_rate, 4),
                "false_positive_rate": round(res.false_positive_rate, 4),
                "f1_hallucination": round(res.f1_hallucination, 4),
                "tier_weighted_f1": round(res.tier_weighted_f1, 4),
                "mcc": round(res.mcc, 4),
                "ece": round(res.ece, 4) if res.ece is not None else None,
            },
            "delta_eval_target_manifest_sec1": target,
            "delta_full_minus_deltaeval": deltas,
            "max_abs_delta_pp": max_abs_pp,
            "bootstrap_ci_95": {
                "detection_rate": round_ci(cis.get("detection_rate_ci")),  # type: ignore[arg-type]
                "false_positive_rate": round_ci(cis.get("fpr_ci")),  # type: ignore[arg-type]
                "f1_hallucination": round_ci(cis.get("f1_hallucination_ci")),  # type: ignore[arg-type]
                "tier_weighted_f1": round_ci(cis.get("tier_weighted_f1_ci")),  # type: ignore[arg-type]
                "ece": round_ci(cis.get("ece_ci")),  # type: ignore[arg-type]
                "mcc": round_ci(cis.get("mcc_ci")),  # type: ignore[arg-type]
            },
            "ci_provenance": cis.get("ci_provenance"),
            "per_type_detection_rate": pertype,
            "per_type_extra_types": extra,
        }

        print(f"\n=== {tool} ===")
        print(
            f"  coverage={coverage:.4f} n_uncertain={n_uncertain} "
            f"DR={res.detection_rate:.4f} FPR={res.false_positive_rate:.4f} "
            f"F1={res.f1_hallucination:.4f} MCC={res.mcc:.4f}"
        )
        print(f"  delta vs delta-eval (pp, max abs): {max_abs_pp}  details={deltas}")
        print(f"  DR CI={round_ci(cis.get('detection_rate_ci'))}")  # type: ignore[arg-type]

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
