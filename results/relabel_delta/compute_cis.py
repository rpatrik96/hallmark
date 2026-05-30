"""Stratified bootstrap CIs + paired significance for the relabel delta.

For every tool that has a FULL-coverage per-entry prediction file we compute
real stratified 95% bootstrap CIs (DR, FPR, F1, TW-F1, MCC) via
``compute_persisted_cis`` against the NEW (post-relabel) labels, plus paired
bootstrap significance (``paired_significance``) for the headline pairs where
*both* tools have per-entry predictions on the *same* split.

For SUMMARY-ONLY tools (no per-entry predictions; the released numbers prove
they were reconstructed from confusion-matrix deltas, not stored per entry) we
emit the point estimate from the regenerated aggregate and mark every CI
not-available with a machine-readable reason — per the delta-eval decision we do
NOT re-run the tool to obtain CIs.

Deterministic: seed=42, n_bootstrap=10_000, 95% CIs (matches compute_persisted_cis
defaults). No API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
sys.path.insert(0, str(REPO / "results/relabel_delta"))

from _rescore import load_pred_map, predictions_aligned  # noqa: E402
from regenerate_offline import JOBS  # noqa: E402

from hallmark.dataset.schema import (  # noqa: E402
    BenchmarkEntry,
    Prediction,
    is_canary_entry,
)
from hallmark.evaluation.metrics import (  # noqa: E402
    _metric_f1,
    build_confusion_matrix,
    compute_persisted_cis,
    paired_bootstrap_test,
    tier_weighted_f1,
)

OUT = REPO / "results/relabel_delta"
BR = REPO / "data/v1.0/baseline_results"
N_BOOT = 10_000
SEED = 42

# ---------------------------------------------------------------------------
# Tool -> per-entry prediction path, per split. Derived from
# regenerate_offline.JOBS (the canonical, verified set of full-coverage
# per-entry prediction files that reproduce the published numbers against the
# pre-relabel labels). These get REAL bootstrap CIs. Deriving from JOBS avoids
# duplicating (and drifting from) the long checkpoint paths.
# ---------------------------------------------------------------------------
PER_ENTRY: dict[str, dict[str, str]] = {"dev_public": {}, "test_public": {}}
for _fname, _pp, _sp, _tool, _pub, _mode in JOBS:
    # JOBS tool names use "gpt-5.4"; AGG/point-estimate keys keep the same name.
    PER_ENTRY.setdefault(_sp, {})[_tool] = _pp

# Map tool-name -> the released aggregate JSON (for the point estimate of
# summary-only tools, and as a cross-check for per-entry tools).
AGG_FILE: dict[str, dict[str, str]] = {
    "dev_public": {
        "bibtexupdater": "bibtexupdater_dev_public.json",
        "doi_only": "doi_only_dev_public.json",
        "llm_openrouter_claude_sonnet_4_6": "llm_openrouter_claude_sonnet_4_6_dev_public.json",
        "llm_openrouter_claude_opus_4_7": "llm_openrouter_claude_opus_4_7_dev_public.json",
    },
    "test_public": {
        "bibtexupdater": "bibtexupdater_test_public.json",
        "doi_only": "doi_only_test_public.json",
    },
}

# Summary-only tools per split (no full per-entry predictions -> CI not-available).
SUMMARY_ONLY: dict[str, list[str]] = {
    "dev_public": [
        "bibtexupdater",
        "doi_only",
        "llm_openrouter_claude_sonnet_4_6",
        "llm_openrouter_claude_opus_4_7",
    ],
    "test_public": ["bibtexupdater", "doi_only"],
}

SUMMARY_REASON = (
    "summary-only tool: no full-coverage per-entry prediction file is stored "
    "(released aggregate was reconstructed from confusion-matrix deltas on the "
    "~27 relabel keys, not from per-entry predictions). Per the delta-eval "
    "decision the tool is NOT re-run to obtain CIs; point estimate only."
)


def load_new_entries(split: str) -> list[BenchmarkEntry]:
    ents = [
        BenchmarkEntry.from_json(line)
        for line in (REPO / f"data/v1.0/{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return [e for e in ents if not is_canary_entry(e)]


def point_estimates(entries: list[BenchmarkEntry], preds) -> dict[str, float | None]:
    cm = build_confusion_matrix(entries, preds)
    num_valid = sum(1 for e in entries if e.label == "VALID")
    return {
        "detection_rate": cm.detection_rate,
        "false_positive_rate": cm.false_positive_rate if num_valid > 0 else None,
        "f1_hallucination": cm.f1,
        "tier_weighted_f1": tier_weighted_f1(entries, preds),
        "mcc": cm.mcc,
    }


def agg_point(split: str, tool: str) -> dict[str, float | None]:
    f = AGG_FILE.get(split, {}).get(tool)
    if not f or not (BR / f).exists():
        return {}
    d = json.loads((BR / f).read_text())
    return {
        "detection_rate": d.get("detection_rate"),
        "false_positive_rate": d.get("false_positive_rate"),
        "f1_hallucination": d.get("f1_hallucination"),
        "tier_weighted_f1": d.get("tier_weighted_f1"),
        "mcc": d.get("mcc"),
    }


def main() -> None:
    split_counts: dict[str, dict[str, int]] = {}
    tools_out: dict[str, dict[str, dict[str, object]]] = {}
    paired_out: dict[str, dict[str, object]] = {}

    # split -> tool -> aligned per-entry predictions (only for tools with preds)
    per_entry_preds: dict[str, dict[str, list[Prediction]]] = {}
    for split in ("dev_public", "test_public"):
        entries = load_new_entries(split)
        split_counts[split] = {
            "num_entries": len(entries),
            "num_valid": sum(1 for e in entries if e.label == "VALID"),
            "num_hallucinated": sum(1 for e in entries if e.label == "HALLUCINATED"),
        }
        per_entry_preds[split] = {}
        tools_out[split] = {}

        # ---- tools WITH full per-entry predictions: real CIs ----
        for tool, rel in PER_ENTRY[split].items():
            pm = load_pred_map(REPO / rel)
            aligned = predictions_aligned(entries, pm)
            per_entry_preds[split][tool] = aligned
            cis = compute_persisted_cis(
                entries, aligned, n_bootstrap=N_BOOT, seed=SEED, confidence=0.95
            )
            tools_out[split][tool] = {
                "ci_available": True,
                "point_estimate": point_estimates(entries, aligned),
                "ci": {
                    "detection_rate": cis["detection_rate_ci"],
                    "false_positive_rate": cis["fpr_ci"],
                    "f1_hallucination": cis["f1_hallucination_ci"],
                    "tier_weighted_f1": cis["tier_weighted_f1_ci"],
                    "mcc": cis["mcc_ci"],
                },
                "ci_provenance": cis["ci_provenance"],
            }

        # ---- summary-only tools: point estimate, CI not-available ----
        for tool in SUMMARY_ONLY[split]:
            pts = agg_point(split, tool)
            tools_out[split][tool] = {
                "ci_available": False,
                "point_estimate": pts,
                "ci": {
                    "detection_rate": None,
                    "false_positive_rate": None,
                    "f1_hallucination": None,
                    "tier_weighted_f1": None,
                    "mcc": None,
                },
                "ci_provenance": {
                    "computed": False,
                    "reason": SUMMARY_REASON,
                    "seed": SEED,
                    "n_bootstrap": N_BOOT,
                    "confidence": 0.95,
                },
            }

    # ---- paired significance for headline pairs (F1) ----
    # Sonnet vs Opus: both have full per-entry preds ONLY on test_public.
    # bibtex-updater vs Sonnet: bibtex-updater is summary-only on both splits ->
    #   no paired test possible (one member lacks per-entry preds).
    headline = [
        (
            "sonnet_vs_opus_f1",
            "test_public",
            "llm_openrouter_claude_sonnet_4_6",
            "llm_openrouter_claude_opus_4_7",
        ),
        (
            "sonnet_vs_opus_f1",
            "dev_public",
            "llm_openrouter_claude_sonnet_4_6",
            "llm_openrouter_claude_opus_4_7",
        ),
        (
            "bibtexupdater_vs_sonnet_f1",
            "dev_public",
            "bibtexupdater",
            "llm_openrouter_claude_sonnet_4_6",
        ),
        (
            "bibtexupdater_vs_sonnet_f1",
            "test_public",
            "bibtexupdater",
            "llm_openrouter_claude_sonnet_4_6",
        ),
    ]
    for name, split, a, b in headline:
        entries = load_new_entries(split)
        preds = per_entry_preds[split]
        have_a = a in preds
        have_b = b in preds
        key = f"{name}__{split}"
        if have_a and have_b:
            # Orient A := the higher-F1 tool so the directional (one-sided) test
            # matches the observed ordering and the conservative one-sided->two-
            # sided doubling is not inflated to 1.0 by an alphabetical accident.
            f1_a = _metric_f1(entries, preds[a])
            f1_b = _metric_f1(entries, preds[b])
            hi, lo = (a, b) if f1_a >= f1_b else (b, a)
            diff_one, p_one, cohens_h = paired_bootstrap_test(
                entries,
                preds[hi],
                preds[lo],
                _metric_f1,
                n_bootstrap=N_BOOT,
                seed=SEED,
                two_sided=False,
            )
            p_two = min(1.0, 2.0 * p_one)
            paired_out[key] = {
                "available": True,
                "pair": f"{hi}_vs_{lo}",
                "higher_f1_tool": hi,
                "lower_f1_tool": lo,
                "metric": "f1_hallucination",
                "observed_diff": diff_one,  # higher - lower (>= 0 by construction)
                "p_value_one_sided": p_one,  # H0: F1(higher) <= F1(lower)
                "p_value_two_sided": p_two,  # conservative (one-sided * 2, capped)
                "cohens_h": cohens_h,
                "significant_at_0.05_two_sided": bool(p_two < 0.05),
                "n_bootstrap": N_BOOT,
                "seed": SEED,
                "note": (
                    "Stratified paired bootstrap on per-entry predictions (NEW "
                    "labels). p-value is conservative (raw-null, not null-centred). "
                    "A oriented as the higher-F1 tool; two-sided is one-sided*2."
                ),
            }
        else:
            missing = [t for t, ok in ((a, have_a), (b, have_b)) if not ok]
            paired_out[key] = {
                "available": False,
                "reason": (
                    f"paired bootstrap test requires per-entry predictions for BOTH "
                    f"tools on {split}; missing per-entry predictions for: {missing}. "
                    f"These are summary-only on this split (delta-eval decision: not "
                    f"re-run). Use point-estimate language for this comparison."
                ),
                "metric": "f1_hallucination",
            }

    out: dict[str, object] = {
        "meta": {
            "description": (
                "Stratified 95% bootstrap CIs (DR/FPR/F1/TW-F1/MCC) + paired "
                "bootstrap significance for headline tool pairs, computed against "
                "the post-relabel labels via compute_persisted_cis() and "
                "paired_bootstrap_test()."
            ),
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "confidence": 0.95,
            "labels": "post-relabel (fix/dev-public-mislabel-audit)",
            "split_counts": split_counts,
        },
        "tools": tools_out,
        "paired_significance": paired_out,
    }

    path = OUT / "cis.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {path}")
    # console summary
    for split in ("dev_public", "test_public"):
        print(f"\n=== {split} ===")
        for tool, blk in tools_out[split].items():
            pe = blk["point_estimate"]
            ci = blk["ci"]
            f1ci = ci["f1_hallucination"] if isinstance(ci, dict) else None
            tag = "CI" if blk["ci_available"] else "NA"
            f1 = pe.get("f1_hallucination") if isinstance(pe, dict) else None
            ci_s = f"[{f1ci[0]:.3f},{f1ci[1]:.3f}]" if f1ci else "       n/a       "
            f1s = f"{f1:.3f}" if f1 is not None else "  n/a"
            print(f"  {tool:36s} F1={f1s} {ci_s} ({tag})")
    print("\n=== paired ===")
    for k, v in paired_out.items():
        if v["available"]:
            print(
                f"  {k:42s} diff={v['observed_diff']:+.4f} "
                f"p={v['p_value_two_sided']:.4f} h={v['cohens_h']:+.4f}"
            )
        else:
            print(f"  {k:42s} NOT AVAILABLE")


if __name__ == "__main__":
    main()
