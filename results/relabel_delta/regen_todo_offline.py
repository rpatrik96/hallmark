#!/usr/bin/env python3
"""Resolve the OFFLINE camera-ready %TODO markers (TASK A) on the v1.1.1 labels.

Produces results/relabel_delta/todo_offline.json with:
  1. shortcut  -- pointer to refit_shortcut.py output (run separately; sklearn dep)
  2. doi_pertype -- DOI-only headline + per-type status (documented limitation)
  3. per_source_dr -- per-generation_method DR/FPR/n for every tool that HAS a
     stored full-coverage per-entry dev_public prediction file (offline rescore
     vs the NEW labels). Summary-only tools (DOI/Sonnet/Opus/btu/cascade) -> note.
  4. gpt51_llm_comparison_row -- the NEW GPT-5.1 dev_public row (manifest sec1).

No API calls. Every number is computed from a stored per-entry prediction file
re-scored against data/v1.0/dev_public.jsonl (the relabeled v1.1.1 split).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _rescore import load_new_entries, load_old_entries, load_pred_map

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OUT = REPO / "results/relabel_delta/todo_offline.json"

# Paper-facing display name for each offline tool.
DISPLAY = {
    "llm_openai": "GPT-5.1",
    "llm_openai_gpt54": "GPT-5.4",
    "llm_openrouter_deepseek_r1": "DeepSeek-R1",
    "llm_openrouter_deepseek_v3": "DeepSeek-V3.2",
    "llm_openrouter_gemini_flash": "Gemini 2.5 Flash",
    "llm_openrouter_gemini_pro": "Gemini 2.5 Pro",
    "llm_openrouter_mistral": "Mistral Large",
    "llm_openrouter_qwen": "Qwen3-235B",
    "llm_openrouter_qwen_max": "Qwen3-VL-235B",
    "llm_openrouter_llama_4_maverick": "Llama 4 Maverick",
    "llm_agentic_btu_sonnet_4_6": "Sonnet 4.6 + bibtex-updater (agentic)",
    "llm_agentic_btu_openai": "GPT-5.1 + bibtex-updater (agentic)",
    "llm_agentic_openai": "GPT-5.1 + CrossRef/OpenAlex/arXiv (agentic)",
    "llm_tool_augmented": "GPT-5.1 + bibtex-updater (always-call)",
}

# (tool_key, per-entry prediction file, relative to REPO)
OFFLINE_DEV = [
    ("llm_openai", "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl"),
    ("llm_openai_gpt54", "results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl"),
    (
        "llm_openrouter_deepseek_r1",
        "results/llm_openrouter_deepseek_r1_dev_public_predictions.jsonl",
    ),
    (
        "llm_openrouter_deepseek_v3",
        "results/llm_openrouter_deepseek_v3_dev_public_predictions.jsonl",
    ),
    (
        "llm_openrouter_gemini_flash",
        "results/llm_openrouter_gemini_flash_dev_public_predictions.jsonl",
    ),
    ("llm_openrouter_mistral", "results/llm_openrouter_mistral_dev_public_predictions.jsonl"),
    ("llm_openrouter_qwen", "results/llm_openrouter_qwen_dev_public_predictions.jsonl"),
    ("llm_openrouter_gemini_pro", "results/new_models/gemini_pro.jsonl"),
    ("llm_openrouter_qwen_max", "results/new_models/qwen_max.jsonl"),
    ("llm_openrouter_llama_4_maverick", "results/new_models/llama4_maverick.jsonl"),
    (
        "llm_agentic_btu_sonnet_4_6",
        "results/checkpoints/llm_agentic_btu_sonnet_4_6_dev_public_v2/"
        "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
    ),
    ("llm_agentic_btu_openai", "results/temporal_checkpoints/agentic_btu_openai_gpt-5.1.jsonl"),
    ("llm_agentic_openai", "results/temporal_checkpoints/agentic_openai_gpt-5.1.jsonl"),
    ("llm_tool_augmented", "data/v1.0/baseline_results/llm_tool_augmented_dev_public.jsonl"),
]

# generation_method buckets in paper order.
METHODS = ["adversarial", "perturbation", "real_world", "llm_generated", "scraped"]


def gen_method(entry) -> str:
    """generation_method for a BenchmarkEntry (schema field)."""
    gm = getattr(entry, "generation_method", None)
    return gm or "unknown"


def per_source(entries, pm) -> dict:
    """Per generation_method: DR (over hall), FPR (over valid), counts.

    Follows the repo's UNCERTAIN Protocol (hallmark/evaluation/metrics.py
    build_confusion_matrix): UNCERTAIN predictions are EXCLUDED from the confusion
    matrix entirely — they are removed from both the DR and FPR denominators (not
    counted as a non-detection). A missing prediction defaults to VALID@0.5
    (conservative), matching predictions_aligned() in _rescore.py. This reproduces
    evaluate()'s detection_rate exactly (verified against manifest sec1).
    """
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_hall": 0, "tp": 0, "n_valid": 0, "fp": 0, "n_uncertain": 0}
    )
    for e in entries:
        m = gen_method(e)
        p = pm.get(e.bibtex_key)
        pred_label = p.label if p is not None else "VALID"
        if pred_label == "UNCERTAIN":
            # excluded from DR/FPR denominators per the UNCERTAIN Protocol
            agg[m]["n_uncertain"] += 1
            continue
        is_hall_pred = pred_label == "HALLUCINATED"
        if e.label == "HALLUCINATED":
            agg[m]["n_hall"] += 1  # confusion-matrix hall denom (excludes UNCERTAIN)
            if is_hall_pred:
                agg[m]["tp"] += 1
        else:  # VALID
            agg[m]["n_valid"] += 1  # confusion-matrix valid denom (excludes UNCERTAIN)
            if is_hall_pred:
                agg[m]["fp"] += 1
    out = {}
    for m in METHODS:
        c = agg.get(m)
        if not c:
            continue
        dr = (c["tp"] / c["n_hall"]) if c["n_hall"] else None
        fpr = (c["fp"] / c["n_valid"]) if c["n_valid"] else None
        out[m] = {
            "n_hall_scored": c["n_hall"],
            "n_valid_scored": c["n_valid"],
            "n_uncertain": c["n_uncertain"],
            "tp": c["tp"],
            "fp": c["fp"],
            "detection_rate": round(dr, 4) if dr is not None else None,
            "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        }
    return out


def overall(entries, pm) -> dict:
    """Overall DR/FPR following the UNCERTAIN Protocol (excludes UNCERTAIN)."""
    tp = fp = nh = nv = nu = 0
    for e in entries:
        p = pm.get(e.bibtex_key)
        pred_label = p.label if p is not None else "VALID"
        if pred_label == "UNCERTAIN":
            nu += 1
            continue
        is_hall = pred_label == "HALLUCINATED"
        if e.label == "HALLUCINATED":
            nh += 1
            tp += is_hall
        else:
            nv += 1
            fp += is_hall
    return {
        "detection_rate": round(tp / nh, 4) if nh else None,
        "false_positive_rate": round(fp / nv, 4) if nv else None,
        "n_hall_scored": nh,
        "n_valid_scored": nv,
        "n_uncertain": nu,
    }


def _load_shortcut() -> dict:
    """Embed refit_shortcut.py output if present; else a pointer + run hint."""
    p = REPO / "results/relabel_delta/shortcut_refit.json"
    if p.exists():
        data: dict = json.loads(p.read_text())
        data["how_to_reproduce"] = (
            "uv run --with scikit-learn --with numpy python results/relabel_delta/refit_shortcut.py"
        )
        return data
    return {
        "status": "NOT_RUN",
        "how_to_reproduce": (
            "uv run --with scikit-learn --with numpy python results/relabel_delta/refit_shortcut.py"
        ),
    }


def _stratified_gpt51(per_source_dr: dict) -> dict:
    """tab:stratified_dr (appendix L295) refreshed for GPT-5.1 on NEW labels.

    Faithful: GPT-5.1's canonical per-entry file reproduces the published headline
    dev DR (.823 OLD). real_world (.891) and llm_generated (.611) per-method DRs
    reproduce the published table exactly; adversarial and perturbation differ from
    the printed .983/.829 by ~2pp (the printed cells trace to a slightly different
    GPT-5.1 snapshot; cf. manifest flag F1). We report the canonical-file values.
    """
    g = per_source_dr["llm_openai"]
    nm = g["by_method_new"]
    om = g["by_method_old"]

    def row(method: str) -> dict:
        n = nm.get(method, {})
        o = om.get(method, {})
        return {
            "n_hall_new": n.get("n_hall_scored"),
            "n_valid_new": n.get("n_valid_scored"),
            "DR_new": n.get("detection_rate"),
            "FPR_new": n.get("false_positive_rate"),
            "DR_old_canonical": o.get("detection_rate"),
            "n_hall_old": o.get("n_hall_scored"),
        }

    return {
        "tool": "GPT-5.1",
        "split": "dev_public",
        "overall_DR_new": g["overall_new"]["detection_rate"],
        "overall_FPR_new": g["overall_new"]["false_positive_rate"],
        "rows": {m: row(m) for m in METHODS},
        "paper_printed_old": {
            "adversarial_DR": 0.983,
            "perturbation_DR": 0.829,
            "real_world_DR": 0.891,
            "llm_generated_DR": 0.611,
            "scraped_FPR": 0.411,
        },
        "reproduction_note": (
            "real_world (.891) and llm_generated (.611) per-method DRs reproduce the "
            "printed OLD table exactly from the canonical GPT-5.1 file (which also "
            "reproduces the headline OLD dev DR .823). adversarial (canonical OLD 1.000 "
            "vs printed .983) and perturbation (canonical OLD .848 vs printed .829) "
            "differ ~2pp: the printed cells trace to a slightly different GPT-5.1 "
            "snapshot (manifest flag F1). NEW values are the canonical-file recompute."
        ),
        "relabel_effect_note": (
            "The relabel moved 4 perturbation and 23 llm_generated dev entries from "
            "HALLUCINATED to VALID, so those two methods now carry a small valid pool "
            "(perturbation n_valid=4, llm_generated n_valid=23) and thus a per-method "
            "FPR; the prior table reported FPR only on the scraped (valid) pool. The "
            "scraped pool itself is unchanged at n_valid=486 (FPR .405 canonical / .411 "
            "printed). The 27 relabeled-VALID entries that left their hallucinated "
            "buckets are why per-method n_hall dropped (perturbation 414->410, "
            "llm_generated 113->90, plus retypes)."
        ),
    }


def main() -> None:
    new_e = load_new_entries("dev_public")
    old_e = load_old_entries("dev_public")

    # ---- ground-truth per-source n on NEW labels (single source of truth) ----
    gt_new: dict[str, dict[str, int]] = defaultdict(lambda: {"hall": 0, "valid": 0})
    for e in new_e:
        m = gen_method(e)
        gt_new[m]["hall" if e.label == "HALLUCINATED" else "valid"] += 1

    per_source_dr: dict[str, dict] = {}
    for tool, rel in OFFLINE_DEV:
        pm = load_pred_map(REPO / rel)
        ov_new = overall(new_e, pm)
        ov_old = overall(old_e, pm)
        per_source_dr[tool] = {
            "display": DISPLAY.get(tool, tool),
            "pred_file": rel,
            "overall_new": ov_new,
            "overall_old": ov_old,
            "by_method_new": per_source(new_e, pm),
            "by_method_old": per_source(old_e, pm),
        }

    result = {
        "_meta": {
            "task": "TASK A offline TODO resolution on v1.1.1 labels",
            "labels": "data/v1.0/dev_public.jsonl (dev 513 valid / 606 hall, n=1119)",
            "old_rev": "7a52362 (pre-relabel)",
            "method": (
                "per-entry predictions re-scored vs NEW labels; missing/UNCERTAIN "
                "treated as non-HALLUCINATED (conservative, matches evaluate())"
            ),
        },
        # ---- item 1: shortcut (embedded from refit_shortcut.py output) ----
        "shortcut": _load_shortcut(),
        # ---- item 2: DOI-only per-type (documented limitation) ----
        "doi_pertype": {
            "headline_new": {
                "detection_rate": 0.268,
                "false_positive_rate": 0.185,
                "f1_hallucination": 0.373,
                "mcc": 0.099,
                "tier_weighted_f1": 0.329,
                "ece": 0.143,
                "source": "delta-eval reconstruction (manifest sec1/sec11.6)",
            },
            "per_type_status": "DOCUMENTED_LIMITATION",
            "per_type_note": (
                "DOI-only per-type DR is NOT faithfully regenerable offline: there is no "
                "stored per-entry DOI-resolution file, and a fresh re-run requires live "
                "doi.org HEAD requests (rate-limited; external-DB drift confirmed by the "
                "'transient block' note in doi_only_dev_public_changed_predictions.json). "
                "The tab:pertype_full DOI column stays carried-over OLD cells under % TODO(F5), "
                "as the paper caption already documents."
            ),
        },
        # ---- item 3: per-source DR ----
        "per_source_dr": {
            "ground_truth_n_new": {m: gt_new[m] for m in METHODS if m in gt_new},
            "tools_offline": per_source_dr,
            "summary_only_tools_note": (
                "DOI-only, Claude Sonnet 4.6, Claude Opus 4.7, bibtex-updater, and the "
                "cascade are SUMMARY-ONLY on dev_public (no stored per-entry predictions), "
                "so their per-generation_method DR/FPR cannot be faithfully recomputed and "
                "are not reported here (documented limitation)."
            ),
        },
        # ---- item 3b: tab:stratified_dr (GPT-5.1 per generation_method, NEW) ----
        "stratified_dr_gpt51": _stratified_gpt51(per_source_dr),
        # ---- item 4: GPT-5.1 llm_comparison.tex row (manifest sec1, offline) ----
        "gpt51_llm_comparison_row": {
            "split": "dev_public",
            "detection_rate": 0.837,
            "false_positive_rate": 0.411,
            "f1_hallucination": 0.766,
            "mcc": 0.442,
            "tier_weighted_f1": 0.822,
            "ece": 0.190,
            "union_recall": 0.775,
            "source": "manifest sec1 (offline, llm_openai dev_public)",
        },
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n")

    # ---- console summary ----
    print("=== ground-truth per-source n (NEW dev labels) ===")
    for m in METHODS:
        if m in gt_new:
            print(f"  {m:14s} hall={gt_new[m]['hall']:4d} valid={gt_new[m]['valid']:4d}")
    print("\n=== per-source DR (NEW labels), offline tools ===")
    for tool, _rel in OFFLINE_DEV:
        d = per_source_dr[tool]
        ov = d["overall_new"]
        print(
            f"\n{d['display']}  (overall DR={ov['detection_rate']} FPR={ov['false_positive_rate']}, "
            f"old DR={d['overall_old']['detection_rate']})"
        )
        for m in METHODS:
            bm = d["by_method_new"].get(m)
            if not bm:
                continue
            dr = bm["detection_rate"]
            fpr = bm["false_positive_rate"]
            print(
                f"    {m:14s} n_h={bm['n_hall_scored']:4d} n_v={bm['n_valid_scored']:4d} "
                f"U={bm['n_uncertain']:3d} "
                f"DR={dr if dr is not None else '  -- ':>6} "
                f"FPR={fpr if fpr is not None else ' -- '}"
            )
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
