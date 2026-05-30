"""Stage-3c regeneration: per-type tables, PPV sweep, cascade T3-F1, UNCERTAIN
counts, and summary-only MCC — all on the NEW (post-relabel) labels, offline.

This consolidates the remaining open-flag work (F4/F5/F7/F9) into one script that
reads the already-regenerated aggregates in ``data/v1.0/baseline_results/`` (which
were rescored against the NEW labels by ``regenerate_offline.py`` /
``regenerate_summary.py`` / ``regenerate_cascade.py``) and derives the paper-facing
numbers from them. No API calls, no re-runs.

Outputs are printed and also written to ``results/relabel_delta/stage3c.json`` for
the manifest append step.

Sections
--------
1. Per-type detection-rate table (tab:pertype_full): every OFFLINE tool's
   ``per_type_metrics`` against NEW labels. DOI/Sonnet/Opus on dev are summary-only
   -> flagged not-recomputable (F5).
2. PPV / base-rate sweep: pure arithmetic PPV = DR*pi / (DR*pi + FPR*(1-pi)) at
   pi in {0.01..0.05} from each tool's NEW (DR, FPR). Re-derives the FPR ordering.
3. Cascade T3-F1 (faithful): from the reconstructed per_tier_metrics["3"]["f1"].
   AUROC is NOT recomputable summary-only (F7) -> flagged.
4. UNCERTAIN counts per offline tool (``num_uncertain``) on NEW labels; DeepSeek-R1
   highlighted.
5. MCC from the NEW confusion matrix for bibtex-updater + doi_only (F4): the dev
   aggregate JSONs store mcc=null; we read the reconstructed value from the
   confusion-matrix reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
BR = REPO / "data/v1.0/baseline_results"
OUT = REPO / "results/relabel_delta/stage3c.json"

# Paper column order + display abbreviations for tab:pertype_full.
PERTYPE_TOOLS = [
    ("doi_only", "DOI", "summary"),
    ("llm_openai", "G51", "offline"),
    ("llm_openai_gpt-5.4", "G54", "offline"),
    ("llm_openrouter_claude_sonnet_4_6", "S4.6", "summary"),
    ("llm_openrouter_claude_opus_4_7", "O4.7", "summary"),
    ("llm_openrouter_deepseek_r1", "R1", "offline"),
    ("llm_openrouter_deepseek_v3", "V3", "offline"),
    ("llm_openrouter_qwen", "Q3", "offline"),
    ("llm_openrouter_mistral", "ML", "offline"),
    ("llm_openrouter_gemini_flash", "GF", "offline"),
    ("llm_openrouter_llama_4_maverick", "L4", "offline"),
    ("llm_openrouter_gemini_pro", "GP", "offline"),
    ("llm_openrouter_qwen_max", "QV", "offline"),
]

# Filenames in baseline_results for the dev aggregate per tool.
DEV_FILE = {
    "doi_only": "doi_only_dev_public.json",
    "llm_openai": "llm_openai_dev_public.json",
    "llm_openai_gpt-5.4": "llm_openai_gpt54_dev_public.json",
    "llm_openrouter_claude_sonnet_4_6": "llm_openrouter_claude_sonnet_4_6_dev_public.json",
    "llm_openrouter_claude_opus_4_7": "llm_openrouter_claude_opus_4_7_dev_public.json",
    "llm_openrouter_deepseek_r1": "llm_openrouter_deepseek_r1_dev_public.json",
    "llm_openrouter_deepseek_v3": "llm_openrouter_deepseek_v3_dev_public.json",
    "llm_openrouter_qwen": "llm_openrouter_qwen_dev_public.json",
    "llm_openrouter_mistral": "llm_openrouter_mistral_dev_public.json",
    "llm_openrouter_gemini_flash": "llm_openrouter_gemini_flash_dev_public.json",
    "llm_openrouter_llama_4_maverick": "llm_openrouter_llama_4_maverick_dev_public.json",
    "llm_openrouter_gemini_pro": "llm_openrouter_gemini_pro_dev_public.json",
    "llm_openrouter_qwen_max": "llm_openrouter_qwen_max_dev_public.json",
}

# Paper row order for tab:pertype_full (11 main types; enum value swapped_authors
# is printed as author_mismatch).
PERTYPE_ROWS = [
    ("1", "fabricated_doi"),
    ("1", "nonexistent_venue"),
    ("1", "placeholder_authors"),
    ("1", "future_date"),
    ("2", "chimeric_title"),
    ("2", "wrong_venue"),
    ("2", "swapped_authors"),  # printed as author_mismatch
    ("2", "preprint_as_published"),
    ("2", "hybrid_fabrication"),
    ("3", "near_miss_title"),
    ("3", "plausible_fabrication"),
]


def load(fname: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((BR / fname).read_text())
    return data


def ppv(dr: float, fpr: float, pi: float) -> float:
    num = dr * pi
    den = dr * pi + fpr * (1.0 - pi)
    return num / den if den > 0 else 0.0


def section_pertype() -> dict[str, Any]:
    """Per-type detection rate (NEW labels) for OFFLINE tools."""
    cols: dict[str, Any] = {}
    for tool, _abbr, kind in PERTYPE_TOOLS:
        d = load(DEV_FILE[tool])
        ptm = d.get("per_type_metrics", {})
        cols[tool] = {
            "kind": kind,
            "per_type_dr": {
                t: round(ptm.get(t, {}).get("detection_rate", 0.0), 3) for _tier, t in PERTYPE_ROWS
            },
            "per_type_count": {t: ptm.get(t, {}).get("count") for _tier, t in PERTYPE_ROWS},
        }
    return cols


def section_codesign_pertype() -> dict[str, Any]:
    """bibtex-updater per-type (tab:codesign_pertype). Summary-only on dev: per-type
    hallucinated buckets are NOT safely decremented -> flag F5. We surface the
    published-carried per_type_metrics (only the 'valid' row FPR was refreshed)."""
    d = load("bibtexupdater_dev_public.json")
    ptm = d.get("per_type_metrics", {})
    return {
        "kind": "summary",
        "per_type_dr": {
            t: (round(ptm[t]["detection_rate"], 3) if t in ptm else None)
            for _tier, t in PERTYPE_ROWS
        },
        "note": (
            "summary-only on dev: per-type hallucinated buckets carried from "
            "published (moved keys are real papers wrongly typed; buckets not "
            "safely decremented). FLAG F5 — treat as approximately unchanged."
        ),
    }


def section_ppv() -> dict[str, Any]:
    """PPV sweep at pi in 0.01..0.05 from NEW (DR, FPR). Re-derive FPR ordering."""
    # Independent zero-shot cohort + DOI + co-designed (the sec:ppv / tab:ppv_sweep set).
    ppv_tools = [
        ("Gemini 2.5 Pro", "llm_openrouter_gemini_pro_dev_public.json"),
        ("Claude Opus 4.7", "llm_openrouter_claude_opus_4_7_dev_public.json"),
        ("Gemini 2.5 Flash", "llm_openrouter_gemini_flash_dev_public.json"),
        ("Claude Sonnet 4.6", "llm_openrouter_claude_sonnet_4_6_dev_public.json"),
        ("Llama 4 Maverick", "llm_openrouter_llama_4_maverick_dev_public.json"),
        ("GPT-5.4", "llm_openai_gpt54_dev_public.json"),
        ("Mistral Large", "llm_openrouter_mistral_dev_public.json"),
        ("GPT-5.1", "llm_openai_dev_public.json"),
        ("Qwen3-235B", "llm_openrouter_qwen_dev_public.json"),
        ("Qwen3-VL-235B", "llm_openrouter_qwen_max_dev_public.json"),
        ("DeepSeek-R1", "llm_openrouter_deepseek_r1_dev_public.json"),
        ("DeepSeek-V3.2", "llm_openrouter_deepseek_v3_dev_public.json"),
        ("DOI-only", "doi_only_dev_public.json"),
        ("bibtex-updater", "bibtexupdater_dev_public.json"),
    ]
    pis = [0.01, 0.02, 0.03, 0.04, 0.05]
    rows: list[dict[str, Any]] = []
    for name, fname in ppv_tools:
        d = load(fname)
        dr = d["detection_rate"]
        fpr = d.get("false_positive_rate") or 0.0
        rows.append(
            {
                "tool": name,
                "dr": round(dr, 3),
                "fpr": round(fpr, 3),
                "ppv": {f"{p:.2f}": round(ppv(dr, fpr, p), 4) for p in pis},
                "ppv_pct_at_2pct": round(100 * ppv(dr, fpr, 0.02), 1),
            }
        )
    # FPR-ascending ordering (the sweep ordering the paper claims).
    fpr_order = sorted(rows, key=lambda r: r["fpr"])
    # PPV-descending at pi=0.02 (PPV ordering claim L12).
    ppv_order = sorted(rows, key=lambda r: -r["ppv"]["0.02"])
    return {
        "rows": rows,
        "fpr_ascending_order": [(r["tool"], r["fpr"]) for r in fpr_order],
        "ppv_at_2pct_descending_order": [(r["tool"], r["ppv_pct_at_2pct"]) for r in ppv_order],
    }


def section_cascade_t3() -> dict[str, Any]:
    """Faithful cascade T3-F1 from reconstructed per_tier_metrics['3']['f1'].
    AUROC NOT recomputable summary-only (F7)."""
    out: dict[str, Any] = {}
    variants = [
        ("dev cons", "cascade_db_diagnosis_dev_public.json"),
        ("test cons", "cascade_db_diagnosis_test_public.json"),
        ("dev agg", "cascade_db_diagnosis_aggressive_dev_public.json"),
        ("test agg", "cascade_db_diagnosis_aggressive_test_public.json"),
    ]
    for label, fname in variants:
        d = load(fname)
        t3 = d.get("per_tier_metrics", {}).get("3", {})
        out[label] = {
            "t3_f1_faithful": round(t3.get("f1", 0.0), 3),
            "t3_dr": round(t3.get("detection_rate", 0.0), 3),
            "t3_precision": round(t3.get("precision", 0.0), 3),
            "t3_num_hallucinated": t3.get("num_hallucinated"),
            "stale_field_tier3_f1": d.get("tier3_f1"),
            "stale_field_auroc": d.get("auroc"),
            "auroc_recomputable": False,
        }
    return out


def section_uncertain() -> dict[str, Any]:
    """num_uncertain per offline tool on NEW labels (dev + test). DeepSeek-R1 first."""
    files = sorted(BR.glob("*_dev_public.json")) + sorted(BR.glob("*_test_public.json"))
    out: dict[str, Any] = {}
    for f in files:
        if "cascade" in f.name or "_ci." in f.name:
            continue
        d = json.loads(f.read_text())
        tool = d.get("tool_name")
        split = d.get("split_name")
        if not tool or split not in ("dev_public", "test_public"):
            continue
        n = d.get("num_uncertain")
        ne = d.get("num_entries")
        out.setdefault(tool, {})[split] = {
            "num_uncertain": n,
            "num_entries": ne,
            "pct": round(100 * n / ne, 1) if (n is not None and ne) else None,
        }
    return out


def section_mcc() -> dict[str, Any]:
    """MCC from the NEW confusion matrix for the summary-only co-designed/DB tools.

    The dev aggregate JSONs store mcc=null (OLD template had null). The
    reconstruction computes MCC exactly; we re-derive it here from the same matrix
    arithmetic the reconstruction used so the value is traceable in this file.
    """
    import sys

    sys.path.insert(0, str(REPO / "results/relabel_delta"))
    from _reconstruct import reconstruct
    from regenerate_summary import (
        load_old_full,
        load_template,
        load_verdicts,
        moved_keys,
    )

    jobs = [
        (
            "bibtexupdater",
            "dev_public",
            "data/v1.0/baseline_results/bibtexupdater_dev_public.json",
            "bibtexupdater_dev_public_changed_predictions.json",
        ),
        (
            "bibtexupdater",
            "test_public",
            "data/v1.0/baseline_results/bibtexupdater_test_public.json",
            "bibtexupdater_test_public_changed_predictions.json",
        ),
        (
            "doi_only",
            "dev_public",
            "results/doi_only_dev_public.json",
            "doi_only_dev_public_changed_predictions.json",
        ),
        (
            "doi_only",
            "test_public",
            "data/v1.0/baseline_results/doi_only_test_public.json",
            "doi_only_test_public_changed_predictions.json",
        ),
    ]
    out: dict[str, Any] = {}
    for tool, split, pub_rel, vf in jobs:
        agg = load_template(pub_rel)
        mv = load_verdicts(vf)
        old_full = load_old_full(split)
        mt = {k: (old_full[k].get("difficulty_tier") or 1) for k in moved_keys(split)}
        rec = reconstruct(agg, mv, mt)
        out[f"{tool}/{split}"] = {
            "TP": rec["TP"],
            "FP": rec["FP"],
            "FN": rec["FN"],
            "TN": rec["TN"],
            "mcc_new": round(rec["mcc"], 3),
            "mcc_old_stored": agg.get("mcc"),
        }
    return out


def main() -> None:
    result: dict[str, Any] = {
        "labels": "post-relabel (fix/dev-public-mislabel-audit)",
        "pi_grid": [0.01, 0.02, 0.03, 0.04, 0.05],
        "pertype_full": section_pertype(),
        "codesign_pertype": section_codesign_pertype(),
        "ppv_sweep": section_ppv(),
        "cascade_t3": section_cascade_t3(),
        "uncertain": section_uncertain(),
        "mcc_confusion": section_mcc(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    # ---- console summary ----
    print("=== MCC from NEW confusion matrix (F4) ===")
    for k, v in result["mcc_confusion"].items():
        print(
            f"  {k:28s} TP={v['TP']:4d} FP={v['FP']:4d} FN={v['FN']:4d} "
            f"TN={v['TN']:4d}  MCC {v['mcc_old_stored']} -> {v['mcc_new']}"
        )

    print("\n=== Cascade T3-F1 (faithful, per_tier['3']['f1']); AUROC not recomputable (F7) ===")
    for label, v in result["cascade_t3"].items():
        print(
            f"  {label:9s} T3-F1={v['t3_f1_faithful']}  (T3 DR={v['t3_dr']}, "
            f"prec={v['t3_precision']}, nh={v['t3_num_hallucinated']})  "
            f"stale AUROC field={v['stale_field_auroc']:.3f}"
        )

    print("\n=== DeepSeek-R1 UNCERTAIN + per-tool coverage (NEW labels) ===")
    r1 = result["uncertain"].get("llm_openrouter_deepseek_r1", {})
    for sp, v in r1.items():
        print(f"  DeepSeek-R1 {sp}: {v['num_uncertain']}/{v['num_entries']} = {v['pct']}%")

    print("\n=== PPV @ pi=2% (FPR-ascending) ===")
    for tool, fpr in result["ppv_sweep"]["fpr_ascending_order"]:
        row = next(r for r in result["ppv_sweep"]["rows"] if r["tool"] == tool)
        print(f"  {tool:20s} FPR={fpr:.3f}  PPV@2%={row['ppv_pct_at_2pct']}%")

    print("\n=== Per-type DR regenerated for OFFLINE tools (summary-only flagged F5) ===")
    for tool, abbr, kind in PERTYPE_TOOLS:
        tag = "OFFLINE" if kind == "offline" else "SUMMARY(F5)"
        print(f"  {abbr:5s} {tool:38s} {tag}")

    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
