#!/usr/bin/env python3
"""Zero-shot LLM verifiers on the Walters & Wilder ChatGPT-citation supplement.

Runs a compact model subset (calibrated <-> aggressive spectrum) on the
341-entry eval-only supplement (172 VALID / 169 HALLUCINATED) with the default
verification prompt, mirroring the main-table protocol (T=0, seed=42,
harness-encoded token budgets). All cited works predate every model's training
cutoff, so this isolates parametric verification on authentic, multidisciplinary
ChatGPT output with no temporal confound.

Headline metrics use hallmark.evaluation.evaluate (UNCERTAIN excluded, the
paper's documented protocol); the by-GPT-version / by-subject-field breakdown
cells use forced-binary scoring (UNCERTAIN -> not flagged) to stay directly
comparable with the bibtexupdater breakdown in docs/walters_wilder_supplement.md.

Usage (from the hallmark repo root):
    .venv/bin/python results/walters_wilder_llm/run_llm_eval.py --models gemini_flash --max-entries 3
    .venv/bin/python results/walters_wilder_llm/run_llm_eval.py --models gpt51
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hallmark.baselines.llm_agentic import verify_agentic_btu_openai  # noqa: E402
from hallmark.baselines.llm_verifier import (  # noqa: E402
    verify_with_openai,
    verify_with_openrouter,
)
from hallmark.dataset.schema import Prediction, load_entries, save_predictions  # noqa: E402
from hallmark.evaluation.metrics import evaluate  # noqa: E402

logger = logging.getLogger(__name__)

WORKSPACE = REPO_ROOT / "results" / "walters_wilder_llm"
SUPPLEMENT = WORKSPACE / "supplement_chatgpt_citations.jsonl"
XLSX = WORKSPACE / "41598_2023_41032_MOESM3_ESM.xlsx"

MODELS: dict[str, dict[str, str]] = {
    "gpt51": {
        "provider": "openai",
        "model_id": "gpt-5.1",
        "key_env": "OPENAI_API_KEY",
        "display": "GPT-5.1",
    },
    "sonnet46": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4.6",
        "key_env": "OPENROUTER_API_KEY",
        "display": "Claude Sonnet 4.6",
    },
    "qwen": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "key_env": "OPENROUTER_API_KEY",
        "display": "Qwen3-235B",
    },
    "gemini_flash": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "key_env": "OPENROUTER_API_KEY",
        "display": "Gemini 2.5 Flash",
    },
    "agentic_btu": {
        "provider": "agentic_btu_openai",
        "model_id": "gpt-5.1",
        "key_env": "OPENAI_API_KEY",
        "display": "GPT-5.1 agentic (BTU-only)",
    },
}


def load_meta(xlsx_path: Path) -> dict[str, dict[str, Any]]:
    """Map bibtex_key -> {gpt_version, subject_field} from Appendix 3.

    Identical to scripts/analyze_walters_wilder.py:load_meta (origin/main).
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Works cited in those papers"]
    rows = [r for r in ws.iter_rows(values_only=True) if r[4]][1:]
    meta: dict[str, dict[str, Any]] = {}
    field_names = {"H": "Humanities", "S": "Social sciences", "N": "Natural sciences"}
    for r in rows:
        if str(r[8]).strip() != "A":
            continue
        gpt = str(r[0]).strip()
        cnum = f"{float(r[3]):.2f}".replace(".", "")
        key = f"ww_{gpt.replace('.', '')}_t{r[2]}_c{cnum}"
        meta[key] = {
            "gpt_version": gpt,
            "subject_field": field_names.get(str(r[1]).strip(), str(r[1]).strip()),
        }
    return meta


def dr_fpr(
    entries: list[Any], preds: dict[str, Prediction]
) -> tuple[float, float, tuple[int, int, int, int]]:
    """Forced-binary DR/FPR (UNCERTAIN/missing -> not flagged)."""
    tp = fn = fp = tn = 0
    for e in entries:
        p = preds.get(e.bibtex_key)
        flagged = bool(p and p.label == "HALLUCINATED")
        if e.label == "HALLUCINATED":
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    dr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return dr, fpr, (tp, fn, fp, tn)


def run_model(key: str, entries: list[Any], max_entries: int | None) -> None:
    cfg = MODELS[key]
    api_key = os.getenv(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"{cfg['key_env']} not set")

    subset = entries[:max_entries] if max_entries else entries
    blind = [e.to_blind() for e in subset]
    ckpt_dir = WORKSPACE / "checkpoints" / key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg["provider"] == "agentic_btu_openai":
        preds_list = verify_agentic_btu_openai(
            blind,
            model=cfg["model_id"],
            api_key=api_key,
            checkpoint_dir=ckpt_dir,
            cache_db_path=WORKSPACE / "agentic_tools.sqlite",
        )
    else:
        verify = verify_with_openai if cfg["provider"] == "openai" else verify_with_openrouter
        preds_list = verify(
            blind,
            model=cfg["model_id"],
            api_key=api_key,
            checkpoint_dir=ckpt_dir,
        )
    save_predictions(preds_list, str(WORKSPACE / f"{key}_predictions.jsonl"))
    preds = {p.bibtex_key: p for p in preds_list}

    result = evaluate(
        subset,
        [preds[e.bibtex_key] for e in subset if e.bibtex_key in preds],
        tool_name=cfg["display"],
        split_name="walters_wilder_chatgpt_citations",
        compute_ci=True,
    )

    uncertain = sum(1 for p in preds_list if p.label == "UNCERTAIN")
    metrics: dict[str, Any] = {
        "tool": cfg["display"],
        "model_id": cfg["model_id"],
        "provider": cfg["provider"],
        "split": "walters_wilder_chatgpt_citations",
        "n_entries": len(subset),
        "coverage": result.coverage,
        "num_uncertain": uncertain,
        "overall": {
            "detection_rate": result.detection_rate,
            "false_positive_rate": result.false_positive_rate,
            "f1_hallucination": result.f1_hallucination,
            "tier_weighted_f1": result.tier_weighted_f1,
            "ece": result.ece,
            "detection_rate_ci": result.detection_rate_ci,
            "fpr_ci": result.fpr_ci,
        },
        "per_tier": result.per_tier_metrics,
        "per_type": result.per_type_metrics,
        "by_gpt_version": {},
        "by_subject_field": {},
        "forced_binary_overall": {},
    }

    dr, fpr, cm = dr_fpr(subset, preds)
    metrics["forced_binary_overall"] = {"detection_rate": dr, "fpr": fpr, "cm": cm}

    if XLSX.exists():
        meta = load_meta(XLSX)
        by_gpt: dict[str, list[Any]] = collections.defaultdict(list)
        by_field: dict[str, list[Any]] = collections.defaultdict(list)
        for e in subset:
            m = meta.get(e.bibtex_key, {})
            by_gpt[m.get("gpt_version", "?")].append(e)
            by_field[m.get("subject_field", "?")].append(e)
        for g, sub in sorted(by_gpt.items()):
            d, f, c = dr_fpr(sub, preds)
            metrics["by_gpt_version"][g] = {"n": len(sub), "detection_rate": d, "fpr": f, "cm": c}
        for fld, sub in sorted(by_field.items()):
            d, f, c = dr_fpr(sub, preds)
            metrics["by_subject_field"][fld] = {
                "n": len(sub),
                "detection_rate": d,
                "fpr": f,
                "cm": c,
            }

    out = WORKSPACE / f"{key}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))

    def fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

    print(
        f"[{cfg['display']}] n={len(subset)} DR={fmt(result.detection_rate)} "
        f"FPR={fmt(result.false_positive_rate)} F1={fmt(result.f1_hallucination)} "
        f"TW-F1={fmt(result.tier_weighted_f1)} ECE={fmt(result.ece)} "
        f"cov={fmt(result.coverage)} uncertain={uncertain} -> {out}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help=f"comma list from {sorted(MODELS)}")
    ap.add_argument("--max-entries", type=int, default=None)
    args = ap.parse_args()

    entries = load_entries(str(SUPPLEMENT))
    print(f"Loaded {len(entries)} supplement entries")
    for key in args.models.split(","):
        run_model(key.strip(), entries, args.max_entries)


if __name__ == "__main__":
    main()
