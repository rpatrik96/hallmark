#!/usr/bin/env python3
"""Comprehensive analysis of LLM baselines on HALLMARK.  [analysis]

Produces stratified results for all LLM baselines (GPT-5.1, Qwen3-235B,
DeepSeek-V3.2, DeepSeek-R1, Mistral Large, Gemini 2.5 Flash) and generates
LaTeX table snippets for the paper appendix.

Analyses:
  a) Per-type detection rate matrix
  b) Per-tier breakdown (DR, FPR, F1 per tier per model)
  c) Generation method stratification
  d) Confidence distribution analysis
  e) Pairwise agreement / consensus analysis (Cohen's kappa)
  f) Sub-test accuracy correlation (skipped if LLM preds lack subtest_results)
  g) Error analysis (FN by type, FP characteristics)

Usage:
    python scripts/analyze_llm_baselines.py
    python scripts/analyze_llm_baselines.py --split dev_public --latex
    python scripts/analyze_llm_baselines.py --output-dir results/llm_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import (
    HALLUCINATION_TIER_MAP,
    BenchmarkEntry,
    Prediction,
    load_predictions,
)
from hallmark.evaluation.metrics import (
    build_confusion_matrix,
    per_generation_method_metrics,
)

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────────────

# Maps model short name -> (display name, prediction file pattern, eval json pattern)
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "llm_openai": {
        "display": "GPT-5.1",
        "pred_pattern": "llm_openai_{split}_predictions.jsonl",
        "eval_pattern": "llm_openai_{split}.json",
    },
    "llm_openrouter_qwen": {
        "display": "Qwen3-235B",
        "pred_pattern": "llm_openrouter_qwen_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_qwen_{split}.json",
    },
    "llm_openrouter_deepseek_v3": {
        "display": "DeepSeek-V3.2",
        "pred_pattern": "llm_openrouter_deepseek_v3_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_deepseek_v3_{split}.json",
    },
    "llm_openrouter_deepseek_r1": {
        "display": "DeepSeek-R1",
        "pred_pattern": "llm_openrouter_deepseek_r1_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_deepseek_r1_{split}.json",
    },
    "llm_openrouter_mistral": {
        "display": "Mistral Large",
        "pred_pattern": "llm_openrouter_mistral_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_mistral_{split}.json",
    },
    "llm_openrouter_gemini_flash": {
        "display": "Gemini 2.5 Flash",
        "pred_pattern": "llm_openrouter_gemini_flash_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_gemini_flash_{split}.json",
    },
}

# Hallucination types in display order (by tier)
TYPE_ORDER: list[str] = [
    # Tier 1
    "fabricated_doi",
    "nonexistent_venue",
    "placeholder_authors",
    "future_date",
    # Tier 2
    "chimeric_title",
    "wrong_venue",
    "swapped_authors",
    "preprint_as_published",
    "hybrid_fabrication",
    # Tier 3
    "near_miss_title",
    "plausible_fabrication",
    # Stress-test
    "merged_citation",
    "partial_author_list",
    "arxiv_version_mismatch",
]

TYPE_DISPLAY: dict[str, str] = {
    "fabricated_doi": "fabricated_doi",
    "nonexistent_venue": "nonexistent_venue",
    "placeholder_authors": "placeholder_authors",
    "future_date": "future_date",
    "chimeric_title": "chimeric_title",
    "wrong_venue": "wrong_venue",
    "swapped_authors": "author_mismatch",
    "preprint_as_published": "preprint_as_pub.",
    "hybrid_fabrication": "hybrid_fabrication",
    "near_miss_title": "near_miss_title",
    "plausible_fabrication": "plausible_fabrication",
    "merged_citation": "merged_citation",
    "partial_author_list": "partial_author_list",
    "arxiv_version_mismatch": "arxiv_version_mismatch",
}

TYPE_TO_TIER: dict[str, int] = {}
for ht, dt in HALLUCINATION_TIER_MAP.items():
    TYPE_TO_TIER[ht.value] = dt.value


# ── Data loading ─────────────────────────────────────────────────────────────


def discover_models(results_dir: Path, split: str) -> dict[str, list[Prediction]]:
    """Discover available LLM model predictions in results_dir.

    Prefers prediction JSONL files; falls back to evaluation JSON if no
    prediction file exists (GPT-5.1 only has eval JSON for dev_public).
    """
    models: dict[str, list[Prediction]] = {}

    for model_key, info in MODEL_REGISTRY.items():
        pred_file = results_dir / info["pred_pattern"].format(split=split)
        eval_file = results_dir / info["eval_pattern"].format(split=split)

        if pred_file.exists():
            try:
                preds = load_predictions(pred_file)
                models[model_key] = preds
                logger.info(
                    "Loaded %d predictions for %s from %s",
                    len(preds),
                    info["display"],
                    pred_file.name,
                )
            except Exception as e:
                logger.warning("Failed to load %s: %s", pred_file, e)
        elif eval_file.exists():
            # Load the eval JSON — no raw predictions, but we can extract
            # per-type/per-tier metrics. We still can't do entry-level analysis.
            logger.info(
                "No prediction JSONL for %s; will use evaluation JSON only",
                info["display"],
            )
            # We store an empty list to signal "eval-only"
            models[model_key] = []
        else:
            logger.debug("No results found for %s", info["display"])

    return models


def load_eval_json(results_dir: Path, model_key: str, split: str) -> dict[str, Any] | None:
    """Load the evaluation JSON for a model."""
    info = MODEL_REGISTRY[model_key]
    eval_file = results_dir / info["eval_pattern"].format(split=split)
    if eval_file.exists():
        with open(eval_file) as f:
            return json.load(f)
    return None


# ── Analysis functions ───────────────────────────────────────────────────────


def per_type_detection_matrix(
    entries: list[BenchmarkEntry],
    model_preds: dict[str, dict[str, Prediction]],
    eval_jsons: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Build type x model detection rate matrix.

    Returns: {hallucination_type: {model_key: detection_rate}}
    """
    # Group entries by hallucination type
    type_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        if entry.label == "HALLUCINATED" and entry.hallucination_type:
            type_entries[entry.hallucination_type].append(entry)

    matrix: dict[str, dict[str, float | None]] = {}

    for htype in TYPE_ORDER:
        matrix[htype] = {}
        for model_key in model_preds:
            preds = model_preds[model_key]
            if preds and htype in type_entries:
                # Compute from raw predictions
                h_entries = type_entries[htype]
                cm = build_confusion_matrix(h_entries, preds)
                matrix[htype][model_key] = cm.detection_rate
            else:
                # Try eval JSON
                ej = eval_jsons.get(model_key)
                if ej and "per_type_metrics" in ej:
                    tm = ej["per_type_metrics"].get(htype, {})
                    matrix[htype][model_key] = tm.get("detection_rate")
                else:
                    matrix[htype][model_key] = None

    return matrix


def per_tier_breakdown(
    entries: list[BenchmarkEntry],
    model_preds: dict[str, dict[str, Prediction]],
    eval_jsons: dict[str, dict[str, Any]],
) -> dict[str, dict[int, dict[str, float | None]]]:
    """Compute DR, FPR, F1 per tier per model.

    Returns: {model_key: {tier: {metric: value}}}
    """
    result: dict[str, dict[int, dict[str, float | None]]] = {}

    for model_key in model_preds:
        result[model_key] = {}
        preds = model_preds[model_key]

        for tier in [1, 2, 3]:
            if preds:
                tier_entries = [
                    e for e in entries if e.difficulty_tier == tier or e.label == "VALID"
                ]
                cm = build_confusion_matrix(tier_entries, preds)
                result[model_key][tier] = {
                    "detection_rate": cm.detection_rate,
                    "fpr": cm.false_positive_rate,
                    "f1": cm.f1,
                }
            else:
                ej = eval_jsons.get(model_key)
                if ej and "per_tier_metrics" in ej:
                    tm = ej["per_tier_metrics"].get(str(tier), {})
                    result[model_key][tier] = {
                        "detection_rate": tm.get("detection_rate"),
                        "fpr": tm.get("false_positive_rate"),
                        "f1": tm.get("f1"),
                    }
                else:
                    result[model_key][tier] = {
                        "detection_rate": None,
                        "fpr": None,
                        "f1": None,
                    }

    return result


def generation_method_stratification(
    entries: list[BenchmarkEntry],
    model_preds: dict[str, dict[str, Prediction]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Compare performance on perturbation vs llm_generated vs real_world.

    Returns: {model_key: {method: {n, detection_rate, false_positive_rate, f1}}}
    Only models with raw predictions are included.
    """
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for model_key, preds in model_preds.items():
        if not preds:
            continue
        result[model_key] = per_generation_method_metrics(entries, preds)
    return result


def confidence_analysis(
    entries: list[BenchmarkEntry],
    model_preds: dict[str, dict[str, Prediction]],
) -> dict[str, dict[str, Any]]:
    """Analyze confidence distributions for correct vs incorrect predictions.

    Returns: {model_key: {correct: {mean, median, std, n}, incorrect: {...}}}
    """
    entry_map = {e.bibtex_key: e for e in entries}
    result: dict[str, dict[str, Any]] = {}

    for model_key, preds in model_preds.items():
        if not preds:
            continue

        correct_confs: list[float] = []
        incorrect_confs: list[float] = []

        for pred in preds.values():
            entry = entry_map.get(pred.bibtex_key)
            if entry is None:
                continue
            if pred.label == "UNCERTAIN":
                continue

            is_correct = pred.label == entry.label
            if is_correct:
                correct_confs.append(pred.confidence)
            else:
                incorrect_confs.append(pred.confidence)

        result[model_key] = {
            "correct": _conf_stats(correct_confs),
            "incorrect": _conf_stats(incorrect_confs),
            "all_mean": (
                sum(correct_confs + incorrect_confs) / len(correct_confs + incorrect_confs)
                if (correct_confs or incorrect_confs)
                else 0.0
            ),
        }

    return result


def _conf_stats(confs: list[float]) -> dict[str, float]:
    """Compute mean/median/std for a list of confidence values."""
    if not confs:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0}
    n = len(confs)
    mean = sum(confs) / n
    sorted_c = sorted(confs)
    median = sorted_c[n // 2] if n % 2 == 1 else (sorted_c[n // 2 - 1] + sorted_c[n // 2]) / 2
    variance = sum((c - mean) ** 2 for c in confs) / n if n > 1 else 0.0
    return {"mean": mean, "median": median, "std": variance**0.5, "n": n}


def agreement_analysis(
    entries: list[BenchmarkEntry],
    model_preds: dict[str, dict[str, Prediction]],
) -> dict[str, Any]:
    """Pairwise agreement and consensus analysis.

    Returns dict with:
      pairwise_agreement: {(model_a, model_b): {agreement_pct, cohens_kappa}}
      consensus: {all_agree_correct, all_agree_wrong, any_disagree, hard_entries}
    """
    entry_map = {e.bibtex_key: e for e in entries}
    # Only models with raw predictions
    active_models = {k: v for k, v in model_preds.items() if v}
    model_keys = sorted(active_models.keys())

    if len(model_keys) < 2:
        return {"pairwise_agreement": {}, "consensus": {}}

    # Pairwise agreement
    pairwise: dict[str, dict[str, float]] = {}
    for a, b in combinations(model_keys, 2):
        preds_a = active_models[a]
        preds_b = active_models[b]
        common_keys = set(preds_a.keys()) & set(preds_b.keys())
        if not common_keys:
            continue

        agree = 0
        n = len(common_keys)
        # For Cohen's kappa
        table = {"HH": 0, "HV": 0, "VH": 0, "VV": 0}
        for key in common_keys:
            la = "H" if preds_a[key].label == "HALLUCINATED" else "V"
            lb = "H" if preds_b[key].label == "HALLUCINATED" else "V"
            table[la + lb] += 1
            if la == lb:
                agree += 1

        po = agree / n if n > 0 else 0.0
        # Expected agreement
        pa_h = (table["HH"] + table["HV"]) / n
        pb_h = (table["HH"] + table["VH"]) / n
        pe = pa_h * pb_h + (1 - pa_h) * (1 - pb_h)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

        pair_key = f"{MODEL_REGISTRY[a]['display']} vs {MODEL_REGISTRY[b]['display']}"
        pairwise[pair_key] = {
            "agreement_pct": po * 100,
            "cohens_kappa": kappa,
            "n": n,
        }

    # Consensus analysis
    all_keys = set()
    for preds in active_models.values():
        all_keys.update(preds.keys())
    # Only consider entries where all models have predictions
    common_all = all_keys.copy()
    for preds in active_models.values():
        common_all &= set(preds.keys())

    all_correct = 0
    all_wrong = 0
    any_disagree = 0
    hard_entries: list[str] = []  # entries no model correctly identifies

    for key in common_all:
        entry = entry_map.get(key)
        if entry is None:
            continue

        labels = [active_models[m][key].label for m in model_keys if key in active_models[m]]
        correct_flags = [
            (active_models[m][key].label == entry.label)
            for m in model_keys
            if key in active_models[m]
        ]

        if len(set(labels)) == 1:
            # All agree
            if all(correct_flags):
                all_correct += 1
            else:
                all_wrong += 1
                if entry.label == "HALLUCINATED":
                    hard_entries.append(key)
        else:
            any_disagree += 1
            if not any(correct_flags) and entry.label == "HALLUCINATED":
                hard_entries.append(key)

    return {
        "pairwise_agreement": pairwise,
        "consensus": {
            "all_agree_correct": all_correct,
            "all_agree_wrong": all_wrong,
            "any_disagree": any_disagree,
            "total_common": len(common_all),
            "hard_entries_count": len(hard_entries),
            "hard_entries": hard_entries[:20],  # sample
        },
    }


def error_analysis(
    entries: list[BenchmarkEntry],
    model_preds: dict[str, dict[str, Prediction]],
) -> dict[str, dict[str, Any]]:
    """Analyze false negatives and false positives per model.

    Returns: {model_key: {fn_by_type: {...}, fp_by_method: {...}, fn_count, fp_count}}
    """
    result: dict[str, dict[str, Any]] = {}

    for model_key, preds in model_preds.items():
        if not preds:
            continue

        fn_by_type: dict[str, int] = defaultdict(int)
        fn_by_method: dict[str, int] = defaultdict(int)
        fp_by_method: dict[str, int] = defaultdict(int)
        fn_count = 0
        fp_count = 0

        for entry in entries:
            pred = preds.get(entry.bibtex_key)
            if pred is None:
                # Missing = VALID default
                if entry.label == "HALLUCINATED":
                    fn_count += 1
                    if entry.hallucination_type:
                        fn_by_type[entry.hallucination_type] += 1
                    fn_by_method[entry.generation_method] += 1
                continue
            if pred.label == "UNCERTAIN":
                continue

            if entry.label == "HALLUCINATED" and pred.label != "HALLUCINATED":
                fn_count += 1
                if entry.hallucination_type:
                    fn_by_type[entry.hallucination_type] += 1
                fn_by_method[entry.generation_method] += 1
            elif entry.label == "VALID" and pred.label == "HALLUCINATED":
                fp_count += 1
                fp_by_method[entry.generation_method] += 1

        result[model_key] = {
            "fn_by_type": dict(fn_by_type),
            "fn_by_method": dict(fn_by_method),
            "fp_by_method": dict(fp_by_method),
            "fn_count": fn_count,
            "fp_count": fp_count,
        }

    return result


# ── Output formatting ────────────────────────────────────────────────────────


def _fmt(val: float | None, decimals: int = 3) -> str:
    """Format a float or return '--' for None."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def print_type_matrix(
    matrix: dict[str, dict[str, float | None]],
    model_keys: list[str],
    type_counts: dict[str, int],
) -> None:
    """Print per-type detection rate matrix as text table."""
    print("\n" + "=" * 120)
    print("PER-TYPE DETECTION RATE MATRIX")
    print("=" * 120)

    header = f"{'Type':<25} {'Tier':>4} {'n':>5}"
    for mk in model_keys:
        header += f"  {MODEL_REGISTRY[mk]['display']:>14}"
    header += f"  {'Best':>14}  {'Worst':>14}"
    print(header)
    print("-" * 120)

    current_tier = None
    for htype in TYPE_ORDER:
        if htype not in matrix:
            continue
        tier = TYPE_TO_TIER.get(htype, 0)
        if tier != current_tier:
            if current_tier is not None:
                print("-" * 120)
            current_tier = tier

        row = f"{TYPE_DISPLAY.get(htype, htype):<25} {tier:>4} {type_counts.get(htype, 0):>5}"
        vals: list[tuple[str, float]] = []
        for mk in model_keys:
            v = matrix[htype].get(mk)
            row += f"  {_fmt(v):>14}"
            if v is not None:
                vals.append((MODEL_REGISTRY[mk]["display"], v))

        if vals:
            best = max(vals, key=lambda x: x[1])
            worst = min(vals, key=lambda x: x[1])
            row += f"  {best[0]:>14}  {worst[0]:>14}"
        print(row)

    print("=" * 120)


def print_tier_breakdown(
    tier_data: dict[str, dict[int, dict[str, float | None]]],
    model_keys: list[str],
) -> None:
    """Print per-tier DR/FPR/F1 for each model."""
    print("\n" + "=" * 100)
    print("PER-TIER BREAKDOWN")
    print("=" * 100)

    tier_labels = {1: "Tier 1 (Easy)", 2: "Tier 2 (Medium)", 3: "Tier 3 (Hard)"}
    for tier in [1, 2, 3]:
        print(f"\n{tier_labels[tier]}")
        print(f"{'Model':<20} {'DR':>8} {'FPR':>8} {'F1':>8}")
        print("-" * 50)
        for mk in model_keys:
            d = tier_data.get(mk, {}).get(tier, {})
            print(
                f"{MODEL_REGISTRY[mk]['display']:<20} "
                f"{_fmt(d.get('detection_rate')):>8} "
                f"{_fmt(d.get('fpr')):>8} "
                f"{_fmt(d.get('f1')):>8}"
            )


def print_generation_method(
    method_data: dict[str, dict[str, dict[str, float | int]]],
    model_keys: list[str],
) -> None:
    """Print generation-method stratification."""
    print("\n" + "=" * 100)
    print("GENERATION METHOD STRATIFICATION")
    print("=" * 100)

    methods = set()
    for md in method_data.values():
        methods.update(md.keys())
    methods_sorted = sorted(methods)

    for method in methods_sorted:
        print(f"\n{method.upper()}")
        print(f"{'Model':<20} {'n':>6} {'DR':>8} {'FPR':>8} {'F1':>8}")
        print("-" * 55)
        for mk in model_keys:
            if mk not in method_data:
                continue
            d = method_data[mk].get(method, {})
            n = d.get("n", 0)
            dr = d.get("detection_rate")
            fpr = d.get("false_positive_rate")
            f1 = d.get("f1")
            print(
                f"{MODEL_REGISTRY[mk]['display']:<20} "
                f"{n:>6} "
                f"{_fmt(dr):>8} "
                f"{_fmt(fpr):>8} "
                f"{_fmt(f1):>8}"
            )


def print_confidence_analysis(
    conf_data: dict[str, dict[str, Any]],
    model_keys: list[str],
) -> None:
    """Print confidence distribution stats."""
    print("\n" + "=" * 100)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 100)
    print(
        f"{'Model':<20} "
        f"{'Correct Mean':>12} {'Correct Med':>12} {'Correct Std':>12} "
        f"{'Incorr Mean':>12} {'Incorr Med':>12} {'Incorr Std':>12} "
        f"{'Gap':>8}"
    )
    print("-" * 100)
    for mk in model_keys:
        if mk not in conf_data:
            continue
        c = conf_data[mk]["correct"]
        i = conf_data[mk]["incorrect"]
        gap = c["mean"] - i["mean"] if c["n"] > 0 and i["n"] > 0 else float("nan")
        print(
            f"{MODEL_REGISTRY[mk]['display']:<20} "
            f"{_fmt(c['mean']):>12} {_fmt(c['median']):>12} {_fmt(c['std']):>12} "
            f"{_fmt(i['mean']):>12} {_fmt(i['median']):>12} {_fmt(i['std']):>12} "
            f"{_fmt(gap):>8}"
        )
    print("\nGap = (correct mean) - (incorrect mean); larger gap = better calibration")


def print_agreement(agree_data: dict[str, Any]) -> None:
    """Print pairwise agreement and consensus."""
    print("\n" + "=" * 100)
    print("PAIRWISE AGREEMENT ANALYSIS")
    print("=" * 100)

    pairwise = agree_data.get("pairwise_agreement", {})
    if pairwise:
        print(f"{'Pair':<45} {'Agree%':>8} {'Kappa':>8} {'n':>6}")
        print("-" * 70)
        for pair, stats in sorted(pairwise.items()):
            print(
                f"{pair:<45} "
                f"{stats['agreement_pct']:>7.1f}% "
                f"{stats['cohens_kappa']:>8.3f} "
                f"{stats['n']:>6}"
            )

    consensus = agree_data.get("consensus", {})
    if consensus:
        total = consensus.get("total_common", 0)
        print(f"\nConsensus on {total} entries with predictions from all models:")
        print(f"  All agree + correct:  {consensus.get('all_agree_correct', 0)}")
        print(f"  All agree + wrong:    {consensus.get('all_agree_wrong', 0)}")
        print(f"  Any disagreement:     {consensus.get('any_disagree', 0)}")
        print(f"  Hard entries (no LLM detects): {consensus.get('hard_entries_count', 0)}")


def print_error_analysis(
    err_data: dict[str, dict[str, Any]],
    model_keys: list[str],
    type_counts: dict[str, int],
) -> None:
    """Print false negative/positive analysis."""
    print("\n" + "=" * 100)
    print("ERROR ANALYSIS — FALSE NEGATIVES BY TYPE")
    print("=" * 100)

    # Collect all types that have FNs
    all_fn_types: set[str] = set()
    for md in err_data.values():
        all_fn_types.update(md.get("fn_by_type", {}).keys())
    fn_types = [t for t in TYPE_ORDER if t in all_fn_types]

    header = f"{'Type':<25} {'n':>5}"
    for mk in model_keys:
        if mk in err_data:
            header += f"  {MODEL_REGISTRY[mk]['display']:>14}"
    print(header)
    print("-" * 100)

    for htype in fn_types:
        row = f"{TYPE_DISPLAY.get(htype, htype):<25} {type_counts.get(htype, 0):>5}"
        for mk in model_keys:
            if mk not in err_data:
                continue
            fn_count = err_data[mk].get("fn_by_type", {}).get(htype, 0)
            total = type_counts.get(htype, 1)
            pct = fn_count / total * 100 if total > 0 else 0.0
            row += f"  {fn_count:>5} ({pct:>5.1f}%)"
        print(row)

    print("\n" + "-" * 80)
    print("FALSE POSITIVE SUMMARY")
    print(f"{'Model':<20} {'FP count':>10} {'Total valid':>12}")
    print("-" * 50)
    for mk in model_keys:
        if mk not in err_data:
            continue
        fp = err_data[mk]["fp_count"]
        # Count valid entries from fp_by_method
        total_valid = sum(err_data[mk].get("fp_by_method", {}).values())
        print(f"{MODEL_REGISTRY[mk]['display']:<20} {fp:>10} {total_valid:>12}")


def print_overall_summary(
    eval_jsons: dict[str, dict[str, Any]],
    model_keys: list[str],
) -> None:
    """Print overall metric summary for all models."""
    print("\n" + "=" * 120)
    print("OVERALL METRICS SUMMARY")
    print("=" * 120)
    print(
        f"{'Model':<20} {'DR':>8} {'FPR':>8} {'F1':>8} {'TW-F1':>8} "
        f"{'MCC':>8} {'ECE':>8} {'AUROC':>8} {'Cov':>6} {'Unc':>5}"
    )
    print("-" * 120)

    for mk in model_keys:
        ej = eval_jsons.get(mk)
        if ej is None:
            continue
        print(
            f"{MODEL_REGISTRY[mk]['display']:<20} "
            f"{_fmt(ej.get('detection_rate')):>8} "
            f"{_fmt(ej.get('false_positive_rate')):>8} "
            f"{_fmt(ej.get('f1_hallucination')):>8} "
            f"{_fmt(ej.get('tier_weighted_f1')):>8} "
            f"{_fmt(ej.get('mcc')):>8} "
            f"{_fmt(ej.get('ece')):>8} "
            f"{_fmt(ej.get('auroc')):>8} "
            f"{_fmt(ej.get('coverage', 1.0)):>6} "
            f"{ej.get('num_uncertain', 0):>5}"
        )
    print("=" * 120)


# ── LaTeX generation ─────────────────────────────────────────────────────────


def generate_latex_overall(
    eval_jsons: dict[str, dict[str, Any]],
    model_keys: list[str],
) -> str:
    """Generate LaTeX table for overall LLM comparison."""
    lines: list[str] = []
    lines.append("\\begin{table}[h]")
    lines.append(
        "\\caption{LLM baseline comparison on \\texttt{dev\\_public}. "
        "Best value per metric is \\textbf{bold}. "
        "$\\uparrow$ = higher is better, $\\downarrow$ = lower is better.}"
    )
    lines.append("\\label{tab:llm_comparison}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lccccccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Model} & \\textbf{DR $\\uparrow$} & \\textbf{FPR $\\downarrow$} "
        "& \\textbf{F1 $\\uparrow$} & \\textbf{MCC $\\uparrow$} "
        "& \\textbf{TW-F1 $\\uparrow$} & \\textbf{ECE $\\downarrow$} "
        "& \\textbf{AUROC $\\uparrow$} & \\textbf{Cov.} & \\textbf{Unc.} \\\\"
    )
    lines.append("\\midrule")

    # Find best values for bolding
    metrics_higher_better = [
        "detection_rate",
        "f1_hallucination",
        "mcc",
        "tier_weighted_f1",
        "auroc",
    ]
    metrics_lower_better = ["false_positive_rate", "ece"]

    best: dict[str, float] = {}
    for metric in metrics_higher_better:
        vals = [ej.get(metric, 0.0) for ej in eval_jsons.values() if ej.get(metric) is not None]
        best[metric] = max(vals) if vals else 0.0
    for metric in metrics_lower_better:
        vals = [ej.get(metric, 1.0) for ej in eval_jsons.values() if ej.get(metric) is not None]
        best[metric] = min(vals) if vals else 1.0

    def _bold(val: float | None, metric: str) -> str:
        if val is None:
            return "--"
        s = f"{val:.3f}"
        if metric in best and abs(val - best[metric]) < 1e-6:
            return f"\\textbf{{{s}}}"
        return s

    for mk in model_keys:
        ej = eval_jsons.get(mk)
        if ej is None:
            continue
        name = MODEL_REGISTRY[mk]["display"]
        lines.append(
            f"{name} "
            f"& {_bold(ej.get('detection_rate'), 'detection_rate')} "
            f"& {_bold(ej.get('false_positive_rate'), 'false_positive_rate')} "
            f"& {_bold(ej.get('f1_hallucination'), 'f1_hallucination')} "
            f"& {_bold(ej.get('mcc'), 'mcc')} "
            f"& {_bold(ej.get('tier_weighted_f1'), 'tier_weighted_f1')} "
            f"& {_bold(ej.get('ece'), 'ece')} "
            f"& {_bold(ej.get('auroc'), 'auroc')} "
            f"& {ej.get('coverage', 1.0):.2f} "
            f"& {ej.get('num_uncertain', 0)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_latex_pertype_heatmap(
    matrix: dict[str, dict[str, float | None]],
    model_keys: list[str],
    type_counts: dict[str, int],
) -> str:
    """Generate LaTeX heatmap-style per-type detection rate table."""
    lines: list[str] = []
    n_models = len(model_keys)

    lines.append("\\begin{table}[h]")
    lines.append(
        "\\caption{Per-type detection rate for LLM baselines on \\texttt{dev\\_public}. "
        "\\colorbox{green!20}{Green} = best per type, "
        "\\colorbox{red!15}{red} = worst. "
        "Main types (Tiers 1--3) and stress-test types are separated.}"
    )
    lines.append("\\label{tab:llm_pertype}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    col_spec = "llr" + "c" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "\\textbf{Tier} & \\textbf{Type} & \\textbf{$n$}"
    for mk in model_keys:
        header += f" & \\textbf{{{MODEL_REGISTRY[mk]['display']}}}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Group by tier
    tier_groups: dict[str, list[str]] = {
        "1": [],
        "2": [],
        "3": [],
        "stress": [],
    }
    stress_types = {"merged_citation", "partial_author_list", "arxiv_version_mismatch"}
    for htype in TYPE_ORDER:
        if htype in stress_types:
            tier_groups["stress"].append(htype)
        else:
            tier = str(TYPE_TO_TIER.get(htype, 0))
            if tier in tier_groups:
                tier_groups[tier].append(htype)

    tier_labels = {
        "1": ("1", len(tier_groups["1"])),
        "2": ("2", len(tier_groups["2"])),
        "3": ("3", len(tier_groups["3"])),
        "stress": ("\\rotatebox{90}{\\scriptsize Stress}", len(tier_groups["stress"])),
    }

    first_section = True
    for tier_key in ["1", "2", "3", "stress"]:
        types = tier_groups[tier_key]
        if not types:
            continue
        if not first_section:
            lines.append("\\midrule")
        first_section = False

        label, count = tier_labels[tier_key]
        for i, htype in enumerate(types):
            vals: list[tuple[str, float]] = []
            for mk in model_keys:
                v = matrix.get(htype, {}).get(mk)
                if v is not None:
                    vals.append((mk, v))

            best_mk = max(vals, key=lambda x: x[1])[0] if vals else None
            worst_mk = min(vals, key=lambda x: x[1])[0] if vals else None

            tier_col = f"\\multirow{{{count}}}{{*}}{{{label}}}" if i == 0 else ""
            type_name = TYPE_DISPLAY.get(htype, htype).replace("_", "\\_")
            n = type_counts.get(htype, 0)
            row = f"{tier_col} & \\texttt{{{type_name}}} & {n}"

            for mk in model_keys:
                v = matrix.get(htype, {}).get(mk)
                if v is None:
                    row += " & --"
                else:
                    cell = f"{v:.3f}"
                    if mk == best_mk and len(vals) > 1:
                        cell = f"\\cellcolor{{green!20}}{cell}"
                    elif mk == worst_mk and len(vals) > 1:
                        cell = f"\\cellcolor{{red!15}}{cell}"
                    row += f" & {cell}"

            row += " \\\\"
            lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_latex_agreement(
    agree_data: dict[str, Any],
) -> str:
    """Generate LaTeX snippet for agreement analysis."""
    pairwise = agree_data.get("pairwise_agreement", {})
    if not pairwise:
        return "% No pairwise agreement data available"

    lines: list[str] = []
    lines.append("\\begin{table}[h]")
    lines.append(
        "\\caption{Pairwise agreement between LLM baselines on \\texttt{dev\\_public}. "
        "Cohen's $\\kappa$ quantifies agreement beyond chance.}"
    )
    lines.append("\\label{tab:llm_agreement}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Model pair} & \\textbf{Agreement (\\%)} & \\textbf{Cohen's $\\kappa$} \\\\"
    )
    lines.append("\\midrule")

    for pair, stats in sorted(pairwise.items()):
        pair_escaped = pair.replace("_", "\\_")
        lines.append(
            f"{pair_escaped} & {stats['agreement_pct']:.1f} & {stats['cohens_kappa']:.3f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of LLM baselines on HALLMARK"
    )
    parser.add_argument(
        "--split",
        default="dev_public",
        help="Benchmark split to analyze (default: dev_public)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing evaluation results (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/llm_analysis"),
        help="Directory for output JSON files (default: results/llm_analysis/)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table snippets to stdout",
    )
    parser.add_argument(
        "--latex-file",
        type=Path,
        default=None,
        help="Write LaTeX to this file instead of stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load benchmark entries
    try:
        entries = load_split(args.split, version="v1.0")
        logger.info("Loaded %d entries from %s", len(entries), args.split)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Discover models
    raw_models = discover_models(args.results_dir, args.split)
    if not raw_models:
        print("Error: No LLM baseline results found", file=sys.stderr)
        sys.exit(1)

    # Build pred_maps (dict keyed by bibtex_key) for models with predictions
    model_preds: dict[str, dict[str, Prediction]] = {}
    for mk, preds in raw_models.items():
        if preds:
            model_preds[mk] = {p.bibtex_key: p for p in preds}
        else:
            model_preds[mk] = {}

    model_keys = [mk for mk in MODEL_REGISTRY if mk in raw_models]

    # Load eval JSONs for all models
    eval_jsons: dict[str, dict[str, Any]] = {}
    for mk in model_keys:
        ej = load_eval_json(args.results_dir, mk, args.split)
        if ej:
            eval_jsons[mk] = ej

    # Compute type counts
    type_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        if entry.label == "HALLUCINATED" and entry.hallucination_type:
            type_counts[entry.hallucination_type] += 1

    # ── Run analyses ─────────────────────────────────────────────────────

    print(f"\nHALLMARK LLM Baseline Analysis — {args.split}")
    print(f"Models found: {', '.join(MODEL_REGISTRY[mk]['display'] for mk in model_keys)}")
    n_with_preds = sum(1 for mk in model_keys if model_preds[mk])
    print(f"Models with raw predictions: {n_with_preds}/{len(model_keys)}")
    print(f"Entries: {len(entries)}")

    # Overall summary
    print_overall_summary(eval_jsons, model_keys)

    # (a) Per-type detection rate matrix
    type_matrix = per_type_detection_matrix(entries, model_preds, eval_jsons)
    print_type_matrix(type_matrix, model_keys, type_counts)

    # (b) Per-tier breakdown
    tier_data = per_tier_breakdown(entries, model_preds, eval_jsons)
    print_tier_breakdown(tier_data, model_keys)

    # (c) Generation method stratification
    method_data = generation_method_stratification(entries, model_preds)
    if method_data:
        print_generation_method(method_data, model_keys)

    # (d) Confidence analysis
    conf_data = confidence_analysis(entries, model_preds)
    if conf_data:
        print_confidence_analysis(conf_data, model_keys)

    # (e) Agreement analysis
    agree_data = agreement_analysis(entries, model_preds)
    print_agreement(agree_data)

    # (g) Error analysis
    err_data = error_analysis(entries, model_preds)
    if err_data:
        print_error_analysis(err_data, model_keys, type_counts)

    # ── Save JSON results ────────────────────────────────────────────────

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "split": args.split,
        "models": {mk: MODEL_REGISTRY[mk]["display"] for mk in model_keys},
        "type_detection_matrix": type_matrix,
        "tier_breakdown": {
            mk: {str(t): v for t, v in tiers.items()} for mk, tiers in tier_data.items()
        },
        "generation_method": method_data,
        "confidence_analysis": conf_data,
        "agreement": agree_data,
        "error_analysis": {mk: v for mk, v in err_data.items()},
        "type_counts": dict(type_counts),
    }

    out_path = args.output_dir / f"llm_analysis_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nDetailed results saved to {out_path}")

    # ── LaTeX output ─────────────────────────────────────────────────────

    if args.latex or args.latex_file:
        latex_parts: list[str] = []

        latex_parts.append("% === LLM Baseline Comparison (auto-generated) ===")
        latex_parts.append("")
        latex_parts.append(generate_latex_overall(eval_jsons, model_keys))
        latex_parts.append("")
        latex_parts.append(generate_latex_pertype_heatmap(type_matrix, model_keys, type_counts))
        latex_parts.append("")
        latex_parts.append(generate_latex_agreement(agree_data))

        latex_content = "\n".join(latex_parts)

        if args.latex_file:
            args.latex_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.latex_file, "w") as f:
                f.write(latex_content + "\n")
            print(f"LaTeX tables written to {args.latex_file}")
        else:
            print("\n" + "=" * 80)
            print("LATEX OUTPUT")
            print("=" * 80)
            print(latex_content)


if __name__ == "__main__":
    main()
