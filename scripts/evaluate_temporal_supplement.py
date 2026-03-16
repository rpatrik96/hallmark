#!/usr/bin/env python3
"""Evaluate LLM baselines on the temporal supplement (2024-2025 valid papers).  [evaluation]

Addresses the temporal memorization shortcut: all v1.0 valid entries are from
2021-2023, so LLMs might use publication year as a proxy for validity.  This
script evaluates models on 2024-2025 valid papers and compares FPR/DR/F1
against the original dev_public split to quantify temporal degradation.

Analyses:
  1. Run all available LLM baselines on the temporal supplement (with checkpointing)
  2. Load existing v1.0 dev_public predictions for comparison
  3. Report DR, FPR, F1 for each model on 2021-2023 vs 2024-2025
  4. Per-tier breakdown for both time periods
  5. Bootstrap significance test for FPR degradation
  6. Generate LaTeX table snippet for paper appendix

Usage::

    python scripts/evaluate_temporal_supplement.py \\
        --supplement-file /tmp/temporal_supplement_2024_2025.jsonl
    python scripts/evaluate_temporal_supplement.py \\
        --supplement-file /tmp/temporal_supplement_2024_2025.jsonl \\
        --models qwen,mistral
    python scripts/evaluate_temporal_supplement.py \\
        --supplement-file /tmp/temporal_supplement_2024_2025.jsonl \\
        --skip-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.llm_verifier import (
    verify_with_openai,
    verify_with_openrouter,
)
from hallmark.dataset.loader import load_split
from hallmark.dataset.schema import (
    BenchmarkEntry,
    Prediction,
    load_entries,
    load_predictions,
    save_predictions,
)
from hallmark.evaluation.metrics import build_confusion_matrix, evaluate

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────

# Maps model key -> (provider, model_id, api_key_env, display_name,
#                    v1.0 pred file pattern, v1.0 eval json pattern)
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "llm_openai": {
        "provider": "openai",
        "model_id": "gpt-5.1",
        "api_key_env": "OPENAI_API_KEY",
        "display": "GPT-5.1",
        "pred_pattern": "llm_openai_{split}_predictions.jsonl",
        "eval_pattern": "llm_openai_{split}.json",
    },
    "llm_openrouter_qwen": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "Qwen3-235B",
        "pred_pattern": "llm_openrouter_qwen_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_qwen_{split}.json",
    },
    "llm_openrouter_deepseek_v3": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v3.2",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "DeepSeek-V3.2",
        "pred_pattern": "llm_openrouter_deepseek_v3_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_deepseek_v3_{split}.json",
    },
    "llm_openrouter_deepseek_r1": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "DeepSeek-R1",
        "pred_pattern": "llm_openrouter_deepseek_r1_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_deepseek_r1_{split}.json",
    },
    "llm_openrouter_mistral": {
        "provider": "openrouter",
        "model_id": "mistralai/mistral-large-2512",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "Mistral Large",
        "pred_pattern": "llm_openrouter_mistral_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_mistral_{split}.json",
    },
    "llm_openrouter_gemini_flash": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "Gemini 2.5 Flash",
        "pred_pattern": "llm_openrouter_gemini_flash_{split}_predictions.jsonl",
        "eval_pattern": "llm_openrouter_gemini_flash_{split}.json",
    },
}


# ── Data loading helpers ──────────────────────────────────────────────────────


def load_supplement(path: Path) -> list[BenchmarkEntry]:
    """Load temporal supplement entries from JSONL."""
    entries = load_entries(path)
    n_valid = sum(1 for e in entries if e.label == "VALID")
    n_hall = sum(1 for e in entries if e.label == "HALLUCINATED")
    logger.info(
        "Loaded %d temporal supplement entries (%d valid, %d hallucinated) from %s",
        len(entries),
        n_valid,
        n_hall,
        path,
    )
    return entries


def load_v1_predictions(
    results_dir: Path, model_key: str, split: str = "dev_public"
) -> list[Prediction] | None:
    """Load existing v1.0 predictions for a model.

    Returns None if no prediction file exists.
    """
    info = MODEL_REGISTRY[model_key]
    pred_file = results_dir / info["pred_pattern"].format(split=split)
    if pred_file.exists():
        try:
            preds = load_predictions(pred_file)
            logger.info(
                "Loaded %d v1.0 predictions for %s from %s",
                len(preds),
                info["display"],
                pred_file.name,
            )
            return preds
        except Exception as e:
            logger.warning("Failed to load %s: %s", pred_file, e)
    return None


def load_v1_eval_json(
    results_dir: Path, model_key: str, split: str = "dev_public"
) -> dict[str, Any] | None:
    """Load existing v1.0 evaluation JSON for a model."""
    info = MODEL_REGISTRY[model_key]
    eval_file = results_dir / info["eval_pattern"].format(split=split)
    if eval_file.exists():
        with open(eval_file) as f:
            result: dict[str, Any] = json.load(f)
            return result
    return None


# ── Model execution ───────────────────────────────────────────────────────────


def run_model_on_supplement(
    model_key: str,
    entries: list[BenchmarkEntry],
    checkpoint_dir: Path,
    api_key_env: str,
) -> list[Prediction]:
    """Run a single LLM model on the temporal supplement entries.

    Uses checkpointing for resilience against API failures.
    """
    info = MODEL_REGISTRY[model_key]
    provider = info["provider"]
    model_id = info["model_id"]
    env_var = api_key_env if api_key_env != "OPENROUTER_API_KEY" else info["api_key_env"]
    api_key = os.environ.get(env_var)

    if not api_key:
        logger.warning(
            "Skipping %s: environment variable %s not set",
            info["display"],
            env_var,
        )
        return []

    blind_entries = [e.to_blind() for e in entries]

    logger.info(
        "Running %s (%s) on %d supplement entries...",
        info["display"],
        model_id,
        len(blind_entries),
    )

    if provider == "openai":
        predictions = verify_with_openai(
            blind_entries,
            model=model_id,
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
        )
    elif provider == "openrouter":
        predictions = verify_with_openrouter(
            blind_entries,
            model=model_id,
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    logger.info("Completed %s: %d predictions", info["display"], len(predictions))
    return predictions


def run_all_models(
    model_keys: list[str],
    entries: list[BenchmarkEntry],
    checkpoint_dir: Path,
    output_dir: Path,
    api_key_env: str,
    max_workers: int = 3,
) -> dict[str, list[Prediction]]:
    """Run all specified models on the temporal supplement.

    Models run sequentially (API rate limits make true parallelism
    counterproductive for OpenRouter). Checkpointing enables resumption.

    Returns dict mapping model_key -> predictions.
    """
    all_predictions: dict[str, list[Prediction]] = {}

    for model_key in model_keys:
        info = MODEL_REGISTRY[model_key]

        # Check for existing predictions (already completed)
        pred_file = output_dir / f"{model_key}_temporal_supplement_predictions.jsonl"
        if pred_file.exists():
            try:
                preds = load_predictions(pred_file)
                logger.info(
                    "Loaded existing predictions for %s from %s (%d entries)",
                    info["display"],
                    pred_file.name,
                    len(preds),
                )
                all_predictions[model_key] = preds
                continue
            except Exception as e:
                logger.warning("Failed to load %s, re-running: %s", pred_file, e)

        predictions = run_model_on_supplement(model_key, entries, checkpoint_dir, api_key_env)

        if predictions:
            # Save predictions
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            save_predictions(predictions, pred_file)
            logger.info("Saved predictions to %s", pred_file)
            all_predictions[model_key] = predictions

    return all_predictions


# ── Bootstrap significance test ───────────────────────────────────────────────


def bootstrap_fpr_difference(
    v1_entries: list[BenchmarkEntry],
    v1_preds: dict[str, Prediction],
    supp_entries: list[BenchmarkEntry],
    supp_preds: dict[str, Prediction],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Test whether FPR on the temporal supplement differs from v1.0.

    Uses a paired bootstrap test on valid entries. For each resample, computes
    FPR on both sets and records the difference (supplement - v1.0).

    Returns:
        Dict with observed_diff, p_value, ci_lower, ci_upper, n_bootstrap.
    """
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available; skipping bootstrap significance test")
        return {}

    # Extract valid-entry outcomes for each set
    v1_valid = [e for e in v1_entries if e.label == "VALID"]
    supp_valid = [e for e in supp_entries if e.label == "VALID"]

    if not v1_valid or not supp_valid:
        logger.warning("Insufficient valid entries for bootstrap test")
        return {}

    # Binary vectors: 1 = false positive, 0 = true negative
    v1_outcomes = []
    for e in v1_valid:
        pred = v1_preds.get(e.bibtex_key)
        if pred is None or pred.label == "UNCERTAIN":
            v1_outcomes.append(0)  # missing = VALID (TN)
        else:
            v1_outcomes.append(1 if pred.label == "HALLUCINATED" else 0)

    supp_outcomes = []
    for e in supp_valid:
        pred = supp_preds.get(e.bibtex_key)
        if pred is None or pred.label == "UNCERTAIN":
            supp_outcomes.append(0)
        else:
            supp_outcomes.append(1 if pred.label == "HALLUCINATED" else 0)

    v1_arr = np.array(v1_outcomes, dtype=np.float64)
    supp_arr = np.array(supp_outcomes, dtype=np.float64)

    observed_v1_fpr = v1_arr.mean()
    observed_supp_fpr = supp_arr.mean()
    observed_diff = float(observed_supp_fpr - observed_v1_fpr)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        v1_sample = v1_arr[rng.integers(0, len(v1_arr), size=len(v1_arr))]
        supp_sample = supp_arr[rng.integers(0, len(supp_arr), size=len(supp_arr))]
        diffs[i] = supp_sample.mean() - v1_sample.mean()

    # Two-sided p-value: fraction of bootstrap diffs at least as extreme
    p_value = float(np.mean(np.abs(diffs) >= np.abs(observed_diff)))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    return {
        "observed_diff": observed_diff,
        "v1_fpr": float(observed_v1_fpr),
        "supplement_fpr": float(observed_supp_fpr),
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "n_v1_valid": len(v1_valid),
        "n_supp_valid": len(supp_valid),
    }


# ── Output formatting ────────────────────────────────────────────────────────


def _fmt(val: float | None, decimals: int = 3) -> str:
    """Format a float or return '--' for None."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def _signed(val: float | None, decimals: int = 3) -> str:
    """Format a float with sign prefix."""
    if val is None:
        return "--"
    return f"{val:+.{decimals}f}"


def print_comparison_table(
    model_keys: list[str],
    v1_metrics: dict[str, dict[str, float | None]],
    supp_metrics: dict[str, dict[str, float | None]],
    bootstrap_results: dict[str, dict[str, float]],
) -> None:
    """Print side-by-side comparison table to stdout."""
    print("\n" + "=" * 130)
    print("TEMPORAL SUPPLEMENT EVALUATION — 2021-2023 (v1.0) vs 2024-2025 (supplement)")
    print("=" * 130)

    header = (
        f"{'Model':<20} "
        f"{'DR (v1)':>8} {'DR (sup)':>9} {'dDR':>7}  "
        f"{'FPR (v1)':>9} {'FPR (sup)':>10} {'dFPR':>7}  "
        f"{'F1 (v1)':>8} {'F1 (sup)':>9} {'dF1':>7}  "
        f"{'p(FPR)':>8}"
    )
    print(header)
    print("-" * 130)

    for mk in model_keys:
        v1 = v1_metrics.get(mk, {})
        sp = supp_metrics.get(mk, {})
        bs = bootstrap_results.get(mk, {})

        v1_dr = v1.get("detection_rate")
        sp_dr = sp.get("detection_rate")
        d_dr = (sp_dr - v1_dr) if (v1_dr is not None and sp_dr is not None) else None

        v1_fpr = v1.get("fpr")
        sp_fpr = sp.get("fpr")
        d_fpr = (sp_fpr - v1_fpr) if (v1_fpr is not None and sp_fpr is not None) else None

        v1_f1 = v1.get("f1")
        sp_f1 = sp.get("f1")
        d_f1 = (sp_f1 - v1_f1) if (v1_f1 is not None and sp_f1 is not None) else None

        p_val = bs.get("p_value")
        p_str = _fmt(p_val) if p_val is not None else "--"
        if p_val is not None and p_val < 0.05:
            p_str += " *"

        display = MODEL_REGISTRY[mk]["display"]
        print(
            f"{display:<20} "
            f"{_fmt(v1_dr):>8} {_fmt(sp_dr):>9} {_signed(d_dr):>7}  "
            f"{_fmt(v1_fpr):>9} {_fmt(sp_fpr):>10} {_signed(d_fpr):>7}  "
            f"{_fmt(v1_f1):>8} {_fmt(sp_f1):>9} {_signed(d_f1):>7}  "
            f"{p_str:>8}"
        )

    print("-" * 130)
    print("dDR/dFPR/dF1 = supplement - v1.0; positive dFPR = worse on new papers")
    print("p(FPR) = bootstrap p-value for FPR difference; * = p < 0.05")
    print("=" * 130)


def print_per_tier_comparison(
    model_keys: list[str],
    v1_tier: dict[str, dict[int, dict[str, float | None]]],
    supp_tier: dict[str, dict[int, dict[str, float | None]]],
) -> None:
    """Print per-tier DR comparison between v1.0 and supplement."""
    print("\n" + "=" * 100)
    print("PER-TIER DETECTION RATE COMPARISON")
    print("=" * 100)

    tier_labels = {1: "Tier 1 (Easy)", 2: "Tier 2 (Medium)", 3: "Tier 3 (Hard)"}
    for tier in [1, 2, 3]:
        print(f"\n{tier_labels[tier]}")
        print(f"{'Model':<20} {'DR (v1)':>8} {'DR (sup)':>9} {'delta':>7}")
        print("-" * 50)
        for mk in model_keys:
            v1_d = v1_tier.get(mk, {}).get(tier, {})
            sp_d = supp_tier.get(mk, {}).get(tier, {})
            v1_dr = v1_d.get("detection_rate")
            sp_dr = sp_d.get("detection_rate")
            delta = (sp_dr - v1_dr) if (v1_dr is not None and sp_dr is not None) else None
            display = MODEL_REGISTRY[mk]["display"]
            print(f"{display:<20} {_fmt(v1_dr):>8} {_fmt(sp_dr):>9} {_signed(delta):>7}")


# ── LaTeX generation ──────────────────────────────────────────────────────────


def generate_latex_table(
    model_keys: list[str],
    v1_metrics: dict[str, dict[str, float | None]],
    supp_metrics: dict[str, dict[str, float | None]],
    bootstrap_results: dict[str, dict[str, float]],
    supp_n_valid: int,
    supp_n_hall: int,
    v1_n_valid: int,
    v1_n_hall: int,
) -> str:
    """Generate LaTeX table comparing v1.0 and temporal supplement performance."""
    lines: list[str] = []
    lines.append("% Auto-generated by scripts/evaluate_temporal_supplement.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Temporal robustness evaluation: LLM baseline performance on")
    lines.append(
        f"  v1.0 \\texttt{{dev\\_public}} (2021--2023; $n_v$={v1_n_valid}, $n_h$={v1_n_hall})"
    )
    lines.append(
        f"  vs.\\ temporal supplement (2024--2025; $n_v$={supp_n_valid}, $n_h$={supp_n_hall})."
    )
    lines.append("  $\\Delta$FPR $>0$ indicates degradation on recent papers.")
    lines.append(
        "  $p$-values from paired bootstrap test ($B=10{,}000$); "
        "\\textbf{bold} = significant at $\\alpha=0.05$.}"
    )
    lines.append("\\label{tab:temporal_supplement}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l cc c cc c cc c}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{2}{c}{\\textbf{DR $\\uparrow$}}"
        " & & \\multicolumn{2}{c}{\\textbf{FPR $\\downarrow$}}"
        " & & \\multicolumn{2}{c}{\\textbf{F1 $\\uparrow$}}"
        " & \\\\"
    )
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){5-6} \\cmidrule(lr){8-9}")
    lines.append(
        "\\textbf{Model}"
        " & {21--23} & {24--25}"
        " & & {21--23} & {24--25}"
        " & & {21--23} & {24--25}"
        " & $p$(FPR) \\\\"
    )
    lines.append("\\midrule")

    for mk in model_keys:
        v1 = v1_metrics.get(mk, {})
        sp = supp_metrics.get(mk, {})
        bs = bootstrap_results.get(mk, {})
        display = MODEL_REGISTRY[mk]["display"]

        v1_dr = v1.get("detection_rate")
        sp_dr = sp.get("detection_rate")
        v1_fpr = v1.get("fpr")
        sp_fpr = sp.get("fpr")
        v1_f1 = v1.get("f1")
        sp_f1 = sp.get("f1")
        p_val = bs.get("p_value")

        def _cell(val: float | None) -> str:
            return f"{val:.3f}" if val is not None else "--"

        # Bold FPR supplement if significantly worse
        sp_fpr_str = _cell(sp_fpr)
        if (
            p_val is not None
            and p_val < 0.05
            and sp_fpr is not None
            and v1_fpr is not None
            and sp_fpr > v1_fpr
        ):
            sp_fpr_str = f"\\textbf{{{sp_fpr_str}}}"

        p_str = f"{p_val:.3f}" if p_val is not None else "--"
        if p_val is not None and p_val < 0.05:
            p_str = f"\\textbf{{{p_str}}}"

        lines.append(
            f"{display}"
            f" & {_cell(v1_dr)} & {_cell(sp_dr)}"
            f" & & {_cell(v1_fpr)} & {sp_fpr_str}"
            f" & & {_cell(v1_f1)} & {_cell(sp_f1)}"
            f" & {p_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Per-tier metrics helper ───────────────────────────────────────────────────


def compute_per_tier(
    entries: list[BenchmarkEntry],
    pred_map: dict[str, Prediction],
) -> dict[int, dict[str, float | None]]:
    """Compute DR, FPR, F1 per difficulty tier."""
    result: dict[int, dict[str, float | None]] = {}
    for tier in [1, 2, 3]:
        tier_entries = [e for e in entries if e.difficulty_tier == tier or e.label == "VALID"]
        if not tier_entries:
            result[tier] = {"detection_rate": None, "fpr": None, "f1": None}
            continue
        cm = build_confusion_matrix(tier_entries, pred_map)
        result[tier] = {
            "detection_rate": cm.detection_rate,
            "fpr": cm.false_positive_rate,
            "f1": cm.f1,
        }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM baselines on the temporal supplement (2024-2025 papers)"
    )
    parser.add_argument(
        "--supplement-file",
        type=Path,
        required=True,
        help="Path to temporal supplement JSONL file",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated model keys to evaluate, or 'all'. "
            f"Available: {', '.join(MODEL_REGISTRY.keys())}"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing existing v1.0 evaluation results (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/temporal_supplement"),
        help="Directory for output files (default: results/temporal_supplement/)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/temporal_checkpoints"),
        help="Directory for checkpoint files (default: /tmp/temporal_checkpoints/)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable for OpenRouter API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip model execution; only compare existing predictions",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples for significance test (default: 10000)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Max concurrent API workers (default: 3; unused in sequential mode)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table to stdout in addition to saving",
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

    # ── Validate inputs ───────────────────────────────────────────────────

    if not args.supplement_file.exists():
        print(f"Error: supplement file not found: {args.supplement_file}", file=sys.stderr)
        sys.exit(1)

    # Determine which models to run
    if args.models == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in MODEL_REGISTRY:
                print(
                    f"Error: unknown model key '{k}'. "
                    f"Valid keys: {', '.join(MODEL_REGISTRY.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────

    supp_entries = load_supplement(args.supplement_file)
    if not supp_entries:
        print("Error: temporal supplement file is empty", file=sys.stderr)
        sys.exit(1)

    supp_n_valid = sum(1 for e in supp_entries if e.label == "VALID")
    supp_n_hall = sum(1 for e in supp_entries if e.label == "HALLUCINATED")

    # Load v1.0 dev_public for comparison
    try:
        v1_entries = load_split("dev_public", version="v1.0")
        logger.info("Loaded %d v1.0 dev_public entries", len(v1_entries))
    except FileNotFoundError as e:
        print(f"Error loading v1.0 dev_public: {e}", file=sys.stderr)
        sys.exit(1)

    v1_n_valid = sum(1 for e in v1_entries if e.label == "VALID")
    v1_n_hall = sum(1 for e in v1_entries if e.label == "HALLUCINATED")

    print("\nTemporal Supplement Evaluation")
    print(
        f"  Supplement: {len(supp_entries)} entries ({supp_n_valid} valid, {supp_n_hall} hallucinated)"
    )
    print(
        f"  v1.0 dev_public: {len(v1_entries)} entries ({v1_n_valid} valid, {v1_n_hall} hallucinated)"
    )
    print(f"  Models: {', '.join(MODEL_REGISTRY[mk]['display'] for mk in model_keys)}")

    # ── Run models on supplement ──────────────────────────────────────────

    supp_predictions: dict[str, list[Prediction]] = {}
    if not args.skip_run:
        supp_predictions = run_all_models(
            model_keys=model_keys,
            entries=supp_entries,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            api_key_env=args.api_key_env,
            max_workers=args.max_workers,
        )
    else:
        # Load existing predictions
        for mk in model_keys:
            pred_file = args.output_dir / f"{mk}_temporal_supplement_predictions.jsonl"
            if pred_file.exists():
                try:
                    supp_predictions[mk] = load_predictions(pred_file)
                    logger.info(
                        "Loaded %d predictions for %s",
                        len(supp_predictions[mk]),
                        MODEL_REGISTRY[mk]["display"],
                    )
                except Exception as e:
                    logger.warning("Failed to load %s: %s", pred_file, e)

    # ── Compute metrics ───────────────────────────────────────────────────

    v1_metrics: dict[str, dict[str, float | None]] = {}
    supp_metrics: dict[str, dict[str, float | None]] = {}
    v1_tier_metrics: dict[str, dict[int, dict[str, float | None]]] = {}
    supp_tier_metrics: dict[str, dict[int, dict[str, float | None]]] = {}
    bootstrap_results: dict[str, dict[str, float]] = {}

    active_models: list[str] = []

    for mk in model_keys:
        info = MODEL_REGISTRY[mk]

        # --- Supplement metrics ---
        supp_preds = supp_predictions.get(mk)
        if not supp_preds:
            logger.info("No supplement predictions for %s; skipping", info["display"])
            continue

        active_models.append(mk)

        supp_result = evaluate(
            supp_entries,
            supp_preds,
            tool_name=f"{mk}_temporal_supplement",
            split_name="temporal_supplement",
        )
        supp_metrics[mk] = {
            "detection_rate": supp_result.detection_rate,
            "fpr": supp_result.false_positive_rate,
            "f1": supp_result.f1_hallucination,
            "tw_f1": supp_result.tier_weighted_f1,
            "ece": supp_result.ece,
        }

        supp_pred_map = {p.bibtex_key: p for p in supp_preds}
        supp_tier_metrics[mk] = compute_per_tier(supp_entries, supp_pred_map)

        # Save per-model eval JSON
        eval_path = args.output_dir / f"{mk}_temporal_supplement_eval.json"
        with open(eval_path, "w") as f:
            json.dump(supp_result.to_dict(), f, indent=2, ensure_ascii=False)

        # --- v1.0 metrics ---
        v1_preds = load_v1_predictions(args.results_dir, mk)
        v1_eval = load_v1_eval_json(args.results_dir, mk)

        if v1_preds:
            v1_result = evaluate(
                v1_entries,
                v1_preds,
                tool_name=f"{mk}_v1",
                split_name="dev_public",
            )
            v1_metrics[mk] = {
                "detection_rate": v1_result.detection_rate,
                "fpr": v1_result.false_positive_rate,
                "f1": v1_result.f1_hallucination,
                "tw_f1": v1_result.tier_weighted_f1,
                "ece": v1_result.ece,
            }
            v1_pred_map = {p.bibtex_key: p for p in v1_preds}
            v1_tier_metrics[mk] = compute_per_tier(v1_entries, v1_pred_map)

            # Bootstrap test
            logger.info("Running bootstrap FPR test for %s...", info["display"])
            bs = bootstrap_fpr_difference(
                v1_entries,
                v1_pred_map,
                supp_entries,
                supp_pred_map,
                n_bootstrap=args.n_bootstrap,
            )
            if bs:
                bootstrap_results[mk] = bs

        elif v1_eval:
            # Fall back to eval JSON (no per-entry bootstrap possible)
            v1_metrics[mk] = {
                "detection_rate": v1_eval.get("detection_rate"),
                "fpr": v1_eval.get("false_positive_rate"),
                "f1": v1_eval.get("f1_hallucination"),
                "tw_f1": v1_eval.get("tier_weighted_f1"),
                "ece": v1_eval.get("ece"),
            }
            # Per-tier from eval JSON
            ptm = v1_eval.get("per_tier_metrics", {})
            v1_tier_metrics[mk] = {}
            for tier in [1, 2, 3]:
                td = ptm.get(str(tier), {})
                v1_tier_metrics[mk][tier] = {
                    "detection_rate": td.get("detection_rate"),
                    "fpr": td.get("false_positive_rate"),
                    "f1": td.get("f1"),
                }
        else:
            logger.warning("No v1.0 results for %s; comparison unavailable", info["display"])
            v1_metrics[mk] = {}

    if not active_models:
        print("\nNo models produced predictions. Nothing to compare.", file=sys.stderr)
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────────

    print_comparison_table(active_models, v1_metrics, supp_metrics, bootstrap_results)
    print_per_tier_comparison(active_models, v1_tier_metrics, supp_tier_metrics)

    # ── Bootstrap details ─────────────────────────────────────────────────

    if bootstrap_results:
        print("\n" + "=" * 80)
        print("BOOTSTRAP SIGNIFICANCE TEST DETAILS (FPR difference)")
        print("=" * 80)
        for mk in active_models:
            bs_detail = bootstrap_results.get(mk)
            if bs_detail is None:
                continue
            display = MODEL_REGISTRY[mk]["display"]
            sig = "SIGNIFICANT" if bs_detail["p_value"] < 0.05 else "not significant"
            print(
                f"\n  {display}:"
                f"\n    v1.0 FPR:         {bs_detail['v1_fpr']:.4f}  (n={bs_detail['n_v1_valid']} valid)"
                f"\n    Supplement FPR:   {bs_detail['supplement_fpr']:.4f}  (n={bs_detail['n_supp_valid']} valid)"
                f"\n    Difference:       {bs_detail['observed_diff']:+.4f}"
                f"\n    95% CI:           [{bs_detail['ci_lower']:+.4f}, {bs_detail['ci_upper']:+.4f}]"
                f"\n    p-value:          {bs_detail['p_value']:.4f}  ({sig})"
            )

    # ── Save combined results JSON ────────────────────────────────────────

    combined = {
        "supplement_file": str(args.supplement_file),
        "supplement_stats": {
            "n_total": len(supp_entries),
            "n_valid": supp_n_valid,
            "n_hallucinated": supp_n_hall,
        },
        "v1_stats": {
            "n_total": len(v1_entries),
            "n_valid": v1_n_valid,
            "n_hallucinated": v1_n_hall,
        },
        "models": {mk: MODEL_REGISTRY[mk]["display"] for mk in active_models},
        "v1_metrics": v1_metrics,
        "supplement_metrics": supp_metrics,
        "bootstrap_fpr": bootstrap_results,
    }

    combined_path = args.output_dir / "temporal_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nCombined results saved to {combined_path}")

    # ── LaTeX output ──────────────────────────────────────────────────────

    latex_content = generate_latex_table(
        active_models,
        v1_metrics,
        supp_metrics,
        bootstrap_results,
        supp_n_valid=supp_n_valid,
        supp_n_hall=supp_n_hall,
        v1_n_valid=v1_n_valid,
        v1_n_hall=v1_n_hall,
    )

    latex_path = args.output_dir / "temporal_comparison.tex"
    with open(latex_path, "w") as f:
        f.write(latex_content + "\n")
    print(f"LaTeX table saved to {latex_path}")

    if args.latex:
        print("\n" + "=" * 80)
        print("LATEX OUTPUT")
        print("=" * 80)
        print(latex_content)


if __name__ == "__main__":
    main()
