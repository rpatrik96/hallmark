#!/usr/bin/env python3
"""Run the cutoff-aware prompt ablation on the temporal supplement.

Tests H2: when explicitly reminded of the training-data cutoff, do LLMs
route post-cutoff citations to UNCERTAIN rather than over-flag them as
HALLUCINATED? Loads the 2024-2025 temporal supplement, re-runs a subset
of LLM baselines with ``cutoff_aware=True``, and compares to the default
predictions already in ``results/temporal_supplement/``.

Subset (by design): GPT-5.1 (OpenAI), Gemini 2.5 Flash and Qwen3-235B
(OpenRouter). DeepSeek variants omitted -- the default-prompt temporal
run already saturates at UNCERTAIN. Claude omitted -- no API key in this
environment.

Usage::

    source /tmp/.openai_env
    source /tmp/.or_env
    python scripts/run_cutoff_aware_ablation.py

Outputs go to ``results/temporal_supplement/``:
  - ``{model_key}_cutoff_aware_temporal_predictions.jsonl``
  - ``{model_key}_cutoff_aware_temporal.json`` (eval metrics)
  - ``cutoff_aware_comparison.json``              (delta vs default)
  - ``cutoff_aware_comparison_table.tex``         (paper-ready LaTeX)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.llm_verifier import (
    verify_with_openai,
    verify_with_openrouter,
)
from hallmark.dataset.schema import (
    Prediction,
    load_entries,
    load_predictions,
    save_predictions,
)
from hallmark.evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


SUBSET: dict[str, dict[str, str]] = {
    "llm_openai": {
        "provider": "openai",
        "model_id": "gpt-5.1",
        "api_key_env": "OPENAI_API_KEY",
        "display": "GPT-5.1",
    },
    "llm_openai_gpt54": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
        "display": "GPT-5.4",
    },
    "llm_openrouter_gemini_flash": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "Gemini 2.5 Flash",
    },
    "llm_openrouter_qwen": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "api_key_env": "OPENROUTER_API_KEY",
        "display": "Qwen3-235B",
    },
}


def run_one(
    model_key: str,
    supplement_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    cutoff_aware: bool = True,
) -> list[Prediction] | None:
    info = SUBSET[model_key]
    api_key = os.environ.get(info["api_key_env"])
    if not api_key:
        logger.warning("Skipping %s: env var %s not set", info["display"], info["api_key_env"])
        return None

    entries = load_entries(supplement_path)
    blind = [e.to_blind() for e in entries]

    suffix = "cutoff_aware" if cutoff_aware else "default"
    pred_file = output_dir / f"{model_key}_{suffix}_temporal_predictions.jsonl"
    if pred_file.exists():
        try:
            existing = load_predictions(pred_file)
            if len(existing) == len(blind):
                logger.info(
                    "Reusing existing cutoff-aware predictions for %s (%d entries)",
                    info["display"],
                    len(existing),
                )
                return existing
            logger.info(
                "Partial predictions for %s (%d/%d) — resuming",
                info["display"],
                len(existing),
                len(blind),
            )
        except Exception as e:
            logger.warning("Failed to read %s; re-running: %s", pred_file, e)

    logger.info(
        "Running %s %s on %d entries (file=%s)...",
        suffix.upper(),
        info["display"],
        len(blind),
        supplement_path.name,
    )

    if info["provider"] == "openai":
        predictions = verify_with_openai(
            blind,
            model=info["model_id"],
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
            cutoff_aware=cutoff_aware,
        )
    elif info["provider"] == "openrouter":
        predictions = verify_with_openrouter(
            blind,
            model=info["model_id"],
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
            cutoff_aware=cutoff_aware,
        )
    else:
        raise ValueError(info["provider"])

    pred_file.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, pred_file)
    logger.info("Saved predictions to %s", pred_file)
    return predictions


def compute_eval(
    predictions: list[Prediction],
    entries_path: Path,
    eval_out: Path,
    model_key: str,
) -> dict[str, Any]:
    entries = load_entries(entries_path)
    result = evaluate(
        entries,
        predictions,
        tool_name=f"{model_key}_cutoff_aware",
        split_name="temporal_2024_2025",
    )
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result.__dict__
    eval_out.write_text(json.dumps(result_dict, indent=2, default=str))
    logger.info("Saved eval to %s", eval_out)
    return result_dict


def _uncertain_rate(preds: list[Prediction]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p in preds if p.label == "UNCERTAIN") / len(preds)


def build_comparison(
    supplement_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build default vs cutoff-aware delta table on the supplement.

    Default eval is recomputed against the (possibly deduped) supplement to
    guarantee apples-to-apples denominators with the cutoff-aware run.
    """
    entries = load_entries(supplement_path)
    entry_keys = {e.bibtex_key for e in entries}

    rows: list[dict[str, Any]] = []
    for model_key, info in SUBSET.items():
        ca_eval_path = output_dir / f"{model_key}_cutoff_aware_temporal.json"
        # Accept either the legacy supplement-run naming or the new --default-run naming.
        default_pred_candidates = [
            output_dir / f"{model_key}_default_temporal_predictions.jsonl",
            output_dir / f"{model_key}_temporal_predictions.jsonl",
        ]
        default_pred_path: Path | None = next(
            (p for p in default_pred_candidates if p.exists()), None
        )
        ca_pred_path = output_dir / f"{model_key}_cutoff_aware_temporal_predictions.jsonl"

        if not (ca_eval_path.exists() and default_pred_path is not None):
            logger.warning(
                "Skipping %s from comparison: missing files (ca_eval=%s, default_preds=%s)",
                info["display"],
                ca_eval_path.exists(),
                default_pred_path is not None,
            )
            continue

        # Recompute default eval on the current supplement for comparability.
        default_preds_all = load_predictions(default_pred_path)
        default_preds = [p for p in default_preds_all if p.bibtex_key in entry_keys]
        default_result = evaluate(
            entries,
            default_preds,
            tool_name=f"{model_key}_default_recomputed",
            split_name="temporal_2024_2025",
        )
        default_eval = (
            default_result.to_dict()
            if hasattr(default_result, "to_dict")
            else default_result.__dict__
        )

        ca_eval = json.loads(ca_eval_path.read_text())
        ca_preds = load_predictions(ca_pred_path) if ca_pred_path.exists() else []

        rows.append(
            {
                "model": info["display"],
                "model_key": model_key,
                "model_id": info["model_id"],
                "default": {
                    "detection_rate": default_eval.get("detection_rate"),
                    "false_positive_rate": default_eval.get("false_positive_rate"),
                    "f1": default_eval.get("f1_hallucination"),
                    "ece": default_eval.get("ece"),
                    "uncertain_rate": _uncertain_rate(default_preds),
                    "num_uncertain": default_eval.get("num_uncertain"),
                    "num_entries": default_eval.get("num_entries"),
                },
                "cutoff_aware": {
                    "detection_rate": ca_eval.get("detection_rate"),
                    "false_positive_rate": ca_eval.get("false_positive_rate"),
                    "f1": ca_eval.get("f1_hallucination"),
                    "ece": ca_eval.get("ece"),
                    "uncertain_rate": _uncertain_rate(ca_preds),
                    "num_uncertain": ca_eval.get("num_uncertain"),
                    "num_entries": ca_eval.get("num_entries"),
                },
            }
        )

    return {"rows": rows, "supplement": str(supplement_path)}


def _fmt(x: Any) -> str:
    if x is None:
        return "--"
    if isinstance(x, (int, float)):
        return f"{x * 100:.1f}" if abs(x) <= 1.0 else f"{x:.1f}"
    return str(x)


def emit_latex(comparison: dict[str, Any], out_path: Path) -> None:
    rows = comparison["rows"]
    lines: list[str] = []
    lines.append("% Auto-generated by scripts/run_cutoff_aware_ablation.py. Do not edit by hand.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{Cutoff-aware prompt ablation on the 2024--2025 temporal supplement "
        "(\\(N = 480\\) entries, all post-cutoff for the three models tested). "
        "For each model we report detection rate (DR, \\%), false-positive rate "
        "(FPR, \\%), F1 on the hallucinated class, Expected Calibration Error (ECE), "
        "and the fraction of entries the model routes to UNCERTAIN. "
        "$\\Delta$ columns show the cutoff-aware minus default change. "
        "Negative $\\Delta$FPR and positive $\\Delta$UNC are consistent with "
        "\\emph{partial metacognition}: reminding the model of its cutoff reduces "
        "confident over-flagging but also inflates abstention.}"
    )
    lines.append("\\label{tab:cutoff-aware-results}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{lrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Model & Prompt & DR (\\%) & FPR (\\%) & F1 & ECE & UNC (\\%) & $\\Delta$FPR & $\\Delta$UNC \\\\"
    )
    lines.append("\\midrule")
    for row in rows:
        d = row["default"]
        c = row["cutoff_aware"]
        d_fpr = d["false_positive_rate"] or 0.0
        c_fpr = c["false_positive_rate"] or 0.0
        d_unc = d["uncertain_rate"] or 0.0
        c_unc = c["uncertain_rate"] or 0.0
        delta_fpr = (c_fpr - d_fpr) * 100
        delta_unc = (c_unc - d_unc) * 100
        lines.append(
            f"{row['model']} & default & {_fmt(d['detection_rate'])} & "
            f"{_fmt(d['false_positive_rate'])} & {_fmt(d['f1'])} & {_fmt(d['ece'])} & "
            f"{_fmt(d['uncertain_rate'])} & -- & -- \\\\"
        )
        lines.append(
            f" & cutoff-aware & {_fmt(c['detection_rate'])} & "
            f"{_fmt(c['false_positive_rate'])} & {_fmt(c['f1'])} & {_fmt(c['ece'])} & "
            f"{_fmt(c['uncertain_rate'])} & {delta_fpr:+.1f} & {delta_unc:+.1f} \\\\"
        )
        lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote LaTeX table to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--supplement-file",
        type=Path,
        default=Path("results/temporal_supplement/temporal_supplement_2024_2025.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/temporal_supplement"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/temporal_checkpoints_cutoff_aware"),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(SUBSET.keys()),
        help="Comma-separated subset of: " + ",".join(SUBSET.keys()),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only compute the comparison table from existing predictions",
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Run the DEFAULT prompt instead of the cutoff-aware prompt "
        "(useful for filling in missing default-prompt baselines on "
        "arbitrary supplement files). Predictions are saved with a "
        "'_default_' infix so they do not collide.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip building the default-vs-cutoff-aware comparison table "
        "(use when running a --default pass that only produces one side).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.supplement_file.exists():
        logger.error("Supplement file not found: %s", args.supplement_file)
        sys.exit(1)

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in model_keys:
        if k not in SUBSET:
            logger.error("Unknown model key: %s", k)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cutoff_aware_flag = not args.default
    suffix = "cutoff_aware" if cutoff_aware_flag else "default"

    if not args.skip_run:
        for mk in model_keys:
            preds = run_one(
                mk,
                args.supplement_file,
                args.output_dir,
                args.checkpoint_dir,
                cutoff_aware=cutoff_aware_flag,
            )
            if preds is None:
                continue
            eval_out = args.output_dir / f"{mk}_{suffix}_temporal.json"
            compute_eval(preds, args.supplement_file, eval_out, mk)

    if args.skip_comparison:
        return

    comparison = build_comparison(args.supplement_file, args.output_dir)
    comp_path = args.output_dir / "cutoff_aware_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    logger.info("Saved comparison to %s", comp_path)

    latex_path = args.output_dir / "cutoff_aware_comparison_table.tex"
    emit_latex(comparison, latex_path)

    # Print console summary
    print("\n=== Cutoff-aware ablation summary ===")
    for row in comparison["rows"]:
        d = row["default"]
        c = row["cutoff_aware"]
        d_fpr = (d["false_positive_rate"] or 0.0) * 100
        c_fpr = (c["false_positive_rate"] or 0.0) * 100
        d_unc = (d["uncertain_rate"] or 0.0) * 100
        c_unc = (c["uncertain_rate"] or 0.0) * 100
        d_dr = (d["detection_rate"] or 0.0) * 100
        c_dr = (c["detection_rate"] or 0.0) * 100
        print(
            f"  {row['model']:20s}  DR {d_dr:5.1f} -> {c_dr:5.1f}  "
            f"FPR {d_fpr:5.1f} -> {c_fpr:5.1f} (Δ{c_fpr - d_fpr:+.1f})  "
            f"UNC {d_unc:5.1f} -> {c_unc:5.1f} (Δ{c_unc - d_unc:+.1f})"
        )


if __name__ == "__main__":
    main()
