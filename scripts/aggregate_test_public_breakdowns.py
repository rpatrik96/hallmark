"""Aggregate per-tier DR/FPR for 6 finished test_public LLM verifiers.

Deliverables:
  A) results/per_split_breakdown/{model}_test_public.txt  — detailed eval text
  B) tables/cross_split_per_tier.csv                     — 18-row comparison
  C) tables/cross_split_per_tier.tex                     — booktabs LaTeX table

Dev-public sources:
  - claude_opus_4_7 : data/v1.0/baseline_results/llm_openrouter_claude_opus_4_7_dev_public.json
  - others          : evaluate from results/llm_openrouter_*_dev_public_predictions.jsonl
                      or results/new_models/*.jsonl (qwen_max, llama_4_maverick)

Test-public sources:
  - per_tier_metrics pulled from results/checkpoints/llm_openrouter_*_test_public/eval.json
  - detailed text produced by running: hallmark evaluate --predictions <jsonl> --split test_public --detailed
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent

MODELS: list[dict] = [
    {
        "key": "gemini_flash",
        "display": "Gemini 2.5 Flash",
        "test_predictions": REPO
        / "results/checkpoints/llm_openrouter_gemini_flash_test_public"
        / "openrouter_google_gemini-2.5-flash.jsonl",
        "test_eval_json": REPO
        / "results/checkpoints/llm_openrouter_gemini_flash_test_public/eval.json",
        "dev_predictions": REPO
        / "results/llm_openrouter_gemini_flash_dev_public_predictions.jsonl",
        "dev_eval_json": None,
    },
    {
        "key": "mistral",
        "display": "Mistral Large 2512",
        "test_predictions": REPO
        / "results/checkpoints/llm_openrouter_mistral_test_public"
        / "openrouter_mistralai_mistral-large-2512.jsonl",
        "test_eval_json": REPO / "results/checkpoints/llm_openrouter_mistral_test_public/eval.json",
        "dev_predictions": REPO / "results/llm_openrouter_mistral_dev_public_predictions.jsonl",
        "dev_eval_json": None,
    },
    {
        "key": "qwen_max",
        "display": "Qwen3-VL-235B",
        "test_predictions": REPO
        / "results/checkpoints/llm_openrouter_qwen_max_test_public"
        / "openrouter_qwen_qwen3-vl-235b-a22b-instruct.jsonl",
        "test_eval_json": REPO
        / "results/checkpoints/llm_openrouter_qwen_max_test_public/eval.json",
        "dev_predictions": REPO / "results/new_models/qwen_max.jsonl",
        "dev_eval_json": None,
    },
    {
        "key": "llama_4_maverick",
        "display": "Llama 4 Maverick",
        "test_predictions": REPO
        / "results/checkpoints/llm_openrouter_llama_4_maverick_test_public"
        / "openrouter_meta-llama_llama-4-maverick.jsonl",
        "test_eval_json": REPO
        / "results/checkpoints/llm_openrouter_llama_4_maverick_test_public/eval.json",
        "dev_predictions": REPO / "results/new_models/llama4_maverick.jsonl",
        "dev_eval_json": None,
    },
    {
        "key": "qwen",
        "display": "Qwen3-235B",
        "test_predictions": REPO
        / "results/checkpoints/llm_openrouter_qwen_test_public"
        / "openrouter_qwen_qwen3-235b-a22b-2507.jsonl",
        "test_eval_json": REPO / "results/checkpoints/llm_openrouter_qwen_test_public/eval.json",
        "dev_predictions": REPO / "results/llm_openrouter_qwen_dev_public_predictions.jsonl",
        "dev_eval_json": None,
    },
    {
        "key": "claude_opus_4_7",
        "display": "Claude Opus 4.7",
        "test_predictions": REPO
        / "results/checkpoints/llm_openrouter_claude_opus_4_7_test_public"
        / "openrouter_anthropic_claude-opus-4.7.jsonl",
        "test_eval_json": REPO
        / "results/checkpoints/llm_openrouter_claude_opus_4_7_test_public/eval.json",
        "dev_predictions": None,
        "dev_eval_json": REPO
        / "data/v1.0/baseline_results/llm_openrouter_claude_opus_4_7_dev_public.json",
    },
]

OUT_BREAKDOWN = REPO / "results/per_split_breakdown"
OUT_TABLES = REPO / "tables"

HALLMARK_CMD = ["uv", "run", "hallmark"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_hallmark_evaluate(
    predictions: Path,
    split: str,
    detailed: bool = True,
    fmt: str = "text",
) -> str:
    """Run 'hallmark evaluate' and return stdout as a string."""
    cmd = [
        *HALLMARK_CMD,
        "evaluate",
        "--predictions",
        str(predictions),
        "--split",
        split,
        "--format",
        fmt,
    ]
    if detailed:
        cmd.append("--detailed")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-800:]}", file=sys.stderr)
        raise RuntimeError(f"hallmark evaluate failed (rc={result.returncode}) for {predictions}")
    return result.stdout


def load_tier_metrics(eval_json: Path) -> dict[str, dict]:  # type: ignore[type-arg]
    """Return per_tier_metrics dict from an already-computed eval JSON."""
    from typing import Any, cast

    with open(eval_json) as f:
        d: dict[str, Any] = cast(dict[str, Any], json.load(f))
    return cast("dict[str, dict]", d["per_tier_metrics"])  # type: ignore[type-arg]


def evaluate_predictions_to_tier_metrics(predictions: Path, split: str) -> dict[str, dict]:  # type: ignore[type-arg]
    """Run hallmark evaluate and extract per_tier_metrics from JSON output."""
    # Use --format text and parse per_tier from the output? No — pipe to a temp
    # JSON output file to get structured data.
    import tempfile
    from typing import Any, cast

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        *HALLMARK_CMD,
        "evaluate",
        "--predictions",
        str(predictions),
        "--split",
        split,
        "--output",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-800:]}", file=sys.stderr)
        raise RuntimeError(f"hallmark evaluate failed (rc={result.returncode}) for {predictions}")
    with open(tmp_path) as f:
        d: dict[str, Any] = cast(dict[str, Any], json.load(f))
    os.unlink(tmp_path)
    return cast("dict[str, dict]", d["per_tier_metrics"])  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_BREAKDOWN.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []  # will hold 18 rows for CSV/LaTeX

    for model in MODELS:
        key = model["key"]
        display = model["display"]
        print(f"\n{'=' * 60}")
        print(f"Processing: {display} ({key})")
        print("=" * 60)

        # --- Deliverable A: detailed text eval on test_public ---
        txt_path = OUT_BREAKDOWN / f"{key}_test_public.txt"
        print(f"  Running detailed eval → {txt_path.relative_to(REPO)}")
        text_out = run_hallmark_evaluate(
            model["test_predictions"],
            split="test_public",
            detailed=True,
            fmt="text",
        )
        txt_path.write_text(text_out)
        print(f"  Saved {len(text_out)} chars")

        # --- Load test per_tier_metrics (already computed) ---
        print("  Loading test_public tier metrics from eval.json")
        test_tiers = load_tier_metrics(model["test_eval_json"])

        # --- Load or compute dev per_tier_metrics ---
        if model["dev_eval_json"] is not None:
            print("  Loading dev_public tier metrics from existing JSON")
            dev_tiers = load_tier_metrics(model["dev_eval_json"])
        else:
            print("  Computing dev_public tier metrics from predictions JSONL")
            dev_tiers = evaluate_predictions_to_tier_metrics(
                model["dev_predictions"], split="dev_public"
            )

        # --- Build rows for CSV/LaTeX ---
        for tier in ["1", "2", "3"]:
            dev_t = dev_tiers[tier]
            test_t = test_tiers[tier]
            dev_dr = dev_t["detection_rate"]
            test_dr = test_t["detection_rate"]
            dev_fpr = dev_t["false_positive_rate"]
            test_fpr = test_t["false_positive_rate"]
            rows.append(
                {
                    "model": display,
                    "key": key,
                    "tier": int(tier),
                    "dev_dr": dev_dr,
                    "test_dr": test_dr,
                    "dev_fpr": dev_fpr,
                    "test_fpr": test_fpr,
                    "delta_dr": test_dr - dev_dr,
                    "delta_fpr": test_fpr - dev_fpr,
                }
            )
            print(
                f"  Tier {tier}: dev DR={dev_dr:.3f} FPR={dev_fpr:.3f} | "
                f"test DR={test_dr:.3f} FPR={test_fpr:.3f} | "
                f"ΔDR={test_dr - dev_dr:+.3f} ΔFPR={test_fpr - dev_fpr:+.3f}"
            )

    if not rows:
        print("No model rows collected — all predictions missing. Skipping CSV/LaTeX output.")
        return

    # --- Deliverable B: CSV ---
    csv_path = OUT_TABLES / "cross_split_per_tier.csv"
    fieldnames = [
        "model",
        "tier",
        "dev_dr",
        "test_dr",
        "dev_fpr",
        "test_fpr",
        "delta_dr",
        "delta_fpr",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path.relative_to(REPO)}")

    # --- Deliverable C: LaTeX ---
    tex_path = OUT_TABLES / "cross_split_per_tier.tex"
    _write_latex(rows, tex_path)
    print(f"Wrote {tex_path.relative_to(REPO)}")

    # --- Summary: largest ΔFPR rows ---
    sorted_by_dfpr = sorted(rows, key=lambda r: abs(r["delta_fpr"]), reverse=True)
    print("\n--- Top 6 rows by |ΔFPR| (test - dev) ---")
    for r in sorted_by_dfpr[:6]:
        print(
            f"  {r['model']} Tier {r['tier']}: "
            f"ΔFPR={r['delta_fpr']:+.3f} (dev={r['dev_fpr']:.3f}, test={r['test_fpr']:.3f})"
        )

    # --- Sanity check ---
    print("\n--- Sanity check (flag suspicious rows) ---")
    issues_found = False
    for r in rows:
        flags = []
        for field in ("dev_dr", "test_dr", "dev_fpr", "test_fpr"):
            v = r[field]
            if v is None or (isinstance(v, float) and (v < 0 or v != v)):
                flags.append(f"{field}={v}")
        if flags:
            print(f"  ISSUE {r['model']} Tier {r['tier']}: {', '.join(flags)}")
            issues_found = True
    if not issues_found:
        print("  All values look sane (no negative, NaN, or None).")


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _fmt_delta(v: float) -> str:
    return f"{v:+.3f}"


def _write_latex(rows: list[dict], path: Path) -> None:
    tier_labels = {1: "Tier 1 (Easy)", 2: "Tier 2 (Medium)", 3: "Tier 3 (Hard)"}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-tier detection rate (DR) and false positive rate (FPR) on "
        r"\texttt{dev\_public} and \texttt{test\_public} for six LLM verifiers. "
        r"$\Delta$DR and $\Delta$FPR measure cross-split robustness "
        r"(test\,$-$\,dev). DeepSeek-R1 omitted (run still in progress).}",
        r"\label{tab:cross-split-per-tier}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Tier "
        r"& \multicolumn{2}{c}{DR} "
        r"& \multicolumn{2}{c}{FPR} "
        r"& $\Delta$DR & $\Delta$FPR \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"& & dev & test & dev & test & & \\",
        r"\midrule",
    ]

    # Group by model
    model_keys_seen: list[str] = []
    model_rows: dict[str, list[dict]] = {}
    for r in rows:
        k = r["key"]
        if k not in model_keys_seen:
            model_keys_seen.append(k)
            model_rows[k] = []
        model_rows[k].append(r)

    for i, key in enumerate(model_keys_seen):
        if i > 0:
            lines.append(r"\midrule")
        model_data = model_rows[key]
        display = model_data[0]["model"]
        for j, r in enumerate(sorted(model_data, key=lambda x: x["tier"])):
            tier_str = tier_labels[r["tier"]]
            model_cell = r"\multirow{3}{*}{" + display + "}" if j == 0 else ""
            lines.append(
                f"{model_cell} & {tier_str} "
                f"& {_fmt(r['dev_dr'])} & {_fmt(r['test_dr'])} "
                f"& {_fmt(r['dev_fpr'])} & {_fmt(r['test_fpr'])} "
                f"& {_fmt_delta(r['delta_dr'])} & {_fmt_delta(r['delta_fpr'])} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
