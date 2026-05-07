"""Compute classification metrics for the thinking-budget smoke-test cells.

Reads ``results/checkpoints/smoke_test_thinking_budget/{model}_{regime}.jsonl``
plus ``data/v1.0/dev_public.jsonl``, computes DR / FPR / F1 / MCC per cell
treating UNCERTAIN and parse-failure entries as non-detections (label=VALID),
and prints a LaTeX-ready table for App.~G of the paper.

Usage:

    uv run python scripts/smoke_test_summarize.py

The script produces:
  - stdout: pretty table
  - tables/smoke_test_thinking_budget.csv
  - tables/smoke_test_thinking_budget.tex (booktabs, ready to paste)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "results" / "checkpoints" / "smoke_test_thinking_budget"
DATA_FILE = ROOT / "data" / "v1.0" / "dev_public.jsonl"
TABLES_DIR = ROOT / "tables"

CELLS: list[tuple[str, str, str, str]] = [
    # (model_key, regime, display_label, budget_str)
    ("gpt-5-5", "a", "GPT-5.5", "256"),
    ("gpt-5-5", "b", "GPT-5.5", "1024"),
    ("gpt-5-5", "c", "GPT-5.5", "4096"),
    ("gemini-3-1-pro", "a", "Gemini 3.1 Pro", "2048 + 1024 reas."),
    ("gemini-3-1-pro", "b", "Gemini 3.1 Pro", "8192 + 4096 reas."),
    ("gemini-3-1-pro", "c", "Gemini 3.1 Pro", "16384 + 8192 reas."),
    ("deepseek-v4-pro", "a", "DeepSeek-V4-Pro", "4096, effort=low"),
    ("deepseek-v4-pro", "b", "DeepSeek-V4-Pro", "8192, effort=high"),
    ("deepseek-v4-pro", "c", "DeepSeek-V4-Pro", "16384, effort=high"),
]


def load_truth() -> dict[str, bool]:
    truth: dict[str, bool] = {}
    for line in DATA_FILE.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        truth[d["bibtex_key"]] = d["label"] == "HALLUCINATED"
    return truth


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    pos = tp + fn
    neg = tn + fp
    dr = tp / pos if pos else 0.0
    fpr = fp / neg if neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * dr / (precision + dr) if (precision + dr) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "n": pos + neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "dr": dr,
        "fpr": fpr,
        "f1": f1,
        "mcc": mcc,
    }


def evaluate_cell(jsonl: Path, truth: dict[str, bool]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gt_pos = truth.get(rec["bibtex_key"])
        if gt_pos is None:
            continue
        # Treat UNCERTAIN and parse-failure as label=VALID (non-detection)
        pred_pos = rec["label"] == "HALLUCINATED"
        if gt_pos and pred_pos:
            tp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        elif (not gt_pos) and pred_pos:
            fp += 1
        else:
            tn += 1
    return metrics(tp, fp, tn, fn)


def main() -> None:
    truth = load_truth()
    print(f"Loaded ground truth for {len(truth)} entries")

    # Use Any so the CSV writer doesn't complain about mixed str/float/int values.
    rows: list[dict[str, Any]] = []

    print(
        f"\n{'Model':<18} {'Bud.':<22} "
        f"{'n':>3} {'PF':>5} {'p95/cap':>8} "
        f"{'DR':>5} {'FPR':>5} {'F1':>5} {'MCC':>5} "
        f"{'mean_t':>7} {'p95_t':>6} {'$':>5}"
    )
    print("-" * 110)

    # Build cap lookup from COHORT in smoke_test_thinking_budget for a single
    # source of truth. Fallback: hardcoded dict typed dict[str, int] that
    # covers every entry in CELLS — an assertion guards against drift.
    _CAP_FALLBACK: dict[str, int] = {
        "gpt-5-5_a": 256,
        "gpt-5-5_b": 1024,
        "gpt-5-5_c": 4096,
        "gemini-3-1-pro_a": 2048,
        "gemini-3-1-pro_b": 8192,
        "gemini-3-1-pro_c": 16384,
        "deepseek-v4-pro_a": 4096,
        "deepseek-v4-pro_b": 8192,
        "deepseek-v4-pro_c": 16384,
    }
    expected_cells = {f"{mk}_{r}" for mk, r, _, _ in CELLS}
    assert expected_cells <= _CAP_FALLBACK.keys(), (
        f"_CAP_FALLBACK is missing cells: {expected_cells - _CAP_FALLBACK.keys()}"
    )

    try:
        from scripts.smoke_test_thinking_budget import COHORT  # type: ignore[import]

        _cap_map: dict[str, int] = {
            f"{mk}_{r}": cfg[f"regime_{r}"]["max_tokens"]
            for mk, cfg in COHORT.items()
            for r in ("a", "b", "c")
            if f"regime_{r}" in cfg
        }
    except Exception:
        _cap_map = _CAP_FALLBACK

    for model_key, regime, display, budget in CELLS:
        cell = f"{model_key}_{regime}"
        jsonl = CHECKPOINT_DIR / f"{cell}.jsonl"
        summary_path = CHECKPOINT_DIR / f"{cell}.summary.json"
        if not jsonl.exists() or not summary_path.exists():
            print(f"!! missing checkpoint for {cell}")
            continue
        m = evaluate_cell(jsonl, truth)
        s = json.loads(summary_path.read_text())
        cap: int = _cap_map.get(cell, _CAP_FALLBACK[cell])
        sat = s["p95_output_tokens"] / cap
        row: dict[str, Any] = {
            "cell": cell,
            "display": display,
            "budget": budget,
            "n": m["n"],
            "parse_failure": s["parse_failure_rate"],
            "p95": s["p95_output_tokens"],
            "cap": cap,
            "saturation": sat,
            "dr": m["dr"],
            "fpr": m["fpr"],
            "f1": m["f1"],
            "mcc": m["mcc"],
            "mean_tokens": s["mean_output_tokens"],
            "p95_tokens": s["p95_output_tokens"],
            "wall_clock_s": s["total_wall_clock_seconds"],
            "cost_usd": s["estimated_cost_usd"],
        }
        rows.append(row)
        print(
            f"{display:<18} {budget:<22} "
            f"{m['n']:>3} {s['parse_failure_rate']:>5.2f} "
            f"{sat:>8.3f} "
            f"{m['dr']:>5.2f} {m['fpr']:>5.2f} {m['f1']:>5.2f} {m['mcc']:>5.2f} "
            f"{s['mean_output_tokens']:>7.0f} {s['p95_output_tokens']:>6} "
            f"{s['estimated_cost_usd']:>5.2f}"
        )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    if not rows:
        print("No cells found — checkpoints missing. Skipping CSV/LaTeX output.")
        return
    TABLES_DIR.mkdir(exist_ok=True)
    csv_path = TABLES_DIR / "smoke_test_thinking_budget.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------
    tex_path = TABLES_DIR / "smoke_test_thinking_budget.tex"
    lines = [
        r"\begin{table}[t]",
        r"\caption{\textbf{Thinking-budget regime boundary smoke test.} "
        r"Stratified $n{=}100$ subsample of \texttt{dev\_public}. "
        r"PF = parse-failure rate; "
        r"$p_{95}/\text{cap}$ = saturation ratio (1.00 = the 95th-percentile output hits the budget cap). "
        r"DR / FPR / F1 / MCC computed treating UNCERTAIN and parse-failure entries as label=\texttt{VALID}. "
        r"\$ is the OpenRouter list-price cost for the cell at May 2026 rates.}",
        r"\label{tab:smoke-results}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Budget} & \textbf{$n$} & \textbf{PF} & "
        r"\textbf{$p_{95}/\text{cap}$} & "
        r"\textbf{DR $\uparrow$} & \textbf{FPR $\downarrow$} & "
        r"\textbf{F1 $\uparrow$} & \textbf{MCC $\uparrow$} & "
        r"\textbf{mean tok} & \textbf{$p_{95}$ tok} & \textbf{\$} \\",
        r"\midrule",
    ]
    prev_display = None
    for r in rows:
        display = r["display"] if r["display"] != prev_display else ""
        prev_display = r["display"]
        lines.append(
            f"{display} & \\texttt{{{r['budget']}}} & {r['n']} & "
            f"{r['parse_failure']:.2f} & {r['saturation']:.3f} & "
            f"{r['dr']:.3f} & {r['fpr']:.3f} & {r['f1']:.3f} & {r['mcc']:.3f} & "
            f"{r['mean_tokens']:.0f} & {r['p95_tokens']} & "
            f"\\${r['cost_usd']:.2f} \\\\"
        )
        if r["cell"].endswith("_c"):
            lines.append(r"\midrule")
    # remove trailing \midrule before \bottomrule
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX: {tex_path}")

    # totals
    tot_cost = sum(r["cost_usd"] for r in rows)  # type: ignore[arg-type]
    tot_min = sum(r["wall_clock_s"] for r in rows) / 60.0  # type: ignore[arg-type]
    print(f"\nTotal cost: ${tot_cost:.2f}, total wall-clock: {tot_min:.0f} min")


if __name__ == "__main__":
    main()
