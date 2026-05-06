"""Per-hallucination-type detection rate breakdown for the smoke-test cells.

n=5 per type means per-type DRs have very wide CIs (±~35pp at 95%). The
breakdown is useful only for surfacing *relative* shape across budget regimes,
not absolute claims. We label this clearly in the rendered table.

Usage:

    uv run python scripts/smoke_test_per_type.py

Outputs:
  - stdout: pretty per-type table
  - tables/smoke_test_per_type.csv
  - tables/smoke_test_per_type.tex
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "results" / "checkpoints" / "smoke_test_thinking_budget"
DATA_FILE = ROOT / "data" / "v1.0" / "dev_public.jsonl"
TABLES_DIR = ROOT / "tables"

CELLS: list[tuple[str, str, str]] = [
    ("gpt-5-5", "a", "GPT-5.5 (256)"),
    ("gpt-5-5", "b", "GPT-5.5 (1024)"),
    ("gpt-5-5", "c", "GPT-5.5 (4096)"),
    ("gemini-3-1-pro", "a", "Gemini 3.1 Pro (2k+1k)"),
    ("gemini-3-1-pro", "b", "Gemini 3.1 Pro (8k+4k)"),
    ("gemini-3-1-pro", "c", "Gemini 3.1 Pro (16k+8k)"),
    ("deepseek-v4-pro", "a", "DS-V4-Pro (4k low)"),
    ("deepseek-v4-pro", "b", "DS-V4-Pro (8k high)"),
    ("deepseek-v4-pro", "c", "DS-V4-Pro (16k high)"),
]

# Display order: tier 1 → tier 2 → tier 3 → stress
TYPES_ORDERED = [
    ("fabricated_doi", 1),
    ("nonexistent_venue", 1),
    ("placeholder_authors", 1),
    ("future_date", 1),
    ("chimeric_title", 2),
    ("wrong_venue", 2),
    ("swapped_authors", 2),
    ("preprint_as_published", 2),
    ("hybrid_fabrication", 2),
    ("near_miss_title", 3),
    ("plausible_fabrication", 3),
    ("merged_citation", "S"),
    ("partial_author_list", "S"),
    ("arxiv_version_mismatch", "S"),
]


def load_truth() -> dict[str, tuple[bool, str | None]]:
    truth: dict[str, tuple[bool, str | None]] = {}
    for line in DATA_FILE.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        truth[d["bibtex_key"]] = (
            d["label"] == "HALLUCINATED",
            d.get("hallucination_type"),
        )
    return truth


def evaluate_cell_per_type(
    jsonl: Path, truth: dict[str, tuple[bool, str | None]]
) -> dict[str, tuple[int, int]]:
    """Return {hallucination_type: (n_detected, n_total)}.

    Treats UNCERTAIN and parse failures as non-detections.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [detected, total]
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gt = truth.get(rec["bibtex_key"])
        if gt is None:
            continue
        gt_pos, htype = gt
        if not gt_pos or htype is None:
            continue
        detected = rec["label"] == "HALLUCINATED"
        counts[htype][1] += 1
        if detected:
            counts[htype][0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def main() -> None:
    truth = load_truth()
    cell_per_type: dict[str, dict[str, tuple[int, int]]] = {}
    for model_key, regime, _ in CELLS:
        cell = f"{model_key}_{regime}"
        jsonl = CHECKPOINT_DIR / f"{cell}.jsonl"
        if not jsonl.exists():
            print(f"!! missing {jsonl}")
            continue
        cell_per_type[cell] = evaluate_cell_per_type(jsonl, truth)

    cell_keys = [f"{m}_{r}" for m, r, _ in CELLS]
    cell_displays = [d for _, _, d in CELLS]

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    header = f"{'Type':<26} {'n':>3} | " + " | ".join(
        f"{d.split(' (')[0][:11]:>11}" for d in cell_displays
    )
    print(header)
    print("-" * len(header))

    rows_csv: list[dict[str, str | float | int]] = []
    rows_tex: list[list[str]] = []

    for htype, tier in TYPES_ORDERED:
        # Use the first cell to obtain n_total per type (should be identical)
        first = cell_per_type[cell_keys[0]].get(htype, (0, 0))
        n_total = first[1]
        if n_total == 0:
            continue
        row_vals_pretty = []
        row_vals_csv: dict[str, str | float | int] = {
            "type": htype,
            "tier": tier,
            "n": n_total,
        }
        row_tex: list[str] = []
        for ck in cell_keys:
            cnt = cell_per_type[ck].get(htype, (0, n_total))
            dr = cnt[0] / cnt[1] if cnt[1] else 0.0
            row_vals_pretty.append(f"{dr:>11.2f}")
            row_vals_csv[ck] = dr
            row_tex.append(f"{dr:.2f}")
        print(f"{htype:<26} {n_total:>3} | " + " | ".join(row_vals_pretty))
        rows_csv.append(row_vals_csv)
        rows_tex.append([f"\\texttt{{{htype}}}", str(tier), str(n_total), *row_tex])

    # CSV
    TABLES_DIR.mkdir(exist_ok=True)
    csv_path = TABLES_DIR / "smoke_test_per_type.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["type", "tier", "n", *cell_keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_csv)
    print(f"\nCSV: {csv_path}")

    # LaTeX
    cap = (
        r"\caption{\textbf{Per-type detection rate for smoke-test cells "
        r"(stratified $n{=}100$ subsample of \texttt{dev\_public}, $\geq 5$ per type).} "
        r"Each cell shows the fraction of hallucinated entries of that type that the model "
        r"labelled \texttt{HALLUCINATED}; UNCERTAIN and parse failures count as non-detections. "
        r"Per-type $n$ is small ($\geq 5$); 95\% binomial CI width is roughly $\pm 35$\,pp at "
        r"$n{=}5$, so within-row differences across budgets are noise unless they exceed that "
        r"band. The table is interpretable for relative shape (which types each model finds "
        r"easiest/hardest at each budget) but not for fine absolute comparisons.}"
    )
    multicol = (
        r" & & & \multicolumn{3}{c|}{\textbf{GPT-5.5}}"
        r" & \multicolumn{3}{c|}{\textbf{Gemini 3.1 Pro}}"
        r" & \multicolumn{3}{c}{\textbf{DeepSeek-V4-Pro}} \\"
    )
    tex_lines = [
        r"\begin{table}[t]",
        cap,
        r"\label{tab:smoke-per-type}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llr|rrr|rrr|rrr}",
        r"\toprule",
        multicol,
        r"\textbf{Type} & \textbf{Tier} & \textbf{$n$} & "
        r"\textbf{A} & \textbf{B} & \textbf{C} & "
        r"\textbf{A} & \textbf{B} & \textbf{C} & "
        r"\textbf{A} & \textbf{B} & \textbf{C} \\",
        r"\midrule",
    ]
    prev_tier: str | int | None = None
    for vals in rows_tex:
        if prev_tier is not None and vals[1] != str(prev_tier):
            tex_lines.append(r"\midrule")
        prev_tier = vals[1]
        tex_lines.append(" & ".join(vals) + r" \\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    tex_path = TABLES_DIR / "smoke_test_per_type.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
