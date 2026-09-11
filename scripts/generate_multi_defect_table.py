#!/usr/bin/env python3
"""Emit the multi- vs single-defect detection table (LaTeX + CSV).

Scoring convention
------------------
``UNCERTAIN`` counts as a **miss**, so every model is scored over the full
denominator (459 single-defect / 147 multi-defect / 513 valid on
``dev_public``). This differs from ``hallmark.evaluation.metrics`` and from
``scripts/analyze_multi_defect_detection.py``, which drop UNCERTAIN from the
denominator entirely.

The strict convention is used here because the drop protocol shrinks each arm
by a *different* amount and so makes the two arms non-comparable: the GPT-5.4
cascade abstains on 21 hallucinated entries, 7 of them multi-defect, which
inflates its multi-defect rate purely by removing hard cases from a 147-entry
arm. Abstention is also a real cost in deployment — an UNCERTAIN verdict does
not catch a hallucinated citation.

Usage
-----
    uv run python scripts/generate_multi_defect_table.py
    uv run python scripts/generate_multi_defect_table.py --split dev_public
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data/v1.2")
DEFAULT_OUT_DIR = Path("tables")

#: ``cross_db_agreement`` is False for 100% of hallucinated entries, so it
#: restates the label rather than observing a defect. See
#: ``scripts/analyze_defect_multiplicity.py`` for the full argument.
CONSTANT_FIELD = "cross_db_agreement"

#: display name -> per-entry prediction dump, relative to the repo root.
CANDIDATES: dict[str, str] = {
    "GPT-5.4 + BTU (cascade)": "results/cascade_gpt54/gpt54_{split}_preds.jsonl",
    "Opus 4.7": (
        "results/checkpoints/llm_openrouter_claude_opus_4_7_{split}/"
        "openrouter_anthropic_claude-opus-4.7.jsonl"
    ),
    "bibtex-updater": "results/relabel_delta/btu_v1_2_0/bibtexupdater_{split}_per_entry.jsonl",
}


def defect_count(entry: dict[str, Any]) -> int:
    """Number of explicitly-failing subtests, excluding the constant field."""
    subtests = entry.get("subtests") or {}
    return sum(1 for name, v in subtests.items() if v is False and name != CONSTANT_FIELD)


def load_predictions(path: Path) -> dict[str, str]:
    """Map bibtex_key -> predicted label, last write per key winning.

    bibtex-updater's dump names its verdict ``pred_label``; everything else
    uses ``label``.
    """
    preds: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        label = rec.get("label") or rec.get("pred_label")
        if rec.get("bibtex_key") and label:
            preds[rec["bibtex_key"]] = label
    return preds


def detected(entries: list[dict[str, Any]], preds: dict[str, str]) -> tuple[int, int]:
    """``(flagged, n)`` — UNCERTAIN and missing predictions both count as misses."""
    flagged = sum(1 for e in entries if preds.get(e["bibtex_key"], "VALID") == "HALLUCINATED")
    return flagged, len(entries)


def build_rows(data_dir: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entries = [
        e
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
        for e in [json.loads(line)]
        if not e["bibtex_key"].startswith("__canary__")
    ]
    hallucinated = [e for e in entries if e["label"] == "HALLUCINATED"]
    single = [e for e in hallucinated if defect_count(e) == 1]
    multi = [e for e in hallucinated if defect_count(e) >= 2]
    valid = [e for e in entries if e["label"] == "VALID"]

    rows: list[dict[str, Any]] = []
    for name, template in CANDIDATES.items():
        path = Path(template.format(split=split))
        if not path.is_file():
            print(f"[skip] {name}: no per-entry dump at {path}")
            continue
        preds = load_predictions(path)
        ds, ns = detected(single, preds)
        dm, nm = detected(multi, preds)
        fp, nv = detected(valid, preds)
        uncertain = sum(
            1 for e in hallucinated + valid if preds.get(e["bibtex_key"]) == "UNCERTAIN"
        )
        rows.append(
            {
                "model": name,
                "dr_single": ds / ns,
                "dr_single_n": f"{ds}/{ns}",
                "dr_multi": dm / nm,
                "dr_multi_n": f"{dm}/{nm}",
                "diff": dm / nm - ds / ns,
                "fpr": fp / nv,
                "fpr_n": f"{fp}/{nv}",
                "uncertain": uncertain,
            }
        )
    counts = {
        "hallucinated": len(hallucinated),
        "single": len(single),
        "multi": len(multi),
        "valid": len(valid),
    }
    return rows, counts


def to_latex(rows: list[dict[str, Any]], counts: dict[str, int], split: str) -> str:
    split_tt = split.replace("_", r"\_")
    lines = [
        r"\begin{table}[t]",
        r"\caption{\textbf{Detection rate on single- vs multi-defect hallucinated "
        r"entries.} A \emph{defect} is a failing per-field subtest, not a distinct "
        r"\texttt{hallucination\_type}: each entry carries one type label (the injected "
        r"cause) but may break several fields. \texttt{cross\_db\_agreement} is excluded "
        r"because it is False for 100\% of hallucinated entries and therefore restates the "
        r"label. UNCERTAIN and missing predictions count as \emph{misses}, so all models "
        r"are scored over the full denominator; the GPT-5.4 cascade is the only candidate "
        r"that abstains. FPR is a single number per model: valid entries have no defects, "
        rf"so it cannot be split by defect count. Split: \texttt{{{split_tt}}} "
        rf"({counts['single']} single-defect, {counts['multi']} multi-defect, "
        rf"{counts['valid']} valid).}}",
        r"\label{tab:multi-vs-single-defect}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{DR single} & \textbf{DR multi} & "
        r"$\Delta$ & \textbf{FPR} & \textbf{UNC} \\",
        r"\midrule",
    ]
    for r in rows:
        model = r["model"].replace("&", r"\&")
        lines.append(
            f"{model} & {r['dr_single']:.3f} & {r['dr_multi']:.3f} & "
            f"{r['diff']:+.3f} & {r['fpr']:.3f} & {r['uncertain']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="dev_public")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows, counts = build_rows(args.data_dir, args.split)
    if not rows:
        print("No candidates scored; nothing written.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.out_dir / "multi_vs_single_defect.tex"
    tex_path.write_text(to_latex(rows, counts, args.split))

    csv_path = args.out_dir / "multi_vs_single_defect.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'model':<26}{'DR single':>18}{'DR multi':>18}{'diff':>9}{'FPR':>18}{'UNC':>6}"
    print(
        f"{args.split}: {counts['hallucinated']} hallucinated "
        f"({counts['single']} single, {counts['multi']} multi), {counts['valid']} valid"
    )
    print("UNCERTAIN counted as a miss\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<26}"
            f"{r['dr_single']:>10.3f} ({r['dr_single_n']}){'':>1}"
            f"{r['dr_multi']:>9.3f} ({r['dr_multi_n']}){'':>1}"
            f"{r['diff']:>+9.3f}"
            f"{r['fpr']:>10.3f} ({r['fpr_n']}){r['uncertain']:>5}"
        )
    print(f"\nWrote {tex_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
