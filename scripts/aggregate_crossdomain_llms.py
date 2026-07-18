"""Aggregate the cross-domain LLM sweep and decompose domain vs recency.

The cross-domain split confounds domain with recency: ALL 116 valid biomedical
entries are dated 2026, while the 84 valid non-ML CS entries span 2023-2025
(and dev_public spans 2021-2023). A raw "cross-domain FPR" therefore mixes a
domain effect with the post-cutoff effect of sec:temporal_robustness.

This script decomposes the two:
  * per-domain FPR (biomed = all-2026; cs_non_ml = 2023-2025)
  * per-year FPR WITHIN cs_non_ml (domain held fixed -> isolates recency)
  * the year-matched contrast: cs_non_ml 2023 vs dev_public (2021-2023)
    (recency approximately held fixed -> isolates domain)
with Wilson 95% intervals throughout, since the year strata are small (n=25-32).

Usage:
    python scripts/aggregate_crossdomain_llms.py
    python scripts/aggregate_crossdomain_llms.py --json results/crossdomain_llms/decomposition.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XD = REPO / "data/v1.0/test_crossdomain.jsonl"
RES = REPO / "results/crossdomain_llms"
BTU_XD = REPO / "results/relabel_delta/btu_v1_2_0/bibtexupdater_test_crossdomain_per_entry.jsonl"

# dev_public FPR from tab:results (main run; the two Anthropic rows carry the
# endpoint-drift caveat of app:coverage).
DEV_FPR = {
    "llm_openai": ("GPT-5.1", 0.411),
    "llm_openai_gpt54": ("GPT-5.4", 0.228),
    "llm_openrouter_qwen": ("Qwen3-235B", 0.533),
    "llm_openrouter_deepseek_v3": ("DeepSeek-V3.2", 0.702),
    "llm_openrouter_deepseek_r1": ("DeepSeek-R1", 0.623),
    "llm_openrouter_mistral": ("Mistral Large", 0.250),
    "llm_openrouter_gemini_flash": ("Gemini 2.5 Flash", 0.100),
    "llm_openrouter_gemini_pro": ("Gemini 2.5 Pro", 0.050),
    "llm_openrouter_llama_4_maverick": ("Llama 4 Maverick", 0.146),
    "llm_openrouter_claude_sonnet_4_6": ("Claude Sonnet 4.6", 0.127),
    "llm_openrouter_claude_opus_4_7": ("Claude Opus 4.7", 0.072),
    "llm_openrouter_qwen_max": ("Qwen3-VL-235B", 0.551),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


def load_entries() -> dict[str, dict]:
    return {json.loads(line)["bibtex_key"]: json.loads(line) for line in XD.open()}


def domain_of(e: dict) -> str:
    return "biomed" if e.get("source") in ("pubmed", "biorxiv") else "cs_non_ml"


def year_of(e: dict) -> str:
    return str(e["fields"].get("year", ""))[:4]


def fpr_cells(preds: dict[str, str], entries: dict[str, dict]) -> dict:
    """FPR over VALID entries, bucketed by domain and (domain, year).

    Abstentions score as committed-VALID, matching sec:main_results.
    """
    cells: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for k, lab in preds.items():
        e = entries.get(k)
        if not e or e["label"] != "VALID":
            continue
        flagged = 1 if lab == "HALLUCINATED" else 0
        for key in (("all",), (domain_of(e),), (domain_of(e), year_of(e))):
            cells[key][0] += flagged
            cells[key][1] += 1
    out = {}
    for key, (k_, n_) in cells.items():
        lo, hi = wilson(k_, n_)
        out["/".join(key)] = {
            "fpr": round(k_ / n_, 4),
            "n": n_,
            "ci95": [round(lo, 4), round(hi, 4)],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    entries = load_entries()

    rows = {}
    for f in sorted(glob.glob(str(RES / "result_*.json"))):
        if "limit" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        key = d["key"]
        pred_file = RES / f"{key}_test_crossdomain_predictions.jsonl"
        if not pred_file.exists():
            continue
        preds = {}
        for line in pred_file.open():
            r = json.loads(line)
            preds[r["bibtex_key"]] = "VALID" if r["label"] == "UNCERTAIN" else r["label"]
        rows[key] = {
            "display": d["display"],
            "overall": d["overall"],
            "fpr_cells": fpr_cells(preds, entries),
        }

    # bibtex-updater comparator (database-backed; already run on this split).
    if BTU_XD.exists():
        btu = {}
        for line in BTU_XD.open():
            r = json.loads(line)
            btu[r["bibtex_key"]] = "VALID" if r["pred_label"] == "UNCERTAIN" else r["pred_label"]
        rows["bibtex_updater"] = {
            "display": "bibtex-updater (co-designed)",
            "overall": None,
            "fpr_cells": fpr_cells(btu, entries),
        }

    header = (
        f"{'Model':26s} {'dev':>6s} {'xd all':>8s} {'biomed26':>9s} "
        f"{'cs23':>7s} {'cs24':>7s} {'cs25':>7s}  {'d(cs23-dev)':>11s}"
    )
    print(header)
    print("-" * 92)
    for key, r in rows.items():
        c = r["fpr_cells"]
        dev = DEV_FPR.get(key, (None, None))[1]
        if key == "bibtex_updater":
            dev = 0.092
        cs23 = c.get("cs_non_ml/2023", {}).get("fpr")
        delta = f"{cs23 - dev:+.3f}" if (cs23 is not None and dev is not None) else "n/a"
        print(
            f"{r['display']:26s} {dev if dev is not None else float('nan'):6.3f} "
            f"{c['all']['fpr']:8.3f} {c.get('biomed', {}).get('fpr', float('nan')):9.3f} "
            f"{c.get('cs_non_ml/2023', {}).get('fpr', float('nan')):7.3f} "
            f"{c.get('cs_non_ml/2024', {}).get('fpr', float('nan')):7.3f} "
            f"{c.get('cs_non_ml/2025', {}).get('fpr', float('nan')):7.3f}  {delta:>11s}"
        )

    print("\nYear-matched domain contrast (cs_non_ml 2023 vs dev_public 2021-23), Wilson 95%:")
    for key, r in rows.items():
        cell = r["fpr_cells"].get("cs_non_ml/2023")
        if not cell:
            continue
        dev = 0.092 if key == "bibtex_updater" else DEV_FPR.get(key, (None, None))[1]
        if dev is None:
            excl = ""
        elif cell["ci95"][0] <= dev <= cell["ci95"][1]:
            excl = " dev inside CI"
        else:
            excl = " dev OUTSIDE CI"
        print(
            f"  {r['display']:26s} cs2023 FPR {cell['fpr']:.3f} CI {cell['ci95']} "
            f"(n={cell['n']}) vs dev {dev}{excl}"
        )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
