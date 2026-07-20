"""Domain x recency 2x2 decomposition for the cross-domain analysis.

The original test_crossdomain split confounds domain (ML vs biomed/CS) with
recency (pre- vs post-cutoff): all valid biomed entries are 2026. This script
assembles the full factorial so each factor can be read alone:

    cell                         data source                         factor level
    ----------------------------------------------------------------------------
    in-domain ML,  pre-cutoff    dev_public (2021-23)                domain-, recency-
    in-domain ML,  post-cutoff   temporal supplement (2024-25)       domain-, recency+
    out-domain,    post-cutoff   test_crossdomain biomed-2026        domain+, recency+
    out-domain,    pre-cutoff    test_crossdomain_matched (2021-23)  domain+, recency-   <-- new

Clean contrasts:
  * DOMAIN effect  = matched(2021-23, out) - dev_public(2021-23, in)   [recency fixed]
  * RECENCY effect = temporal(24-25, in)  - dev_public(2021-23, in)    [domain fixed]
  * the confounded biomed-2026 cell should be ~ domain+recency combined.

It also classifies each false positive's reason as date-heuristic ("2026 is in
the future / beyond cutoff") vs recall-failure, since the adversarial review
found the biomed-2026 FPR is almost entirely date-driven. In the matched split
(all pre-cutoff) the date-heuristic share should collapse -- that is the test.

Usage:
    python scripts/analyze_domain_recency_2x2.py
    python scripts/analyze_domain_recency_2x2.py --json results/crossdomain_matched_llms/domain_recency_2x2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATCHED_SPLIT = REPO / "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl"
V2026_SPLIT = REPO / "data/v1.0/test_crossdomain.jsonl"
MATCHED_RES = REPO / "results/crossdomain_matched_llms"
V2026_RES = REPO / "results/crossdomain_llms"

# dev_public (in-domain ML, pre-cutoff) FPR from tab:results, and temporal
# supplement (in-domain ML, post-cutoff) FPR from tab:temporal_supplement
# (canonical 448-subset numbers). Keyed by the runner's model key.
DEV_FPR = {
    "llm_openai": 0.411,
    "llm_openai_gpt54": 0.228,
    "llm_openrouter_qwen": 0.533,
    "llm_openrouter_deepseek_v3": 0.702,
    "llm_openrouter_deepseek_r1": 0.623,
    "llm_openrouter_mistral": 0.250,
    "llm_openrouter_gemini_flash": 0.100,
    "llm_openrouter_gemini_pro": 0.050,
    "llm_openrouter_llama_4_maverick": 0.146,
    "llm_openrouter_claude_sonnet_4_6": 0.127,
    "llm_openrouter_claude_opus_4_7": 0.072,
    "llm_openrouter_qwen_max": 0.551,
}
TEMPORAL_FPR = {
    "llm_openai": 0.759,
    "llm_openrouter_gemini_flash": 0.595,
    "llm_openrouter_mistral": 0.793,
    "llm_openrouter_qwen": 0.809,
    "llm_openrouter_deepseek_v3": 0.759,
    "llm_openrouter_deepseek_r1": 0.856,
    "llm_openrouter_claude_opus_4_7": 0.073,
    "llm_openrouter_claude_sonnet_4_6": 0.120,
    "llm_openrouter_gemini_pro": 0.250,
    "llm_openrouter_llama_4_maverick": 0.763,
    "llm_openrouter_qwen_max": 0.887,
}

DISPLAY = {
    "llm_openai": "GPT-5.1",
    "llm_openai_gpt54": "GPT-5.4",
    "llm_openrouter_qwen": "Qwen3-235B",
    "llm_openrouter_deepseek_v3": "DeepSeek-V3.2",
    "llm_openrouter_deepseek_r1": "DeepSeek-R1",
    "llm_openrouter_mistral": "Mistral Large",
    "llm_openrouter_gemini_flash": "Gemini 2.5 Flash",
    "llm_openrouter_gemini_pro": "Gemini 2.5 Pro",
    "llm_openrouter_llama_4_maverick": "Llama 4 Maverick",
    "llm_openrouter_claude_sonnet_4_6": "Claude Sonnet 4.6",
    "llm_openrouter_claude_opus_4_7": "Claude Opus 4.7",
    "llm_openrouter_qwen_max": "Qwen3-VL-235B",
}

# reason strings that indicate the date/cutoff heuristic rather than a recall failure
DATE_PAT = re.compile(
    r"\b(future|futuristic|has not yet|not yet (been )?(published|occurred)|"
    r"beyond (my )?(training|knowledge) cutoff|after (my )?cutoff|"
    r"post-cutoff|year 20\d\d is|in the future|yet to (be|occur)|"
    r"cannot be verified.*20\d\d|20\d\d.*not yet)\b",
    re.IGNORECASE,
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


def diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Two-proportion difference (p1 - p2) with a Wald CI. Returns (diff, lo, hi)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return (d, d - z * se, d + z * se)


def load_entries(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.open():
        if line.strip():
            d = json.loads(line)
            out[d["bibtex_key"]] = d
    return out


def domain_of(e: dict) -> str:
    return "biomed" if e.get("source") in ("pubmed", "biorxiv", "medrxiv") else "cs_non_ml"


def load_preds(res_dir: Path, key: str) -> dict[str, dict]:
    hits = glob.glob(str(res_dir / f"{key}_test_crossdomain_predictions.jsonl"))
    if not hits:
        return {}
    out = {}
    with Path(hits[0]).open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["bibtex_key"]] = r
    return out


def fp_stats(preds: dict, entries: dict, sources: set[str] | None = None) -> dict:
    """Over VALID entries (optionally restricted to `sources`): FP count, n, and
    the share of FPs whose reason is date-heuristic. UNCERTAIN -> committed-VALID."""
    fp = n = date_fp = 0
    for k, e in entries.items():
        if e["label"] != "VALID":
            continue
        if sources is not None and domain_of(e) not in sources:
            continue
        p = preds.get(k)
        if p is None:
            continue
        n += 1
        if p["label"] == "HALLUCINATED":
            fp += 1
            if DATE_PAT.search(p.get("reason", "") or ""):
                date_fp += 1
    lo, hi = wilson(fp, n)
    return {
        "fpr": round(fp / n, 4) if n else None,
        "fp": fp,
        "n": n,
        "ci95": [round(lo, 4), round(hi, 4)],
        "date_driven_share_of_fp": round(date_fp / fp, 3) if fp else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not MATCHED_SPLIT.exists():
        raise SystemExit(f"matched split not built yet: {MATCHED_SPLIT}")
    matched_e = load_entries(MATCHED_SPLIT)
    v2026_e = load_entries(V2026_SPLIT) if V2026_SPLIT.exists() else {}

    rows = {}
    for key in DEV_FPR:
        matched_p = load_preds(MATCHED_RES, key)
        v2026_p = load_preds(V2026_RES, key)
        # out-domain pre-cutoff (matched): overall + per-domain
        m_all = fp_stats(matched_p, matched_e) if matched_p else None
        m_bio = fp_stats(matched_p, matched_e, {"biomed"}) if matched_p else None
        m_cs = fp_stats(matched_p, matched_e, {"cs_non_ml"}) if matched_p else None
        # out-domain post-cutoff (2026 biomed) for the additivity check
        v_bio = fp_stats(v2026_p, v2026_e, {"biomed"}) if v2026_p else None

        entry = {
            "display": DISPLAY[key],
            "dev_fpr_in_pre": DEV_FPR.get(key),
            "temporal_fpr_in_post": TEMPORAL_FPR.get(key),
            "matched_out_pre_all": m_all,
            "matched_out_pre_biomed": m_bio,
            "matched_out_pre_cs": m_cs,
            "v2026_out_post_biomed": v_bio,
        }
        # clean DOMAIN effect: matched(out, pre) vs dev(in, pre). dev is a point
        # estimate (per-entry preds not re-scored here), so we report the matched
        # CI and whether dev sits inside it, plus the raw difference.
        if m_all and entry["dev_fpr_in_pre"] is not None:
            dev = entry["dev_fpr_in_pre"]
            lo, hi = m_all["ci95"]
            entry["domain_effect"] = {
                "matched_fpr": m_all["fpr"],
                "dev_fpr": dev,
                "diff": round(m_all["fpr"] - dev, 4) if m_all["fpr"] is not None else None,
                "dev_inside_matched_ci": bool(lo <= dev <= hi)
                if m_all["fpr"] is not None
                else None,
            }
        rows[key] = entry

    # pooled date-driven share, matched vs 2026, to show the artifact collapses
    def pooled_date_share(getter) -> dict:
        fp = date = 0
        for key in DEV_FPR:
            st = getter(rows[key])
            if not st or not st.get("fp"):
                continue
            fp += st["fp"]
            date += (
                round(st["date_driven_share_of_fp"] * st["fp"])
                if st["date_driven_share_of_fp"]
                else 0
            )
        return {
            "total_fp": fp,
            "date_driven_fp": date,
            "share": round(date / fp, 3) if fp else None,
        }

    summary = {
        "matched_biomed_date_share": pooled_date_share(lambda r: r["matched_out_pre_biomed"]),
        "v2026_biomed_date_share": pooled_date_share(lambda r: r["v2026_out_post_biomed"]),
    }

    header = (
        f"{'Model':20s} {'dev(in,pre)':>11s} {'matched(out,pre)':>16s} "
        f"{'diff':>7s} {'temporal(in,post)':>17s} {'2026bio(out,post)':>17s}"
    )
    print(header)
    print("-" * 96)
    for key in DEV_FPR:
        r = rows[key]
        m = r["matched_out_pre_all"]
        mfpr = f"{m['fpr']:.3f}(n{m['n']})" if m else "PENDING"
        diff = r.get("domain_effect", {}).get("diff")
        diff_s = f"{diff:+.3f}" if diff is not None else "-"
        vb = r["v2026_out_post_biomed"]
        vb_s = f"{vb['fpr']:.3f}" if vb and vb["fpr"] is not None else "-"
        print(
            f"{r['display']:20s} {r['dev_fpr_in_pre']!s:>11s} {mfpr:>16s} "
            f"{diff_s:>7s} {r['temporal_fpr_in_post']!s:>17s} {vb_s:>17s}"
        )
    print("\nDate-driven share of biomedical false positives (the artifact):")
    print(f"  2026 (post-cutoff): {summary['v2026_biomed_date_share']}")
    print(
        f"  matched (pre-cutoff): {summary['matched_biomed_date_share']}  <- should collapse toward 0"
    )

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
