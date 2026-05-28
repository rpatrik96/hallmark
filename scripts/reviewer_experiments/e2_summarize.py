#!/usr/bin/env python3
"""E2: summarize the non-Anthropic/non-OpenAI late-cutoff control.

Reads the per-model result_*.json files in e2_latecutoff_control/ and writes
SUMMARY.md. The decisive question: does DeepSeek-V4-Pro (a non-Anthropic,
non-OpenAI model with a late training cutoff) RESIST the post-cutoff FPR
collapse on 2024-2025 papers (like the Anthropic pair) or COLLAPSE (like
GPT-5.1)? We surface UNCERTAIN/coverage because parse failures route to
UNCERTAIN and can artificially deflate FPR.
"""

import json
from pathlib import Path

D = Path("results/reviewer_experiments/e2_latecutoff_control")
# alias -> (provider, training-cutoff note). Cutoffs are provider-reported;
# DeepSeek-V4-Pro's exact cutoff MUST be verified before citing in the paper.
META = {
    "gpt-5.1": ("OpenAI", "Sep 2024 (paper roster)"),
    "claude-sonnet-4-6": ("Anthropic", "<= Aug 2025"),
    "claude-opus-4-7": ("Anthropic", "<= Oct 2025"),
    "deepseek-v4-pro": ("DeepSeek", "late-2025/2026, provider-reported -- VERIFY"),
}
ORDER = ["gpt-5.1", "claude-sonnet-4-6", "deepseek-v4-pro", "claude-opus-4-7"]

rows = {}
for f in D.glob("result_*.json"):
    d = json.loads(f.read_text())
    alias = d.get("alias") or f.stem.replace("result_", "")
    ov = d.get("overall") or {}
    rows[alias] = {
        "model_id": d.get("model_id"),
        "n": d.get("n"),
        "DR": ov.get("detection_rate"),
        "FPR": ov.get("false_positive_rate"),
        "FPR_ci": ov.get("fpr_ci"),
        "F1": ov.get("f1_hallucination"),
        "num_uncertain": ov.get("num_uncertain"),
        "coverage": ov.get("coverage"),
        "dist": d.get("predicted_label_dist"),
        "api_calls": d.get("api_calls"),
    }

sub = json.loads((D / "subsample_n300_seed42.json").read_text())


def fmt(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


def fmtci(ci):
    if not ci or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    return f" [{ci[0]:.3f},{ci[1]:.3f}]"


L = []
L.append("# E2 -- Non-Anthropic/non-OpenAI late-cutoff control\n")
L.append(
    "**Question.** Failure mode (iii) shows most LLMs over-flag real 2024-2025 papers while the two "
    "Anthropic models resist -- but recency and the Anthropic pipeline are confounded. Does a "
    "non-Anthropic, non-OpenAI model with a *late* cutoff resist the FPR collapse, or collapse like "
    "the mid-2025 models? The control model selected from the OpenRouter catalogue is "
    "**DeepSeek-V4-Pro**.\n"
)
L.append(
    f"**Sample.** Fixed stratified subsample of the 858-entry temporal supplement: n={sub['n']} "
    f"({sub['label_dist']['VALID']} VALID + {sub['label_dist']['HALLUCINATED']} HALLUCINATED; "
    f"year split {sub['year_dist']}), seed {sub['seed']}. FPR is computed on the VALID half "
    f"(real 2024-2025 papers); DR on the HALLUCINATED half. NOTE: the paper text reports a 448-entry "
    f"supplement -- this is the current 858-entry file, so numbers are not directly comparable to the "
    f"paper's printed values; GPT-5.1 and Sonnet 4.6 are re-run here on the SAME subsample for an "
    f"apples-to-apples comparison on this data version.\n"
)
L.append("## Results (same n=300 subsample, this data version)\n")
L.append("| Model | Provider | Cutoff | FPR (2024-25 valid) | DR | F1 | UNCERTAIN | Coverage |")
L.append("|---|---|---|---|---|---|---|---|")
for a in ORDER:
    if a not in rows:
        continue
    r = rows[a]
    prov, cut = META.get(a, ("?", "?"))
    L.append(
        f"| {a} | {prov} | {cut} | {fmt(r['FPR'])}{fmtci(r['FPR_ci'])} | {fmt(r['DR'])} | "
        f"{fmt(r['F1'])} | {r['num_uncertain']} | {fmt(r['coverage'])} |"
    )
L.append("")
L.append("Predicted-label distributions (n=300):")
for a in ORDER:
    if a in rows:
        L.append(f"- {a}: {rows[a]['dist']}")

# Interpretation
v4 = rows.get("deepseek-v4-pro", {})
gpt = rows.get("gpt-5.1", {})
son = rows.get("claude-sonnet-4-6", {})
L.append("\n## Interpretation\n")
if v4.get("FPR") is not None:
    verdict = (
        "RESISTS (low FPR, like the Anthropic pair)"
        if v4["FPR"] < 0.35
        else "COLLAPSES (high FPR, like the OpenAI/open-weight cluster)"
        if v4["FPR"] > 0.55
        else "is INTERMEDIATE"
    )
    L.append(
        f"On the same subsample, GPT-5.1 FPR={fmt(gpt.get('FPR'))} (collapse exemplar) and "
        f"Sonnet 4.6 FPR={fmt(son.get('FPR'))} (resist exemplar). DeepSeek-V4-Pro "
        f"FPR={fmt(v4.get('FPR'))} -> **DeepSeek-V4-Pro {verdict}.**"
    )
    L.append("")
    if v4.get("num_uncertain"):
        L.append(
            f"**Caveat (UNCERTAIN inflation):** DeepSeek-V4-Pro returned {v4['num_uncertain']} UNCERTAIN "
            f"verdicts (coverage {fmt(v4.get('coverage'))}); the run log shows repeated 'Failed to parse "
            f"LLM response'. UNCERTAIN is not scored as a false positive in conservative mode, so a low "
            f"FPR here may partly reflect parse failures routing to UNCERTAIN rather than genuine "
            f"acceptance of valid papers. Read FPR together with the UNCERTAIN count and coverage."
        )
    L.append(
        "\n**Bearing on the recency-vs-pipeline question.** "
        "If DeepSeek-V4-Pro resists despite NOT being an Anthropic model, that weakens the "
        "'Anthropic-pipeline-specific' reading and is consistent with later training recency "
        "(and/or DBLP contamination, which scales with capability) helping; if it collapses despite a "
        "late cutoff, that supports pipeline/post-training calibration over recency alone. Either way "
        "this is a single non-Anthropic control (N=1 provider) and shares the DBLP-contamination "
        "confound that E1 probes directly."
    )
else:
    L.append("DeepSeek-V4-Pro result not found -- run incomplete.")

L.append(
    "\n## Cost\n- API calls per model run = 300; total across models reported in api_calls_total.json."
)
L.append(
    "- Cutoffs are provider-reported; **DeepSeek-V4-Pro's exact training cutoff must be verified "
    "before being cited in the paper.**"
)

(D / "SUMMARY.md").write_text("\n".join(L) + "\n")
print("wrote", D / "SUMMARY.md")
print("\n".join(L))
