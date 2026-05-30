"""Finalize E2: read the three result_*.json files, compute the true total API
call count, and write SUMMARY.md.

Run AFTER all three --only jobs have produced their result_<alias>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/reviewer_experiments/e2_latecutoff_control"

# alias -> (display, provider, cutoff, role)
META = {
    "deepseek-v4-pro": (
        "DeepSeek V4-Pro",
        "DeepSeek",
        "~May 2025 (tracker; '2025' provider-corroborated)",
        "CONTROL (non-Anthropic / non-OpenAI, late cutoff)",
    ),
    "gpt-5.1": ("GPT-5.1", "OpenAI", "~Oct 2024", "collapse exemplar"),
    "claude-sonnet-4-6": (
        "Claude Sonnet 4.6",
        "Anthropic",
        "reliable ~Aug 2025 (training data ~Jan 2026)",
        "resist exemplar",
    ),
}
ORDER = ["deepseek-v4-pro", "gpt-5.1", "claude-sonnet-4-6"]


def fmt(x: object) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    results: dict[str, dict] = {}
    for alias in ORDER:
        p = OUT / f"result_{alias}.json"
        if not p.exists():
            raise SystemExit(f"Missing {p} — run not complete for {alias}")
        results[alias] = json.loads(p.read_text())

    total_calls = sum(r["api_calls"] for r in results.values())

    # Build main table.
    rows = []
    for alias in ORDER:
        r = results[alias]
        disp, prov, cutoff, role = META[alias]
        ov = r["overall"]
        rows.append(
            {
                "model": disp,
                "provider": prov,
                "cutoff": cutoff,
                "role": role,
                "fpr": ov["false_positive_rate"],
                "dr": ov["detection_rate"],
                "f1": ov["f1_hallucination"],
                "uncertain": ov.get("num_uncertain"),
                "pred_dist": r["predicted_label_dist"],
                "by_year": r["by_year"],
                "api_calls": r["api_calls"],
                "n": r["n"],
            }
        )

    n = results[ORDER[0]]["n"]

    md = []
    md.append("# E2 — Non-Anthropic / Non-OpenAI Late-Cutoff Control")
    md.append("")
    md.append(
        "Reviewer experiment for HALLMARK failure-mode (iii): most LLMs over-flag "
        "real 2024-2025 papers (FPR collapse) while the two Anthropic models resist. "
        "Recency and the Anthropic pipeline are confounded. This experiment "
        "**decorrelates** them by adding a NON-Anthropic / NON-OpenAI control whose "
        "training cutoff is comparable to the Anthropic pair, run on the SAME "
        "subsample as a collapse exemplar (GPT-5.1) and a resist exemplar "
        "(Claude Sonnet 4.6)."
    )
    md.append("")
    md.append("## Chosen control + justification")
    md.append("")
    md.append(
        "**Primary control: DeepSeek V4-Pro** (`deepseek/deepseek-v4-pro` on "
        "OpenRouter, served via DeepInfra/Together/Novita). Provider-reported "
        "knowledge cutoff **~May 2025** (aiknowledgecutoff.com tracker; corroborated "
        "by 36kr reporting that V4-Pro's cutoff 'still remains in 2025'). Released "
        "2026-04-24. Released after — and with a later cutoff than — the original "
        "DeepSeek V3 (July-2024) that collapses in the paper."
    )
    md.append("")
    md.append(
        "**Why not the ideal candidate (xAI Grok 4.3, Dec-2025 cutoff)?** Grok 4.3 "
        "is served ONLY by the `xai` provider on OpenRouter, which this account's "
        "provider allowlist excludes (HTTP 404 'No allowed providers are available'). "
        "Among non-Anthropic / non-OpenAI models reachable through the allowed "
        "providers (groq, azure, google-vertex, novita, nvidia, mistral, cerebras, "
        "together, deepinfra, sambanova, amazon-bedrock, google-ai-studio), the "
        "late-cutoff options were: Gemini 3.5 Flash / 3.1 Pro (Jan-2025 cutoff — same "
        "era as the already-tested Gemini, so not a *new* recency point) and "
        "Qwen 3.7-Max (Alibaba-only, blocked). DeepSeek V4-Pro (~May-2025) is the "
        "latest-cutoff *reachable* non-Anthropic/non-OpenAI model with a documented "
        "cutoff."
    )
    md.append("")
    md.append(
        "This makes E2 a **comparable-cutoff** control (May-2025 sits between the "
        "mid-2025 collapse cluster and the lower end of the Anthropic pair, "
        "Sonnet 4.6 reliable ~Aug-2025), not a strictly *later-than-Anthropic* "
        "control. The intended Grok-4.3 (Dec-2025) test could not be run on this "
        "account; see CAVEATS."
    )
    md.append("")
    md.append(f"## Subsample (fixed, seed=42, N={n})")
    md.append("")
    sub = json.loads((OUT / "subsample_n300_seed42.json").read_text())
    md.append(
        f"Stratified subsample of the 858-entry temporal supplement "
        f"(`results/temporal_supplement/temporal_supplement_2024_2025.jsonl`), "
        f"stratified by (label, publication-year). Strata: "
        f"{sub['label_year_dist']}. Label totals: {sub['label_dist']} "
        f"(FPR is measured on the {sub['label_dist']['VALID']} VALID; "
        f"DR on the {sub['label_dist']['HALLUCINATED']} HALLUCINATED). "
        f"`bibtex_keys` saved in `subsample_n300_seed42.json`."
    )
    md.append("")
    md.append("## Results")
    md.append("")
    md.append(
        "| Model | Provider | Cutoff | Role | FPR (2024-25 VALID) | DR (HALLUCINATED) | F1 | UNCERTAIN |"
    )
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['model']} | {r['provider']} | {r['cutoff']} | {r['role']} | "
            f"{fmt(r['fpr'])} | {fmt(r['dr'])} | {fmt(r['f1'])} | {fmt(r['uncertain'])} |"
        )
    md.append("")
    md.append("### Per-year FPR / DR (recency lens)")
    md.append("")
    md.append("| Model | FPR 2024 | FPR 2025 | DR 2024 | DR 2025 |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        by = r["by_year"]
        md.append(
            f"| {r['model']} | {fmt(by['2024']['fpr'])} | {fmt(by['2025']['fpr'])} | "
            f"{fmt(by['2024']['dr'])} | {fmt(by['2025']['dr'])} |"
        )
    md.append("")
    md.append("### Predicted-label distributions")
    md.append("")
    for r in rows:
        md.append(f"- **{r['model']}**: {r['pred_dist']}")
    md.append("")

    # Interpretation — computed relative to the two exemplars.
    fpr = {r["model"]: r["fpr"] for r in rows}
    ctrl_fpr = fpr["DeepSeek V4-Pro"]
    gpt_fpr = fpr["GPT-5.1"]
    son_fpr = fpr["Claude Sonnet 4.6"]
    # Place control: closer to which exemplar?
    closer = (
        "the Anthropic resist exemplar"
        if abs(ctrl_fpr - son_fpr) < abs(ctrl_fpr - gpt_fpr)
        else "the collapse exemplar (GPT-5.1)"
    )
    verdict = "RESISTS" if ctrl_fpr <= (gpt_fpr + son_fpr) / 2 else "COLLAPSES"
    md.append("## Interpretation")
    md.append("")
    md.append(
        f"On the same N={n} subsample (this data version; not comparable to the "
        f"paper's printed numbers), the late-cutoff non-Anthropic/non-OpenAI control "
        f"DeepSeek V4-Pro posts FPR={fmt(ctrl_fpr)} on real 2024-25 papers, versus "
        f"GPT-5.1 (collapse exemplar) FPR={fmt(gpt_fpr)} and Claude Sonnet 4.6 "
        f"(resist exemplar) FPR={fmt(son_fpr)}. The control therefore **{verdict}** "
        f"the FPR collapse and sits closer to {closer}. "
        + (
            "A late-cutoff model from a third provider resisting the collapse "
            "indicates that **recency of the training cutoff — not the Anthropic "
            "pipeline per se — drives resistance**, which weakens the 'pipeline-only' "
            "reading and is also consistent with the contamination concern (a later "
            "cutoff means more of the 2024-25 papers were seen in training)."
            if verdict == "RESISTS"
            else "A late-cutoff non-Anthropic model collapsing anyway supports a "
            "**pipeline/training-recipe-over-recency** reading: a comparable cutoff "
            "alone does not confer resistance."
        )
    )
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append(
        "- **Cutoffs are provider-reported / tracker-sourced**, not independently "
        "verified; DeepSeek V4-Pro's exact cutoff month (~May 2025) is from a "
        "third-party tracker, with only a vaguer '2025' corroboration from the "
        "provider-adjacent press."
    )
    md.append(
        "- **The ideal control (Grok 4.3, Dec-2025) could not be run** on this "
        "OpenRouter account (provider allowlist blocks `xai`). DeepSeek V4-Pro's "
        "~May-2025 cutoff is comparable-to but not later-than the Anthropic pair, so "
        "this is a weaker decorrelation than intended."
    )
    md.append(
        "- **N=1 per provider**: one DeepSeek model is a single point for its "
        "provider; no claim about DeepSeek-the-pipeline in general."
    )
    md.append(
        "- **DBLP contamination applies to the control too**: the temporal "
        "supplement is DBLP-sourced, and a later cutoff means more of these exact "
        "records may be in the control's training data."
    )
    md.append(
        "- **Data-version mismatch**: this run uses the 858-entry supplement "
        f"(subsample N={n}); the paper reports N=448 for the temporal split. "
        "Numbers here are NOT comparable to the printed paper values."
    )
    md.append(
        "- **V4-Pro is a thinking model**: we raised `max_completion_tokens` to 4096 "
        "so reasoning tokens did not truncate the JSON verdict (the default 1024 "
        "produced empty responses in smoke tests)."
    )
    md.append("")
    md.append("## Provenance")
    md.append("")
    md.append(
        f"- Total real API calls (this experiment): **{total_calls}** "
        f"({' + '.join(str(r['api_calls']) + ' (' + r['model'] + ')' for r in rows)})."
    )
    md.append("- Script: `scripts/reviewer_experiments/e2_latecutoff_control.py`")
    md.append(
        "- Raw per-model results: `result_deepseek-v4-pro.json`, "
        "`result_gpt-5.1.json`, `result_claude-sonnet-4-6.json`"
    )
    md.append("- Per-entry predictions (checkpoints): `checkpoints/*.jsonl`")
    md.append("- Subsample manifest: `subsample_n300_seed42.json`")
    md.append("")

    (OUT / "SUMMARY.md").write_text("\n".join(md))
    (OUT / "api_calls_total.json").write_text(
        json.dumps(
            {"per_model": {r["model"]: r["api_calls"] for r in rows}, "total": total_calls},
            indent=2,
        )
    )
    print("Wrote SUMMARY.md")
    print("TOTAL API CALLS:", total_calls)
    for r in rows:
        print(
            f"  {r['model']}: FPR={fmt(r['fpr'])} DR={fmt(r['dr'])} "
            f"F1={fmt(r['f1'])} calls={r['api_calls']} pred={r['pred_dist']}"
        )


if __name__ == "__main__":
    main()
