"""Render results/relabel_delta/cis.json as a human-readable markdown table.

Emits cis_table.md: a per-tool CI table (F1 with 95% CI, plus DR/FPR/MCC/TW-F1
CIs) for each split, a summary-only roster with point estimates and the
not-available reason, and the headline paired-significance verdicts with explicit
"backed by a real paired test" vs "point-estimate only" labelling.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
CIS = REPO / "results/relabel_delta/cis.json"
OUT = REPO / "results/relabel_delta/cis_table.md"


def fmt_ci(ci: list[float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_pt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    d = json.loads(CIS.read_text())
    meta = d["meta"]
    lines: list[str] = []
    lines.append("# Relabel-delta bootstrap CIs and paired significance\n")
    lines.append(
        f"Stratified 95% bootstrap CIs (seed={meta['seed']}, "
        f"n_bootstrap={meta['n_bootstrap']:,}), computed against the "
        f"{meta['labels']} via `compute_persisted_cis()` and stratified paired "
        f"bootstrap via `paired_bootstrap_test()`. No API calls; per-entry "
        f"predictions only.\n"
    )
    for split in ("dev_public", "test_public"):
        sc = meta["split_counts"][split]
        lines.append(
            f"\n## {split} (n={sc['num_entries']}, "
            f"{sc['num_valid']} VALID / {sc['num_hallucinated']} HALL)\n"
        )
        # tools with CIs first, then summary-only
        ci_tools = {t: b for t, b in d["tools"][split].items() if b["ci_available"]}
        na_tools = {t: b for t, b in d["tools"][split].items() if not b["ci_available"]}

        lines.append("Tools with per-entry predictions (real 95% CIs):\n")
        lines.append(
            "| tool | F1 [95% CI] | DR [95% CI] | FPR [95% CI] | TW-F1 [95% CI] | MCC [95% CI] |"
        )
        lines.append("|---|---|---|---|---|---|")
        for t in sorted(ci_tools, key=lambda x: -ci_tools[x]["point_estimate"]["f1_hallucination"]):
            b = ci_tools[t]
            pe, ci = b["point_estimate"], b["ci"]
            lines.append(
                f"| {t} | {fmt_pt(pe['f1_hallucination'])} {fmt_ci(ci['f1_hallucination'])} "
                f"| {fmt_pt(pe['detection_rate'])} {fmt_ci(ci['detection_rate'])} "
                f"| {fmt_pt(pe['false_positive_rate'])} {fmt_ci(ci['false_positive_rate'])} "
                f"| {fmt_pt(pe['tier_weighted_f1'])} {fmt_ci(ci['tier_weighted_f1'])} "
                f"| {fmt_pt(pe['mcc'])} {fmt_ci(ci['mcc'])} |"
            )

        if na_tools:
            lines.append("\nSummary-only tools (point estimate only; CI not available):\n")
            lines.append("| tool | F1 | DR | FPR | TW-F1 | MCC | CI |")
            lines.append("|---|---|---|---|---|---|---|")
            for t in sorted(
                na_tools,
                key=lambda x: -(na_tools[x]["point_estimate"].get("f1_hallucination") or 0),
            ):
                pe = na_tools[t]["point_estimate"]
                lines.append(
                    f"| {t} | {fmt_pt(pe.get('f1_hallucination'))} "
                    f"| {fmt_pt(pe.get('detection_rate'))} "
                    f"| {fmt_pt(pe.get('false_positive_rate'))} "
                    f"| {fmt_pt(pe.get('tier_weighted_f1'))} "
                    f"| {fmt_pt(pe.get('mcc'))} | not available |"
                )

    lines.append("\n## Headline paired-significance verdicts (F1)\n")
    lines.append(
        "| comparison | split | backed by real paired test? | "
        "diff (higher-lower) | one-sided p | two-sided p | Cohen's h |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in d["paired_significance"].items():
        comp, split = k.rsplit("__", 1)
        if v["available"]:
            lines.append(
                f"| {comp} | {split} | YES | +{v['observed_diff']:.4f} "
                f"({v['higher_f1_tool'].replace('llm_openrouter_', '')} higher) "
                f"| {v['p_value_one_sided']:.4f} | {v['p_value_two_sided']:.4f} "
                f"| {v['cohens_h']:+.4f} |"
            )
        else:
            lines.append(f"| {comp} | {split} | NO (point-estimate only) | — | — | — | — |")

    lines.append("\n### How to phrase these in the paper\n")
    lines.append(
        "- **Sonnet vs Opus F1, test_public**: backed by a real stratified paired "
        "bootstrap. Sonnet F1=0.866 [0.847,0.884] vs Opus F1=0.846 [0.828,0.863]; "
        "diff +0.020, one-sided p=0.024, conservative two-sided p=0.049, Cohen's "
        "h=+0.057 (negligible). This is *borderline* significant at 0.05 and the "
        "effect is negligible — do NOT claim a clean within-CI tie; the honest "
        "statement is 'Sonnet edges Opus on F1 by ~2pp, at the 0.05 significance "
        "boundary with a negligible effect size; the gap is within the noise that "
        "a different split or seed could flip.'\n"
    )
    lines.append(
        "- **Sonnet vs Opus F1, dev_public**: NO paired test possible (both are "
        "summary-only on dev_public — no stored per-entry predictions). Point "
        "estimates are nearly identical (Sonnet 0.827 vs Opus 0.830); any "
        "'indistinguishable' claim here MUST be softened to point-estimate "
        "language — there is no CI or p-value to cite.\n"
    )
    lines.append(
        "- **bibtex-updater vs Sonnet F1, either split**: NO paired test possible "
        "(bibtex-updater has no full per-entry prediction file; it is summary-only "
        "on both splits per the delta-eval reconstruction). Report point estimates "
        "only: dev 0.909 vs 0.827, test 0.863 vs 0.866. No significance claim can "
        "be backed; phrase as point-estimate comparisons.\n"
    )

    OUT.write_text("\n".join(lines) + "\n")
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
