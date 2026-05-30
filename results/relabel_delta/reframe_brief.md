# Reframe brief — Opus-leads narrative (canonical; use across abstract/intro/experiments/analysis)

The ground-truth relabel recovered 52 real papers that were wrongly labeled HALLUCINATED (dev+test). On the corrected labels the dev headline ordering inverts. Render the facts below in the author's voice (declarative, practitioner-first; em-dash interpolations and colon-payoffs are his; NO slogans, NO hype adjectives, NO theatrical punchlines). Change only what the new numbers/framing require — keep his existing sentences wherever the number is unchanged.

## The facts (new dev_public, independent tools)
- **Opus 4.7 leads F1 (0.830) and MCC (0.683)**; Sonnet 4.6 is a near-tie behind (F1 0.827, MCC 0.652) and remains **best-calibrated (ECE 0.066)**.
- Sonnet FPR rose 0.095 → **0.127**; Opus 0.060 → **0.072**. Why: of the 27 recovered real dev papers, **Sonnet flagged 19 as hallucinated, Opus only 8** — so Sonnet's earlier F1 lead was partly inflated by the very over-flagging failure mode HALLMARK measures.
- FPR-ascending order: Gemini 2.5 Pro (0.050) < Opus (0.072) < Gemini Flash (0.100) < Sonnet (0.127) — Sonnet and Gemini Flash swap vs the old table.
- test_public: bibtex-updater's F1 lead over Sonnet **reverses** (btu 0.863 < Sonnet 0.866) — this *strengthens* the existing "the rule-based lead is a `dev_public` artifact" takeaway.

## The framing (what the reframe should say, in his voice)
- The thesis is unchanged and in fact sharpened: **FPR / calibration is the deployment-decisive lever, and once the labels are clean the best-calibrated model wins.**
- Be scope-honest: the Sonnet/Opus dev gap is a point-estimate near-tie with no dev paired CI (both summary-only); the only headline pair with a real paired test is Sonnet vs Opus F1 on test_public (diff +0.020, p=0.049, negligible effect). State the inversion as a point-estimate reordering, not a significant separation.
- The label-correction itself is worth one honest sentence: a benchmark for detecting real-papers-flagged-as-hallucinated had committed that exact error on ~52 of its own entries; correcting it both fixes the ground truth and removes an artifact that had flattered the more aggressive verifier.

## Canonical sentences (adapt, don't paste verbatim if it breaks local flow)
- Experiments takeaway: "Among independently-evaluated tools, Opus 4.7 now leads on F1 and MCC (0.830 / 0.683) with Sonnet 4.6 a near-tie behind and best-calibrated (ECE 0.066) — an ordering that inverts only once we correct the ground-truth mislabels: Sonnet flags 19 of the 27 recovered real papers as hallucinated to Opus's 8, so its earlier F1 edge partly reflected the over-flagging failure mode the benchmark is built to measure."
- Abstract (compact): "two later-cutoff Anthropic models lead the independent cohort — Opus 4.7 on F1/MCC, Sonnet 4.6 on calibration (ECE 0.066)".
- Cross-split (analysis): keep the existing "rule-based lead is a dev artifact" takeaway and note it now *reverses* on test_public (btu F1 0.863 < Sonnet 0.866), not merely narrows.

## Honest consequences to reflect (not optional)
- New totals everywhere: **2,526 entries / 1,036 valid / 1,490 hallucinated** (was 2,525 / 974 / 1,551); test_hidden 453 → 454; public total 2,072 (valid/hall 826/1,246).
- `n >= 30 per type` is now violated in a few types (dev 1, test 6, hidden 13) because relabeling pulled mistyped real papers out of their hallucination-type buckets — soften the MDE/power language in benchmark.tex (L80), the tab:stats caption, and limitations.tex (L5) accordingly; do not claim a uniform >=30 floor.
- Pre-screening reframe: the released main numbers were computed WITHOUT the pre-screening layer (bibtex-updater dev FPR = exactly 87/486 proves it was never applied); correct experiments.tex L11 + limitations.tex L26 to present pre-screening as an optional, now-corrected ablation rather than a uniformly-applied layer.
