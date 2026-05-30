# Reviewer-experiment findings (run 2026-05-28)

Three experiments addressing the arXiv must-haves on failure mode (iii)
(post-cutoff calibration breakdown) and on single-run statistical stability.
All run from the `hallmark` code repo via the Python API (OpenAI direct +
Anthropic/DeepSeek via OpenRouter).

**Data-version note (MUST reconcile in paper):** the temporal supplement in the
repo is **858 entries (440 VALID / 418 HALLUCINATED)**; the paper text says
**448 (300/148)**. Numbers below are on the current 858-file (and subsamples of
it), so they are NOT drop-in replacements for the paper's printed temporal-table
values. Where a comparison is needed, paper models (GPT-5.1, Sonnet 4.6) were
re-run on the SAME subsample for an apples-to-apples read on this data version.

---

## E1 — Contamination / recall probe  (the decisive one)

n = 150 VALID 2024–2025 papers (stratified by year, seed 42). For each model:
a VERIFY pass (FPR on these real papers) and a RECALL pass (given title+year,
reproduce authors+venue with no lookup; "recalled" = author last-name Jaccard ≥ 0.5).

| Model | recall rate | verify FPR | P(VALID\|recalled) | P(VALID\|NOT recalled) | gap |
|---|---|---|---|---|---|
| GPT-5.1 | 0.0% | 92.0% | — | 8.0% | — |
| Claude Sonnet 4.6 | 10.0% | 24.0% | 86.7% | 49.6% | 37.0 pp |
| Claude Opus 4.7 | 28.7% | 12.7% | 90.7% | **84.1%** | 6.6 pp |

**Finding.** Opus 4.7's low FPR is **genuine calibration**: it accepts 84% of
real papers it cannot even recall. Sonnet 4.6 is **partly memorization-assisted**
(big accept gap by recall status). GPT-5.1 is the clean control: recalls none,
flags 92%. → The DBLP-contamination alternative is **largely ruled out for Opus,
partly implicated for Sonnet.**
Caveats: title-given recall is a lower bound on memorization; abstention vs
"cannot recall" not perfectly separable; N=150, single sample/entry (temp=1.0).

---

## E2 — Non-Anthropic / non-OpenAI late-cutoff control

Fixed stratified n=300 subsample of the 858 set (150 VALID + 150 HALL, seed 42).
Control = **DeepSeek-V4-Pro** (latest-cutoff non-Anthropic/non-OpenAI model on
OpenRouter). GPT-5.1 + Sonnet 4.6 re-run on the same subsample.

| Model | Provider | FPR (2024-25 valid) | DR | UNCERTAIN | Coverage |
|---|---|---|---|---|---|
| GPT-5.1 | OpenAI | 0.926 | 0.973 | 1 | 1.00 |
| Claude Sonnet 4.6 | Anthropic | 0.287 | 0.937 | 43 | 1.00 |
| DeepSeek-V4-Pro | DeepSeek | **0.265** (102 valid) | — (run halted) | 23 | 0.78 |

**Finding.** A late-cutoff, non-Anthropic, non-OpenAI model **also resists** the
FPR collapse (0.265 ≈ Sonnet's 0.287, ≪ GPT-5.1's 0.926). → Resistance is **not
Anthropic-pipeline-specific**; it tracks training recency across providers.
Caveats: DeepSeek-V4-Pro was slow/parse-flaky (23% UNCERTAIN, coverage 0.78 →
UNCERTAIN may deflate apparent FPR); run halted at 103/300 (valids processed
first, so FPR is well-estimated; DR not); N=1 provider; its exact cutoff is
provider-reported and **must be verified before citing**; contamination is not
separable for V4-Pro (no recall probe run on it).
Also note: on this larger 858-derived subsample, even Sonnet's FPR is ~0.29
(vs the paper's 0.12 on the 448 set) — resistance is real but more modest on the
bigger/newer data.

---

## E3 — GPT-5.1 run-to-run variance (temperature=1.0, forced by API)

n=150 dev_public subsample, 3 independent runs (distinct checkpoint dirs).

| metric | mean | sample std |
|---|---|---|
| DR | 0.965 | 0.000 |
| FPR | 0.528 | 0.0089 |
| F1 | 0.815 | **0.0023** |
| MCC | 0.518 | 0.0074 |

Label flips across runs: 1/150. **Finding.** GPT-5.1's single-run F1 is stable
(std 0.0023). The published Sonnet−GPT-5.1 gap (0.069, 30×) and Sonnet−Opus gap
(0.016, 7×) both exceed the noise → single-run rankings hold.
Caveat: N=3 runs; n=150 subsample; measures GPT-5.1's stochasticity as the
noise-floor proxy.

---

## Implications for the paper (failure mode iii)

The experiments let us **upgrade (iii) from "N=2 Anthropic anecdote / open
question" to a tested, decomposed result**:

1. **Not Anthropic-specific.** Reframe "Anthropic exception (N=2, one provider)"
   → "later-cutoff models across *providers* resist; resistance correlates with
   training recency" (E2: DeepSeek-V4-Pro resists too).
2. **Calibration vs contamination, decomposed (E1).** Opus's resistance survives
   a recall probe (calibration); Sonnet's is partly recall. The contamination
   confound is now *partially addressed* — state it as such, not fully resolved
   (Sonnet and V4-Pro remain consistent with some memorization).
3. **Variance defused (E3).** Add the F1 std=0.0023 result; the limitations
   "GPT-5.1 temp=1.0 variance uncharacterized" gap is now filled.
4. **Reconcile 858 vs 448** temporal-supplement counts regardless.

These are robustness checks on a different data version/N; integrate as a clearly
labeled subsection (or appendix) rather than overwriting the printed temporal
table. Total API spend ≈ 2,050 calls (E1 900 + E2 ~700 + E3 450).
