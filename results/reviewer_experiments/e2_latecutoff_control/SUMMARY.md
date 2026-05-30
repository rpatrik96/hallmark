# E2 -- Non-Anthropic/non-OpenAI late-cutoff control

**Question.** Failure mode (iii) shows most LLMs over-flag real 2024-2025 papers while the two Anthropic models resist -- but recency and the Anthropic pipeline are confounded. Does a non-Anthropic, non-OpenAI model with a *late* cutoff resist the FPR collapse, or collapse like the mid-2025 models? The control model selected from the OpenRouter catalogue is **DeepSeek-V4-Pro**.

**Sample.** Fixed stratified subsample of the 858-entry temporal supplement: n=300 (150 VALID + 150 HALLUCINATED; year split {'2024': 177, '2025': 123}), seed 42. FPR is computed on the VALID half (real 2024-2025 papers); DR on the HALLUCINATED half. NOTE: the paper text reports a 448-entry supplement -- this is the current 858-entry file, so numbers are not directly comparable to the paper's printed values; GPT-5.1 and Sonnet 4.6 are re-run here on the SAME subsample for an apples-to-apples comparison on this data version.

## Results (same n=300 subsample, this data version)

| Model | Provider | Cutoff | FPR (2024-25 valid) | DR | F1 | UNCERTAIN | Coverage |
|---|---|---|---|---|---|---|---|
| gpt-5.1 | OpenAI | Sep 2024 (paper roster) | 0.926 | 0.973 | 0.673 | 1 | 1.000 |
| claude-sonnet-4-6 | Anthropic | <= Aug 2025 | 0.287 | 0.937 | 0.864 | 43 | 1.000 |
| deepseek-v4-pro | DeepSeek | late-2025/2026, provider-reported -- VERIFY | 0.265 | 1.000 | n/a | 23 | 0.775 |

Predicted-label distributions (n=300):
- gpt-5.1: {'HALLUCINATED': 284, 'VALID': 15, 'UNCERTAIN': 1}
- claude-sonnet-4-6: {'VALID': 91, 'UNCERTAIN': 43, 'HALLUCINATED': 166}
- deepseek-v4-pro: {'VALID': 52, 'HALLUCINATED': 28, 'UNCERTAIN': 23}

## Interpretation

On the same subsample, GPT-5.1 FPR=0.926 (collapse exemplar) and Sonnet 4.6 FPR=0.287 (resist exemplar). DeepSeek-V4-Pro FPR=0.265 -> **DeepSeek-V4-Pro RESISTS (low FPR, like the Anthropic pair).**

**Caveat (UNCERTAIN inflation):** DeepSeek-V4-Pro returned 23 UNCERTAIN verdicts (coverage 0.775); the run log shows repeated 'Failed to parse LLM response'. UNCERTAIN is not scored as a false positive in conservative mode, so a low FPR here may partly reflect parse failures routing to UNCERTAIN rather than genuine acceptance of valid papers. Read FPR together with the UNCERTAIN count and coverage.

**Bearing on the recency-vs-pipeline question.** If DeepSeek-V4-Pro resists despite NOT being an Anthropic model, that weakens the 'Anthropic-pipeline-specific' reading and is consistent with later training recency (and/or DBLP contamination, which scales with capability) helping; if it collapses despite a late cutoff, that supports pipeline/post-training calibration over recency alone. Either way this is a single non-Anthropic control (N=1 provider) and shares the DBLP-contamination confound that E1 probes directly.

## Cost
- API calls per model run = 300; total across models reported in api_calls_total.json.
- Cutoffs are provider-reported; **DeepSeek-V4-Pro's exact training cutoff must be verified before being cited in the paper.**
