# E1 -- Contamination / Recall Probe

Research question: is the low post-cutoff false-positive rate (FPR) of Claude Sonnet 4.6 and Opus 4.7 on real 2024-2025 papers explained by **training-data recall** (memorization of these DBLP papers) rather than genuine calibration / abstention?

## Setup

- Sample: **150** VALID-only entries from the temporal supplement (`temporal_supplement_2024_2025.jsonl`, 858 entries / 440 VALID), stratified by year, seed=42 (84 from 2024, 66 from 2025). All entries are truly VALID, so any HALLUCINATED verdict is a false positive.

- Pass A (verify): standard HALLMARK verification on blinded entries (full BibTeX minus url). FPR = fraction predicted HALLUCINATED.

- Pass B (recall probe): model is given **only the title + year** and asked, with no external lookup, for the author list and venue it believes the paper has. `recalled`=1 when Jaccard(predicted last-names, true last-names) >= 0.50. Venue match recorded as a secondary signal.

- Note on N: the paper text reports 448 temporal entries; this data file has 858 (440 VALID). Numbers here are for the N actually used and are not directly comparable to the paper's printed values.


## Headline numbers

| Model | Recall rate | Verify FPR | Venue-match rate |
|---|---|---|---|
| `gpt-5.1` | 0.0% (0/150) | 92.0% (138/150) | 0.0% |
| `anthropic/claude-sonnet-4.6` | 10.0% (15/150) | 24.0% (36/150) | 9.3% |
| `anthropic/claude-opus-4.7` | 28.7% (43/150) | 12.7% (19/150) | 29.3% |

Verify-verdict breakdown (all 150 are truly VALID):

| Model | pred VALID | pred HALLUCINATED | pred UNCERTAIN |
|---|---|---|---|
| `gpt-5.1` | 12 | 138 | 0 |
| `anthropic/claude-sonnet-4.6` | 80 | 36 | 34 |
| `anthropic/claude-opus-4.7` | 129 | 19 | 2 |

## Memorization 2x2: P(verify = VALID | recall status)

If low FPR comes from memorization, predicted-VALID should concentrate on recalled entries (high P(VALID|recalled), lower P(VALID|not recalled)). If it comes from genuine calibration, the model holds VALID even when it cannot recall the paper.

| Model | P(VALID \| recalled) | P(VALID \| not recalled) | gap |
|---|---|---|---|
| `gpt-5.1` | -- (0/0) | 8.0% (12/150) | -- |
| `anthropic/claude-sonnet-4.6` | 86.7% (13/15) | 49.6% (67/135) | 37.0% |
| `anthropic/claude-opus-4.7` | 90.7% (39/43) | 84.1% (90/107) | 6.6% |

## Interpretation

GPT-5.1 recalls 0% of these real papers and false-flags 92% of them as HALLUCINATED. Sonnet 4.6 recalls 10% (FPR 24%) and Opus 4.7 recalls 29% (FPR 13%). For Sonnet 4.6, predicted-VALID concentrates on recalled entries (P(VALID|recalled)=87% vs P(VALID|not recalled)=50%), which is the signature of memorization driving the low FPR. For Opus 4.7, predicted-VALID holds at 84% even on NON-recalled entries (vs 91% on recalled ones), so the low FPR is not explained by memorization alone -- consistent with genuine calibration / abstention. The GPT-5.1 contrast is the key control: a model that neither recalls these papers nor extends them the benefit of the doubt over-flags real post-cutoff work, which is exactly the failure mode the Anthropic models avoid.


## Caveats

- Title-given recall is a **lower bound** on memorization: a model may have memorized a paper yet decline to reproduce its authorship from the title alone, so the true contamination rate is at least the measured recall rate.
- Abstention and recall are not perfectly separable: a model that declines to guess authors (empty list) is scored not-recalled, which conflates 'cannot recall' with 'chooses not to assert'.
- N=150, one provider pair (OpenAI direct + Anthropic via OpenRouter); the recall probe is single-sample per entry (gpt-5.x is forced to temp=1.0).
- Jaccard on last-names ignores first names and ordering; common surnames (e.g. Wang, Li, Zhang) can inflate overlap, so a 0.5 threshold is deliberately lenient toward counting an entry as recalled.


## Cost

- `gpt-5.1`: verify 150 calls + recall 150 calls = 300
- `anthropic/claude-sonnet-4.6`: verify 150 calls + recall 150 calls = 300
- `anthropic/claude-opus-4.7`: verify 150 calls + recall 150 calls = 300
- **Total API calls: 900** (cap = 150 x 3 x 2 = 900). Smoke tests during setup add ~4 calls not checkpointed.
