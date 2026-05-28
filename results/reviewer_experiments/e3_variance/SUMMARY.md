# E3: GPT-5.1 Run-to-Run Variance

GPT-5.1 runs at **temperature=1.0** (forced by the OpenAI API), so a single baseline run is one stochastic draw. We re-ran `verify_with_openai(model="gpt-5.1")` **3 times** on the **same** fixed stratified subsample of `dev_public` (N=150, seed=42) to quantify run-to-run variance. Each run used a **distinct `checkpoint_dir`** (run1/run2/run3) so checkpoint caching could not replay run 1 — every run actually re-called the API.

## Subsample

- Stratified: **85 HALLUCINATED + 65 VALID** = 150 entries.
- Subsample hallucinated fraction: **0.5667** (population: 0.5657).
- Sampled `bibtex_keys` saved to `sample_keys.json`.

## Per-run metrics

| Run | DR | FPR | F1 (hall) | MCC |
|----:|---:|----:|----------:|----:|
| 1 | 0.9647 | 0.5385 | 0.8119 | 0.5099 |
| 2 | 0.9647 | 0.5231 | 0.8159 | 0.5227 |
| 3 | 0.9647 | 0.5231 | 0.8159 | 0.5227 |

## Mean +/- sample std (across 3 runs)

| Metric | Mean | Sample std | Min | Max |
|--------|-----:|-----------:|----:|----:|
| Detection rate (DR) | 0.9647 | 0.0000 | 0.9647 | 0.9647 |
| False positive rate (FPR) | 0.5282 | 0.0089 | 0.5231 | 0.5385 |
| F1 (hallucination) | 0.8146 | 0.0023 | 0.8119 | 0.8159 |
| MCC | 0.5184 | 0.0074 | 0.5099 | 0.5227 |

## Sanity check: predictions are not identical across runs

- Entries whose predicted label **flipped** across the 3 runs: **1 / 150**.
- Prediction sets identical across runs: **False** (False confirms each run genuinely re-called the API and drew a fresh sample).
- Pairwise label disagreements: run1_vs_run2=1, run1_vs_run3=1, run2_vs_run3=0.

## Ranking-stability conclusion

GPT-5.1's **F1 run-to-run sample std is 0.0023** (on this N=150 subsample, 3 runs). Comparing against the published full-split F1 gaps:

- **Sonnet 4.6 - GPT-5.1 gap = 0.069** (29.6x the F1 std): **outside** GPT-5.1's run-to-run std (stable).
- **Sonnet 4.6 - Opus 4.7 gap = 0.016** (6.9x the F1 std): **outside** GPT-5.1's run-to-run std (stable).

## Caveats

- **N=3 runs** is a small variance estimate (sample std on 3 points; wide uncertainty on the std itself).
- The **N=150** subsample means absolute metrics differ from the full-split paper numbers. For context, run 1 on this subsample gave DR=0.9647, FPR=0.5385, F1=0.8119, MCC=0.5099.
- The published gaps (Sonnet-GPT-5.1 0.069, Sonnet-Opus 0.016) are full-split numbers; this experiment measures only **GPT-5.1's** stochasticity, not the other models'. The comparison treats GPT-5.1's std as a proxy for the noise floor on temperature=1.0 single-run F1 estimates.

## Provenance

- Total API calls: **450** (cap = 150 x 3 = 450).
- Raw per-run metrics + predictions: `run1/`, `run2/`, `run3/`.
- Machine-readable aggregate: `summary_data.json`.
