# Relabel-delta bootstrap CIs and paired significance

Stratified 95% bootstrap CIs (seed=42, n_bootstrap=10,000), computed against the post-relabel (fix/dev-public-mislabel-audit) via `compute_persisted_cis()` and stratified paired bootstrap via `paired_bootstrap_test()`. No API calls; per-entry predictions only.


## dev_public (n=1119, 513 VALID / 606 HALL)

Tools with per-entry predictions (real 95% CIs):

| tool | F1 [95% CI] | DR [95% CI] | FPR [95% CI] | TW-F1 [95% CI] | MCC [95% CI] |
|---|---|---|---|---|---|
| llm_agentic_btu_sonnet_4_6 | 0.841 [0.828, 0.854] | 0.990 [0.982, 0.997] | 0.431 [0.388, 0.474] | 0.913 [0.904, 0.921] | 0.630 [0.595, 0.665] |
| llm_agentic_btu_openai | 0.824 [0.811, 0.838] | 0.980 [0.969, 0.990] | 0.470 [0.427, 0.513] | 0.900 [0.889, 0.909] | 0.584 [0.545, 0.622] |
| llm_agentic_openai | 0.816 [0.802, 0.830] | 0.967 [0.954, 0.979] | 0.478 [0.433, 0.520] | 0.892 [0.882, 0.903] | 0.558 [0.517, 0.598] |
| llm_openai_gpt-5.4 | 0.783 [0.761, 0.805] | 0.767 [0.739, 0.797] | 0.228 [0.193, 0.265] | 0.807 [0.784, 0.829] | 0.538 [0.491, 0.585] |
| llm_openai | 0.766 [0.747, 0.785] | 0.837 [0.812, 0.861] | 0.411 [0.370, 0.454] | 0.822 [0.803, 0.840] | 0.442 [0.393, 0.490] |
| llm_openrouter_qwen | 0.744 [0.726, 0.762] | 0.860 [0.835, 0.884] | 0.533 [0.489, 0.576] | 0.821 [0.804, 0.838] | 0.358 [0.305, 0.409] |
| llm_openrouter_mistral | 0.742 [0.717, 0.766] | 0.716 [0.684, 0.748] | 0.250 [0.213, 0.289] | 0.765 [0.738, 0.791] | 0.465 [0.416, 0.513] |
| llm_openrouter_qwen_max | 0.740 [0.722, 0.758] | 0.860 [0.835, 0.884] | 0.551 [0.508, 0.594] | 0.818 [0.800, 0.835] | 0.342 [0.290, 0.394] |
| llm_openrouter_deepseek_r1 | 0.739 [0.722, 0.755] | 0.896 [0.872, 0.918] | 0.623 [0.580, 0.664] | 0.825 [0.809, 0.840] | 0.324 [0.271, 0.375] |
| llm_openrouter_deepseek_v3 | 0.727 [0.713, 0.742] | 0.911 [0.889, 0.932] | 0.702 [0.661, 0.741] | 0.821 [0.806, 0.836] | 0.268 [0.215, 0.321] |
| llm_openrouter_llama_4_maverick | 0.707 [0.680, 0.732] | 0.614 [0.581, 0.645] | 0.146 [0.117, 0.177] | 0.709 [0.679, 0.737] | 0.476 [0.430, 0.520] |
| llm_openrouter_gemini_flash | 0.631 [0.598, 0.662] | 0.500 [0.465, 0.535] | 0.100 [0.075, 0.127] | 0.628 [0.592, 0.662] | 0.429 [0.383, 0.473] |
| llm_openrouter_gemini_pro | 0.627 [0.595, 0.658] | 0.476 [0.442, 0.510] | 0.050 [0.032, 0.070] | 0.609 [0.573, 0.644] | 0.473 [0.434, 0.511] |

Summary-only tools (point estimate only; CI not available):

| tool | F1 | DR | FPR | TW-F1 | MCC | CI |
|---|---|---|---|---|---|---|
| bibtexupdater | 0.909 | 0.969 | 0.193 | 0.944 | n/a | not available |
| llm_openrouter_claude_opus_4_7 | 0.830 | 0.752 | 0.072 | 0.851 | 0.683 | not available |
| llm_openrouter_claude_sonnet_4_6 | 0.827 | 0.781 | 0.127 | 0.834 | 0.652 | not available |
| doi_only | 0.373 | 0.268 | 0.185 | 0.329 | n/a | not available |

## test_public (n=831, 312 VALID / 519 HALL)

Tools with per-entry predictions (real 95% CIs):

| tool | F1 [95% CI] | DR [95% CI] | FPR [95% CI] | TW-F1 [95% CI] | MCC [95% CI] |
|---|---|---|---|---|---|
| llm_agentic_btu_sonnet_4_6 | 0.902 [0.888, 0.916] | 0.990 [0.981, 0.998] | 0.343 [0.292, 0.397] | 0.946 [0.937, 0.955] | 0.721 [0.678, 0.763] |
| llm_agentic_btu_openai | 0.883 [0.867, 0.899] | 0.960 [0.942, 0.975] | 0.356 [0.301, 0.410] | 0.927 [0.914, 0.939] | 0.661 [0.610, 0.711] |
| llm_openrouter_claude_sonnet_4_6 | 0.866 [0.847, 0.884] | 0.821 [0.794, 0.846] | 0.125 [0.090, 0.163] | 0.872 [0.852, 0.890] | 0.679 [0.634, 0.722] |
| llm_tool_augmented | 0.851 [0.831, 0.871] | 0.855 [0.827, 0.882] | 0.256 [0.208, 0.308] | 0.878 [0.857, 0.897] | 0.601 [0.546, 0.655] |
| llm_openrouter_claude_opus_4_7 | 0.846 [0.828, 0.863] | 0.763 [0.737, 0.787] | 0.067 [0.042, 0.096] | 0.871 [0.852, 0.889] | 0.673 [0.638, 0.709] |
| llm_agentic_openai | 0.827 [0.812, 0.842] | 0.942 [0.925, 0.960] | 0.558 [0.503, 0.612] | 0.888 [0.874, 0.901] | 0.464 [0.404, 0.520] |
| llm_openai_gpt-5.4 | 0.815 [0.791, 0.837] | 0.780 [0.748, 0.811] | 0.224 [0.179, 0.272] | 0.833 [0.809, 0.855] | 0.544 [0.488, 0.597] |
| llm_openrouter_deepseek_r1 | 0.812 [0.785, 0.837] | 0.809 [0.772, 0.843] | 0.319 [0.261, 0.380] | 0.842 [0.815, 0.867] | 0.488 [0.418, 0.556] |
| llm_openrouter_qwen_max | 0.804 [0.789, 0.820] | 0.927 [0.906, 0.948] | 0.628 [0.574, 0.683] | 0.875 [0.861, 0.888] | 0.372 [0.308, 0.434] |
| llm_openrouter_qwen | 0.798 [0.781, 0.814] | 0.909 [0.886, 0.932] | 0.615 [0.561, 0.670] | 0.865 [0.849, 0.880] | 0.355 [0.290, 0.419] |
| llm_openai | 0.796 [0.775, 0.816] | 0.852 [0.823, 0.879] | 0.481 [0.426, 0.535] | 0.846 [0.825, 0.865] | 0.397 [0.332, 0.460] |
| llm_openrouter_deepseek_v3 | 0.776 [0.760, 0.791] | 0.911 [0.888, 0.934] | 0.728 [0.676, 0.776] | 0.851 [0.835, 0.866] | 0.244 [0.176, 0.311] |
| llm_openrouter_mistral | 0.741 [0.713, 0.767] | 0.688 [0.651, 0.724] | 0.282 [0.234, 0.333] | 0.761 [0.732, 0.790] | 0.394 [0.334, 0.453] |
| llm_openrouter_llama_4_maverick | 0.729 [0.700, 0.758] | 0.631 [0.595, 0.668] | 0.167 [0.125, 0.208] | 0.728 [0.696, 0.759] | 0.452 [0.399, 0.506] |
| llm_openrouter_gemini_flash | 0.644 [0.610, 0.677] | 0.505 [0.466, 0.543] | 0.106 [0.074, 0.141] | 0.635 [0.597, 0.672] | 0.404 [0.354, 0.453] |
| llm_openrouter_gemini_pro | 0.613 [0.575, 0.650] | 0.458 [0.418, 0.498] | 0.059 [0.034, 0.088] | 0.610 [0.569, 0.649] | 0.420 [0.374, 0.465] |

Summary-only tools (point estimate only; CI not available):

| tool | F1 | DR | FPR | TW-F1 | MCC | CI |
|---|---|---|---|---|---|---|
| bibtexupdater | 0.863 | 0.913 | 0.337 | 0.904 | 0.608 | not available |
| doi_only | 0.498 | 0.387 | 0.279 | 0.455 | 0.110 | not available |

## Headline paired-significance verdicts (F1)

| comparison | split | backed by real paired test? | diff (higher-lower) | one-sided p | two-sided p | Cohen's h |
|---|---|---|---|---|---|---|
| sonnet_vs_opus_f1 | test_public | YES | +0.0200 (claude_sonnet_4_6 higher) | 0.0243 | 0.0486 | +0.0571 |
| sonnet_vs_opus_f1 | dev_public | NO (point-estimate only) | — | — | — | — |
| bibtexupdater_vs_sonnet_f1 | dev_public | NO (point-estimate only) | — | — | — | — |
| bibtexupdater_vs_sonnet_f1 | test_public | NO (point-estimate only) | — | — | — | — |

### How to phrase these in the paper

- **Sonnet vs Opus F1, test_public**: backed by a real stratified paired bootstrap. Sonnet F1=0.866 [0.847,0.884] vs Opus F1=0.846 [0.828,0.863]; diff +0.020, one-sided p=0.024, conservative two-sided p=0.049, Cohen's h=+0.057 (negligible). This is *borderline* significant at 0.05 and the effect is negligible — do NOT claim a clean within-CI tie; the honest statement is 'Sonnet edges Opus on F1 by ~2pp, at the 0.05 significance boundary with a negligible effect size; the gap is within the noise that a different split or seed could flip.'

- **Sonnet vs Opus F1, dev_public**: NO paired test possible (both are summary-only on dev_public — no stored per-entry predictions). Point estimates are nearly identical (Sonnet 0.827 vs Opus 0.830); any 'indistinguishable' claim here MUST be softened to point-estimate language — there is no CI or p-value to cite.

- **bibtex-updater vs Sonnet F1, either split**: NO paired test possible (bibtex-updater has no full per-entry prediction file; it is summary-only on both splits per the delta-eval reconstruction). Report point estimates only: dev 0.909 vs 0.827, test 0.863 vs 0.866. No significance claim can be backed; phrase as point-estimate comparisons.
