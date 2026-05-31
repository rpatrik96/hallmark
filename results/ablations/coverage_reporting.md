# Coverage / abstention-aware reporting (selective prediction)

**Offline.** No API calls. Stored per-entry predictions re-scored on the v1.1.1
corrected labels (`dev_public` = 513 VALID / 606 HALL, N=1119; `test_public` =
312 / 519, N=831) via `hallmark.evaluation.evaluate`. Machine-readable companion:
`coverage_reporting.json`. Reusable script: `coverage_reporting.py`. Design:
`coverage_reporting_design.md`.

Definitions used throughout:

- **Coverage** = `#committed(VALID|HALL) / N` = `1 − UNCERTAIN-rate`. Missing
  predictions are treated as committed-VALID (the repo's conservative default),
  so they do *not* lower coverage.
- **Conservative DR/FPR/F1**: UNCERTAIN excluded from the confusion matrix
  (`build_confusion_matrix` protocol; `evaluate(eval_mode="conservative")`).
- **Aggressive DR/FPR/F1**: UNCERTAIN + missing → HALLUCINATED@0.55
  (`evaluate(eval_mode="aggressive")`).

Every conservative cell below reproduces the regenerated post-relabel aggregates
in `data/v1.0/baseline_results/` to within 5e-4 (the manifest §1 check;
verified tool-by-tool), so this surfaces existing numbers rather than recomputing
the benchmark.

---

## (a) Table-2 addition — per-tool Coverage + dual scoring

### dev_public (N=1119)

| Tool | Coverage | Cons. DR/FPR/F1 | Aggr. DR/FPR/F1 |
|---|---|---|---|
| llm_openrouter_deepseek_r1 | 0.984 | 0.896 / 0.623 / 0.739 | 0.898 / 0.628 / 0.739 |
| llm_openrouter_deepseek_v3 | 1.000 | 0.911 / 0.702 / 0.727 | 0.911 / 0.702 / 0.727 |
| llm_openrouter_gemini_flash | 0.988 | 0.500 / 0.100 / 0.631 | 0.508 / 0.105 / 0.636 |
| llm_openrouter_mistral | 0.989 | 0.716 / 0.251 / 0.742 | 0.721 / 0.253 / 0.745 |
| llm_openrouter_qwen | 0.998 | 0.860 / 0.533 / 0.744 | 0.860 / 0.534 / 0.744 |
| llm_openrouter_gemini_pro | 0.967 | 0.476 / 0.050 / 0.627 | 0.497 / 0.074 / 0.637 |
| llm_openrouter_qwen_max | 0.999 | 0.860 / 0.551 / 0.740 | 0.860 / 0.552 / 0.739 |
| llm_openrouter_llama_4_maverick | 1.000 | 0.614 / 0.146 / 0.707 | 0.614 / 0.146 / 0.707 |
| llm_openrouter_claude_opus_4_7 | n/a (summary-only / drift) | 0.752 / 0.072 / 0.830 | n/a |
| llm_openrouter_claude_sonnet_4_6 | n/a (summary-only / drift) | 0.781 / 0.127 / 0.827 | n/a |
| llm_openai (gpt-5.1) | 1.000 | 0.837 / 0.411 / 0.766 | 0.837 / 0.411 / 0.766 |
| llm_openai_gpt_5_4 | 1.000 | 0.767 / 0.228 / 0.783 | 0.767 / 0.228 / 0.783 |
| llm_agentic_openai | 1.000 | 0.967 / 0.478 / 0.816 | 0.967 / 0.478 / 0.816 |
| llm_agentic_btu_openai | 1.000 | 0.980 / 0.470 / 0.824 | 0.980 / 0.470 / 0.824 |
| llm_agentic_btu_sonnet_4_6 | 1.000 | 0.990 / 0.431 / 0.841 | 0.990 / 0.431 / 0.841 |
| **bibtexupdater (v1.2.0)** | **0.746** | **0.979 / 0.045 / 0.962** | **0.987 / 0.140 / 0.937** |

The two Anthropic dev rows carry their **published delta-eval point estimates**
(`results_manifest §1` / `todo_sonnet_opus_dev.json`), not a fresh re-score: the
OpenRouter Anthropic endpoint drifted since the published run and the original
per-entry predictions were never persisted, so coverage and the aggressive stance
are not recoverable for these two cells. Their *risk-coverage curves* are reported
in §(b) (drift-immune for the curve shape).

### test_public (N=831)

| Tool | Coverage | Cons. DR/FPR/F1 | Aggr. DR/FPR/F1 |
|---|---|---|---|
| llm_openrouter_deepseek_r1 | 0.783 | 0.809 / 0.319 / 0.812 | 0.848 / 0.481 / 0.793 |
| llm_openrouter_deepseek_v3 | 1.000 | 0.911 / 0.728 / 0.776 | 0.911 / 0.728 / 0.776 |
| llm_openrouter_gemini_flash | 1.000 | 0.505 / 0.106 / 0.644 | 0.505 / 0.106 / 0.644 |
| llm_openrouter_mistral | 1.000 | 0.688 / 0.282 / 0.741 | 0.688 / 0.282 / 0.741 |
| llm_openrouter_qwen | 0.999 | 0.909 / 0.615 / 0.798 | 0.909 / 0.615 / 0.798 |
| llm_openrouter_gemini_pro | 0.907 | 0.458 / 0.059 / 0.613 | 0.512 / 0.135 / 0.643 |
| llm_openrouter_qwen_max | 1.000 | 0.927 / 0.628 / 0.804 | 0.927 / 0.628 / 0.804 |
| llm_openrouter_llama_4_maverick | 0.999 | 0.631 / 0.167 / 0.729 | 0.632 / 0.167 / 0.730 |
| llm_openrouter_claude_opus_4_7 | 0.999 | 0.762 / 0.067 / 0.846 | 0.763 / 0.067 / 0.846 |
| llm_openrouter_claude_sonnet_4_6 | 1.000 | 0.821 / 0.125 / 0.866 | 0.821 / 0.125 / 0.866 |
| llm_openai (gpt-5.1) | 1.000 | 0.852 / 0.481 / 0.796 | 0.852 / 0.481 / 0.796 |
| llm_openai_gpt_5_4 | 1.000 | 0.780 / 0.224 / 0.815 | 0.780 / 0.224 / 0.815 |
| llm_agentic_openai | 1.000 | 0.942 / 0.558 / 0.827 | 0.942 / 0.558 / 0.827 |
| llm_agentic_btu_openai | 1.000 | 0.960 / 0.356 / 0.883 | 0.960 / 0.356 / 0.883 |
| llm_agentic_btu_sonnet_4_6 | 1.000 | 0.990 / 0.343 / 0.902 | 0.990 / 0.343 / 0.902 |
| llm_tool_augmented | 1.000 | 0.856 / 0.256 / 0.851 | 0.856 / 0.256 / 0.851 |
| **bibtexupdater (v1.2.0)** | **PENDING** | **PENDING** | **PENDING** |

**bibtexupdater test_public is NOT reportable yet.** The v1.2.0 raw status file
covers only ~500/831 entries (the count is still growing — the GEN workflow
`wkp97jqbb` is actively writing it). Unscored entries silently default to
committed-VALID, which deflates DR and inflates apparent coverage; the script
flags this split `INCOMPLETE` and refuses to ship it. The current partial numbers
(coverage ≈0.82, cons DR ≈0.45) are recorded in the JSON only as a moving
provisional value, not for the paper. **Recompute once `wkp97jqbb` completes.**

The conservative→aggressive gap is the abstention's cost. For full-coverage tools
it is ~0 (nothing to flip). It matters exactly where abstention is real:
bibtex-updater dev (coverage 0.746) pays **+9.5pp FPR** (0.045→0.140) for +0.8pp
DR when its abstentions are force-flagged; DeepSeek-R1 test (coverage 0.783) pays
**+16.2pp FPR** (0.319→0.481) for +3.9pp DR. The aggressive column is the
guardrail number (see Note 1).

---

## (b) Risk–coverage / selective-prediction view (appendix figure)

For each tool with graded confidences we abstain on the least-confident entries
first (smallest `|P(hallucinated) − 0.5|`), widening an abstention band around the
decision boundary, and score FPR/DR/F1 on the committed entries at the model's
native 0.5 decision. This reuses the **same per-entry confidences the A3 ablation
consumed** (`a3_threshold_full/`); AUROC is carried over from A3 where available.
Full curves are in `coverage_reporting.json["risk_coverage"][tool]["curve"]`
(coverage grid 1.00→0.05 in 0.05 steps) — that is the figure-ready data.

### FPR@90%-coverage (dev_public)

| Tool | AUROC | FPR@90% | DR@90% | F1@90% |
|---|---|---|---|---|
| llm_openrouter_claude_sonnet_4_6 † | 0.928 | 0.120 | 0.923 | 0.914 |
| llm_openrouter_claude_opus_4_7 † | 0.906 | 0.095 | 0.903 | 0.909 |
| llm_openai_gpt_5_4 | 0.834 | 0.222 | 0.811 | 0.816 |
| llm_openrouter_mistral | 0.744 | 0.241 | 0.705 | 0.736 |
| llm_openrouter_deepseek_r1 | 0.741 | 0.581 | 0.888 | 0.752 |
| llm_openrouter_gemini_flash | 0.739 | 0.097 | 0.445 | 0.578 |
| llm_openrouter_qwen | 0.695 | 0.507 | 0.853 | 0.742 |
| llm_openrouter_deepseek_v3 | 0.609 | 0.699 | 0.900 | 0.711 |
| llm_openrouter_gemini_pro | — | 0.043 | 0.486 | 0.637 |
| llm_openrouter_qwen_max | — | 0.530 | 0.858 | 0.743 |
| llm_openrouter_llama_4_maverick | — | 0.121 | 0.604 | 0.706 |
| llm_openai (gpt-5.1) | — | 0.373 | 0.826 | 0.774 |
| llm_agentic_openai | — | 0.423 | 0.964 | 0.833 |
| llm_agentic_btu_openai | — | 0.427 | 0.985 | 0.834 |
| llm_agentic_btu_sonnet_4_6 | — | 0.342 | 0.995 | 0.887 |

† Anthropic dev curves are **appendix-only with a drift caveat**: the operating
point is drift-affected, but the curve *shape* and AUROC are deterministic
re-scores of the stored confidences (drift-immune, per A3's snapshot note). They
do not feed the Table-2 coverage cells. AUROC `—` denotes tools A3 did not sweep
(agentic / gpt-5.1 / new_models); their curves are still in the JSON.

The curve confirms the selective-prediction story: abstaining on the
least-confident decile buys a real FPR reduction for the better-calibrated
verifiers. gpt-5.4, for instance, drops FPR 0.228→0.150 and lifts F1 0.783→0.845
between 100% and 75% coverage. The high-AUROC verifiers (Sonnet/Opus, AUROC ≈0.91–0.93)
sit at the favourable corner — high DR with FPR ≈0.10–0.12 at 90% coverage.

---

## (c) Three honest notes (non-negotiable, from the design)

**1. Coverage prevents gaming.** Coverage must sit next to every
abstain-excluded metric, otherwise abstention is trivially gamed: abstain on
everything and the conservative precision-on-committed is perfect. The Coverage
column plus the aggressive number close that loophole — the aggressive stance
prices in every abstention as a flag, so a tool cannot hide a weak verdict behind
UNCERTAIN. This is why bibtex-updater's headline (cons F1 0.962 on dev) is
reported *with* coverage 0.746 and the aggressive F1 0.937: it earns its precision
on the 74.6% it commits to, not on the whole split.

**2. LLM abstention is prompt-dependent.** Coverage is not a fixed tool property.
The A1 prompt-sensitivity ablation (`a1_prompt_full/`, n=150 stratified sample)
shows Sonnet 4.6's UNCERTAIN rate moving from 2.0% (default / terse variants) to
8.7% (the abstention-encouraging "uncertain" variant) — a ~7pp coverage swing
from prompt wording alone. The headline coverage numbers here use the fixed
default prompt (`VERIFICATION_PROMPT`, identical to the paper); the A1 sensitivity
must be cited alongside so coverage is read as "coverage under the default
prompt", not an intrinsic constant.

**3. BTU vs LLM abstention differ in mechanism (same column, asymmetric meaning).**
bibtex-updater abstains because *no record was found* — a data-coverage gap in the
backing databases (`not_found` / `partial_match` / `unconfirmed`, mapped to
UNCERTAIN via the cascade `ROUTE_TO_STAGE2` set). An LLM abstains because it is
*not confident* — an epistemic gap. Both occupy the Coverage column, but BTU's
26% abstention on dev is "the databases don't know", whereas an LLM's is "the
model won't commit". Under the route-to-human framing both are the same useful
action (defer rather than false-flag), which is why the column is shared — but the
asymmetry should be stated in one sentence so readers don't equate the two
mechanisms. The selective-prediction lineage (El-Yaniv & Wiener 2010; Geifman &
El-Yaniv 2017) is worth citing: no peer citation-benchmark reports coverage, so
this is ahead-of-field.

---

## (d) Mapping to the paper

**Table 2 (`tab:results`)** gains one **Coverage** column and reports DR/FPR/F1 in
two stances. The conservative triple is the existing headline number (already what
the table shows); the aggressive triple and Coverage are the additions. Transcribe
the §(a) grids directly — conservative cells already match
`data/v1.0/baseline_results/`. Hold bibtex-updater's **test_public** row until GEN
workflow `wkp97jqbb` finishes; its **dev_public** row (coverage 0.746) is final.
The two Anthropic **dev** rows keep their published point estimates with Coverage
printed as "n/a (drift)" and no aggressive cell.

**Appendix selective-prediction figure** plots the per-tool risk–coverage curves
from `coverage_reporting.json["risk_coverage"][*]["curve"]` (FPR vs coverage), with
**FPR@90%-coverage** from the §(b) table as the single summary number per tool.
Mark the two Anthropic curves with the drift caveat. Cite the three notes in the
figure/table caption and the selective-prediction lineage in the surrounding text;
tie it to the reviewer-bound, FPR-decisive framing (abstain = route-to-human).
