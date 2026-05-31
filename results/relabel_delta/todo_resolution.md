# Camera-ready %TODO resolution map (consolidated)

**Date:** 2026-05-30
**Benchmark branch:** `chore/paper-todo-resolution` (v1.1.1 corrected data: dev 513 valid / 606 hall, n=1119; test 312/519; hidden 210/244)
**Paper branch:** `arxiv-prep` (`/Users/patrik.reizinger/Documents/GitHub/hallmark-paper`)

Consolidates the three Generate outputs:
- `results/relabel_delta/todo_offline.json` — offline re-score of persisted per-entry predictions vs NEW labels (per-source DR, per-type, shortcut refit, DOI per-type status)
- `results/relabel_delta/todo_sonnet_opus_dev.json` — full-coverage Sonnet 4.6 / Opus 4.7 dev re-run (TASK B; endpoint-drift finding)
- `results/relabel_delta/todo_hard.json` — F10 / codesign per-type / cascade AUROC (TASK C)

Authoritative anchor: `results/reviewer_experiments/results_manifest.md` §11 (Stage-3c). All "resolved" numbers below either (a) are faithfully recomputed offline from persisted per-entry predictions vs the corrected labels, or (b) are carried-over pre-relabel values that §11 certifies as not faithfully regenerable, replaced by a one-sentence documented limitation. **No fabricated or unfaithful number is used to remove a TODO.**

---

## Summary of the 7 markers

| # | File:line | Marker | Disposition |
|---|-----------|--------|-------------|
| 1 | `tables/llm_comparison.tex:31` | `% TODO(orphaned-schema)` | DOCUMENTED LIMITATION (drop row OR re-source) — no faithful backing file exists |
| 2 | `tables/llm_comparison.tex:111` | `% TODO(F5)` per-source rows | RESOLVED (faithful offline per-source DR/FPR) + one scraped-row caveat |
| 3 | `appendix.tex:304` | `% TODO(F5)` GPT-5.1 stratified_dr | RESOLVED (faithful offline) + scraped-row caveat |
| 4 | `appendix.tex:553` | `% TODO(relabel)` shortcut refit | RESOLVED (faithful offline refit) — direction unchanged, see coefficient caveat |
| 5 | `appendix.tex:690` | `% TODO(F5)` DOI/S4.6/O4.7 per-type | DOCUMENTED LIMITATION (summary-only; Sonnet/Opus re-run drifted, do NOT ship) |
| 6 | `appendix.tex:982` | `% TODO(F5)` bibtex-updater per-type | DOCUMENTED LIMITATION (summary-only; raw-tool config differs from published) |
| 7 | `appendix.tex:1058` | `% TODO(F5)` tool-augmented per-type | DOCUMENTED LIMITATION (summary-only augmented-eval subset; not re-scorable) |

Plus one **cross-file**: cascade AUROC (`sections/experiments.tex` `tab:cascade`) carries the `% TODO(F7)` per `todo_hard.json`/§11.4 — included below for completeness even though it did not surface in the grep (it is a `% TODO(F7)` comment; verify its exact line in `experiments.tex`).

---

## 1. `tables/llm_comparison.tex:31` — `% TODO(orphaned-schema)` (GPT-5.1 row in `tab:llm_comparison`)

**What it governs:** the `GPT-5.1` row in `tab:llm_comparison` printing `DR 0.797 / FPR 0.171 / F1 0.822 / TW-F1 0.846 / ECE 0.107 / AUROC 0.829 / MCC —`.

**Disposition: DOCUMENTED LIMITATION — these numbers cannot be made faithful.** They come from a retired GPT-5.1 evaluation schema with no surviving backing file, and they contradict the main-results GPT-5.1 (`tab:results`: dev_public NEW = DR 0.837 / FPR 0.411 / F1 0.766 / MCC 0.442 / TW-F1 0.822 / ECE 0.190 / union_recall 0.775). There is no faithful path to the printed 0.171/0.822/0.107/0.829 schema-mismatched row.

**Two clean actions (author's choice — both faithful):**

- **(A, preferred) Drop the GPT-5.1 row** from `tab:llm_comparison` and the appendix-comparison framing (this table already declares it focuses on "three representative LLMs" — keep Qwen3-235B and DeepSeek-V3.2, which ARE backed by persisted per-entry files). Adjust L10-14 prose accordingly ("two representative LLMs … high-detection (Qwen3-235B) and coverage-aggressive (DeepSeek-V3.2)").

- **(B) Replace the row with the main-results GPT-5.1 numbers** and drop the AUROC cell (no per-entry score distribution is persisted for GPT-5.1 zero-shot dev, so AUROC is not recomputable):
  `GPT-5.1 & 0.837 & 0.411 & 0.766 & 0.442 & 0.822 & 0.190 & --- & 1.00 & 0`
  (DR 0.837, FPR 0.411, F1 0.766, MCC 0.442, TW-F1 0.822, ECE 0.190, AUROC —, Cov 1.00, Unc 0). Source: `todo_offline.json:gpt51_llm_comparison_row` / manifest §1. Note this makes GPT-5.1 no longer "best F1/ECE/AUROC" — the bold markers and the L19/L48-50 narrative ("GPT-5.1 achieves the best F1 and calibration") would need revising, because under the corrected schema DeepSeek-V3.2/Qwen relationships change. Action (A) avoids that prose churn.

**One-sentence limitation note (if neither row replacement is desired, replace the bare TODO comment with):**
> The GPT-5.1 aggregate previously reported here came from a retired evaluation schema (FPR 0.171, AUROC 0.829) that no surviving per-entry file reproduces and that disagrees with the main-results GPT-5.1 (FPR 0.411; \cref{tab:results}); we therefore omit GPT-5.1 from this representative-model comparison and refer to \cref{tab:results} for its corrected aggregate.

---

## 2. `tables/llm_comparison.tex:111` — `% TODO(F5)` per-source rows (`tab:llm_gen_method`)

**What it governs:** the four hallucinated-source DR rows + scraped FPR row for Qwen3-235B and DeepSeek-V3.2, currently printing pre-relabel values.

**Disposition: RESOLVED — faithful offline per-source recompute vs NEW labels** (`todo_offline.json:per_source_dr.tools_offline`, persisted per-entry files `results/llm_openrouter_qwen_dev_public_predictions.jsonl` and `results/llm_openrouter_deepseek_v3_dev_public_predictions.jsonl`).

**Replace the table body (L115-119) with NEW values:**

| Generation method | n | Qwen3-235B | DeepSeek-V3.2 |
|---|---|---|---|
| Adversarial | 60 | 0.917 | 1.000 |
| Perturbation | 410 | 0.866 | 0.937 |
| Real-world | 46 | 0.978 | 0.935 |
| LLM-generated | 90 | 0.730 | 0.722 |
| Scraped (FPR) | 486 | 0.551 | 0.731 |

LaTeX rows:
```
Adversarial       & 60  & 0.917 & 1.000 \\
Perturbation      & 410 & 0.866 & 0.937 \\
Real-world        & 46  & 0.978 & 0.935 \\
LLM-generated     & 90  & 0.730 & 0.722 \\
Scraped (FPR)     & 486 & 0.551 & 0.731 \\
```

**Caveats the author must reconcile (do NOT silently ship without addressing):**
- **n column changed**: perturbation 414→410, LLM-generated 113→90, scraped 513→486. The corrected relabel moved 4 perturbation + 23 LLM-generated entries into the VALID pool (so those two methods now also carry a small per-method FPR — Qwen perturbation-FPR 0.75 / LLM-gen-FPR 0.091; DeepSeek perturbation-FPR 1.0 / LLM-gen-FPR 0.043 over n_valid 4 and ~23). The table as structured shows DR for hallucinated rows and FPR only for scraped; that structure is still defensible, but the **"Scraped (FPR) 513"** denominator in the current draft is wrong post-relabel — the scraped *valid* pool is **486**, not 513 (the other 27 valid entries are 4 perturbation + 23 LLM-generated). Use **486**.
- **L123 footnote** "LLM-generated entries include the GPT-5.1 baseline evaluation result (DR = 0.611)": GPT-5.1 LLM-generated NEW DR is **0.656** (`stratified_dr_gpt51.rows.llm_generated.DR_new = 0.6556`), up from 0.611. Update the parenthetical to **0.656**.
- **L126 prose** "DR 0.584-0.604 (LLM-generated) … perturbation 0.865-0.937": NEW LLM-generated range is **0.722-0.730**; perturbation range **0.866-0.937**. The qualitative claim (LLM-generated substantially harder than perturbation; rules out self-recognition bias) still holds — update the numeric range to **0.722-0.730** vs **0.866-0.937**.
- **L128 prose** "Adversarial … 0.917-1.000": unchanged (still 0.917-1.000). Keep.

---

## 3. `appendix.tex:304` — `% TODO(F5)` GPT-5.1 `tab:stratified_dr`

**What it governs:** the GPT-5.1 per-generation-method DR/FPR rows (L309-313).

**Disposition: RESOLVED — faithful offline recompute vs NEW labels** (`todo_offline.json:stratified_dr_gpt51`, persisted per-entry `results/checkpoints/llm_openai/openai_gpt-5.1.jsonl`).

**Replace the table body (L309-313) with NEW values:**

| Generation method | n (hall.) | n (valid) | DR | FPR |
|---|---|---|---|---|
| Adversarial | 60 | --- | 1.000 | --- |
| Perturbation | 410 | 4 | 0.846 | 1.000 |
| Real-world | 46 | --- | 0.891 | --- |
| LLM-generated | 90 | 23 | 0.656 | 0.435 |
| Scraped (valid) | --- | 486 | --- | 0.405 |

LaTeX rows:
```
Adversarial      & 60  & ---  & 1.000 & ---   \\
Perturbation     & 410 & 4    & 0.846 & 1.000 \\
Real-world       & 46  & ---  & 0.891 & ---   \\
LLM-generated    & 90  & 23   & 0.656 & 0.435 \\
Scraped (valid)  & --- & 486  & ---   & 0.405 \\
```

**Caveats (must reconcile):**
- **Scraped FPR is 0.405 over n_valid=486**, NOT 0.411 over 513. The headline GPT-5.1 dev FPR of **0.411** is computed over ALL 513 valid entries; the *scraped-only* FPR is 0.4053. The current draft's F5 comment claims "Only the Scraped (valid) row is updated (n=486 → 513; FPR 0.405 → 0.411)" — that conflates the scraped pool with the full valid pool. Faithful per-source values: scraped n_valid=**486**, scraped FPR=**0.405**. If the table is meant to show per-source breakdown, use 486/0.405 and add the 4 perturbation-valid + 23 LLM-gen-valid (with their per-method FPR 1.000 / 0.435) as shown above; the four per-method FPR values and 0.405 reconcile to the headline 0.411 over 513. **Do not print "Scraped (valid) 513 / 0.411" — that is internally inconsistent under the corrected per-source split.**
- The table currently has only a Scraped FPR column for valid; the relabel introduced 4+23 valid entries in perturbation/LLM-generated, so those rows now legitimately carry an FPR. Including them (as above) is the faithful representation.
- The L296 caption and the adversarial DR: NEW adversarial DR=**1.000** (offline canonical recompute), vs printed 0.983. Per manifest §1/F1 the printed 0.983 traces to a slightly different GPT-5.1 snapshot; the canonical persisted file gives 1.000. Recommend **1.000** (faithful to the shipped per-entry file). Perturbation similarly: canonical NEW=0.846 vs printed 0.829. Use **0.846** (canonical-file recompute). Real-world (0.891) and LLM-generated headline both reproduce.

---

## 4. `appendix.tex:553` — `% TODO(relabel)` shortcut logistic-regression refit

**What it governs:** L554 "cross-validated accuracy of 58.5\% (majority-class baseline: 56.6\%), a margin of 1.9pp", L555 "below 5pp", L558 "has_doi (coefficient 0.12)", L559 "All other features have coefficients < 0.05".

**Disposition: RESOLVED — faithful refit on NEW labels** (`todo_offline.json:shortcut`, reproducible via `results/relabel_delta/refit_shortcut.py`).

**Headline Variant B refit (StandardScaler + StratifiedKFold(5, shuffle, seed=42), LogisticRegression):**
- CV accuracy = **0.588 (58.8%)**
- Majority-class baseline = **0.542 (54.2%)** (post-relabel dev majority class is HALLUCINATED, 606/1119)
- Margin = **0.046 (4.6pp)**

**Replace L554 with:**
> The logistic regression achieves a cross-validated accuracy of 58.8\% (majority-class baseline: 54.2\%), a margin of 4.6pp.

L555 "Since this margin is below 5pp, metadata features provide negligible predictive signal" — **still holds** (4.6pp < 5pp). Keep. Direction of the conclusion is unaffected, exactly as the TODO anticipated.

**Coefficient caveat (must reconcile — methodology mismatch):**
- The original "has_doi coefficient 0.12 … all others < 0.05" came from the **unscaled** shipped script (`scripts/analyze_shortcuts.py`). The faithful Variant B refit standardizes features, so its `has_doi` coefficient is **1.19** (standardized), not 0.12, and the "<0.05" claim does not translate. Two faithful options:
  - **(A)** Keep the unscaled shipped-script methodology for the *coefficient* sentences only and re-run it on NEW labels to get the new raw `has_doi` coefficient (the script exists; raw-coefficient refit was not separately persisted in the JSON — a ~5-second re-run). The unscaled CV variant, however, gives 54.2% / margin 0.08pp because its **fold 5 collapses** on the all-HALLUCINATED tail of the dev file (`variant_shipped_script_unshuffled_raw.note`) — that 54.2% is NOT a faithful signal estimate, so do not use it for the headline accuracy.
  - **(B, preferred for consistency)** Report the Variant B headline (58.8% / 54.2% / 4.6pp) and **soften the coefficient sentences** to the qualitative claim that survives standardization: `has_doi` is the single most informative feature (consistent with the DOI-only baseline's non-zero DR) and no single feature dominates — dropping the specific "0.12"/"<0.05" magnitudes, which were scale-dependent artifacts of the unscaled fit.

  Recommended L558-559 rewrite (Option B):
  > The most informative feature is \texttt{has\_doi}, consistent with the DOI-only baseline's non-zero detection rate; no single metadata feature carries enough weight to push CV accuracy meaningfully above the class-prevalence baseline.

---

## 5. `appendix.tex:690` — `% TODO(F5)` DOI / S4.6 / O4.7 per-type cells (`tab:pertype_full`)

**What it governs:** the `DOI`, `S4.6` (Sonnet 4.6), `O4.7` (Opus 4.7) columns of the 13-column per-type heatmap table (L670-684). The other 10 columns (G51, G54, R1, V3, Q3, ML, GF, L4, GP, QV) are already faithfully recomputed (manifest §11.1) and need no change.

**Disposition: DOCUMENTED LIMITATION — keep printed OLD cells, keep the `% TODO(F5)` as a permanent caveat (it is already mirrored in the table caption L659).** Three independent reasons make a faithful regeneration impossible:
1. **DOI-only:** no persisted per-entry DOI-resolution file exists; a fresh recompute needs live `doi.org` HEAD requests (rate-limited, external-DB drift confirmed by the "transient block" note in `doi_only_dev_public_changed_predictions.json`). `todo_offline.json:doi_pertype.per_type_status = DOCUMENTED_LIMITATION`.
2. **Sonnet 4.6 / Opus 4.7:** these were summary-only on dev (no persisted per-entry predictions from the original 2026-05-04 run, commit 493cbb3). TASK B DID produce a full 1119-entry re-run with persisted per-entry files (`results/checkpoints/llm_openrouter_claude_sonnet_4_6_dev_public/…jsonl`, `…claude_opus_4_7_dev_public_zeroshot/…jsonl`), **but those re-runs DIVERGE from the published aggregates beyond tolerance due to OpenRouter Anthropic endpoint drift** (Opus dev DR .909 vs published .752, +15.7pp; Sonnet DR .916 vs .781, +13.5pp; controlled replay shows only 90% Opus / 75% Sonnet label agreement on identical inputs — `endpoint_drift_probe/drift_summary.json`). **Shipping the re-run per-type cells would contradict the delta-eval that all of manifest §11 (PPV, Pareto, takeaways) rests on.** `todo_sonnet_opus_dev.json` explicitly verdicts: "do NOT ship these as the paper's Sonnet/Opus dev numbers."

**Action:** No cell changes for DOI/S4.6/O4.7. The current caption text (L659) already documents this exactly. Keep the `% TODO(F5)` comment OR (cleaner for camera-ready) replace the bare TODO comment with this one-sentence limitation note so no `TODO` token ships:
> % The DOI-only, Sonnet~4.6, and Opus~4.7 per-type cells are carried over from the pre-relabel evaluation: these tools are summary-only on \texttt{dev\_public} (no persisted per-entry verdicts), and a faithful re-run is precluded by external-database drift (DOI resolution) and OpenRouter Anthropic endpoint drift (Sonnet/Opus re-runs diverge >13pp from the published aggregates the delta-eval depends on), so we report their per-type rates at the pre-relabel values consistent with the carried-forward aggregates.

(The persisted drifted re-run + CIs live at `todo_sonnet_opus_dev.json` for reproducibility, NOT for paper substitution.)

---

## 6. `appendix.tex:982` — `% TODO(F5)` bibtex-updater per-type (`tab:codesign_pertype`)

**What it governs:** all 14 per-type DR cells for bibtex-updater (L995-1014).

**Disposition: DOCUMENTED LIMITATION — keep printed OLD per-type cells (caption L985 already states this).** Per `todo_hard.json:F5_bibtexupdater_codesign_pertype` + manifest §11.2: bibtex-updater is summary-only on dev for per-type purposes. The released RAW bibtex-check per-entry file (`data/v1.0/baseline_results/bibtexupdater_raw_dev_public.jsonl`) is offline-rescorable but reflects a **tool-only configuration ~6 false positives short of the PUBLISHED config** (raw DR .972 / FPR .181 vs published .969 / .193) — substituting raw-tool per-type DRs would be inconsistent with the published bibtex-updater aggregate the paper reports in `tab:results`/`tab:codesign`. A fresh re-run is unfaithful (Semantic Scholar rate-limit timeouts: 8/12 probe entries timed out; DB drift; no S2_API_KEY). The aggregate metrics in `tab:codesign` ARE updated (DR .969, FPR .193, MCC **.794** — see manifest §11.6); only the per-type cells are carried over.

**Action:** No per-type cell changes. The caption (L985) already documents it. Replace the bare `% TODO(F5)` comment (if removing all TODO tokens for camera-ready) with:
> % bibtex-updater is summary-only on \texttt{dev\_public}: the deployed (DOI-resolution-inclusive) configuration stored no per-entry verdicts, the released raw bibtex-check file reflects a tool-only configuration ${\sim}6$ false positives short of the published aggregate, and a fresh re-run is precluded by Semantic Scholar rate-limiting and database drift, so these per-type rates are carried forward from the pre-relabel evaluation.

Transparency artifact (NOT for substitution): `results/relabel_delta/bibtexupdater_pertype_rawtool_dev.json` records the raw-tool-only per-type DR on NEW labels, documenting the direction of movement only.

---

## 7. `appendix.tex:1058` — `% TODO(F5)` tool-augmented per-type (`tab:tool_augmented_pertype`)

**What it governs:** the per-type cells of the tool-augmented GPT-5.1 table (GPT-5.1 / BTU / Augmented columns, L1069+).

**Disposition: DOCUMENTED LIMITATION — keep printed OLD per-type cells (caption L1062 already states this).** This table's per-type buckets are computed over the augmented-eval subset; the augmented baseline is summary-only per-type (the persisted aggregate file `data/v1.0/baseline_results/llm_tool_augmented_dev_public.jsonl` carries 79 UNCERTAIN and was not re-scored cell-by-cell against NEW labels). The aggregate rows in `tab:tool_augmented` (L1042-1044) are updated (manifest §1: tool-augmented dev NEW DR 0.843 / FPR 0.144 / ECE 0.078); the per-type cells stay carried over. The BTU column inherits the same §11.2 bibtex-updater summary-only constraint, and the GPT-5.1 per-type column here is the pre-relabel set (the faithfully-recomputed NEW GPT-5.1 per-type DRs live in `tab:pertype_full` / `tab:llm_pertype`, e.g. author_mismatch 0.448, near_miss 0.577, chimeric 0.894, plausible 0.934 — if desired the author may sync the GPT-5.1 column of THIS table to those faithful values, but the BTU and Augmented columns remain carried-over).

**Action:** No required cell changes (caption documents it). Replace the bare `% TODO(F5)` comment for camera-ready with:
> % The per-type cells for the tool-augmented baseline are summary-only (the persisted aggregate carries UNCERTAIN verdicts and was not re-scored cell-by-cell); the aggregate rows in \cref{tab:tool_augmented} are recomputed against the corrected labels, but these per-type buckets are carried over from the pre-relabel evaluation.

(Optional faithful improvement: replace ONLY the GPT-5.1 column with its NEW per-type DRs from `tab:pertype_full`; the BTU/Augmented columns stay carried-over. The L1053 prose "author_mismatch remains at 63.3%" refers to the *Augmented* column, which is carried-over — leave it.)

---

## Cross-file: cascade AUROC `% TODO(F7)` — `sections/experiments.tex` `tab:cascade`

**Not surfaced by this repo's grep run** (verify the exact line in `sections/experiments.tex`; per `todo_hard.json` it is around the `tab:cascade` AUROC cells / L118-128 and the takeaway). Included here so the F7 marker is not lost.

**What it governs:** the 4 AUROC cells (test cons .867 / agg .805, dev cons .833 / agg .740) and the L128 takeaway AUROC deltas (−6.2pp test, −9.3pp dev).

**Disposition: DOCUMENTED LIMITATION — keep printed AUROC, keep/convert the `% TODO(F7)`.** Per `todo_hard.json:F7_cascade_auroc` + manifest §11.4: AUROC needs the full per-entry score distribution, which was never persisted for the cascade (summary-only; the 27 stored confidences are quantized stage constants with no ROC gradient), and a fresh run is unfaithful (Stage-1 bibtex-check DB drift: a fresh run reports 16/27 dev moved keys VALID vs the released FN=11). **T3-F1 IS faithful and unchanged to 3dp** (dev cons .417, dev agg .570, test cons .596, test agg .707 — recomputed from reconstructed per-tier counts; keep printed). The L128 T3-F1 deltas (+15.3pp dev = .570−.417, +11.1pp test = .707−.596) are UNCHANGED. The AUROC deltas (−6.2/−9.3pp) are arithmetic on the carried-over AUROC cells; keep as printed, self-consistent.

**One-sentence limitation note (replace the bare `% TODO(F7)`):**
> % Cascade AUROC is reported at its pre-relabel value: AUROC requires the full per-entry score distribution, which was not persisted for the (summary-only) cascade, and the cascade's Stage-1 database state has drifted, so a fresh run is not faithful to the released per-key verdicts; the relabel-affected keys were all cascade true-positives that become false positives under the corrected label, which leaves Tier-3 F1 (recomputed exactly from the reconstructed per-tier counts) unchanged to three decimals, so only the rank-discrimination AUROC is carried over.

---

## Also-resolved (faithful, no longer carries a TODO comment in the grep, anchored by `todo_hard.json`)

- **F10 — GPT-5.4 temporal 448 (FPR 0.413)** (`analysis.tex`, `appendix.tex` `tab:gpt54-probe`): **RESOLVED faithful**. The per-entry predictions exist and reproduce 0.413/89.9/−32.9 exactly (`results/gpt54_probe/llm_openai_gpt54_default_temporal_predictions.jsonl`; self-contained export + offline rescore asserting FPR 0.413/DR 0.899 at `results/temporal_supplement/gpt54_448/`). No number changes; the value is now offline-reproducible. If a `% TODO(F10)` comment still exists near `tab:gpt54-probe`, it is removable (cite the per-entry file as the reproducibility source).

## Persisted artifacts (offline-rescorable, for reproducibility)

- Per-source/per-type offline re-score: `results/relabel_delta/todo_offline.json` (+ `refit_shortcut.py`)
- Sonnet/Opus full re-run + CIs (drifted; NOT for substitution): `results/relabel_delta/todo_sonnet_opus_dev.json`, predictions under `results/checkpoints/llm_openrouter_claude_{sonnet_4_6,opus_4_7}_dev_public*/`, drift probe `results/relabel_delta/endpoint_drift_probe/drift_summary.json`
- Hard TODOs (F10/codesign/cascade): `results/relabel_delta/todo_hard.json`; GPT-5.4 448 export `results/temporal_supplement/gpt54_448/`; bibtex-updater raw-tool per-type `results/relabel_delta/bibtexupdater_pertype_rawtool_dev.json`
- Authoritative map: `results/reviewer_experiments/results_manifest.md` §11
