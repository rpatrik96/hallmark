# HALLMARK Benchmark Hardening Brief

*Prepared 2026-05-30 from a 58-agent stress-test workflow (8 audit dimensions → adversarial verification of every finding → synthesis), 49 findings, all 49 survived verification. The load-bearing numbers below were re-reproduced by hand (independent of the agents) against the working tree of `fix/dev-public-mislabel-audit`.*

**Independently verified facts (reproduced in this session, not just agent-reported):**

- Working-tree label counts: `dev_public` **503 VALID / 616 HALL**, `test_public` **302 / 529**. Paper `tab:stats` still prints 486/633 and 287/544. 17 dev + 15 test labels were flipped `HALLUCINATED→VALID` (all in that direction).
- **Residual mislabels still in the released splits: 56 entries** (`HALLUCINATED` + `plausible_fabrication` + an arXiv DataCite DOI whose ID is a real landmark paper) — e.g. **OPT** (2205.01068), **PaLM** (2204.02311), **Flamingo** (2204.14198), **Make-A-Video** (2209.14792), **Linformer** (2006.04768), **SCAFFOLD** (1910.06378), **Concrete Problems in AI Safety** (1606.06565), **FedProx** (1812.06127), **Learning to Summarize w/ Human Feedback** (2009.01325), **Benchmarking GNNs** (2003.00982), **Sparks of AGI** (2303.12712). Agent census of the arXiv-DOI `plausible_fabrication` cohort: ~71% (39/55) resolve to a real paper with title match ≥0.80.
- **Re-score (fixed labels) is benign:** reproduced the paper's published OLD Table-2 cells exactly (deepseek_r1 .871/.640/.737, deepseek_v3 .880/.730/.721, gemini_flash .482/.101/.617, mistral .691/.258/.731, qwen .832/.551/.737), and the NEW-label deltas are **|ΔFPR| ≤ 0.016, |ΔF1| ≤ 0.007 with no ranking change**.
- **Hidden split is not contamination-resistant:** 109/253 (**43%**) of `test_hidden` HALL entries share a normalized title with a public-split entry; 51/253 (20%) share a DOI.
- **Canary detector does not exist:** `is_canary_entry()` only checks a `__canary__` `bibtex_key` prefix to *exclude* canaries from eval — there is no scanner for verbatim canary-GUID emission in model outputs.
- **Pre-screening contradiction:** bibtex-updater dev FPR is exactly **87/486 = 0.179**.
- **VALID-with-type inconsistency:** ≥13 entries are now labeled `VALID` while retaining `hallucination_type` (9 `plausible_fabrication`, 3 `swapped_authors`, 1 `wrong_venue` among arXiv-DOI ones alone) and stale sub-tests (`title_exists=false`).

---

## 1. Executive summary — what most threatens acceptance

Five issues dominate, and they share a single root cause: **a CrossRef-only auto-labeller that cannot resolve arXiv DataCite DOIs (`10.48550/arXiv.*`)**, whose blind spot propagated into the labels → into every metric → into the headline integrity claim.

1. **The ground truth is still contaminated, and the fix was opportunistic, not systematic.** The LLM "fabrication" generator hard-codes `label=HALLUCINATED` (`generate_llm_hallucinations.py:547`); when it accidentally regurgitated a real paper, the CrossRef-only validator declared "DOI does not resolve" and kept the HALLUCINATED label. A 32-key hand allow-list (`patch_mislabels.py`, four batches) flipped the famous ones that were *noticed*; 56 same-signature suspects remain. **So what:** the benchmark's core value proposition — trustworthy labels — rests on labels that are demonstrably ~1.5–2% wrong and were never swept end-to-end.

2. **Every headline table was computed on the buggy pre-relabel labels and never regenerated.** Stored aggregate JSONs predate all five relabel commits; `tab:stats`/datasheet/`metadata.json` print stale counts. **So what:** as written, no reported number is reproducible from the released artifacts — a reject trigger for any reviewer who re-scores. **Mitigant (verified):** re-scoring shifts cells ≤1.7pp FPR / ≤4.5pp MCC and *reorders nothing*, so this is a must-fix-but-survivable correctness defect, not a conclusion-killer.

3. **The temporal table is unreproducible from the released 858-entry supplement, and Sonnet's headline FPR is unstable.** Every printed temporal FPR (Sonnet 0.12, Opus 0.073, GPT-5.4 0.41) is from an n=448 subset; the released file is 858; the one model re-run on 858 (GPT-5.1) jumped +16pp; Sonnet's FPR ranges 0.12 (paper) → 0.24 (E1) → 0.287 (E2). **So what:** the abstract's four temporal FPRs cannot be reproduced from any released artifact.

4. **The "uniform pre-screening" control is internally contradicted by its own numbers.** The author-name check `check_capitalized_unknown_authors` flags ~21% of *real* middle-initial papers; yet released bibtex-updater FPR = exactly 87/486 = 0.179, arithmetically impossible if that check had been applied (floor would be ≥105/486 = 0.216). **So what:** the layer was silently not applied (or the FPRs are wrong) — discoverable with a calculator.

5. **"Contamination-resistant" is overclaimed and the canary mechanism is vapor.** 43% of the held-out hidden split's hallucinations share a title with a public VALID entry (recoverable because keys are content hashes); the advertised canary detector does not exist in code, and canaries are absent from the two splits carrying every headline number.

**Throughline:** issue 1 causes 2, 3, and most of the construct-validity/circularity findings. Fix the labels systematically and re-score, and the bulk of this brief collapses to a v1.0→v1.1 erratum plus honest reframing. **The central conclusion — conservative low-FPR verifiers beat recall-aggressive ones; FPR is the deployment-decisive lever — is robust to the relabel (ranking unchanged) and to the temporal data-version change.**

---

## 2. Lessons learned

- **A resolution-based auto-labeller inherits its resolver's blind spots as ground-truth errors.** CrossRef does not index arXiv DataCite DOIs, so a CrossRef-only labeller emits "does not resolve → HALLUCINATED" on exactly the real papers most likely to be memorized. *Encode it:* a GT verifier must be multi-source (CrossRef + OpenAlex + DBLP + arXiv DataCite + S2) before you trust a single negative.
- **Opportunistic fixes masquerade as audits.** The commit log says it plainly ("surfaced by post-fix re-run", "leak-followup"): each flip happened because the co-designed tool *happened* to flip an entry. *Encode it:* an allow-list of hand-confirmed keys is not an audit. If you can't produce the script that re-resolved the entire cohort and its output log, you cannot report an agreement rate.
- **Data-version drift between artifact and paper is silent and lethal.** The supplement grew 448→858; the printed table is still the 448 subset; only GPT-5.1 was re-run (abandoned in `regen858/`). Two `% TODO: regenerate before camera-ready` comments shipped anyway. *Encode it:* every printed number must be regenerated from the same released file in one pass; a TODO is a debt that compounds into "nothing is reproducible."
- **Co-design contaminates labels, not just scores — and only the score risk was disclosed.** The 203 `llm_generated` entries' *types/sub-tests* were assigned by the same CrossRef/title-similarity logic bibtex-updater runs at test time. The redeeming twist: bibtex-updater's arXiv-DataCite handling made it *diverge* from the buggy GT on 12/17 flips — that divergence is a stronger validity argument than any disclosure paragraph. *Encode it:* audit which *labels* a co-designed tool's logic touched, and run a divergence test on known-bad labels.
- **Single-sample-per-entry + one-model variance study cannot underwrite "rankings are stable."** E3 measured only GPT-5.1, n=150, and run2≡run3 (std driven by one flipped entry). The saving grace surfaced only on inspection: all non-GPT models run at temp=0 with a fixed seed, so they're near-deterministic — the unmeasured models aren't the high-variance ones. *Encode it:* name which model was measured under which decoding config; "stable to sampling noise" ≠ "bootstrap CIs over entries."
- **Don't claim reproducibility you didn't ship.** Every `_ci` field in the released artifacts is null; the one printed bootstrap interval is stale (its F1 CI doesn't contain the current point estimate). The CI-computing code exists but was never invoked. *Encode it:* gate a build step that fails when intervals are absent.

---

## 3. Mitigation strategy (improve / reframe / abandon)

`Blocks` = must land before submission.

| # | Weak spot | Decision + tradeoff | Effort | Blocks |
|---|-----------|--------------------|--------|--------|
| 1 | Residual mislabels in released splits | **IMPROVE.** Bounded (~1.5–2%) and the cohort is genuinely *mixed* (real chimeras coexist with mislabeled real papers), so abandoning the benchmark is unwarranted — but the 56 suspects make any agreement claim indefensible until swept. Replace the allow-list with a deterministic **multi-source** labeller (CrossRef+OpenAlex+DBLP+arXiv DataCite+S2) over **every** HALL entry incl. no-DOI title-search; flip-to-VALID on full title+author+year+venue match, re-type to `wrong_venue`/`future_date` on metadata-only defects, keep `plausible_fabrication` only when no real paper matches; commit the script + full output log. | L | **Yes** |
| 2 | All headline tables stale | **IMPROVE → robustness claim.** Re-score is deterministic for tools with stored per-entry preds, shifts ≤4.5pp with zero ranking change — so the fix *strengthens* the paper ("correcting 32 labels leaves the ranking unchanged"). Regenerate Table 2, temporal, cascade, co-design tables, `tab:stats` (incl. tier cells + Total), datasheet, `metadata.json`; add a CI check that fails if any aggregate JSON predates the jsonl it scores. | M | **Yes** |
| 3 | Bolded winners (Sonnet/Opus) lack dev per-entry preds | **IMPROVE.** Gemini-Pro is re-scorable; only Sonnet+Opus aren't, and they're bolded. Re-run them on the 32 flipped entries (a minutes-long delta-eval) for exact corrected cells; persist all per-entry preds so future re-scores are offline. | S | **Yes** |
| 4 | Temporal table unreproducible from 858 | **IMPROVE (regenerate) or REFRAME (downsize to n=448).** Qualitative conclusion survives (Sonnet 0.287 ≪ GPT-5.1 0.926 on 858), but no printed number reproduces. Finish `regen858/` for all 12 models, OR formally ship the 448 prediction files as canonical and relabel the artifact n=448. Resolve the `% TODO` at `analysis.tex:45`. | M–L | **Yes** |
| 5 | Pre-screening contradiction | **IMPROVE + REFRAME.** Fix `check_capitalized_unknown_authors` (require the whole author field to be a lone initial); re-run DB baselines with the layer actually applied; report `compute_prescreening_breakdown` per tool; reconcile `experiments.tex:11` / `limitations.tex:26` wording to what was run. | M | **Yes** |
| 6 | "Contamination-resistant" hidden split leaks 43% | **REFRAME (cheap) or IMPROVE (regen).** Headline tables don't use `test_hidden`, so this falsifies an *advertised property*, not a number. For submission: rename to "label-withheld held-out split" and report the 109/253 overlap; schedule regenerate-from-disjoint-pool for v1.1; enforce cross-split title/DOI disjointness as a CI invariant. | S (reframe) | **Yes (the word)** |
| 7 | "Resistance spans providers" N=1 halted control | **REFRAME → near-ABANDON.** One comparable-cutoff model, 22% parse-failure abstention, imputed full-coverage FPR ~0.34–0.36, unverified cutoff, run halted at 103/300. Drop "spans providers"; downgrade to "one comparable-cutoff non-Anthropic model showed a low partial-run FPR (0.27 on 102/150 valids; 22% abstention), consistent with but not establishing a cross-provider effect." | S | **Yes** |
| 8 | Sonnet FPR reported as 0.12 point estimate | **REFRAME.** 0.12 is the most-favorable subset+run+denominator; report a range (~0.12–0.31) with a fixed UNCERTAIN convention. Conclusion (≪ collapse cluster) is invariant. | S | Strongly advised |
| 9 | Bootstrap CIs / p-values absent + stale | **IMPROVE.** Code exists (`paired_bootstrap_test`, `compute_pairwise_significance`); never invoked/shipped. Compute + ship populated CIs and paired p-values for all 12 tools on fixed labels; report the actual p-value for each "not significant" pair; verify the printed GPT-5.1 interval against the current run. | M | **Yes** |
| 10 | VALID entries retain `hallucination_type`/tier + stale sub-tests | **IMPROVE (data hygiene).** On relabel-to-VALID: null `hallucination_type`+tier, recompute the 6 sub-tests from the confirmed real paper, preserve old values in a provenance field; **extend `verify_subtests.py`** to check `title_exists`/`authors_match`/`cross_db_agreement` (it currently checks only 3) and gate in CI. | M | No |
| 11 | Co-design label provenance under-disclosed | **REFRAME (liability → strength).** In `app:codesign`: state the 203 `llm_generated` entries' types/sub-tests were CrossRef-assigned; report the 12/17 divergence + DR +1.96pp post-fix as direct non-circularity evidence; acknowledge the new btu FPs on real papers (not total exoneration). | S | No |
| 12 | Taxonomy validation: 53% clean map, no IAA | **REFRAME.** State "57/108 (53%) map cleanly; 33% unmapped; 14% compound without IAA"; reconcile the 55%-vs-70% `plausible_fabrication` discrepancy; add a second annotator + Cohen's κ; frame taxonomy as a working hypothesis. | S–M | No |
| 13 | Canary detector doesn't exist | **REFRAME or IMPROVE.** Soften `benchmark.tex:86` to a forward-looking watermark claim, OR implement the ~20-line prediction-JSONL GUID scanner + seed one canary per split. | S | No |
| 14 | Printed prompt ≠ code prompt | **IMPROVE.** Replace `app:prompt-template` with the verbatim `VERIFICATION_PROMPT` (it omits UNCERTAIN + the 14-type instruction), or mark it abridged + cite the source file at a commit. | S | No |
| 15 | Agentic "any-DB-miss flags" strawman | **REFRAME (writing).** The harness doesn't auto-flag — the *LLM* treats DB-absence as fabrication. The misdescription is in the abstract + conclusion; correct the mechanism wording. The "agentic doesn't lower FPR" conclusion stands. | S | Strongly advised (it's in the abstract) |
| 16 | Decoding heterogeneity confound | **IMPROVE.** Run one Anthropic model at temp=1.0 / native API to bound the contribution; same-temp models already span the full FPR range, so decoding isn't the dominant driver. | M | No |

---

## 4. What to abandon (or hard-downgrade)

- **The "resistance spans providers" cross-provider causal claim** (`limitations.tex:18`). Saving it needs a re-run to full n=300 with a non-failing parser *plus* a verified-cutoff later-cutoff model; the imputed FPR (~0.34–0.36) already undercuts "out-resists like Anthropic." Keep only "a single comparable-cutoff non-Anthropic model showed a low partial-run FPR," explicitly N=1 and partial. *Clearest negative-EV claim in the paper.*
- **The unqualified ">98% agreement with ground truth"** (`benchmark.tex:79`). Until a systematic re-resolution audit (action #1) *measures* an agreement rate with a CI, the number is an opportunistic lower bound against a non-independent gold standard. Replace with an empirically-measured rate after #1 and disclose the arXiv-DataCite mechanism as a v1.0→v1.1 erratum. Do not ship the old phrasing.
- **The "controlled diagnostic" (unconditional) framing of perturbations** (`benchmark.tex:77`). The generator emitted real, still-valid citations via no-op perturbations ("3 authors reduced to 3"); 2 survive unscrubbed in `stress_test`. Downgrade to "controlled in intent, verified in outcome"; fix the N→N no-op author-injection generator.
- **The "uniform pre-screening applied to every tool" wording** (`prescreening-baselines-1`). The released numbers prove it wasn't applied. Either fix-and-re-run (#5) or delete the claim; it cannot stand as written.
- **The PPV "prevalence-invariant ranking" as a *finding*.** It's an algebraic identity (π cancels), not an empirical result. Downgrade from "the sweep shows" to "a property of the PPV identity." Keep the absolute-PPV sweep, which has content.
- **The advertised canary contamination *detector*.** It does not exist. Either implement (~20 lines) or call canaries a forward-looking watermark.

**Net:** nothing requires abandoning the benchmark or the central conclusion. Five overclaimed framings must be cut/softened; seven blocking actions (#1–7, #9) must land before submission; everything else is data-card hygiene that strengthens the paper without re-running experiments.

---

## Key files for the blocking work

- Relabeller target + current allow-list: `data/v1.0/dev_public.jsonl`, `data/v1.0/test_public.jsonl`, `scripts/patch_mislabels.py`, read-only suspect-lister `scripts/check_mislabeled_entries.py`
- Generator bug (hard-coded HALL label): `scripts/generate_llm_hallucinations.py:547`
- Pre-screening bug: `hallmark/baselines/prescreening.py` (~316–323, 593)
- Temporal regen: `results/temporal_supplement/temporal_supplement_2024_2025.jsonl` (858), `.../regen858/`
- Stale counts: `data/v1.0/metadata.json`; paper `sections/benchmark.tex` (`tab:stats`), `sections/experiments.tex` (`tab:results`, prescreening L11), `sections/abstract.tex` (L9 temporal FPRs), `appendix.tex` (datasheet, `app:bootstrap`, `app:codesign`), `sections/limitations.tex` (L18 spans-providers, L26 uniform pre-screening), `sections/analysis.tex:45` (the `% TODO`)
- Sub-test QA to extend + CI-gate: `scripts/verify_subtests.py`; canary code: `hallmark/dataset/schema.py:626–653`
