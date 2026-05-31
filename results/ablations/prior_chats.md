# Prior Chats — Ablations & Reviewer-Defense History (R1)

Mined from claude-mem (projects `hallmark` + `bibtexupdater`) on 2026-05-30. Goal:
catalog every ablation / robustness experiment we have **already discussed, planned,
or run**, plus the **reviewer concerns** we have responded to, so the new
small-scale-ablation plan does not re-propose work that already exists, and so it
matches what NeurIPS D&B reviewers have actually pushed on.

IDs are claude-mem observation IDs (`#N`) or session goals (`#SN`). "On disk" notes
where artifacts already live in the repo.

---

## 1. Ablations / robustness experiments ALREADY done or planned

These are real, executed (or fully scaffolded) experiments. **Do not re-propose them as
"new" small-scale ablations** — the new plan should either (a) extend them on a larger
sample, or (b) fill a genuinely uncovered axis (see §4).

| # / id | Experiment | What it tested | Status / artifact |
|--------|-----------|----------------|-------------------|
| #16436, #16441, #16457, #S3444–#S3467 | **Cutoff-aware prompt ablation (H2)** | Does appending a "you have a training cutoff, use UNCERTAIN for post-cutoff papers" addendum reduce FPR? | DONE, integrated into appendix. Outcome = **"partial metacognition"**: GPT-5.1 FPR −72.6pp but pre-cutoff UNCERTAIN inflated to 52.7%; Qwen3-235B −78.5pp but 95.3% pre-cutoff UNCERTAIN; Gemini 2.5 Flash most discriminative (−10.5pp, 0.7% pre-cutoff UNCERTAIN). Pre-registered interpretation table (commit 47bd7da) **before** results. Variants live in registry as `llm_*_cutoff_aware`. |
| #16481 | **Post-freeze model probe (H1)** — GPT-5.4 | Does a *newer training cutoff* reduce the temporal FPR? | DONE, appendix subsection. GPT-5.4 (Aug-2025 cutoff): 2024 stratum FPR 28.0% vs GPT-5.1 53.1% (−25.2pp); 2025 stratum 56.8% vs 99.3% (−42.4pp). Key claim: cutoff alone does **not** explain full FPR; residual 28% on pre-cutoff data → mitigation needs retrieval, not newer models / prompt tricks. UNCERTAIN drops to 0% (genuine familiarity, not prompted caution). |
| #17002, #17100 | **Model-drift re-evaluation** (GPT-5.1 Feb vs Mar) | Stability of a fixed model over one month | DONE, appendix. DR +2.1pp but **FPR +21.9pp** (0.171→0.390) over one month; test_public FPR 0.455. Motivates versioned predictions + model-version IDs. This is a *temporal stability* ablation already in the paper. |
| #17281, #17663, #17327 (W1) | **Thinking-budget / reasoning-effort smoke test** (App. G) | 3 models × 3 budget regimes (A/B/C), n=100 stratified | DONE, App. G + `tables/smoke_test_thinking_budget.{csv,tex}`. Three archetypes: GPT-5.5 clean ceiling (PF 66%→12%→1%), Gemini 3.1 Pro non-converging tail (saturation never <0.96 up to 16k), DeepSeek-V4-Pro clean recovery (PF 6%→0% at 8k). Cost $10.77, 303 min. This is the **decoding/budget** axis already covered. |
| #17356, #17327 (W6) | **Format-tell / shortcut audit** | Can a model exploit surface cues in synthetic perturbations? | DONE, `scripts/audit_format_tells.py` + `tables/format_tells_audit.csv`. Static audit: 33 potential tells affecting 3/14 types; only 4.4% (28/633) hallucinated entries exploitable; **conservative ≤17.4pp DR upper bound**. |
| #17356, #17327 (W3) | **Co-design / construct-overfitting bound** | Does bibtex-updater overfit to design-aligned types vs design-orthogonal stress types? | DONE, `scripts/compute_codesign_bound.py` + `tables/codesign_bound.csv`. Stress-test DR=0.956 vs design-aligned 0.945; gap **+1.1pp favoring stress types**, bootstrap 95% CI [−3.6, +5.8]pp includes zero → no overfitting at n=90. |
| #17276, #17327 (W8) | **Cost / latency audit** | Per-entry $ and seconds for all 18 baselines | DONE, `scripts/compute_baseline_costs.py` + `tables/baseline_cost_latency.csv`. $/entry $0–$0.0062; s/entry 0.3–25s; DeepSeek-R1 ~25s/entry CoT; agentic 2–4× cost multiplier. |
| #7394, #17781, #4102 | **Pre-screening lift ablation** | Incremental detection from local DOI/year/author checks before tools | DONE, reported with/without pre-screening; **~5pp lift on Tier 1**. `no_prescreening` variants registered (`doi_only_no_prescreening`, `bibtexupdater_no_prescreening`, `harc_no_prescreening`). #17781 added capitalized-token fake-author check (from CheckIfExist). |
| #13014 | **Water-filling / strategic-gaming diagnostic** | Do tools concentrate on easy Tier-1 types and neglect Tier-3? | DONE, `hallmark/evaluation/water_filling.py` (Gini, T1/T3 ratio, Shannon entropy; flag when Gini>0.3 or ratio>3.0). Observed 30–50× T1/T3 ratios for doi_only/harc/bibtexupdater. |
| #7503 | **Canary contamination filtering** | Exclude canary entries so contamination probes don't skew metrics | DONE, `evaluate()` auto-filters canaries. (Contamination-detection axis already handled.) |

### Reviewer-experiments harness (on disk, newer than most chat history)
These dirs are **uncommitted** (git status `??`) under `results/reviewer_experiments/` and
`results/temporal_supplement/regen858*` — they are a fresh, larger re-run of the above and
must be reconciled with whatever the new plan proposes:

- **E1 `e1_recall_probe/`** — *parametric-recall vs verify-label* probe, n=150 sample,
  3 models (gpt-5.1, claude-sonnet-4.6, claude-opus-4.7). Two passes per model:
  `recall_*.jsonl` (asks the model whether it *knows* the paper — `model_says_known`,
  `author_jaccard`, `venue_match`, `recalled`) and `verify_*.jsonl` (the actual
  HALLUCINATED/VALID verdict). `per_entry_combined.json` joins them. **This directly tests
  whether FPR tracks parametric recall** — i.e. the mechanism behind H1. New ablation plan
  should treat E1 as the canonical recall-mechanism experiment, not duplicate it.
- **E2 `e2_latecutoff_control/`** — *late-cutoff control* on a 300-entry temporal
  2024–2025 subsample (seed 42). Results: GPT-5.1 FPR **0.926** (DR 0.973, MCC 0.108,
  ECE 0.351 — pathological over-flagging); Claude Sonnet 4.6 FPR **0.287** (DR 0.937,
  MCC 0.675); DeepSeek-V4-Pro run **halted partial** (103/300, "too slow via OpenRouter").
  This is a larger redo of the GPT-5.4 / Opus-4.7 post-cutoff probes.
- **E3 `e3_variance/`** — *seed/run variance* of GPT-5.1, n=150 dev_public, 3 runs, seed 42.
  Across runs DR is identical (0.9647) and FPR is near-identical (0.538 / 0.523 / 0.523);
  MCC 0.510–0.523. Note temperature is **"1.0 (forced by API)"** — GPT-5.1 does not expose
  a temperature knob, so cross-run variance is *sampling-only*, very small. **This is the
  variance ablation already run** — the new plan's E3-variance pilot is redundant with this
  unless extended to more models/larger n.

### Parallel pilot (the OTHER workflow, `results/ablations/`)
For dedup awareness — these pilots are in-flight and **mostly degenerate so far**:
- `e_prompt_pilot/summary.json` — 4 prompt variants (default / notaxo / uncertain / terse),
  n=60: **all four collapsed to 100% UNCERTAIN via error-fallback** (`num_error_fallback: 60`).
  Pilot is currently broken (API/parse failure), not a real signal.
- `e_decoding/result_combined.json` — temperature t0 vs t1 on deepseek-v3.2, n=60:
  **0 flips, all UNCERTAIN** — same fallback bug.
- `a3_threshold_aggregation_result.json` — **offline** threshold sweep on v1.1.1 labels
  (no API calls), n=60. Claude Opus 4.7 AUROC 0.9225; verdict is flat across thr 0.1–0.7
  (DR 0.9412 / FPR 0.2308) — the confidence scores are quantized, so threshold tuning buys
  little. This is the one pilot that produced a usable curve.
- `ablation4_input_format/` — input-format ablation, only 15 lines run so far (incomplete).

---

## 2. Reviewer comments we have responded to (NeurIPS D&B)

The weakness labels W1/W3/W6/W8 recur across sessions #S3737, #S3770, #S3789, #S3798,
#S3828 (May 2026 reviewer-defense sprint). Mapping:

| Weakness | What the reviewer pushed on | Our empirical response (committed) |
|----------|----------------------------|-----------------------------------|
| **W1 — model coverage / selection bias** | Why exclude newer thinking-tier models (GPT-5.5, Gemini 3.1 Pro, DeepSeek-V4-Pro) from the main table? Is the model set cherry-picked? | Thinking-budget smoke test (App. G) reframes exclusion as a *shared-budget protocol boundary*, not a coverage gap. Plus **cross-split test_public** runs (#17593, #S3761, #S3774, #S3780) to show ΔFPR generalizes across splits and isn't selection bias. |
| **W3 — co-design / construct overfitting** | bibtex-updater is co-developed with the benchmark → does it overfit to the design-aligned hallucination types? | Co-design bound: stress-test (design-orthogonal) DR ≥ design-aligned DR, 95% CI includes zero. No overfitting at n=90. |
| **W6 — synthetic-data quality / format tells** | Are perturbations detectable via surface artifacts (format shortcuts) rather than genuine reasoning? | Static format-tell audit: ≤17.4pp DR exploitability upper bound; 4.4% of hallucinated entries exploitable. |
| **W8 — cost / latency / reproducibility** | D&B reviewers want resource disclosure ($/entry, latency, throughput) for practicality. | Cost-latency table + inline Experiments paragraph; abstract/caption anchors added (#17356 flagged the missing touchpoints). |

Other reviewer-adjacent threads:
- Multi-persona **audience-accessibility** checks (#4168, #4755, #4952) — readability, not ablations.
- Earlier internal **code-review / devil's-advocate** audits surfaced their own "R-1…R-7"
  items (#4674 ranking/UNCERTAIN bug, #7543 bootstrap bug, #7600 CLI fixes, #4837 PDF-breaking
  errors) — these are code-correctness, not reviewer-requested ablations, but they show the
  evaluation internals have already been hardened.
- #S3599 — applying a NeurIPS D&B reviewer's commented-out "L12" themes to the abstract.

---

## 3. Comparison / positioning vs other benchmarks (already discussed)

Relevant because reviewers expect the ablation set to *match peer-benchmark norms*, and
because novelty is contested:

- **GPTZero NeurIPS-2025 report** (#3160, #3161, #3886): in-house agent vs 220M articles,
  >99% catch rate w/ human review; public Google-Sheet of ~100 real hallucinations across
  51–53 papers. Three categories map onto our 14 types. This is our real-world validation
  anchor and a *performance ceiling* reference. Surfaced "incomplete arXiv ID" pattern not
  originally in our taxonomy.
- **Competing benchmarks/tools** (#16826, #16842): CiteAudit (6-field check, ≤EMNLP-2025),
  a 931-paper / 9-field-taxonomy benchmark, HalluCitation (arXiv:2601.18724, 300 ACL
  hallucinations), CheckIfExist (arXiv:2602.15871), HalluCiteChecker (arXiv:2604.26835),
  Rao & Callison-Burch BibTeX hallucinations (arXiv:2604.03159), Rao-Wong URL verification
  (arXiv:2604.03173). **Flagged as a novelty risk** — "first benchmark" claim must be
  softened; the space is more crowded than the paper acknowledged. CheckIfExist's
  capitalized-token heuristic was *adopted* into prescreening (#17781).
- **Related-work heritage** (#4657): design principles borrowed from HumanEval (sub-tests),
  SWE-bench (temporal segmentation), LiveCodeBench (continuous updates), ONEBench
  (Plackett-Luce ranking for incomplete eval). GhostCite (604/56k papers), Mysterious
  Citations, RefChecker as audits. General hallucination benchmarks (HaluEval, HaloGen)
  explicitly out of scope (factual not bibliographic).
- **Benchmark-science framing** (#12926, Hardt "Emerging Science of Benchmarks"): the
  "iron rule", ranking validity > absolute numbers, annotation-noise tolerance, multi-task
  aggregation impossibility (Arrow). Justifies our Plackett-Luce ranking and motivates
  *ranking-stability* robustness checks. The water-filling module (#13014) operationalizes
  Hardt's Gibbard–Satterthwaite gaming concern.
- **Title/framing decision** (#16973): committed to the *structural precision-ceiling*
  finding (base-rate × FPR) rather than the brittle "rule-based beats LLM" claim, precisely
  because the latter is vulnerable to model drift. **Implication for ablations:** reviewers
  will expect robustness evidence that the precision-ceiling finding is model-/seed-/
  prompt-invariant — which is what E1/E2/E3 + the format-tell / co-design bounds collectively
  argue. The ablation plan should be framed as *defending the durability of the precision
  ceiling*, not as tuning for best numbers.

---

## 4. What this implies for the NEW small-scale ablation plan

**Already covered — do NOT re-propose as novel:**
1. Temperature / decoding variance → E3 (`e3_variance/`, GPT-5.1, 3 runs) + decoding pilot.
   GPT-5.1 temperature is API-forced to 1.0, so "temperature ablation" is near-vacuous for it.
2. Thinking-budget / reasoning-effort → App. G smoke test (3×3 regimes).
3. Cutoff-aware prompting → done, "partial metacognition" outcome, pre-registered.
4. Newer-cutoff / late-cutoff control → GPT-5.4 probe + E2 (`e2_latecutoff_control/`).
5. Pre-screening lift → ~5pp Tier-1, with/without variants registered.
6. Format-tell / shortcut exploitability → ≤17.4pp bound (W6).
7. Co-design / construct-overfitting bound → CI includes zero (W3).
8. Cost / latency → full table (W8).
9. Parametric-recall mechanism → E1 recall probe (recall vs verify join).
10. Strategic gaming / tier concentration → water-filling module.

**Genuinely uncovered axes the new plan COULD add (low redundancy):**
- **Threshold/operating-point sweep reported as a curve for ALL tools** (not just the offline
  Opus pilot) — peer benchmarks report DR–FPR / ROC curves; we currently quantize confidences.
  This is the cleanest "small-scale" win and the offline pilot shows it's cheap.
- **Prompt-robustness done correctly** — the prompt pilot is currently 100% error-fallback;
  a *working* paraphrase/format-variant prompt ablation (n≥150) would be new signal, but only
  after fixing the fallback bug.
- **Seed/sampling variance for MULTIPLE models** — E3 only covers GPT-5.1 (whose temp is
  fixed). Extending to a model with a real temperature knob (DeepSeek, open models) would make
  the variance claim general rather than single-model.
- **Input-format / metadata-field ablation** (`ablation4_input_format/` started but only 15
  lines) — e.g. BibTeX vs JSON vs prose rendering, or ablating individual fields (DOI present
  vs absent) to see which field drives detection.
- **Aggregation/ensemble-rule ablation** — `a3_threshold_aggregation` touches thresholds; an
  explicit ensemble voting-rule sweep (majority vs confidence-weighted vs union) isn't yet a
  standalone ablation.

**Framing note:** every new ablation should be positioned as *robustness evidence for the
precision-ceiling finding* (model-, seed-, prompt-, format-invariance), matching the title
decision in #16973 and the reviewer-defense logic of W1/W3/W6/W8.
