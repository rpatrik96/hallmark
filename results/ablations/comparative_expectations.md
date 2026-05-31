# Comparative + Historical Grounding for the Small-Scale Ablation Plan

**Purpose.** Merge R1 (`prior_chats.md`, our own ablation/rebuttal history) with R2
(`peer_norms.md`, what the cited peer benchmarks actually report) into one decision-ready
view. Answers three questions for the ablation go/no-go:

1. **What will reviewers expect that is MISSING** from both the paper and the current pilots
   (ranked, with the peer paper that sets the precedent).
2. **What is already SETTLED** by prior work / rebuttal — do NOT re-run (with the chat id to cite).
3. **Where prior reviewer asks (W1/W6/W8) map** onto the candidate ablations.

**Sources merged:** `results/ablations/prior_chats.md` (R1), `results/ablations/peer_norms.md` (R2).
**Pilot state verified on disk** (2026-05-30); no `ABLATION_PLAN.md` exists yet, so the four
pilot dimensions are referenced by name: **(P1) prompt-structure**, **(P2) decoding/self-consistency**,
**(P3) threshold/aggregation**, **(P4) input-format/field**.

---

## 0. Current pilot state (verified on disk — read this before deciding)

| Pilot | Dir | State | Usable signal? |
|-------|-----|-------|----------------|
| **P1 prompt-structure** | `e_prompt_pilot/` | 4 variants (default/notaxo/uncertain/terse), n=60 on deepseek-v3.2: **60/60 UNCERTAIN, `num_error_fallback: 60`** | **NO** — parse/API fallback bug, carries no signal |
| **P2 decoding** | `e_decoding/` | t0 vs t1 on deepseek-v3.2, n=60: **0 flips, 60/60 UNCERTAIN** | **NO** — same fallback bug |
| **P3 threshold/aggregation** | `a3_threshold_aggregation_result.json` | offline re-score on v1.1.1 labels, n=60, Claude Opus 4.7 **AUROC 0.9225**; curve **flat** thr 0.1–0.7 (DR 0.9412 / FPR 0.2308) | **YES** — only fully-usable pilot; confidences are quantized so threshold tuning buys little |
| **P4 input-format/field** | `ablation4_input_format/` | full vs structured + **leave-one-field-out** (title/author/venue/year/doi), deepseek-v3.2, **24/60 each (partial), real verdicts** | **PARTIAL but REAL** — produces genuine HALLUCINATED/VALID labels, just incomplete; LOO design is sound |

**Decision-relevant fact:** two of the four pilots (P1, P2) are *degenerate* (error-fallback to
UNCERTAIN on deepseek-v3.2), not informative. P3 produced a flat curve (quantized confidences).
P4 is the only run producing real differential signal and is the most novel axis. **Any go/no-go
must first decide whether to fix the fallback bug + re-run on a model that actually answers**,
because three of the four pilots cannot be cited as-is.

---

## 1. MISSING ablations reviewers will expect (ranked) — gaps in BOTH paper and pilots

Ranked by reviewer salience × novelty × cost-to-close. "Precedent" = the peer paper that makes
the omission conspicuous.

### Rank 1 — Inter-annotator agreement (κ) on a re-annotated real-world subset  **[NOT in any pilot]**
- **Precedent:** CiteAudit (Shi et al., arXiv:2602.23452) uses ≥2 reviewers + consensus but
  reports **NO κ**; *no surveyed peer reports κ at all* (peer_norms #12). GhostCite/HalluCitation/
  Mysterious are prevalence studies without IAA.
- **Why it's the top miss:** it is the single dimension where HALLMARK could move *ahead of the
  entire field* rather than reach parity. The real-world set (108 entries) currently has **zero
  reported IAA**, and the relabel audit on this very branch (`fix/dev-public-mislabel-audit`,
  52 recovered real papers wrongly flagged HALLUCINATED) is the natural re-annotation substrate.
- **Cost:** small — re-annotate a real-world subset with 2–3 raters, report Cohen's/Fleiss' κ.
- **Caveat (R1):** the recent mislabel-fix commits (c9915d2, 84f9fe4, 6eced89, …) prove our own
  *single-pass* labels were noisy; an honest κ is therefore both expected AND defensible, and it
  retroactively justifies the audit. Not in P1–P4. **Highest-leverage missing analysis.**

### Rank 2 — Working prompt-sensitivity sweep on a model that actually answers  **[P1 exists but degenerate]**
- **Precedent:** *No peer reports a prompt sweep* (peer_norms #10); this is an open differentiator,
  and our own design uses a single prompt (`sec:baselines`), which a reviewer will probe.
- **Why expected:** R2 ranks prompt-sensitivity the *highest-salience open dimension* precisely
  because a clean "rankings are robust to phrasing / taxonomy-in-prompt / abstention wording"
  result defends the single-prompt design. agrawal2024 (direct/indirect query consistency) is the
  self-consistency-under-reformulation anchor.
- **Status:** P1 is scaffolded but **100% error-fallback** — it carries no signal yet. This is a
  gap *in practice* even though the pilot nominally targets it.
- **Action:** fix the parse/fallback bug, re-run n≥150 on ≥1 model that answers (GPT-5.1 / Claude
  Sonnet 4.6 / Gemini Flash), report verdict-flip rate and ranking stability across prompt variants.

### Rank 3 — Confidence-threshold / DR–FPR operating curve for ALL zero-shot tools  **[P3 partial: one model, flat]**
- **Precedent:** CiteAudit hard-codes a 0.92 cosine match threshold **without justification or a
  sweep** (peer_norms §1, #8); Hardt Ch.12 motivates operating-point sensitivity.
- **Why expected:** peer benchmarks report DR–FPR / ROC curves; the paper currently reports fixed
  operating points (@0.5 / cascade @0.55). The offline P3 pilot shows a per-tool curve is *cheap*
  (no new API calls — re-score stored confidences).
- **Status:** P3 covers only Claude Opus 4.7 and the curve is **flat** because confidences are
  quantized. **Extend to all zero-shot tools**; if curves stay flat, report that as the finding
  (quantized confidences → threshold tuning buys little — itself a clean, honest result).

### Rank 4 — Input-format / field-ablation completed (full vs structured + leave-one-field-out)  **[P4 partial but real]**
- **Precedent:** CiteAudit *motivates* the benchmark with "citation-format variability" but
  **never ablates format** (peer_norms #11); rao2026bibtex ablates *architecture* (1-stage vs
  2-stage) not input format. HALLMARK could be the **first** to ablate format/field directly.
- **Why expected:** "which field drives detection?" is a natural mechanistic question; the LOO
  design (drop title/author/venue/year/doi) answers it and complements the format-tell audit (W6).
- **Status:** P4 is the only degenerate-looking pilot that actually returns real verdicts — just
  24/60 complete on deepseek-v3.2. **Finish it** (n=60 full, ≥1 answering model). Low extra cost,
  genuinely novel.

### Rank 5 — Cross-model decoding/temperature sweep  **[P2 exists but degenerate; partially settled]**
- **Precedent:** HumanEval (pass@k via repeated sampling) establishes multi-sample decoding as a
  recognized robustness device; CiteAudit fixes T=0 (peer_norms #9).
- **Why only Rank 5:** R1 shows this is *largely settled* for our headline model — GPT-5.1's
  temperature is **API-forced to 1.0** (E3 variance: DR identical 0.9647, FPR 0.538/0.523/0.523
  across 3 runs), so a "temperature sweep" is near-vacuous for it. The genuine gap is a model with
  a *real* temperature knob (DeepSeek / open models). P2 targets exactly this but is degenerate.
- **Action (optional):** only worth it if the fallback bug is fixed anyway for P1; then add a
  cross-model temp sweep cheaply. Low marginal salience — note the GPT-5.1 single-draw caveat and
  cite E3 rather than over-investing.

### Rank 6 (consider, lower priority) — Explicit ensemble voting-rule ablation
- **Precedent:** Hardt Ch.12 (aggregation = voting-system analogy). No peer reports this.
- **Gap (R1 §4):** P3 touches thresholds; an explicit ensemble rule sweep (majority vs
  confidence-weighted vs union) is not yet a standalone ablation. Cheap if we already have
  per-tool predictions. Lower reviewer-salience than Ranks 1–4.

---

## 2. SETTLED by prior work / rebuttal — do NOT re-run (cite the chat id)

These are real, executed (or fully scaffolded) experiments. Re-proposing them as "new" wastes
budget and invites a reviewer to ask why we duplicated our own appendix.

| Axis | Settled by | Outcome / artifact | Cite |
|------|-----------|--------------------|------|
| **Cutoff-aware prompting** (abstention-wording variant) | H2 ablation | "Partial metacognition": GPT-5.1 FPR −72.6pp but 52.7% pre-cutoff UNCERTAIN; Gemini Flash most discriminative. **Pre-registered** interpretation (commit 47bd7da). Registry `llm_*_cutoff_aware`. | **#16436, #16441, #16457, #S3444–#S3467** |
| **Newer-cutoff / late-cutoff control** | H1 GPT-5.4 probe + **E2** | GPT-5.4 2024-stratum FPR 28.0% vs GPT-5.1 53.1%; cutoff alone does NOT explain full FPR (residual 28%). E2 300-entry redo: GPT-5.1 FPR 0.926, Sonnet-4.6 0.287. | **#16481** + `results/reviewer_experiments/e2_latecutoff_control/` |
| **Model-drift / temporal stability** (fixed model over time) | GPT-5.1 Feb-vs-Mar | DR +2.1pp but **FPR +21.9pp** (0.171→0.390) in one month. Motivates versioned predictions. | **#17002, #17100** |
| **Thinking-budget / reasoning-effort** (decoding-budget axis) | App. G smoke test, 3×3 | Three archetypes (clean ceiling / non-converging tail / clean recovery). $10.77, 303 min. | **#17281, #17663, #17327 (W1)** |
| **Format-tell / shortcut exploitability** | Static audit | ≤17.4pp DR upper bound; 4.4% (28/633) hallucinated entries exploitable. | **#17356, #17327 (W6)** |
| **Co-design / construct-overfitting bound** | Stress vs design-aligned | Stress DR 0.956 ≥ aligned 0.945; 95% CI [−3.6,+5.8] includes zero. | **#17356, #17327 (W3)** |
| **Cost / latency** | 18-baseline audit | $0–$0.0062/entry; 0.3–25s/entry; agentic 2–4× cost. | **#17276, #17327 (W8)** |
| **Pre-screening lift** | with/without ablation | ~5pp Tier-1 lift; `*_no_prescreening` variants registered. | **#7394, #17781, #4102** |
| **Parametric-recall mechanism** (recall vs verify join) | **E1** recall probe | n=150, 3 models; `per_entry_combined.json` joins `model_says_known`/`author_jaccard` to the verdict — tests whether FPR tracks recall. **Canonical recall experiment; do not duplicate.** | `results/reviewer_experiments/e1_recall_probe/` |
| **Seed/sampling variance** (GPT-5.1) | **E3** | 3 runs, n=150: DR identical 0.9647; FPR 0.538/0.523/0.523. **Temp API-forced to 1.0** → sampling-only variance, tiny. Redundant unless extended to a real-temp model (→ Rank 5). | `results/reviewer_experiments/e3_variance/` |
| **Strategic gaming / tier concentration** | Water-filling module | Gini / T1:T3 ratio / Shannon entropy; 30–50× T1/T3 ratios observed. | **#13014** (`hallmark/evaluation/water_filling.py`) |
| **Canary contamination filtering** | `evaluate()` auto-filter | Canary entries excluded from metrics. | **#7503** |
| **Synthetic-vs-real fidelity** | KS test | 3 surface features, 6/9 types n.s.; flags underpower + TOST need. (Peer parity: CiteAudit χ², p=0.994.) | `sec:synth_vs_real` |
| **Statistical significance / CIs** | Stratified bootstrap | 10k resamples, paired bootstrap p-values, per-type power/MDE. **Ahead of cohort.** | `app:bootstrap`, `app:statistics` (#5025) |
| **Calibration (ECE)** | `evaluate()` | ECE in `tab:results` + calibration paragraph. **Differentiator** (no peer surfaces ECE). | `sec:calibration` |
| **Pipeline-stage / architecture** | Cascade + agentic + prescreening | Conservative-vs-aggressive @0.55; agentic vs zero-shot. Matches/exceeds rao2026bibtex 1-vs-2-stage. | `sec:cascade` |
| **Cross-split generalization** | ΔFPR column | dev→test FPR shift. **Differentiator.** | `sec:crosssplit_robustness` (#17593, #S3761/#S3774/#S3780) |
| **Holdout / adaptive-overfitting discipline** | hidden split + caveat | dwork2015 / Hardt Ch.5 anchors; "we don't tune on test." | — |

**Bottom line from R2:** HALLMARK already over-delivers on the analyses reviewers most expect for
this class (model breadth, temporal/contamination control, recall disentanglement, significance,
calibration, per-tier, cross-split). On every dimension where CiteAudit reports something, HALLMARK
reports an equal-or-stronger version; on recall (#3), significance (#5), ECE (#6), cross-split (#15)
it leads the whole surveyed cohort. The settled list is long *because the paper is already strong* —
the ablation budget should go to the §1 gaps, not to re-running §2.

---

## 3. Prior reviewer asks (W1/W6/W8) mapped onto candidate ablations

The NeurIPS D&B weakness labels W1/W3/W6/W8 recur across the May-2026 reviewer-defense sprint
(#S3737, #S3770, #S3789, #S3798, #S3828). How each maps onto the §1 candidates:

| Weakness | Reviewer push | Already answered by | New candidate it touches |
|----------|---------------|---------------------|--------------------------|
| **W1 — model coverage / selection bias** | Why exclude thinking-tier models (GPT-5.5, Gemini 3.1 Pro, DeepSeek-V4-Pro)? Cherry-picked set? | Thinking-budget smoke test (App. G) reframes exclusion as a shared-budget protocol boundary; cross-split test_public runs. **#17281/#17327** | **Rank 2 (prompt sweep)** and **Rank 5 (cross-model decoding)** reinforce W1: a *model-invariant* prompt/decoding result shows the finding isn't a single-model artifact. **Rank 3 (per-tool DR–FPR curves)** also broadens the model/tool view. Frame these as "the precision-ceiling is model- and prompt-invariant," matching the title decision **#16973**. |
| **W3 — co-design / construct overfitting** | bibtex-updater co-developed with benchmark → overfits design-aligned types? | Co-design bound, CI includes zero. **#17356/#17327** | **Rank 4 (input-format/field LOO)** complements W3: showing detection isn't driven by a single co-designed field (or by format) is further construct-validity evidence. |
| **W6 — synthetic-data quality / format tells** | Are perturbations detectable via surface artifacts, not reasoning? | Static format-tell audit, ≤17.4pp bound. **#17356/#17327** | **Rank 4 (input-format/field)** is the *dynamic* counterpart to the static W6 audit — LOO/format ablation empirically tests whether the model leans on a surface field. Strongest W6 reinforcement. **Rank 1 (κ)** indirectly supports W6: IAA on real-world entries shows the labels themselves are reliable, not artifact-driven. |
| **W8 — cost / latency / reproducibility** | D&B reviewers want resource disclosure. | Cost-latency table + inline paragraph. **#17276/#17327** | No new ablation needed; but **keep new pilots cheap and report their $/runtime** (P3 is offline = $0; P4/P1 should log cost) so they extend the W8 story rather than undercut it. |

*Note:* W3 was a reviewer ask but is not in the W1/W6/W8 set the task named; included for completeness
since Rank 4 touches it. No prior reviewer explicitly asked for κ (Rank 1), prompt-sweep (Rank 2), or
operating-curve (Rank 3) — these are *peer-norm-driven* expectations, not yet-raised asks, which is why
they are the cleanest pre-emptive additions.

---

## 4. Decision-ready summary (fold into the go/no-go)

**GO — add (ranked):**
1. **κ / IAA on a re-annotated real-world subset** — not in any pilot; moves HALLMARK ahead of the
   field; substrate already exists from the relabel audit. *No precedent to match — sets one.*
2. **Working prompt-sensitivity sweep** — fix the P1 fallback bug, re-run n≥150 on an answering model;
   no peer reports this; defends the single-prompt design.
3. **Per-tool DR–FPR operating curve** (extend P3 from one model to all zero-shot tools) — cheap
   (offline re-score); differentiates from CiteAudit's unjustified fixed 0.92 threshold.
4. **Finish input-format/field LOO** (extend P4 to full n on an answering model) — first in the field;
   reinforces W6/W3.

**NO-GO / DOWNWEIGHT:**
- **Temperature/decoding sweep as a standalone headline** — largely settled (E3; GPT-5.1 temp forced
  to 1.0). Only add a cross-model temp sweep *opportunistically* once the fallback bug is fixed for P1.
- **Anything in §2** — re-running cutoff-aware, late-cutoff, thinking-budget, format-tell, co-design,
  cost, recall (E1), variance (E3), water-filling, KS, bootstrap, ECE, cross-split, cascade duplicates
  committed work / appendix; cite the chat id instead.

**Blocking prerequisite:** P1 and P2 are 100% UNCERTAIN error-fallback and P4 is only 24/60 — **fix the
parse/API fallback before citing any of these three.** P3 (offline) is the only directly-usable pilot.

**Framing (all new ablations):** position as *robustness evidence for the precision-ceiling finding*
(model-, seed-, prompt-, format-invariance), per title decision **#16973** and the W1/W3/W6/W8
defense logic — not as tuning for best numbers.
