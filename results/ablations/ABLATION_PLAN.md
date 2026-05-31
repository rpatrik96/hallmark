# Ablation Plan — evidence-backed go/no-go (ablation scout)

**Date:** 2026-05-30 · **Branch:** `fix/dev-public-mislabel-audit` (v1.1.1 corrected labels)
**Fixed pilot sample:** `results/ablations/pilot_sample_dev60.jsonl` (n=60 dev_public, 34 HALL / 26 VALID, 13 types).
All four pilots scored against this sample with `hallmark.evaluation` via the repo verify path
(`_verify_with_openai_compatible` / a custom `prompt_fn`), temp+seed fixed (temp=0, seed=42) unless noted.

**Scope of this doc.** Decide, per candidate ablation, whether it is worth a FULL-scale run, backed by the
cheap pilot already on disk. Ranked by **reviewer-value × cheapness × novelty**. The headline-fragility flag
(§0) is the highest-priority signal and drives the shortlist.

---

## 0. HEADLINE FRAGILITY FLAG (read first)

> **The paper's headline FPR / over-flagging numbers are prompt- and format-tunable by 20–35 pp on a single
> open model — the over-flagging magnitude is partly a prompt artifact, not solely a model property.**

The exposed claim (abstract claim 3; `conclusion.tex` (iii); `analysis.tex` L48/L50/L53):
*"on 448 post-cutoff papers most LLMs over-flag sharply (FPR up to ~0.89) … later-cutoff models hold FPR low."*
FPR is the load-bearing metric. Two pilots move it by more than the cross-model gaps the paper reports:

| Pilot | Same sample, same model | What moves FPR | Magnitude |
|---|---|---|---|
| **P1 prompt** | deepseek-v3.2, n=60 | default → "abstention-permitted" wording | **FPR 0.885 → 0.538 (−34.6 pp)**, DR 1.000 → 0.853 |
| **P4 input-format** | deepseek-v3.2, n=24×7 | full BibTeX block → structured field list | **FPR 0.90 → 0.70 (−20 pp)**; dropping authors → FPR 1.00 |

Raw-label check (P1): **11 of 26 VALID entries (42%) flip label across the four prompt variants.** Confirmed
directly from the checkpoint JSONL, not a scoring artifact.

**Why this is the top signal, and how to defuse it (not bury it).** The variant that *lowers* FPR is exactly
the abstention/"route to UNCERTAIN" wording the paper already advocates as the mitigation (`app:cutoff-aware`,
the closing line of `conclusion.tex`). So the fragility is *consistent with* the paper's thesis — but its size
means a reviewer who reruns with a different prompt can legitimately report a very different FPR. The paper
currently leans on a **single prompt** (`sec:baselines`) with no sensitivity reported (peer_norms #10: no peer
reports a prompt sweep either). **A full prompt-sensitivity sweep is therefore both the highest-value pre-empt
and the fix for the fragility:** show that the *ranking* of tools (and the later-cutoff-resistance pattern) is
prompt-invariant even though the *absolute* FPR is not, and state explicitly that absolute FPR is prompt-dependent.

---

## 1. Pilot state on disk (corrects the stale `comparative_expectations.md`)

`comparative_expectations.md` / `peer_norms.md` (written from R1 `prior_chats.md` + R2) assert P1 and P2 are
"60/60 UNCERTAIN error-fallback (degenerate)" and P4 is "24/60 partial." **The on-disk results contradict
this — the pilots completed with real verdicts:**

| Pilot | Dir | Real state on disk | Usable? |
|---|---|---|---|
| **P1 prompt** | `e_prompt_pilot/` | 4 variants × n=60 on deepseek-v3.2, **`num_error_fallback: 0`, coverage 1.0**, real HALL/VALID splits | **YES** — and carries the fragility signal |
| **P2 decoding** | `e_decoding/` | t0 vs t1 × n=60 (real verdicts) + GPT-5.1 3-draw self-consistency (E3 reuse) + deepseek 3-draw crosscheck | **YES** |
| **P3 threshold/agg** | `a3_threshold_aggregation_result.json` | offline re-score, n=60, **9 tools** (Opus/Sonnet/5 open + agentic), AUROC + 19-point DR–FPR curves + subtest aggregation | **YES** ($0, no API) |
| **P4 input-format** | `ablation4_input_format/` | full vs structured + leave-one-field-out (title/author/venue/year/doi), deepseek-v3.2, **n=24×7=168 calls (by design, under the 180 cap — not "partial")** | **YES** |

All four are decision-ready. The "degenerate" warning was written against an earlier run and is no longer true.

---

## 2. Candidate ablations — pilot effect, reviewer concern, full-run cost, go/no-go

Cost model (OpenRouter deepseek-v3.2 ≈ $0.0003/call; GPT-5.1 ≈ $0.003/call; Sonnet 4.6 ≈ $0.004/call;
~850 tok/call). dev_public = 1,119 entries.

### A1 — Prompt-sensitivity sweep  →  **GO (run full)**  ★ highest priority
- **Reviewer concern:** single-prompt design (`sec:baselines`); "is the over-flagging finding a prompt artifact?"
  No peer reports a prompt sweep (peer_norms #10). Maps to **W1** (single-model/-config artifact).
- **Pilot effect (n=60, deepseek-v3.2):** FPR default 0.885 / notaxo 0.692 / uncertain 0.538 / terse 0.654 —
  **34.6 pp spread**; DR 1.000 / 0.941 / 0.853 / 0.882. 11/26 VALID entries flip. F1 stays in a tight band
  (0.741–0.773), so *the ranking metric is far more stable than FPR* — the defensible result is already visible.
- **Full run:** 4 prompt variants × n=150 stratified dev_public × 3 answering models (GPT-5.1, Sonnet 4.6,
  Gemini Flash) = 1,800 calls. Cost ≈ GPT-5.1 600×$0.003 + Sonnet 600×$0.004 + Gemini 600×$0.0006 ≈ **$4.5**,
  ~60–90 min. Report verdict-flip rate, per-model FPR band, and **tool-ranking stability** (Spearman) across prompts.
- **Go reasoning:** directly closes the §0 fragility; cheapest possible defense of the headline; novel (no peer
  has it). The pilot *raises* the question but cannot *settle* it — it is one open model, and the claim is about
  the 12-model cohort. **This is the one ablation the pilot does NOT settle and most needs the full run.**

### A4 — Input-format / leave-one-field-out  →  **GO (run full, cheap)**  ★ second priority
- **Reviewer concern:** **W6** (do models exploit surface format tells rather than reason?) and **W3** (does
  detection lean on one co-designed field?). CiteAudit *motivates* the benchmark with "format variability" but
  never ablates it (peer_norms #11); HALLMARK would be **first in the field**.
- **Pilot effect (n=24×7, deepseek-v3.2):** full→structured **FPR −20 pp** (0.90→0.70); LOO vs structured:
  drop-author **+30 pp FPR (→1.00, flags everything)**, drop-year +20 pp, drop-doi +20 pp, drop-title +10 pp,
  drop-venue +10 pp. Clear mechanistic signal: **authors are the dominant valid-entry anchor**; format rendering
  alone shifts FPR 20 pp.
- **Full run:** 7 conditions × n=60 (full sample) × 1 cheap model = 420 calls ≈ **$0.15**, ~20 min. Optionally add
  GPT-5.1 for a headline-model confirm: +420 calls ≈ $1.3. Recommend cheap-model n=60 + a single GPT-5.1 pass.
- **Go reasoning:** novel, dirt-cheap, and it is the *dynamic* counterpart to the paper's static format-tell audit
  (W6, ≤17.4 pp bound) — empirically shows whether a model leans on one field. n=24 is a real pilot, not partial,
  but n is small (14 HALL / 10 VALID); the +30 pp author effect rests on 10 valid entries, so a full-n confirm is
  warranted before it goes in the paper.

### A3 — Threshold sweep + aggregation-rule ablation  →  **PILOT MOSTLY SETTLES IT (extend offline only, $0)**
- **Reviewer concern:** **#8** operating-point sensitivity; CiteAudit hard-codes a 0.92 threshold with no sweep
  (peer_norms §1). Paper reports fixed @0.5 / cascade @0.55.
- **Pilot effect (offline, n=60, 9 tools, $0):** per-tool 19-point DR–FPR curves. **Confidence-threshold tuning
  buys little** — curves are flat across thr 0.1–0.7 for the high-confidence models (Opus: identical DR 0.941 /
  FPR 0.231 for thr 0.1–0.8; quantized to 6 confidence levels), which is itself a clean, honest finding. The
  *aggregation* result is stronger and publishable: subtest "**any_miss (k≥1 of 4)**" gives **FPR 0.0 / DR 0.971 /
  F1 0.985**, and "cross_db_agreement only" gives FPR 0.038 / DR 1.0 / F1 0.986 — the rule, not the threshold,
  is the lever. Noisy-voter ensemble (7 LLMs): majority (≥4/7) cuts FPR 57.7 pp vs any-miss at −11.8 pp DR.
- **Full run:** **none needed for the curve** — it re-scores stored confidences ($0). To make it cohort-complete,
  re-score the *already-stored* dev_public predictions for all zero-shot tools (offline, $0, ~5 min script).
- **Go reasoning:** the expensive part is done and the pilot settles the substantive question (thresholds are
  quantized → tuning is near-vacuous; aggregation rule matters). **Extend offline to all tools on the full split
  for a paper-ready DR–FPR figure; do not spend API budget.** Differentiates cleanly from CiteAudit's unjustified
  fixed threshold.

### A2 — Decoding / temperature / self-consistency  →  **PILOT SETTLES IT (skip full run)**
- **Reviewer concern:** **#9** decoding robustness (HumanEval sampling precedent; CiteAudit fixes T=0).
- **Pilot effect:** deepseek-v3.2 t0→t1 FPR 0.808→0.885 (+7.7 pp), **8.3% verdict-flip**, F1 0.764→0.733 — a real
  but second-order temperature effect on an open model. GPT-5.1 (E3 reuse, n=150, 3 draws): **flip rate 0.67%,
  F1 std 0.002** — temperature is **API-forced to 1.0**, so a "sweep" is near-vacuous for the headline model.
  deepseek 3-draw majority vote does not rescue FPR (maj3 FPR 0.923).
- **Full run:** a cross-model temp sweep would be 2 temps × n=150 × ≥2 real-temperature open models ≈ 600 calls
  ≈ $0.4. Low marginal value.
- **No-go reasoning:** the headline model has no temperature knob (already documented, `limitations.tex` L18,
  E3). The open-model effect is small and already captured by the pilot. **Cite E3 + this pilot; do not run more.**
  Only worth a cross-model temp pass *opportunistically* if A1 is running anyway on the same models.

### A5 (peer-norm gap, NOT in any pilot) — Inter-annotator agreement (κ) on a re-annotated real-world subset  →  **GO (no API, human/agent annotation)**
- **Reviewer concern:** **#12** — *no surveyed peer reports κ* (CiteAudit uses ≥2 reviewers + consensus but no
  statistic). The relabel audit on this very branch (52 recovered real papers, commits 84f9fe4/c9915d2/6eced89/…)
  proves the single-pass labels were noisy, which makes an honest κ both expected and the natural justification
  for the audit.
- **Pilot effect:** none — not a pilot dimension; flagged here because R1/R2 rank it the single highest-leverage
  *missing* analysis and it is the one axis where HALLMARK could move *ahead of the field*, not to parity.
- **Cost:** re-annotate a real-world subset (~108 entries) with 2–3 raters; report Cohen's/Fleiss' κ. No API
  spend; cost is annotation effort. Out of scope for *this* API-pilot phase but belongs at the top of the plan.
- **Go reasoning:** sets a precedent rather than matching one; substrate already exists. Surface to the user as
  the highest-value *non-API* addition.

---

## 3. Already SETTLED elsewhere — do NOT re-run (cite, don't spend)

Confirmed against `comparative_expectations.md` §2 (chat ids) — these duplicate committed work / appendix:
cutoff-aware prompting (H2, `app:cutoff-aware`), late-cutoff control (E2, `results/reviewer_experiments/e2_*`),
model-drift (#17002), thinking-budget (App. G), format-tell static audit (W6 ≤17.4 pp), co-design bound (W3),
cost/latency (#17276), pre-screening lift, **recall probe E1** (`e1_recall_probe/`), **seed variance E3**
(`e3_variance/`), water-filling, KS synth-vs-real, bootstrap CIs, ECE, cascade, cross-split ΔFPR. The paper
already over-delivers on the analyses this benchmark class expects; budget goes to §2 gaps, not these.

---

## 4. RANKED candidate list (reviewer-value × cheapness × novelty)

| Rank | Ablation | Pilot verdict | Reviewer value | Novelty | Full-run cost | Decision |
|---|---|---|---|---|---|---|
| 1 | **A1 prompt sweep** | **fragile headline exposed** | Very high (W1; defends single-prompt) | High (no peer) | ~$4.5, 60–90 min | **RUN FULL NOW** |
| 2 | **A4 input-format / LOO** | real signal, n small | High (W6/W3) | Highest (first in field) | ~$0.15 (cheap) / +$1.3 (GPT-5.1) | **RUN FULL NOW** |
| 3 | **A3 threshold/agg** | pilot mostly settles | Medium-high (#8) | Medium | **$0 (offline re-score)** | **EXTEND OFFLINE ONLY** |
| 4 | **A5 κ / IAA** | not a pilot | Very high (#12, ahead of field) | Highest | $0 API (annotation effort) | **GO — non-API, escalate to user** |
| 5 | **A2 decoding/temp** | **pilot settles** | Low (temp API-forced; E3) | Low | ~$0.4 if forced | **SKIP — cite E3 + pilot** |

---

## 5. Decision in one approval

**RUN NOW (one full-ablation batch, total ≈ $6, ≈ 2 h wall clock):**
1. **A1 prompt sweep** — 4 variants × n=150 × {GPT-5.1, Sonnet 4.6, Gemini Flash} = 1,800 calls ≈ **$4.5**.
   Deliver: per-model FPR band + verdict-flip rate + **Spearman tool-ranking stability across prompts**.
   This converts the §0 fragility from a liability into a robustness result ("ranking is prompt-invariant; absolute
   FPR is prompt-dependent — we say so").
2. **A4 input-format / LOO** — 7 conditions × n=60 cheap model = 420 calls ≈ **$0.15**, plus one GPT-5.1
   confirmation pass = 420 calls ≈ **$1.3**. Deliver: full→structured FPR delta + per-field LOO FPR/DR deltas.

**PILOT ALREADY SETTLES (no API spend):**
- **A3 threshold/aggregation** — extend the offline re-score to all zero-shot tools on full dev_public for a
  paper-ready DR–FPR figure ($0, ~5 min). Finding: thresholds quantized → tuning near-vacuous; aggregation rule
  ("any_miss") is the real lever.
- **A2 decoding/temperature** — cite E3 (flip 0.67%, F1 std 0.002, temp forced to 1.0) and this pilot. Done.

**GO BUT OUT-OF-SCOPE FOR THIS API PHASE (escalate to user):**
- **A5 κ / inter-annotator agreement** on the re-annotated real-world subset — highest-leverage *missing*
  analysis; no API cost, needs 2–3 annotators on ~108 entries.

**SKIP:** everything in §3 (cite the chat id / appendix instead of re-running).

**One-line approval ask:** approve the **~$6 A1+A4 batch** (+ the $0 A3 offline extension) and the **non-API A5 κ**
study; skip A2 and all of §3.
