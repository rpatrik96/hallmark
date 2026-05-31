# Peer-benchmark ablation/robustness norms (Task R2)

**Scope.** What ablation/robustness analyses do the benchmark and citation-audit papers HALLMARK cites actually report, and which of those does HALLMARK already cover? This is the comparative + historical grounding for the small-scale-ablation decision; the parallel pilot workflow owns the new prompt/decoding/threshold/format runs under `results/ablations/`.

**Method.** Read `references.bib` + `sections/related_work.tex`, then surveyed each reachable peer via arXiv MCP (full-text where downloadable) and WebSearch/WebFetch. CiteAudit (closest peer) was read in full from the downloaded PDF; the audit papers and methodology anchors from abstracts + landing pages (full PDFs were stream-compressed and not text-extractable via WebFetch, flagged below).

---

## 1. Per-source survey: what each peer actually reports

### CiteAudit — `citeaudit` (Shi et al., arXiv:2602.23452v3) — CLOSEST PEER
Full text read (`/Users/patrik.reizinger/mcp-servers/arxiv/storage/2602.23452_*.pdf`, 67k chars).
- **Dataset:** 9,442 citations — 3,356 real-world + 6,086 human-synthesized; "diverse domains, citation formats, and hallucination types." Taxonomy (Fig. 2): Title errors (keyword-sub / paraphrase / topic-conditioned fabrication), Author errors (4 perturbation ops: redundant add, deletion, name-level swap, fully-synthetic), Metadata errors (venue / year / DOI-identifier). Coarser than HALLMARK's 14-type / 3-tier taxonomy — title/author/metadata groups, no difficulty-tier or sub-test decomposition.
- **Models evaluated:** 6 systems as baselines — Mixtral-8x7B, Llama-3.3-70B, Qwen3-Next-80B, Gemini-3-Pro, GPT-5.2, Claude-Sonnet-4.5, plus GPTZero; 5 commercial tools cited (GPTZero, CiteCheck, Citely, RefCheck_ai, SwanRef). Their own multi-agent pipeline (Gemini-3-Flash planner/judge + Qwen3-VL-235B extractor) is the proposed method, not an independent baseline.
- **Decoding:** temperature = 0.0 on planning/judgment stages; memory-match cosine threshold 0.92 (a *fixed* operating point, not swept). §5.1.
- **Synthetic-vs-real fidelity:** chi-square test of GPTZero fake-detection behavior, generated vs real-world fake citations: χ²=5.6e-5, p=0.994 (df=1) → "no significant difference." (Table 2.) **Directly analogous to HALLMARK's KS test in `sec:synth_vs_real`.**
- **Human validation:** author-team annotation, ≥2 independent reviewers per citation, consensus conflict-resolution, no-consensus items dropped, random re-check subset for QA (Appendix C). **No κ / IAA statistic reported** — agreement is procedural, not quantified.
- **Metrics:** Accuracy / Precision / Recall / F1 (fake = positive class) + runtime + API price per 1M tok (Tables 3,4). No ECE/calibration, no MCC, no tier-weighted metric.
- **"Additional Experiment Analysis" (§5.4):** qualitative diagnosis only — proprietary models don't reliably execute verifiable retrieval; black-box provenance. §5.5 = 2 case studies.
- **NOT reported:** pipeline-stage ablation (despite the multi-agent design), prompt-sensitivity sweep, decoding/temperature sweep, detection-threshold sweep, contamination/temporal split, IAA statistic, significance test or CIs on the main model comparison.

### GhostCite — `ghostcite2026` (Xu et al., arXiv:2602.06718)
- Scale: 2.2M citations / 56,381 papers, AI/ML + Security venues 2020–2025; "CiteBench" verification framework.
- Temporal trend: +80.9% increase in invalid-citation papers in 2025; 1.07% of papers contain invalid citations; per-model citation-generation hallucination 14.23%–94.93% across **13 LLMs**.
- Survey of 97 researchers (practice, not annotation). No ablation/robustness, no IAA, no significance/CIs surfaced (abstract-level; full PDF not extracted).
- Prevalence study, not detection-tool benchmark — confirms HALLMARK's framing in related_work.

### HalluCitation Matters — `hallucitation2026` (Sakai et al., arXiv:2601.18724)
- ~300 hallucinated-citation papers across ACL/NAACL/EMNLP 2024–2025; >100 at EMNLP-2025 main+Findings. Strong temporal-increase signal.
- Marked "Work In Progress." No taxonomy detail, IAA, tool/LLM eval, ablation, or significance surfaced (abstract-level).

### The Case of the Mysterious Citations — `mysterious2026` (Bienz et al., arXiv:2602.05867)
- 4 HPC venues, 2021 vs 2025 proceedings; automated pipeline; 2–6% of papers affected; every 2025 proceeding affected vs none in 2021 (temporal contrast). No IAA/ablation/significance surfaced (abstract-level).

### rao2026bibtex (Rao & Callison-Burch, arXiv:2604.03159) — concurrent detection/mitigation
- 3 search-enabled frontier models (GPT-5, Claude Sonnet-4.6, Gemini-3-Flash).
- **Ablation of integration architecture:** single-stage vs two-stage (separating search from revision) — two-stage gives larger gains + lower regression (0.8% vs 4.8% regression). This is an *architecture* ablation, the kind HALLMARK does in its cascade (`sec:cascade`) and agentic-harness analysis.
- **Contamination/temporal control:** 3 citation tiers — popular / low-citation / recent-post-cutoff — to "disentangle parametric memory from search dependence"; accuracy drops 27.7pp popular→recent. **Directly parallels HALLMARK's temporal supplement + recall probe.**
- Mitigation: `clibib` deterministic BibTeX retrieval; two-stage integration +8.0pp to 91.5% acc.
- No prompt/decoding/threshold sweep, no IAA, no significance/CIs surfaced.

### raowong2026urls / DRBench (Rao, Wong, Callison-Burch, arXiv:2604.03173) — concurrent, orthogonal scope (URL liveness)
- **10 models/agents** on DRBench (53,090 URLs; 32 academic fields; 168k+ URLs total), 3 on ExpertQA.
- **Temporal/contamination control:** Wayback-Machine validation to split hallucinated vs link-rot (stale).
- **Per-domain breakdown:** 32 fields, 5.4% (Business) – 11.4% (Theology). Releases `urlhealth`.
- Self-correction reduces non-resolving URLs 6–79× to <1%. No prompt/decoding/threshold sweep, no IAA, no significance/CIs surfaced.

### Methodology anchors
- **HumanEval** (`humaneval`, Chen et al., arXiv:2107.03374): pass@k via repeated sampling (n=100 samples/problem → 70.2% vs 28.8% pass@1) — establishes that **multi-sample decoding analysis** is a recognized robustness device; HALLMARK's single-draw caveat (GPT-5 forced temp=1.0) is the analogue dimension. Multi-criteria functional sub-tests motivate HALLMARK's sub-test decomposition.
- **LiveCodeBench** (`livecodebench2024`, Jain et al., arXiv:2403.07974): 29–52 LLMs; **release-date-windowed contamination analysis** (problems annotated with release dates; evaluate models only on post-cutoff problems; DeepSeek drops on post-Sep-2023 LeetCode). Holistic multi-scenario eval. This is the canonical contamination-free template HALLMARK's temporal split cites.
- **Hardt 2025, "The Emerging Science of ML Benchmarks"** (`hardt2025benchmarks`): Ch.5 adaptive overfitting / holdout discipline ("iron vault for test data" vs reality of repeated adaptive testing); Ch.9 labeling & annotation; Ch.12 aggregation = voting-system analogy, "greater diversity comes at the cost of greater sensitivity to artifacts" → motivates HALLMARK's **tier-weight sensitivity** check and dwork-grounded holdout.
- **dwork2015** reusable holdout (Science): adaptive-data-analysis validity — the formal backing for HALLMARK's dev/test/hidden split discipline; reviewers in this class expect an explicit "we don't tune on test" statement.
- **agrawal2024** "Do LMs know when they're hallucinating references?": direct/indirect query consistency as a detection signal — a probe-style robustness analysis (self-consistency under query reformulation).

---

## 2. Reviewer-expectation checklist for a citation/LLM-eval benchmark of this class

Legend: ✅ HALLMARK covers · ◑ partial · ❌ missing. "Peers" = how common among the surveyed set.

| # | Expected analysis | Peers that report it | HALLMARK status | Where in HALLMARK / gap |
|---|---|---|---|---|
| 1 | **Model-family breadth** (several LLM families + a rule/DB tool) | CiteAudit (6), GhostCite (13), DRBench (10), rao2026bibtex (3), LiveCodeBench (29+) | ✅ | 12 full-coverage tools across 6+ families + DOI-only + bibtex-updater (`sec:baselines`, `tab:results`). Strongest in the cohort. |
| 2 | **Contamination / temporal (training-cutoff) control** | LiveCodeBench, rao2026bibtex, DRBench, GhostCite/HalluCitation/Mysterious (trend) | ✅ | 448-entry temporal supplement + 60-entry 2026 probe + per-model cutoff table (`sec:temporal_robustness`, `app:cutoffs`). Among the most thorough. |
| 3 | **Memorization-vs-calibration disentanglement** (recall probe) | rao2026bibtex (citation-tier proxy) | ✅ | Recall probe E1 + cross-provider control E2 (`app:reviewer_experiments`, `FINDINGS.md`). HALLMARK goes further than any peer here. |
| 4 | **Synthetic-vs-real fidelity test** | CiteAudit (χ², p=0.994) | ✅ | KS test on 3 surface features, 6/9 types n.s. (`sec:synth_vs_real`). HALLMARK is *more* honest (explicitly flags underpower + TOST need). |
| 5 | **Statistical significance / bootstrap CIs** | (rare — none of the audits) | ✅ | Stratified bootstrap 10k resamples, paired bootstrap p-values, per-type power/MDE (`app:bootstrap`, `app:statistics`). HALLMARK leads the cohort. |
| 6 | **Calibration (ECE) reporting** | (rare — none surfaced) | ✅ | ECE in `tab:results`, calibration paragraph (`sec:calibration`). Differentiator. |
| 7 | **Per-type / per-tier / per-domain breakdown** | CiteAudit (per-type), DRBench (per-domain) | ✅ | Per-tier + per-type heatmap + Gini tier-concentration (`sec:per_tier`, `sec:per_type`, `sec:water_filling`). |
| 8 | **Operating-point / metric-weight sensitivity** | (Hardt Ch.12 motivates; CiteAudit fixes one threshold) | ◑ | Tier-weight sensitivity across 5 schemes (`app:tw-sensitivity`) + conservative/aggressive cascade @0.55 (`sec:cascade`). **Gap:** no explicit confidence-THRESHOLD sweep / DR–FPR operating-curve per zero-shot tool. (Pilot A3 in `results/ablations/a3_threshold_aggregation*` targets this.) |
| 9 | **Decoding / temperature sensitivity** | HumanEval (sampling), CiteAudit (fixes T=0) | ◑ | Decoding config documented (T=0; GPT-5 forced T=1.0) + N=3 GPT-5.1 run-variance (`app:reviewer_experiments` E3). **Gap:** no temperature/sampling *sweep* across models. (Pilot `e_decoding` targets this.) |
| 10 | **Prompt sensitivity** (phrasing / taxonomy-in-prompt / abstention wording) | (none of the peers report a sweep) | ◑ | Cutoff-aware prompting ablation on GPT-5.1 (`app:cutoff-aware`) — one prompt variant, temporal-only. **Gap:** no general prompt-phrasing/taxonomy/abstention sweep. (Pilot `e_prompt_pilot` targets this — but current run is degenerate: all 60 entries returned UNCERTAIN/error-fallback on deepseek-v3.2, so it carries no signal yet.) |
| 11 | **Input-format variation** (BibTeX vs plaintext, with/without abstract) | (none of the peers; CiteAudit notes "format variability" as motivation but doesn't ablate) | ◑/❌ | **Gap:** not in the paper. (Pilot `ablation4_input_format` targets this; only a partial deepseek-v3.2 checkpoint exists so far.) |
| 12 | **Inter-annotator agreement (κ) on human labels** | CiteAudit (procedural ≥2-reviewer + consensus, NO κ); none report κ | ❌ | **Gap & opportunity:** HALLMARK's real-world set (108 entries) has no reported IAA. *No peer reports κ either*, so a κ on a re-annotated real-world subset would put HALLMARK **ahead of the field**, not merely at parity. The relabel audit (52 recovered real papers) is the natural substrate. |
| 13 | **Pipeline-stage / architecture ablation** | rao2026bibtex (1-stage vs 2-stage), CiteAudit (multi-agent but NOT ablated) | ✅ | Stage-2 cascade conservative-vs-aggressive (`sec:cascade`), agentic-harness vs zero-shot, prescreening on/off (`sec:prescreening`). HALLMARK matches/exceeds rao2026bibtex. |
| 14 | **Cost / runtime reporting** | CiteAudit (time + $/1M tok), DRBench | ✅ | Cost–accuracy tradeoff $/entry (`sec:cost`, `fig:cost`). |
| 15 | **Cross-split generalization (dev→test FPR shift)** | (rare; rao2026bibtex tiers are the closest) | ✅ | ΔFPR column + `sec:crosssplit_robustness`. Differentiator. |
| 16 | **Holdout / adaptive-overfitting discipline statement** | Hardt Ch.5, dwork2015 (anchors) | ✅ | hidden split + co-design caveat + dwork/Hardt citations. |

---

## 3. Bottom line for the ablation-plan decision

**HALLMARK already over-delivers on the analyses reviewers most expect for this benchmark class** — model breadth (#1), temporal/contamination control (#2,#3), significance/CIs (#5), calibration (#6), per-type/tier (#7), cross-split (#15). On every dimension where the *closest* peer (CiteAudit) reports something, HALLMARK reports an equal-or-stronger version, and on #3/#5/#6/#15 it is ahead of the entire surveyed cohort.

**The genuinely open dimensions the small-scale ablations should close are exactly the four the parallel pilot is running, in priority order:**
1. **Prompt sensitivity (#10)** — *highest reviewer-salience*: no peer reports a prompt sweep, and a clean "rankings are robust to prompt phrasing / taxonomy-in-prompt / abstention wording" result directly defends the single-prompt design choice in `sec:baselines`. **Caveat: the current `e_prompt_pilot` run is degenerate (60/60 UNCERTAIN error-fallback on deepseek-v3.2) — it must be re-run on a model that actually answers before it can be cited.**
2. **Threshold / operating-point sweep (#8)** — turns the fixed-@0.5/@0.55 reporting into a DR–FPR curve; CiteAudit hard-codes a 0.92 match threshold without justification, so a sweep is a clean differentiator.
3. **Decoding/temperature sweep (#9)** — extends the existing N=3 GPT-5.1 variance note into a cross-model sensitivity check; HumanEval precedent makes this expected.
4. **Input-format variation (#11)** — CiteAudit *motivates* the benchmark with "formatting variability" but never ablates it; HALLMARK could be the first to.

**The one ablation no peer reports but that would move HALLMARK ahead of the field: inter-annotator agreement (κ) on a re-annotated real-world subset (#12).** This is not in the current pilot set and is the single highest-leverage *missing* analysis — the relabel audit already produced the re-annotation substrate.

**Things reviewers in this class do NOT generally demand (so do not over-invest):** exhaustive per-domain breakdown beyond per-type (DRBench-style 32-field is specific to a URL-liveness corpus), multi-sample pass@k (citation verification is a binary verdict, not generation), and ablating every multi-agent component individually (rao2026bibtex's single-vs-two-stage contrast is sufficient precedent; HALLMARK's cascade already matches it).

---

## 4. Citations used
- citeaudit — Shi et al., CiteAudit, arXiv:2602.23452v3 (full text)
- ghostcite2026 — Xu et al., GhostCite, arXiv:2602.06718 (abstract)
- hallucitation2026 — Sakai et al., HalluCitation Matters, arXiv:2601.18724 (abstract)
- mysterious2026 — Bienz et al., The Case of the Mysterious Citations, arXiv:2602.05867 (abstract)
- rao2026bibtex — Rao & Callison-Burch, arXiv:2604.03159 (abstract)
- raowong2026urls — Rao, Wong & Callison-Burch, DRBench, arXiv:2604.03173 (abstract)
- humaneval — Chen et al., arXiv:2107.03374 (full metadata)
- livecodebench2024 — Jain et al., arXiv:2403.07974 (landing page + metadata)
- hardt2025benchmarks — Hardt, The Emerging Science of ML Benchmarks, mlbenchmarks.org (TOC)
- dwork2015 — Dwork et al., The Reusable Holdout, Science 349 (cited anchor)
- agrawal2024 — Agrawal et al., EACL Findings 2024 (cited anchor)

*Caveat: full PDFs of the audit papers (ghostcite/hallucitation/mysterious) and the two concurrent Rao papers were not text-extractable via WebFetch (compressed PDF streams); their robustness/IAA/significance lines are reported as "not surfaced at abstract level," not as confirmed absent. CiteAudit, HumanEval, LiveCodeBench were verified at full-text/landing-page level.*
