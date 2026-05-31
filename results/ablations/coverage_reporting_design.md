# Design: abstention/coverage-aware reporting (selective prediction)

**Approved 2026-05-31.** Make LLM verifiers comparable to bibtex-updater (which abstains ~26% in v1.2.0) and stop penalizing honest abstention, by treating UNCERTAIN as a first-class third outcome under the selective-prediction framing.

## What to add
**A — main table (Table 2 / tab:results):** add a **Coverage** column (fraction of entries the tool commits a VALID/HALL verdict on; 1 − UNCERTAIN-rate) and report DR/FPR/F1 in **two stances for every tool** (incl. bibtex-updater): *conservative* (UNCERTAIN excluded — the existing `build_confusion_matrix` protocol) and *aggressive* (UNCERTAIN→flagged). This is the cascade's existing two-stance scoring generalized to all tools.

**B — appendix selective-prediction view:** a **risk–coverage curve** (FPR vs coverage as the abstain threshold sweeps) per tool, plus a single **FPR@90%-coverage** number. Data already produced by the A3 threshold/aggregation ablation — reuse it.

## Guardrails / honesty (non-negotiable)
1. **Coverage must sit next to every abstain-excluded metric** — otherwise abstention is gameable (abstain-on-all ⇒ perfect precision-on-committed). Reporting coverage + the aggressive number prevents this.
2. **LLM abstention is prompt-dependent** (A1: Sonnet UNCERTAIN 2%→9% across prompt variants) — fix the default prompt for the headline coverage numbers and cite the A1 sensitivity; don't present coverage as a fixed tool property without that caveat.
3. **BTU vs LLM abstention differ in mechanism** — BTU abstains because *no record found* (data-coverage gap); an LLM abstains because *not confident* (epistemic). Same column, one honest sentence on the asymmetry.
4. Frame to the thesis: abstention = *route-to-human*, a useful action in the reviewer-bound regime — a tool that abstains rather than false-flags is better for reviewer effort. Sharpens the FPR-is-decisive story.
5. Cite the selective-prediction / risk–coverage lineage (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017); no peer citation-benchmark reports it ⇒ ahead-of-field (like κ).

## Cost / reuse
Near-zero new compute: `build_confusion_matrix` already excludes UNCERTAIN; cascade already has conservative/aggressive; A3 already computed per-tool risk-coverage/AUROC. This is mostly *surfacing* existing numbers + one appendix figure. Integrated in the consolidated paper pass (after GEN).
