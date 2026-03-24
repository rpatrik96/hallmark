# Plan: Addressing Benchmark Science Shortcomings in HALLMARK

Based on Moritz Hardt's "Emerging Science of Benchmarks" (ICLR 2024). Three planning agents produced detailed implementation plans for each area. This document consolidates them.

---

## A. Aggregation Vulnerability

### Problem
Tier-weighted F1 aggregates 14 types × 3 tiers into one number. Arrow's impossibility theorem guarantees aggregation artifacts. Normalization/weighting changes can flip rankings (cf. OpenLLM Leaderboard).

### Implementation Plan

#### A1. Ranking Sensitivity Analysis (Tier Weight Sweep)

**New function in `hallmark/evaluation/metrics.py`:**
```python
def ranking_sensitivity_analysis(
    entries, tool_predictions, n_samples=1000, seed=42
) -> dict
```
- Sample 1000 weight vectors from Dirichlet(1,1,1) over the 3-tier simplex
- For each, compute TW-F1 for all tools, extract ordinal ranking
- Output: `rankings_stable`, `rank_inversions`, `per_tool_range`, `kendall_tau_min`, `concordance_fraction`
- **CLI**: `hallmark sensitivity --predictions-dir ... --n-samples 1000`
- **Figure**: Ternary plot colored by which tool ranks first; parallel coordinates across 5 named schemes

#### A2. IIA Check

**New function in `hallmark/evaluation/metrics.py`:**
```python
def iia_violation_check(entries, tool_predictions, ranking_method="score") -> dict
```
- For score-based ranking (TW-F1): IIA trivially holds (each tool's score is independent). Document this as a strength.
- For Plackett-Luce: leave-one-out test — remove each tool, re-rank, check for pair-order flips.
- Inject synthetic weak baselines (random, always-valid, always-hallucinated) to stress-test.

#### A3. Reporting: Per-Tier/Type as Primary

- **CLI changes**: `--per-tier` and `--per-type` flags on `leaderboard`. Ranking concordance footer.
- **New functions**: `per_tier_rankings()`, `per_type_rankings()`, `ranking_concordance()` in `metrics.py`
- **Paper**: Restructure experiments.tex to lead with per-tier comparison table. Move aggregate TW-F1 to secondary with Hardt citation caveat.
- **Figure**: Bump chart showing tool rank across Tier 1/2/3.

#### A4. Promote Plackett-Luce as Arrow-Robust Alternative

PL satisfies IIA by construction (Luce choice axiom). Rankings invariant to tool addition/removal.
- **CLI**: `--ranking-method {score, plackett_luce, both}` on `leaderboard`
- **Paper**: Expand Section 3.2 to frame PL as Arrow-robust, not just for incomplete data. Add: "The Plackett-Luce model satisfies IIA by construction (Luce, 1959), making rankings invariant to tool set changes — a property weighted score aggregation cannot guarantee (Hardt, 2024)."

### Phase Order
1. Core analysis functions (metrics.py)
2. Tests
3. CLI integration
4. Figures
5. Paper text

---

## B. External Validity

### Problem
HALLMARK uses synthetic hallucinations. Do tool rankings generalize to real citation errors in actual papers?

### Implementation Plan

#### B1. Real-World Validation Set Curation

**Target**: 150-200 hallucinated + 100 valid entries = 250-300 total.

**Sources (ranked by yield):**
1. **GhostCite** (arXiv:2602.06718) — 604 papers with invalid citations from 2.2M analyzed. Currently only 5 entries extracted.
2. **HalluCitation** (arXiv:2601.18724) — ~300 papers with hallucinated citations in ACL/NAACL/EMNLP. Currently only 3 entries extracted.
3. **GPTZero NeurIPS 2025** — 100+ hallucinations in 53 papers. 98 entries exist but skewed easy.
4. **Retraction Watch Database** — 40K+ retracted papers, filter for citation issues.
5. **OpenReview errata** — post-publication comments flagging citation errors.
6. **arXiv version diffs** — changelogs mentioning "corrected references."

**Type distribution target (realistic, not uniform):**
- plausible_fabrication: ~40, near_miss_title: ~20, swapped_authors: ~20, wrong_venue: ~15, chimeric_title: ~15, placeholder_authors: ~15, fabricated_doi: ~10, preprint_as_published: ~10, others: ~5-10 each

**Format**: Standard `BenchmarkEntry` with `generation_method="real_world"`, stored in `data/v1.0/external_validation.jsonl`

**Script**: `scripts/curate_external_validation.py` with per-source functions, validation via `validate_entry()`, deduplication.

**Loader**: Add `"external_validation"` to `SPLIT_PATHS` in `hallmark/dataset/loader.py`.

#### B2. Cross-Validation Analysis

**New script: `scripts/analyze_external_validity.py`**
1. Run all baselines on external_validation split
2. Compute per-tool metrics on both HALLMARK and external sets
3. Compute Kendall's tau-b and Spearman's rho between rankings
4. Per-type concordance analysis
5. Scatter plot: HALLMARK metric vs external metric, one point per tool

**Interpretation framework:**
- tau > 0.7: strong external validity
- 0.4 < tau < 0.7: partial — identify discordant types
- tau < 0.4: weak — report honestly, investigate synthetic-vs-real differences

**Additional analyses:**
- Per-generation-method performance gap: `DR(real_world) / DR(perturbation)` per tool
- Type-prevalence reweighting: reweight HALLMARK DRs by real-world type distribution, check if correlation improves

#### B3. Low-Rank / Single-Factor Analysis (PCA)

**New file: `hallmark/evaluation/factor_analysis.py`**

```python
def compute_score_matrix(entries, tool_predictions) -> (tool_names, type_names, np.ndarray)
def pca_analysis(tool_names, type_names, score_matrix) -> dict
```

- Build tools × hallucination_types detection rate matrix
- PCA via `numpy.linalg.svd`
- Report: explained_variance_ratio, effective_rank (90% variance), PC1 loadings, PC1 correlation with tool features (is_llm, has_api, etc.)
- **Script**: `scripts/analyze_factor_structure.py` → figures: scree plot, biplot
- **Expected**: PC1 ~60-70% variance correlating with `is_llm`. Effective rank 2-3 (not 1) because tier structure adds orthogonal signal.

#### B4. Paper Integration

- **limitations.tex**: New paragraph on external validity with rank correlation results
- **analysis.tex**: Factor structure paragraph with PCA results
- **appendix.tex**: New section with full correlation tables, scatter plots, scree/biplot

### Phase Order
1. PCA analysis (no new data needed) — immediate
2. External validation curation — medium effort
3. Cross-validation analysis — depends on Phase 2
4. Paper integration — depends on 1 + 3

---

## C. Sample Size, Reuse, and Strategic Gaming

### Problem
- Some subtypes have n < 60, too small for reliable ranking
- Dev set reused ~28 times (adaptive overfitting budget consumed)
- Traditional tools show 5x-31x Tier 1/Tier 3 detection rate ratios (water-filling)

### Implementation Plan

#### C1. Bootstrap CIs on Per-Subtype Rankings

**New file: `hallmark/evaluation/ranking_stability.py`**

```python
def per_subtype_ranking_stability(
    entries, tool_predictions, n_bootstrap=1000, seed=42
) -> dict[str, SubtypeRankingResult]
```

Per subtype: filter entries, run Plackett-Luce with bootstrap, track rank (not just score) per iteration, report rank CIs and stability flag.

**Current subtype sizes (smallest → largest across public splits):**
- future_date: 59, preprint_as_published: 60, chimeric_title: 67, fabricated_doi: 68
- Largest: plausible_fabrication: 219, swapped_authors: 123

#### C2. Power Audit (Hardt's Quadratic Scaling)

**New file: `hallmark/evaluation/power.py`**

Refactor from existing `scripts/power_analysis.py`:
```python
def mde_two_proportion(n, p0=0.5, alpha=0.05, power=0.80) -> float
def required_n(delta, p0=0.5, alpha=0.05, power=0.80) -> int
def subtype_power_audit(entries, target_deltas=[0.05, 0.10, 0.15, 0.20]) -> dict
```

**Key numbers:**
| n | MDE (80% power) |
|---|-----------------|
| 30 | 36.2 pp |
| 60 | 25.6 pp |
| 100 | 19.8 pp |
| 200 | 14.0 pp |
| 785 | 10.0 pp (target for 10pp detection) |

**Recommendation**: Types with n < 60 (future_date, preprint_as_published) can only distinguish effect sizes > 25 pp. Expand these rather than improving label quality.

#### C3. Test Set Reuse Tracking

**New file: `hallmark/evaluation/reuse_tracker.py`**

```python
def compute_reuse_budget(history_path, split_sizes, max_budget_ratio=2.0) -> dict[str, SplitReuseBudget]
def log_evaluation(history_path, result) -> None
def estimate_remaining_budget(n, k_current, max_budget_ratio=2.0) -> int
```

**Current budget status:**
- dev_public (n=1119, k=28): heavily reused, adaptive bound ~14x non-adaptive (expected for dev)
- test_public (n=831, k=0): fresh
- hidden (n=453): intact

**Integration**: Wire `log_evaluation()` into `evaluate()`. Add `--budget` flag to CLI.

#### C4. Water-Filling Detection

**New file: `hallmark/evaluation/water_filling.py`**

```python
def water_filling_analysis(entries, tool_predictions, ratio_threshold=3.0, gini_threshold=0.3) -> dict
def water_filling_gini(tier_drs, tier_weights=None) -> float
def plot_water_filling(profiles, output_path) -> None
```

**Empirical finding (already confirmed):**
| Tool | T1 DR | T3 DR | Ratio |
|------|-------|-------|-------|
| bibtexupdater | 0.411 | 0.013 | 30.6x |
| harc | 0.437 | 0.037 | 11.6x |
| doi_only | 0.500 | 0.094 | 5.3x |
| llm_openai | 0.873 | 0.785 | 1.1x |
| deepseek_r1 | 0.993 | 0.832 | 1.2x |

Traditional tools show extreme water-filling; LLMs are nearly uniform across tiers.

#### C5. Tier 3 "Hard Subset" Metric

- Add `tier3_f1: float = 0.0` to `EvaluationResult` in `schema.py`
- Compute in `evaluate()` from existing `per_tier_metrics`
- New function `hard_subset_report()` in `metrics.py`
- CLI: `--hard-subset` flag on leaderboard (sort by Tier 3 F1)
- Warning when tool's Tier 3 F1 is > 20 pp below overall F1

### Phase Order
1. power.py + water_filling.py (no dependencies) — immediate
2. reuse_tracker.py (standalone)
3. ranking_stability.py (builds on ranking.py bootstrap)
4. tier3_f1 in EvaluationResult + hard_subset_report
5. CLI integration
6. Paper figures and text

---

## Overall Priority Ranking

| Priority | Item | Effort | Impact | Dependency |
|----------|------|--------|--------|------------|
| 1 | PCA factor analysis (B3) | Low | High — answers "is HALLMARK one-dimensional?" | None |
| 2 | Water-filling detection (C4) | Low | High — already confirmed, needs formalization | None |
| 3 | Tier 3 hard subset metric (C5) | Low | Medium — simple addition | None |
| 4 | Ranking sensitivity sweep (A1) | Medium | High — directly tests aggregation robustness | None |
| 5 | Power audit (C2) | Low | Medium — quantifies per-subtype limitations | None |
| 6 | Per-tier/type as primary reporting (A3) | Medium | High — paper framing change | A1 |
| 7 | Plackett-Luce promotion (A4) | Medium | Medium — Arrow-robust alternative | None |
| 8 | Test set reuse tracking (C3) | Low | Low — mostly documentation | None |
| 9 | IIA check (A2) | Medium | Medium — validates score-based robustness | A4 |
| 10 | External validation curation (B1-B2) | High | Very High — the real external validity answer | None |
| 11 | Bootstrap ranking stability (C1) | Medium | Medium — per-subtype confidence | None |

Items 1-5 can all proceed in parallel with no dependencies.
