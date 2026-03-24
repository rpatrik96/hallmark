# Hardt's "Emerging Science of Benchmarks" — HALLMARK Implications

Source: Moritz Hardt, ICLR 2024 Invited Talk + manuscript at mlbenchmarks.org

## Summary of Key Arguments

### 1. Rankings > Absolute Numbers
Individual accuracy/F1 numbers don't replicate across datasets, but **model rankings do** — even under major dataset variation (ImageNet V2, ObjectNet, ImageNot). Benchmarks succeed because they produce valid ordinal rankings, not cardinal measurements.

**Goodhart's Law reversal:** "Benchmarks make a virtue out of gaming the metric. Rather than fighting Goodhart's law, they lean into it. The numbers are off, but the ranking is fine."

### 2. Don't Label Twice (Dorner & Hardt, ICML 2024)
Given a fixed labeling budget, it's optimal to collect **one label per sample across more samples** rather than multiple labels per sample + majority vote. Sample size beats label quality for model comparison. Uses Cramér's theorem.

### 3. Multi-Task Aggregation is Fundamentally Limited
Arrow's impossibility theorem applies directly: no ranked aggregation of multiple tasks can simultaneously satisfy unanimity and independence of irrelevant alternatives (IIA) without being dictatorial. Practical consequence: normalization/weighting choices change rankings dramatically (OpenLLM Leaderboard rank-57-to-top-10 example).

**Diversity-stability tradeoff:** more diverse tasks → more sensitivity to aggregation artifacts. No escape.

**Strategic gaming (Gibbard-Satterthwaite):** rational agents concentrate effort on easy tasks with large marginal returns (water-filling), neglecting harder tasks.

### 4. Equalization is Required for Valid Rankings
Rankings are valid only when all competitors face equal conditions (like runners on the same starting line). ImageNet era had this (shared training data). LLM era doesn't.

### 5. Low-Rank Structure in Benchmarks
Benchmark-model score matrices are surprisingly low-rank. First principal component often correlates with a single dominant factor (e.g., pretraining compute), suggesting benchmarks may measure one capability rather than many.

---

## Direct Implications for HALLMARK

### What HALLMARK already does well

- **Tiered difficulty (Easy/Medium/Hard):** Creates richer ordinal signal beyond a single aggregate. Directly addresses Hardt's aggregation concerns.
- **Hidden test split (453 entries):** Proper holdout method. Protects against adaptive overfitting.
- **Per-type and per-tier metrics (`subtest_accuracy_table()`, `source_stratified_metrics()`):** More scientifically meaningful than a single number, exactly as Hardt recommends.
- **Pre-screening transparency (`[Pre-screening override]` tags):** Clean equalization — all baselines benefit from the same lightweight checks, reported honestly.
- **Stress test split:** Probes edge cases beyond the main evaluation.

### Potential shortcomings to address

#### A. Aggregation Vulnerability
**Risk:** Tier-weighted F1 aggregates across 14 hallucination types and 3 tiers into a single number. Different weighting schemes would produce different rankings (cf. OpenLLM Leaderboard normalization fiasco).

**TODO:**
- [ ] Test ranking sensitivity to tier weights: sweep weight vectors and check if tool rankings change
- [ ] Check IIA: does adding a new weak baseline change existing tool rankings?
- [ ] Report per-tier and per-type rankings as primary results, aggregate as secondary
- [ ] Consider reporting Plackett-Luce rankings (already in `evaluation/ranking.py`) as aggregation-robust alternative

#### B. External Validity
**Risk:** HALLMARK uses synthetic hallucinations. Do rankings generalize to real hallucinations found in actual papers?

**TODO:**
- [ ] Curate a small external validation set from real citation errors in published papers
- [ ] Check if HALLMARK tool rankings predict performance on real errors
- [ ] Document the external validity question explicitly in the paper's limitations

#### C. Low-Rank / Single-Factor Dominance
**Risk:** If all tools correlate with a single factor (e.g., "has API access to Crossref/Semantic Scholar"), HALLMARK may measure one capability, not diverse citation verification skills.

**TODO:**
- [ ] Compute PCA on the tool × hallucination-type score matrix
- [ ] Report effective rank and what the first principal component correlates with
- [ ] If low-rank, discuss whether tier structure adds orthogonal signal

#### D. Strategic Gaming / Water-Filling
**Risk:** Tools could optimize for easy Tier 1 types (fabricated_doi, future_date) and neglect hard Tier 3 types, inflating aggregate scores without improving on challenging cases.

**TODO:**
- [ ] Analyze whether current baselines show water-filling pattern (high Tier 1, low Tier 3)
- [ ] Consider reporting Tier 3 performance separately as a "hard subset" metric
- [ ] Evaluate if tier weights sufficiently penalize Tier 1-only strategies

#### E. Sample Size vs. Label Quality Tradeoff
**Risk:** HALLMARK has 2,525 entries. For some hallucination subtypes, counts may be small enough that rankings are unstable.

**TODO:**
- [ ] Compute per-subtype sample sizes and bootstrap confidence intervals on rankings
- [ ] Identify subtypes where n is too small for reliable ranking (apply Hardt's quadratic scaling rule)
- [ ] Consider expanding thin subtypes rather than improving annotation quality

#### F. Test Set Reuse / Adaptive Overfitting
**Risk:** If tools are iteratively developed against dev/test splits, adaptive overfitting applies. Theoretical bound degrades from O(√(log k / n)) to O(√(k log n / n)).

**TODO:**
- [ ] Track how many times each split has been evaluated (k parameter)
- [ ] Estimate remaining statistical budget using Dwork et al. bounds
- [ ] Reserve hidden split for final evaluation only; enforce in documentation

---

## Key References

- Hardt, M. (2025). *The Emerging Science of Machine Learning Benchmarks*. mlbenchmarks.org.
- Dorner, F. & Hardt, M. (2024). "Don't Label Twice: Quantity Beats Quality when Comparing Binary Classifiers on a Budget." ICML 2024. arXiv:2402.02249.
- Recht, B. et al. (2019). "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019.
- Zhang, L. & Hardt, M. (2025). "Beyond Arrow: From Impossibility to Possibilities in Multi-Criteria Benchmarking." arXiv:2602.07593.
