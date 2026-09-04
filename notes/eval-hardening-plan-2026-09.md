# Plan: eval hardening and bibtexupdater robustness (September 2026)

> **Superseded as a status document.** This is the plan as written on the
> morning of 2026-09-04, kept because it records what was believed before the
> work was done and several of its claims turned out wrong. For what was
> actually found, fixed, and left open, see
> [`eval-hardening-2026-09-04.md`](eval-hardening-2026-09-04.md).

Inputs: two peer sessions that ran HALLMARK's own cascade over 5,043 real
references from 267 NeurIPS-workshop submissions, plus a code audit of the
evaluation layer and the tool. Everything marked **verified** below was checked
in this repo or computed from its artifacts; peer measurements are attributed.

Repo health at the time of writing: 1,105 tests pass, ruff clean, mypy clean on
58 files. The defects below are not code-quality defects — they are measurement
and provenance defects, which is why a green suite does not catch them.

---

## Tier 0 — metric-correctness bugs in shipped code

Four analysis functions are silently wrong, and the wrongness is in the direction
that flatters the results. Each is verified by execution against the committed
artifacts. These come first because they change numbers already in print.

### 0.1 The two-sided p-value can only detect a difference in one direction

`hallmark/evaluation/metrics.py:1228-1231` computes
`p = min(1, 2 · P(bootstrap_diff ≤ 0))`. When tool A is *worse* than B every
bootstrap difference is ≤ 0, so `P → 1` and `p → 1.0` however large the gap. The
correct form is `2 · min(P(d ≤ 0), P(d ≥ 0))`.

`compare_tools` (`:1725`) and `paired_significance` (`:1431`) both enumerate
pairs by `sorted()` tool name with `i < j`, so **which tool is A is decided
alphabetically**. Roughly half of every pairwise leaderboard comparison is
reported as non-significant by construction. On a synthetic pair separated by
30.8 F1 points at n=800, the same data gives p = 0 named one way and p = 1 named
the other. Call sites: `cli.py:1037`, `scripts/run_all_baselines.py:445`,
`scripts/compute_bootstrap_ci.py:233`.

The in-code comment calls this "conservative … a deliberate design choice". It is
not conservative; it is directionally broken, and a reviewer who re-runs
`compare_tools` on the released artifact finds it in one command.

### 0.2 Per-type F1 and per-type FPR are non-metrics, and they are shipped

`per_type_metrics` (`:402`) groups entries by `hallucination_type`, so a type's
group contains only hallucinated entries. Precision is therefore always 1.0 and
FPR always 0.0, which makes per-type F1 the deterministic function
`2·DR/(1+DR)`. Verified across `data/v1.0/baseline_results/`: of 638 per-type
rows the only nonzero FPR is the `"valid"` pseudo-type bucket that
`h_type = entry.hallucination_type or "valid"` creates — every actual
hallucination-type row reports FPR 0.000, including for tools whose overall FPR
is 0.405.

`per_tier_metrics` was explicitly fixed for exactly this and documents the fix in
its docstring (`:366-374`): VALID entries go into *every* tier's FPR denominator.
The same fix was never applied to `per_type_metrics`, nor to `hard_subset_report`
(`:2438`), which inherits it. Either apply it or drop the two columns and report
DR with the Wilson interval `:452` already computes.

### 0.3 The tier-weight sensitivity analysis ignores all false positives

`ranking_stability.py:233` reads `tm.get("fpr", 0.0)`, but `per_tier_metrics`
emits the key `false_positive_rate` — verified: the keys present across every
committed result are `count, detection_rate, f1, false_positive_rate,
num_hallucinated, num_valid, precision`, and `fpr` appears nowhere. The default
fires every time, so `total_fp ≡ 0`, precision ≡ 1.0, and the quantity being
swept is weighted recall, not TW-F1. Two tools identical except that one has 25
false positives on 30 valid entries come out with identical ranges and
concordance 1.0.

Any "rankings are stable under tier reweighting" claim resting on this is
unsupported. The same key typo makes `per_tier_rankings(metric="fpr")` return 0.0
for every tool, and `metrics.py:2347` documents `"fpr"` as a valid argument.
`tier_weight_sensitivity` (`metrics.py:1458`) is a separate, correct
implementation that is not reported.

### 0.4 Abstention is free: coverage counts UNCERTAIN as covered

UNCERTAIN is dropped from the confusion matrix (`:177`), from TW-F1 (`:238`),
from ECE (`:563`) and from AUROC/AUPRC — but `coverage` counts it as covered
(`:2170`), so `coverage_adjusted_f1 = f1 × coverage` (`:2274`) applies no
penalty. A tool that abstains whenever unsure has its metrics computed on its
easy subset at full coverage.

This is live in a committed result, not hypothetical. DeepSeek-R1 on
`test_crossdomain` answers 68 of 500 entries — verified label counts are
UNCERTAIN 432, HALLUCINATED 64, VALID 4 — and the framework reports DR 0.953 at
coverage 1.000. Cross-tool DR comparison is invalid whenever abstention rates
differ, which they do by two orders of magnitude across the registry.

Fix by defining coverage to exclude UNCERTAIN, making `coverage_adjusted_f1`
bite, and reporting `(coverage, accuracy-on-answered, AURC)` as a triple. The
`eval_mode="aggressive"` path (`:1956`) is a good counterweight but is opt-in and
the README table does not carry it.

### 0.5 Table 1 carries no intervals, and the headline claim does not survive one

The main results table (`README.md:348-377`) is 18 rows × 6 metrics of bare point
estimates with no interval or test anywhere, while the repo contains stratified
bootstrap CIs, paired bootstrap with Cohen's *h*, Holm/Bonferroni/BH correction,
TOST equivalence and a power audit — all unreported. Only 2 of ~21 committed
`dev_public` result files carry any `*_ci` field.

The specific consequence: `README.md:357-358` bolds Claude Sonnet 4.6 (F1 0.840)
over Opus 4.7 (F1 0.824) as best independent tool. A paired bootstrap clustered
on source paper gives diff +0.0025, 95% CI [−0.0156, +0.0203], p = 0.775 — a
statistical tie. Reframing takeaway 1 from "Sonnet leads" to "Sonnet and Opus are
tied and lead the rest" is the stronger claim, because it survives scrutiny.

Two further problems with the intervals that do exist. The bootstrap is
stratified on hallucination type (`:1035-1052`), resampling VALID and
HALLUCINATED entries independently — but 218 normalized titles in `dev_public`
carry both a VALID and a HALLUCINATED entry, covering 550 of 1,119 entries
(49.2%), because 322 of the 414 perturbation-derived hallucinations sit beside
their unperturbed twin. The correct unit is the source paper; clustering there
widens the F1 interval 1.34× (0.0324 → 0.0433). **Every published interval is
roughly a third too narrow.**

And the committed `dev_public` results were computed against three different
ground truths — 7 files at 633/486, 6 at 582/486, 4 at 593/486 — while the
shipped `data/v1.0/dev_public.jsonl` is 606/513 and matches none of them. The
relabelling is documented (`docs/MISLABEL_AUDIT.md`); the consequence is not:
**Table 1 cannot be regenerated from the shipped data plus the shipped
predictions.** `results/llm_openai_dev_public.json` (FPR 0.405, F1 0.771 — the
README row) and `results/llm_openai_dev_public_ci.json` (FPR 0.171, F1 0.822) are
two incompatible numbers for the same tool on the same split, and the one
carrying the confidence interval is not the one in the paper.

### 0.6 Smaller, same class

`reuse_tracker.py:137` solves `k_max = ratio²/ln(n)`, which is 0.57–0.83 for
every shipped split, so `estimate_remaining_budget` returns 0 before the first
evaluation — the Dwork bound is being compared to `1/√n` with all constants set
to 1. `iia_violation_check` (`ranking_stability.py:290`) verifies that removing
an element from a score-sorted list leaves it sorted, a tautology, and never
tests the Plackett-Luce ranking, which is the one method here where IIA could
fail. `union_recall_at_k` is `{}` in all 67 released result files while detect@k
is advertised at `README.md:27` and `:318`. And `_build_comparisons`
(`ranking.py:139`) feeds `choix` 23,107 pairwise comparisons from 1,119
independent entries — a 21× overcount — of which 19.3% are exact ties broken by a
hash coin-flip (`ranking.py:172`) keyed so that a resampled entry always breaks
the same way, so the bootstrap CI never sees tie-break uncertainty.

---

## Tier 0b — provenance failures in released material

These four are cheap and they are what a reproducer hits first.

### 0b.1 The headline table is not reproducible from the repo

`README.md` "Headline cascade results (v1.1)" reports `test_public` aggressive at
DR 0.978 / FPR 0.456 / F1 0.882, and links `data/v1.2/baseline_results/` for the
full JSONs. Those JSONs report DR 0.992 / FPR 0.160 / F1 0.950. The FPR differs
by a factor of 2.85. No file under `results/` or `data/` reproduces FPR 0.456,
and no `v1.1` `baseline_results` directory exists on any branch. The public site
carries the README's numbers (`site/data/site_data.js` has 0.559, the table's
`dev_public` conservative FPR), so the site and README agree with each other and
disagree with every committed artifact.

Either the run behind the reader-facing numbers was never committed, or the JSONs
were regenerated without updating the README and site. Resolve by re-running the
cascade under pinned versions and republishing all three surfaces from one run.

### 0b.2 No result records which tool version produced it

`EvaluationResult` (`hallmark/dataset/schema.py:535`) has no `tool_version`
field, and `hallmark/evaluation/validate.py` never checks for one. All 47 files
in `data/v1.0/baseline_results/` carry no version, no split name and no
timestamp. The repo currently names three different bibtex-updater versions as
the one that produced results — `README.md:79` and `pyproject.toml:68` say
v1.2.0, `docs/walters_wilder_supplement.md:80` says 1.4.0, and the machine has
1.10.1 installed — while bibtexupdater 1.11.0 shipped on 2026-09-04 and changes
verdict-adjacent output.

Add `tool_version`, `split_name` and `run_timestamp` to `EvaluationResult`, make
`validate-results --strict` require them, and re-emit every reference result.
Report bibtexupdater at two pinned versions in the final table, so the co-design
is handled in the open rather than by a version that drifts under the paper.

### 0b.3 The staleness guard cannot work, and is switched off anyway

`scripts/check_results_freshness.py` decides staleness by comparing file mtimes.
Git does not preserve mtimes, so on any fresh clone the ordering is checkout
order, not causal order — the guard's signal is noise for everyone but the
author. Run here it reports 46 stale files off mtime differences of 0.06 s. CI
invokes it with `--warn-only` (`.github/workflows/tests.yml:48`) and the
real-repo pytest guard is `xfail(strict=False)` with the reason "released result
JSONs predate the relabel; regenerate them in a later stage"
(`tests/test_results_freshness.py:280`).

So the repo already knows its released results predate the May-2026 relabel, and
the mechanism that would enforce it is disabled at both ends. Replace mtime with
a content hash of the split file recorded inside each result, make CI fail on
mismatch, and regenerate.

### 0b.4 The co-design defence was never actually run

`results/bibtexupdater_no_prescreening_dev_public.json` reports DR 0.000,
FPR 0.000 and `mean_api_calls: 0.0` over 1,079 entries — the signature of
`fallback_predictions` firing because `bibtex-check` was not on PATH.
`harc_no_prescreening_dev_public.json` is identically null. The only real
prescreening ablation is `doi_only` (0.203 → 0.256 DR), and the "~5 pp Tier-1
lift" attributed to the layer generalises from that one baseline.

This is the load-bearing evidence for the pre-screening design. Run both
ablations for real and publish the rows beside the composite. The expected result
vindicates the current design — the status histogram already shows bibtexupdater
catching all 30 `future_date` entries itself — but it is presently an inference,
not a measurement.

---

## Tier 1 — the measurements that change what the paper claims

### 1.1 Precision at a realistic base rate

`test_public` is 62.5% hallucinated by construction. The wild base rate measured
by the peer run is indistinguishable from zero: over 5,043 real references, Stage
1 found a matching record for every reference it ruled on, and after
re-adjudication and hand audit **not one of the surviving accusations was a work
that does not exist**. Three were corrupt OpenAlex index records, two were false
positives verified against an independent index, two were genuine metadata
errors, two were unverifiable by construction.

Computed from the committed results, holding DR and FPR fixed:

| assumed prevalence | cascade precision | flags per true finding |
|---|---|---|
| 0.1% | 0.6% | 162 |
| 1% | 5.9% | 17 |
| 5% | 24.6% | 4.1 |
| 20% | 60.8% | 1.6 |
| 62.5% (benchmark) | 91.2% | 1.1 |

The benchmark's own prevalence is the only point on that curve where the
instrument looks good. This table is arithmetic over numbers already in the repo
and belongs in the main paper, not an appendix.

Report alongside it the absolute quantity a program chair can reason about:
**false accusations per 1,000 references**. Accusing an author of fabricating a
citation is a serious claim, and F1 does not let anyone price it.

### 1.2 Abstention as a first-class outcome, and risk–coverage curves

`coverage` in `hallmark/evaluation/metrics.py` currently means "the tool
responded", not selective-prediction coverage. There is no risk–coverage curve,
no AURC, no Brier, and no reliability diagram; ECE is the only calibration
number, and it is high (bibtexupdater 0.399, cascade aggressive 0.113).

Report accuracy on the covered set as a function of coverage, and calibrate **on
the HALLUCINATED predictions only** — the peer's wrong verdicts sat at 0.93–0.99
confidence, which an ECE averaged over a mostly-VALID, mostly-correct population
hides. The positive class is where the decision costs something.

### 1.3 A third substantive class: miscitation

Filed as issue #36. Four of the taxonomy's hallucination modes are explicitly
about real papers — `wrong_venue`, `preprint_as_published`, `partial_author_list`,
`arxiv_version_mismatch`. On the wild corpus, 63% of flags were "real work,
described wrong", 47% from venue and preprint status alone.

The label space does already offer `UNCERTAIN` (`hallmark/baselines/llm_agentic.py:102`,
`:155`), so the model is not forced into a binary. But `UNCERTAIN` is emitted on
tool-call exhaustion and the other give-up paths, which makes it definitionally
an abstention, and "the work exists and the venue string is stale" is a confident
determination, not a failure to determine. Routing it there would corrupt the
abstention rate, which is itself a reported metric.

A detector that correctly identifies a stale venue string is currently scored as
having found a fabrication. Any ranking over that target ranks a conflated
construct. Add the third class and report the taxonomy ablation: score with the
four real-paper modes folded into `miscitation` versus collapsed into
`HALLUCINATED`. If the ranking moves, the taxonomy is doing work the detectors
should be doing, and that is a finding either way.

### 1.4 Source availability is an uncontrolled confound

Same 119 entries, same model, same prompt, same code: **12% of flags cleared with
arXiv starved, 24% with arXiv answering.** Availability is worth a factor of two
on the outcome, so any live-API baseline score is partly a measurement of the
network that day and a re-run months later is not comparable.

The starvation was silent and self-inflicted: arXiv rate-limits per caller while
Stage 2 spent the budget per worker, so eight workers starved it within a minute
and the pipeline recorded "not found" for entries whose only confirming source
was dark. arXiv answered 21 of 377 calls unpaced and 101 of 101 paced. The fix is
PR #38 (`fix/agentic-per-service-rate-limit`), now on origin.

Every published agentic number predates that commit. Re-run the agentic
baselines under stated per-service budgets, log per entry which sources answered,
and report source availability as an experimental condition the way a seed is
reported. Then run the source-ablation grid: each source held out, and each
source *silently failing*, which is the realistic mode.

### 1.5 The generator shortcut probe

`scripts/analyze_shortcuts.py` and `scripts/audit_format_tells.py` already exist
and `tables/format_tells_audit.csv` is committed. Both peers independently named
this as the first experiment they would run. Re-run it against the current splits
and report the result in the paper whichever way it comes out — a negative result
is the strongest available answer to "show me a detector cannot exploit the
perturbation process", and a positive one is better found now.

### 1.6 Frozen reference date for the LLM paths

`hallmark/baselines/prescreening.py:29` already freezes
`_BENCHMARK_REFERENCE_YEAR = 2026` and threads a `reference_year` override
through `check_year_bounds` and `prescreen_entry`. The LLM and agentic paths have
no frozen reference date at all, so the model's training cutoff does the work:
`FUTURE_DATED` was the single largest residual class on the wild corpus at 42.2%
after sources were healthy, rejecting entries that are recent but past.

Thread the value the prescreener already has into the LLM prompts rather than
adding a second source of truth, and report `future_date` separately regardless —
a class whose size moves with a model refresh should never sit inside an
aggregate.

---

## Tier 2 — tests

Ordered by the real bug each would have caught.

1. **Tri-state source contract, asserted.** Every source tool returns
   found / not-found / **unavailable**, and no checker returns "not found" when a
   required source errored. The interface alone will be honoured inconsistently;
   the assertion is what makes it hold. This is the test that would have caught
   the entire arXiv-starvation night.
2. **Source-failure invariance, as a hard gate.** Injecting a failure into any
   source may move a verdict toward abstention and must never move one from VALID
   to HALLUCINATED. Implement with a fault-injecting fake per source; needs no
   network.
3. **Venue-form invariance (metamorphic).** Rewriting a venue to an equivalent
   form must not change the verdict: `arXiv preprint arXiv:NNNN.NNNNN` ≡
   `arXiv:NNNN.NNNNN` ≡ `CoRR abs/NNNN.NNNNN`; `ICLR` ≡ the expanded name ≡ the
   name with an acronym gloss ≡ the brace-protected form Zotero writes.
   bibtexupdater fails the last: `The {{Twelfth International Conference}} on
   {{Learning Representations}}` scores 0.09 against ICLR and produces a hard
   `VENUE_MISMATCH` on a correctly cited paper, because no LaTeX pass runs before
   normalisation.
4. **Corrupt-index-record regression.** OpenAlex serves records with the correct
   DOI, the correct author list and the wrong title: `10.48550/arxiv.2307.16789`
   (ToolLLM), `10.48550/arxiv.2212.08073` (Constitutional AI),
   `10.48550/arxiv.2106.09685` (LoRA). Title similarity 0.384 / 0.348 / 0.385, so
   the blended `0.7·title + 0.3·author` lands at 0.568 / 0.543 / 0.570, above the
   0.50 abstention bar; and `wrong_paper_signature` cannot fire because it
   requires the authors *not* to match, and here they match perfectly. **A
   corroborating author list defeats the only guard present** — and synthetic
   perturbation will never produce that shape. Fixture from the recorded payloads
   in bibtexupdater `8e3c8d1`; **pin them, do not fetch at test time**, or
   OpenAlex repairing the records makes the test pass for the wrong reason.
5. **Ground-truth drift.** Hash the label files; fail CI on an unannounced
   change. Assert every entry's type is in the taxonomy and that tier is a pure
   function of type.
6. **Any regex scanning text that contains identifiers gets a test with a DOI in
   it.** Two independent bugs in one night shared this shape: a `[^.]` gap pattern
   that could not span `2307.16789`, and a `\b` that matched "arxiv" inside
   `10.48550/arxiv.2106.09685` so every OpenAlex failure also reported arXiv as
   failed. Both passed their author's inspection.
7. **Recorded HTTP fixtures for the full agentic path**, so a scored run is
   deterministic. It is not today.
8. **Stamp cache entries with source health at write time** and allow
   invalidation on it. A cache written during an outage is otherwise trusted
   forever.

---

## Tier 3 — bibtexupdater

Ranked by impact. Line references are `bibtexupdater/src/bibtex_updater/`.

1. **`venues_match` must abstain, not mismatch, on unrecognised venue pairs** —
   `fact_checker.py:1893-1958`. The terminal branch returns `MISMATCH`; the alias
   map covers ~45 ML/CS venues and everything else falls to a 0.70 token-sort.
   ISO-4 abbreviated journal names — the standard form in a large share of real
   `.bib` files — fail: `ACM Trans. Graph.` scores 0.70 against `ACM Transactions
   on Graphics`, `Proc. Natl. Acad. Sci. U.S.A.` scores 0.60. These are correct
   citations of real papers reported as a problem at p_valid 0.11. Worse, this
   fires on the single-source pre-cascade path (`_id_anchored_field_mismatch`,
   `:3123-3184`), which returns before the cross-source consensus check and its
   FPR guard ever run. `venue_mismatch` is 66 of 1,112 verdicts on dev_public and
   the second-largest problem bucket. Add an LTWA/ISO-4 expansion route and an
   ISSN / `venue_source_id` / `venue_key` identity route (the union-find logic
   already exists at `:4360-4374`), and return `NON_COMPARABLE` when neither side
   canonicalises.
2. **Give `_detect_chimeric_title` an exact-match veto** — `:4092-4172`. It never
   checks whether any candidate actually matches the entry title, so a Crossref
   hit at score 1.0 does not suppress it, and it returns the tool's most severe
   label at a hardcoded 0.95 with `best_match=None` and no surviving evidence.
   One `if` at the top of the function. Currently fires 0/1,112, so the risk is
   unrealised — but the guard is missing and the verdict is unauditable.
3. **Fix the `preprint_only` polarity/confidence mismatch** —
   `calibration.py:62-72` lists it as clearly-correct at base confidence 0.88,
   while `P_VALID_PROBLEM_STATUSES` (`:474`) gives it problem polarity. The
   result is p_valid 0.060 — the most confident "invalid" verdict the system can
   emit — from a prior that means the opposite. `doi_not_found` has the mirror
   defect. Add a unit test asserting every status in `P_VALID_PROBLEM_STATUSES`
   draws its base confidence from a problem tier.
4. **Record the five silent lookup failures in `sources_failed`** —
   `:1631-1651`, `:1680-1704`, `:1758-1772`, `:3245-3248`, `:2025-2036`. All five
   swallow exceptions and return `None`, so the entry is stamped as a *clean*
   abstention and `coverage_incomplete` under-reports. That flag is exactly what
   HALLMARK keys on to tell a throttled miss from a real one
   (`hallmark/baselines/bibtexupdater.py:438-452`). Same change should make
   `_structured_author_recheck` (`:3401-3491`) return `UNCONFIRMED` rather than
   let an `AUTHOR_MISMATCH` stand when Crossref was unreachable — the one place
   in the tool where a transient failure hardens into a flag.
5. **Add the retraction check** — `:1631-1651` already fetches the Crossref
   message; read `update-to` and emit a `RETRACTED` status. No new API, no new
   rate-limit budget, and it catches an integrity failure the tool is blind to.

Also worth doing: `_check_claimed_venue_exists` (`:2236-2292`) applies a weaker
corroboration gate than its sibling at `:4340-4349`, so a real-but-obscure venue
unknown to both DBLP and OpenAlex `/sources` can be minted `NONEXISTENT_VENUE` —
COLM is absent from `EXPANDED_VENUE_ALIASES` and a correctly cited COLM 2024
paper was called fabricated at 0.97 confidence on the wild corpus. Absence from
an index is grounds to abstain, not evidence a venue does not exist. Any thinly
indexed venue is exposed.

Auditability, not correctness (issue #37): `mismatched_fields` conflates
abstentions with findings — 956 abstentions against 68 hard mismatches on the
wild corpus, 14:1, all rendering identically as `Mismatched: ['venue']`.
HALLMARK's verdict does **not** read it (`bibtexupdater.py:436` derives the label
from `status` alone; `mismatched` only reaches the `reason` string at `:481`), so
no metric is affected — but any error-mode taxonomy, hand-audit sampling frame or
per-type discussion drawn from reason text inherits the conflation. Read
`unconfirmed_fields` alongside it and render the two distinctly.

---

## Tier 4 — external validity

Patrik has ruled that the wild corpus is shared and that the finding goes in the
paper as a joint result, credited.

`~/Documents/GitHub/interpscience-bib-check/corpus_export.jsonl`: 5,043
references, 3.9 MB, carrying bibliographic fields, Stage-1 status, which sources
answered, the reason string, the classification and the Stage-2 verdict where one
exists. De-identification verified independently here — no `cited_by`, no
`"paper"` key; the keys are `bibtex_type`, `classification`, `fields`,
`stage1_reason`, `stage1_sources_answered`, `stage1_status`, `stage2`, `uid`.

Two caveats that must travel with it. The Stage-1 verdicts were produced by
bibtex-updater 1.10.3 and are being regenerated against 1.11.0, with 806 of the
956 venue abstentions expected to confirm outright — treat venue-derived columns
as stale. And source conditions differ across rows: some Stage-2 verdicts come
from the starved pass and some from the paced re-run. Each verdict knows which,
and they must not be pooled without stratifying, because the clearance rate
doubles between conditions.

The claim to make plainly, without softening: HALLMARK's own cascade, run over
5,043 real references, found **zero fabricated works**, and the flag list was
dominated by citation-style artefacts and tool bugs. A benchmark paper reporting
that its instrument produces mostly false positives at the wild base rate is
making the strongest honesty claim available to it, and it motivates the
abstention-first framing better than any synthetic result can.

One example is worth a figure on its own. For the entry "Chain-of-thought is not
explainability", the model's stated grounds for calling it fabricated was that
the closest real work it found is "Is Chain-of-Thought Really Not
Explainability? …" — a rebuttal that names the cited work in its own title. A
rebuttal is proof its target exists. The strongest available counter-evidence was
read as confirming the opposite conclusion. The evidence was not missing; it was
present and inverted.

---

## Already done — do not redo

The Hardt benchmark-science plan (`notes/hardt-shortcomings-plan.md`) is
implemented apart from its item 10. Present and working: `ranking_sensitivity_analysis`
and `iia_violation_check` (`ranking_stability.py:184`, `:310`), `per_tier_rankings`
and `ranking_concordance` (`metrics.py:2337`, `:2392`), `pca_analysis`
(`factor_analysis.py:87`), `water_filling_analysis` (`water_filling.py:144`),
`subtype_power_audit` (`power.py:121`), `compute_reuse_budget`
(`reuse_tracker.py:50`), `per_subtype_ranking_stability` (`ranking_stability.py:76`),
`hard_subset_report` (`metrics.py:2438`). Also already present as scripts:
`analyze_shortcuts.py`, `audit_format_tells.py`, `compute_codesign_bound.py`,
`compute_bootstrap_ci.py`, `analyze_synthetic_vs_real.py`,
`cross_verify_valid_entries.py`, `analyze_seed_sensitivity.py`.

Item 10 — the external validation set — is the gap, and Tier 4 fills it with real
data rather than curated incidents.

## Open provenance questions

- Which of `results/bibtexupdater_dev_public.json` (DR 0.946) and
  `data/v1.0/baseline_results/bibtexupdater_dev_public.json` (DR 0.865) is
  canonical, and how the other is signposted.
  `results/reviewer_experiments/results_manifest.md:83` adds a third pair from
  the relabelled ground truth (0.969 / 0.193).
- Whether keyed HaRC returns to the main table. `harc.py:132` takes
  `api_key: str | None = None` and the registry passes none, while
  `bibtexupdater.py:332-336` self-serves `S2_API_KEY` from the environment. HaRC
  is then excluded from the main table for Semantic Scholar throttling
  (`README.md:377`), but `data/v1.0/baseline_results/harc_with_s2key_dev_public.json`
  is a keyed run at coverage 1.0. The exclusion is defensible on the merits
  (DR 0.209 against bibtexupdater's 0.865) but the stated reason is contradicted
  by a committed artifact.
- `results/walters_wilder_llm/41598_2023_41032_MOESM3_ESM.xlsx` is the input to
  `scripts/ingest_walters_wilder.py`, untracked and not ignored, so the ingest is
  not reproducible by a third party. It is CC BY 4.0; commit it with attribution
  or document the download URL.

---

## Status — 2026-09-04

Committed as `6b1abd8`: all four Tier 0 metric fixes plus the provenance fields,
with 11 regression tests in `tests/test_metric_correctness_regressions.py`. Nine
of the eleven fail against the previous code; the other two are sanity guards on
behaviour that was already right. Suite is 1,116 passing, ruff and mypy clean.

Two corrections to what is written above, both found while implementing.

`split_name` is present in 46 of 47 reference results — the earlier claim that it
was missing came from reading the key `split` rather than `split_name`. Only
`tool_version` and a run timestamp were genuinely absent, and both are now on
`EvaluationResult` alongside `split_sha256`.

The released results are also not as detached from the shipped data as feared.
`llm_openrouter_deepseek_r1_dev_public.json` and the v3 file both record
n=1119, 606 hallucinated, 513 valid, matching `data/v1.0/dev_public.jsonl`
exactly, and both reproduce to three decimals when recomputed from the shipped
predictions. The divergence is between `results/` and
`data/v1.0/baseline_results/`, not between the released directory and the data.
`results/llm_openrouter_claude_sonnet_4_6_dev_public_predictions.jsonl` is a
different run from the one behind the released Sonnet numbers (recomputing gives
DR 0.916 against the released 0.781), so the two directories hold different runs
under the same names.

### What the fixes move, measured

Detection rate, FPR and F1 are unchanged — verified by recomputing DeepSeek-R1
and DeepSeek-V3 on `dev_public` and matching the released values to three
decimals. Coverage now falls below 1.0 wherever a tool abstained (0.984 and
0.981 on the two runs that have UNCERTAIN predictions), which is the intended
change.

Per-type metrics move a great deal, and the direction needs a decision. With the
valid entries borrowed into each type's denominator, DeepSeek-R1's per-type F1
falls from a 0.844–1.000 band to 0.129–0.318, and every type now reports the
tool's overall FPR of 0.623 rather than 0.000. The new numbers are correct, but
with roughly 30–120 hallucinated entries per type against 513 valid ones,
precision is dominated by the shared false-positive count, so per-type F1 now
largely tracks how common a type is rather than how well the tool handles it.

That is an argument for reporting per-type **detection rate with its Wilson
interval** as the primary per-type number — `per_type_metrics(compute_ci=True)`
already computes it — and treating the corrected F1 and FPR as secondary, or
dropping those two columns entirely. Reporting FPR 0.000 and a deterministic
`2·DR/(1+DR)` was never an option; which of the two replacements to publish is.

### Next, in order

1. Add `tool_version`, `split_sha256` and `run_timestamp` to the writers that
   emit reference results, and make `validate-results --strict` require them.
2. Replace the mtime arm of `check_results_freshness.py` with the recorded split
   hash, then drop `--warn-only` from `.github/workflows/tests.yml:48` and the
   `xfail` at `tests/test_results_freshness.py:280`.
3. Reconcile `results/` against `data/v1.0/baseline_results/` and retire one.
4. Regenerate the reference results, the README table and the site from one run.
5. Base-rate/precision table and the wild-corpus case study (Tier 1.1, Tier 4).
6. bibtexupdater: venue abstention, ISSN identity, and the `preprint_only`
   calibration polarity (Tier 3, items 1 and 3).

### The ratchet measures a different population in CI than locally (issue #40)

`scripts/verify_subtests.py:54` puts `test_hidden` in `DEFAULT_SPLITS`, and
`.gitignore:42` ignores `data/hidden/`, so the split exists only on a machine
holding the full dataset. Scanning the CI-visible splits gives 2,572 entries and
exactly 99 mismatches; including the hidden split gives 3,026 and 211.
`MAX_MISMATCHES = 99` is precisely the CI-visible count, so the repo's one real
data-integrity ratchet passes in CI by construction while 112 sub-test
ground-truth mismatches in the hidden split are invisible to every CI run — in
the split whose labels no external contributor can inspect.

This is the same shape as the freshness guard and the `[Error fallback]`
records: the mechanism exists, it reports, and the report reaches nobody. The
CI log for the run on this branch prints `ERROR: Freshness check FAILED: 46
stale file(s)` and the job passes, because `.github/workflows/tests.yml:48`
carries `--warn-only`. That invocation also passes `--results-dir
data/v1.2/baseline_results` with `--version v1.0`: the results path was updated
in the v1.0 → v1.2 rename and the version flag was not, so the guard resolves
split files under a directory the rename removed. Fixing the flag is a
one-token change and belongs with the hash-based rework, not before it.
