# Multi-Defect Entries: What HALLMARK Scores When More Than One Thing Is Wrong

**Status:** design rationale / FAQ. Written 2026-08-26; revised same day after
auditing which diagnostic metrics are actually populated in our results.
**Audience:** paper reviewers, benchmark contributors, and anyone asking: "each
entry has a single `hallucination_type` — does the eval still measure real-world
detection, where a bad reference often has several things wrong at once?"

## A worked example first

Take a `hybrid_fabrication` entry: the DOI is real, but the title and the
authors are both wrong. One entry, two defects.

A tool examines it. It never notices the title problem, but it does notice the
author mismatch, so it outputs `HALLUCINATED`.

**It gets full detection credit.** The scorer
(`build_confusion_matrix()`, `hallmark/evaluation/metrics.py:139`) compares only
the tool's label against the ground-truth label. It never asks *which* defect
the tool found, whether it found *all* of them, or whether it found the
*hardest* one. Catching at least one defect is enough.

This mirrors deployment: once a screening tool flags a reference for any
reason, a human reviews it. The flag is the decision that matters, so the
benchmark scores the flag.

Sections: §1 does the benchmark contain multi-defect entries (yes); §2 what
each metric does with them; §3 which diagnostic metrics are actually live in
our results (one of the three is not); §4 why results transfer to real-world
hallucinations; §5 how the numbers in §1 were measured; §6 limitations;
§7 the measured multi- vs single-defect detection comparison (added
2026-08-31) — which confirms §4(b)'s direction but shows the aggregate gap is
mostly a provenance-composition effect.

---

## 1. One *type* label does not mean one *defect*

Each hallucinated `BenchmarkEntry` (`hallmark/dataset/schema.py`) has:

- **One `hallucination_type`.** This names the perturbation the generator
  applied — the *cause* we injected — not a claim that only one field is wrong.
- **A multi-valued `subtests` dict.** Six per-field checks (`doi_resolves`,
  `title_exists`, `authors_match`, `venue_correct`, `fields_complete`,
  `cross_db_agreement`), each True/False/None. This is where every broken
  field is recorded. The expected pattern per type is declared in
  `EXPECTED_SUBTESTS` (`hallmark/dataset/schema.py:91`).

Some types break several fields *by definition*:

| Type | Broken fields (excluding `cross_db_agreement`, see §5) |
|---|---|
| `hybrid_fabrication` | title + authors (real DOI, wrong metadata) |
| `merged_citation` | authors + venue |
| `plausible_fabrication` | title + authors (+ often identifier) |
| `partial_author_list` | authors, often + one more |

**Measured distribution** of defective fields per hallucinated entry
(reproduce with `uv run python scripts/analyze_defect_multiplicity.py`):

| Split | 1 | 2 | 3 | 4 | ≥2 (share) |
|---|---|---|---|---|---|
| dev_public (606 hall.) | 437 | 116 | 52 | 1 | 169 (27.9%) |
| test_public (519 hall.) | 362 | 117 | 39 | 1 | 157 (30.3%) |
| stress_test (121 hall.) | 75 | 46 | 0 | 0 | 46 (38.0%) |

The aggregate hides a large split by provenance — the synthetic strata are
mostly single-defect, the *empirically sourced* strata are mostly multi-defect:

| Generation method | dev_public | test_public |
|---|---|---|
| `perturbation` | 56/410 (13.7%) | 55/361 (15.2%) |
| `llm_generated` | 18/90 (20.0%) | 18/65 (27.7%) |
| `adversarial` | 55/60 (91.7%) | 58/59 (98.3%) |
| `real_world` | 40/46 (87.0%) | 26/34 (76.5%) |

So the benchmark is not "one error per entry". It is "one *attributed cause*
per entry", and the entries drawn from real hallucinations look exactly like
the multi-defect citations the concern is about.

## 2. What each metric does with a multi-defect entry

**Detection Rate / FPR / F1 / tier-weighted F1: catching any one defect earns
full credit.** As in the worked example — only the binary labels are compared.

**Tier weighting follows the injected type, and this is safe.** The worry: what
if a Tier-3 entry carries an easy Tier-1 giveaway, letting tools earn the 3×
hard-tier weight for spotting something trivial? The data says it doesn't
happen — 47 of 52 dev `near_miss_title` entries fail *only* the title check.
Where an entry does have multiple defects they are of comparable difficulty: a
`plausible_fabrication` breaks title *and* authors, but noticing either
requires the same hard capability (establishing the paper does not exist).

**Type diagnosis is a separate score that cannot cost detection credit.** See
§3 for what these two functions do and where they come from. A wrong type guess
never turns a true positive into a miss.

**Subtest accuracy would be the "did you find *everything*" score — but it is
currently inert.** See §3.

| Question | Metric | Status in our results |
|---|---|---|
| Would this reference get flagged? | DR / FPR / F1 / TW-F1 | live, headline |
| Can the tool name the failure mode? | `hallucination_type_accuracy`, `type_confusion_matrix` | live for LLM baselines (53/219 files) |
| Did it localize every bad field? | `subtest_accuracy_table` | **inert — 0/219 files** |

## 3. The three diagnostic functions: origin, inputs, and whether they run

### `subtest_accuracy_table()` — `metrics.py:857`

**Origin.** Introduced in `7094a6e` ("extend benchmark taxonomy, populate all
types, add ECE + diagnostic metrics", PR #1) — part of the original
HumanEval-inspired multi-criteria design.

**What it computes.** For each of the six `SUBTEST_NAMES`, it walks entries that
have that subtest in ground truth, finds the tool's `Prediction.subtest_results`
for the same key, and tallies accuracy plus a TP/FP/TN/FN breakdown. Entries
where the tool reports `None` for a subtest are skipped, as are entries with no
prediction.

**Where its input comes from — and why the table is empty.** It reads
`Prediction.subtest_results`, which the *tool wrapper* must populate. Only five
wrappers write that field at all (`doi_only`, `hallucitechecker`, `prescreening`,
`cascade`, `checkifexist`), and `doi_only` writes just `doi_resolves`.
Critically, `llm_verifier.py` — the wrapper behind every LLM baseline in the
paper — **never sets it**.

I scanned all 219 prediction files under `results/`: **none** has a populated
`subtest_results`. So although the function is implemented, unit-tested
(`tests/test_metrics.py:401`), and wired into the CLI (`hallmark/cli.py:829`),
it currently prints nothing for our experiments. Treat it as available
infrastructure, not as evidence. Any claim that HALLMARK *measures* per-field
defect localization requires first emitting `subtest_results` from
`llm_verifier.py`; until then the honest statement is that the benchmark
*encodes* multi-defect ground truth but our runs do not yet score against it.

### `hallucination_type_accuracy()` — `metrics.py:1781`

**Origin.** Added in `c81debe` ("DB-first cascade with hallucination-mode
diagnosis + dual-mode eval"), to score the cascade's stage-2 diagnosis step.

**What it computes.** Among ground-truth-hallucinated entries that the tool
*correctly flagged*, the fraction whose `predicted_hallucination_type` equals
the injected `hallucination_type`. It skips: valid entries, entries the tool
missed, and entries where the tool gave no type. It returns `overall`,
`tier_1/2/3`, `main_types_only`, `stress_test_only`, and the counts
`num_evaluated` / `num_correct`; partitions with no qualifying entries return
`NaN` rather than 0.0, so "no data" is distinguishable from "all wrong".

Because the denominator is *detected* entries only, this is a conditional
metric — "given that you caught it, did you know what it was?" — which is why
it cannot penalize detection.

**Where its input comes from.** `Prediction.predicted_hallucination_type`, set
by seven wrappers including `llm_verifier.py` (parsed out of the model's JSON
verdict). This one *is* live: 53 of 219 prediction files populate it, e.g.
1099/1119 predictions in the dev-split Qwen3-14B run. Coverage is not 100%
within a file because malformed model replies fall back to the regex salvage
path, which keeps only label and confidence.

### `type_confusion_matrix()` — `metrics.py:1857`

**Origin.** Same commit as above (`c81debe`).

**What it computes.** A full contingency table: rows are the 14 hallucination
types plus `valid`; columns are the 14 predicted types plus `VALID`,
`UNCERTAIN`, and `HALLUCINATED_unknown` (flagged but no type given). Missing
predictions are counted in the `VALID` column, consistent with the conservative
default used everywhere else.

**Why it matters here.** It is the audit trail for the single-label scoring
limitation. On a multi-defect entry a *defensible* alternative diagnosis — e.g.
calling a `hybrid_fabrication` a `fabricated_doi` — is scored as wrong by
`hallucination_type_accuracy`, so that headline number is a floor. The
confusion matrix shows *where* the mass went, letting a reader distinguish
"confused two plausible readings of the same broken entry" from "had no idea".

**How both reach the results.** `evaluate()` calls them unconditionally
(`metrics.py:2291-2292`) and stores the output in `EvaluationResult.type_accuracy`
and `.type_confusion`. For baselines that never set a predicted type, the
accuracy dict comes back with `num_evaluated=0` and `NaN` rates.

## 4. Why results transfer to real-world (multi-defect) hallucinations

The concern: real hallucinated references — especially LLM-fabricated ones —
are usually wrong in several fields at once. Does performance on mostly
single-defect synthetic entries predict performance there? Three answers:

**(a) Real multi-defect hallucinations are in the benchmark, scored
separately.** `generation_method="real_world"` entries are actual
hallucinations harvested from published papers (46 dev, 34 test), dominated by
`plausible_fabrication`, and 87% / 77% of them break ≥2 fields — exactly the
profile the concern describes. `per_generation_method_metrics()`
(`metrics.py:457`) reports each tool's detection rate on this stratum on its
own, so "synthetic performance predicts real performance" is *checked per tool
in the results table*, not assumed: compare a tool's `real_world` rate against
its `perturbation` rate. Agreement validates transfer; divergence measures the
sim-to-real gap. The `adversarial` stratum (92% / 98% multi-defect) gives a
second, harder multi-defect comparison point.

**(b) Single-defect entries are the hard case, so headline scores are a lower
bound.** *Measured in §7: the direction holds for 25 of 27 model x split cells
and no model is significantly worse on multi-defect entries — but the size of
the gap is mostly composition, not defect count.* Every additional defect is
one more chance for the tool to trip over the entry. An entry with one defect is therefore the hardest version of itself;
a real citation carrying that defect *plus others* is easier to catch. A
benchmark whose synthetic strata are ~86% single-defect understates, rather
than overstates, detection on the real-world mixture — the safe direction for a
benchmark's claims. The false-positive side transfers directly, since valid
entries are real scraped references in both worlds. (Tier-weighted F1 is
deliberately exempt from this argument: it measures a capability ceiling —
"can you catch a Tier-3 defect when it is the *only* tell?" — not deployment
yield. Both numbers are reported; they answer different questions.)

**(c) One injected cause per entry is what makes the per-type table
trustworthy.** If entries bundled several unrelated defects, per-type detection
rates would be confounded: a "detected" `near_miss_title` entry might really
have been caught via an incidental DOI error, silently inflating apparent
hard-type capability. Injecting one cause at a time isolates *which defect
classes a tool can detect* — and that profile is what predicts mixture
performance, since a real citation is caught precisely when the tool catches at
least one of its constituent classes.

## 5. How the §1 numbers were measured

Reproducible via `scripts/analyze_defect_multiplicity.py` (`--json` for
machine-readable output). Method:

1. **Load** `data/v1.2/{dev_public,test_public,stress_test}.jsonl`, dropping
   rows whose `bibtex_key` starts with `__canary__` (the watermark entries that
   `load_entries()` also filters). This is why stress_test is 121 here and 122
   in `CLAUDE.md`. `hidden_test` is not in `data/v1.2/` and was not measured.
2. **Keep** entries with `label == "HALLUCINATED"`.
3. **Count defective fields** as `sum(1 for v in subtests.values() if v is False)`.
   The `is False` test matters: `None` means *not applicable* (e.g.
   `doi_resolves` on an entry with no DOI) and must not count as a defect.
4. **Exclude `cross_db_agreement`** — justified below.
5. **Tabulate** the count distribution, overall and grouped by
   `hallucination_type` and `generation_method`.

### Why `cross_db_agreement` is excluded

`EXPECTED_SUBTESTS` declares `cross_db_agreement: False` for all 14 types, and
the data confirms it holds with no exceptions: **606/606, 519/519, and 121/121**
hallucinated entries have it False, in dev, test, and stress respectively. (The
script re-verifies this on every run and prints a warning if the assumption ever
breaks.)

A field that is constant across the entire hallucinated class carries zero
information about *how many* fields are defective. Including it would add
exactly 1 to every entry's count, shifting the whole distribution one place
right and making genuinely single-defect entries appear to have two — e.g. dev
becomes `{2: 437, 3: 116, 4: 52, 5: 1}`, with "0 single-defect entries", which
would be an artifact, not a finding. Semantically it is a restatement of the
label ("some database disagrees, i.e. this entry is hallucinated") rather than
an independently observable broken field. The script prints both distributions
so the choice is auditable rather than buried.

Note it is *not* constant on valid entries (18 of 513 dev and 18 of 312 test
valid entries also have it False — real references over which databases
genuinely disagree), so the field is not vacuous overall; it is just
uninformative *within* the hallucinated class, which is the only class this
measurement ranges over.

## 6. Known limitations and future work

- **Per-field localization is unmeasured.** `subtest_accuracy_table()` is
  implemented and tested but no baseline we run emits `subtest_results`
  (0/219 prediction files). Populating it in `llm_verifier.py` is the
  highest-value fix if we want to claim HALLMARK *measures* multi-defect
  localization rather than merely encoding it.
- **Type accuracy is single-label.** On multi-defect entries a defensible
  alternate diagnosis counts as wrong, so reported type accuracy is a floor for
  the multi-defect types; read it alongside `type_confusion`. A multi-label
  diagnosis metric (predicted type set vs. failing-subtest set) would fix this;
  it is not currently planned in any roadmap doc.
- **The synthetic defect-count mix is a design choice, not a field estimate.**
  The ~86% single-defect rate in the `perturbation` stratum was chosen for
  clean attribution; it is not an estimate of the real-world defect-count
  distribution (which our own `real_world` stratum puts at ~80% *multi*-defect).
  Deployment-yield estimates should reweight per-type detection rates by an
  externally measured defect mixture rather than reading the headline detection
  rate as a field estimate.
- **The lower-bound argument assumes defects don't interfere.** §4(b) assumes a
  tool's checks do at least as well in combination as alone. A tool could in
  principle be *distracted* by an easy defect and downgrade its verdict. §7 now
  tests this directly across 9 zero-shot models and finds no case of
  significant interference, so the assumption survives; the `real_world` and
  `adversarial` stratum comparisons in §4(a) remain the standing guard.
- **The defect-count effect is barely identified.** §7's within-type estimate
  rests almost entirely on `partial_author_list`, the one type with both
  single- and multi-defect entries in usable numbers (11-12 vs 20). Every other
  type is ~all-single or ~all-multi by construction, so "does an extra defect
  help?" cannot be separated from "is this type easier?" anywhere else. Any
  future generator change that deliberately varied defect count *within* a type
  would make this measurable.

## 7. Measured: do tools actually do better on multi-defect entries?

**Added 2026-08-31.** §4(b) argued that single-defect entries are the harder
case, making headline detection rates a lower bound on real-world (mostly
multi-defect) performance. That was an argument, not a measurement. This
section measures it.

Reproduce with:

```
uv run python scripts/analyze_multi_defect_detection.py --strata \
  --out results/multi_defect/multi_vs_single_defect_detection.json
```

**Method.** For every zero-shot model with a committed per-entry prediction
dump, split the hallucinated entries into *single-defect* (exactly one subtest
False, excluding `cross_db_agreement` per §5) and *multi-defect* (≥2), and
compare detection rate. Scoring follows `build_confusion_matrix()` exactly:
`UNCERTAIN` leaves the denominator, a missing prediction counts as a miss.
Verified against the CLI — the script reproduces `hallmark evaluate` to the
digit (Claude Haiku 4.5 dev: DR 0.8663 / FPR 0.4444; Gemini 2.5 Pro dev: DR
0.476 / FPR 0.050).

### 7.1 Headline: the direction holds, with no sign of interference

Raw contrast, discriminating models only (FPR < 50%; see §7.4 for why the rest
are excluded from this reading):

| Split | Model | FPR | DR 1-defect | DR ≥2-defect | Δ (95% CI) | Fisher p |
|---|---|---|---|---|---|---|
| dev_public | Gemini 2.5 Pro | 5.0% | 43.2% | 58.8% | **+15.6** [+6.6, +24.2] | <.001 |
| dev_public | Llama 4 Maverick | 14.6% | 57.4% | 71.6% | **+14.2** [+5.6, +22.0] | .002 |
| dev_public | Claude Haiku 4.5 | 44.4% | 86.0% | 88.2% | +2.1 [−4.3, +7.5] | .595 |
| test_public | GPT-5.4 | 22.4% | 74.6% | 86.0% | **+11.4** [+3.8, +18.0] | .004 |
| test_public | Claude Haiku 4.5 | 46.8% | 85.9% | 89.8% | +3.9 [−2.7, +9.4] | .256 |
| stress_test | Claude Haiku 4.5 | — | 66.7% | 97.8% | **+31.2** [+17.9, +42.6] | <.001 |

Across **all 27 model × split cells** (17 distinct models over 4 splits) the
multi-defect arm is at least as good in 25; the two exceptions are Qwen3-14B on
test_public (−0.1 pp, p=1.0) and DeepSeek R1 on crossdomain (−11.5 pp, p=.30,
and see §7.4 — only 43 of 300 entries were scored). **No model is significantly
worse on multi-defect entries anywhere.** Sign test over models: p=.008 on
dev_public, p=.021 on crossdomain.

This settles the §6 "defects might interfere" worry in the benchmark's favour:
we find no case where an additional defect distracts a model into downgrading
its verdict.

### 7.2 But the size of the gap is provenance, not defect count

The raw contrast is badly confounded. Defect count is nearly a deterministic
function of `hallucination_type` (§1): `plausible_fabrication`,
`hybrid_fabrication` and `merged_citation` are 100% multi-defect;
`wrong_venue`, `future_date`, `nonexistent_venue`, `arxiv_version_mismatch` are
100% single. Multi-defect entries are also concentrated in the `adversarial`
and `real_world` strata (§1), which these models flag readily.

Mantel-Haenszel pooled risk differences, adjusting two different ways:

| Split | Model | Raw Δ | MH within *type* | MH within *generation method* |
|---|---|---|---|---|
| dev_public | Gemini 2.5 Pro | +15.6 | **+17.7** [+6.7, +28.7] | −1.4 [−12.7, +9.8] |
| dev_public | Llama 4 Maverick | +14.2 | **+20.0** [+6.9, +33.1] | +2.7 [−8.6, +14.0] |
| dev_public | Claude Haiku 4.5 | +2.1 | +10.5 [−4.9, +26.0] | −0.2 [−8.2, +7.9] |
| test_public | GPT-5.4 | +11.4 | **+25.2** [+10.1, +40.3] | −1.4 [−11.7, +9.0] |
| test_public | Claude Haiku 4.5 | +3.9 | +12.8 [−2.6, +28.3] | −4.3 [−13.4, +4.8] |

The two adjustments disagree, and the per-stratum detail (`--strata`) says why:
they are identified on different subpopulations.

- **Within generation method**, the weight sits on `perturbation` (MH weight
  46.6 of ~67 on test_public; 306 single vs 55 multi). There the effect is
  **≈0 and slightly negative** — GPT-5.4 −2.9 pp, Haiku −4.3 pp, both n.s.
- **Within type**, the surviving strata are the handful of types with both
  arms, and the weight sits on `partial_author_list` (7.1) and `near_miss_title`
  (2.8). There the effect is large — GPT-5.4 on `partial_author_list`: 9.1%
  (n=11 single) vs 60.0% (n=20 multi), Δ = +50.9 pp; Haiku 36.4% vs 65.0%,
  Δ = +28.6 pp.

Read together: **the aggregate multi-vs-single gap is mostly a composition
effect.** Multi-defect entries are disproportionately drawn from provenances
and types these models happen to catch. Hold provenance fixed and the gap
vanishes; hold type fixed and it reappears, because the only place defect count
genuinely varies within a type is `partial_author_list` — where dropping one
author (single defect) really is much harder to catch than dropping authors
*and* breaking a second field.

`test_crossdomain` is the one split where provenance is controlled by
construction — all 300 hallucinated entries are `perturbation`. There the
median gap across 12 models is +3.0 pp, consistent with the near-zero
within-`perturbation` estimate above (though at ceiling; see §7.4).

### 7.3 Dose-response

Detection rate by exact defect count is not monotone. On dev_public, Gemini 2.5
Pro goes 43.2% → 60.2% → 54.9% (1 → 2 → 3 defects) and Llama 4 Maverick 57.4% →
73.3% → 67.3%: the jump is from one defect to two, then it flattens or dips.
Cochran-Armitage trend tests reach p<.05 for 5 of 8 dev models, but the shape is
a step, not a gradient — consistent with §7.2's reading that the "dose" is
standing in for entry provenance rather than acting as a count.

### 7.4 What this evidence cannot carry

- **Most models are at ceiling.** All four Qwen3 dense models (FPR 91-98% on
  dev/test) and every one of the 12 models on `test_crossdomain` (FPR 60.6-100%)
  flag nearly everything, which pins both arms near 100% and makes the contrast
  uninformative by construction. The script marks these `*`. Only Gemini 2.5 Pro
  (FPR 5.0%), Llama 4 Maverick (14.6%), GPT-5.4 (22.4%) and Claude Haiku 4.5
  (44.4%) are discriminating enough to read.
- **DeepSeek R1 on crossdomain abstains on 86% of entries** (257/300 UNCERTAIN,
  leaving 43 scored, 7 of them multi-defect). Its −11.5 pp is noise; the script
  marks this `†`.
- **Model coverage is limited by what is committed.** `results/*.jsonl` is
  gitignored, so only 9 models have per-entry dumps on dev/test: the four Qwen3
  dense models, Claude Haiku 4.5, GPT-5.4, Gemini 2.5 Pro, Llama 4 Maverick and
  Qwen3 Max. GPT-5.1, Claude Sonnet 4.6, Claude Opus 4.7, DeepSeek R1/V3.2,
  Mistral Large and Qwen3-235B-A22B are covered on `test_crossdomain` only —
  i.e. only in the ceiling-bound regime. Extending §7.1 to them on dev/test
  needs a re-run of those zero-shot evals.
- **The within-type estimate rests on one type.** See the new §6 bullet.
