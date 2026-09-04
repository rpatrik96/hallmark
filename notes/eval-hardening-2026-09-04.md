# Eval hardening, 4 September 2026 — what was found and what now prevents it

One day's work across three sessions, on HALLMARK and on bibtex-updater. This is
the record: what was wrong, what is fixed, what each new guard prevents, what I
got wrong along the way, and what is still open. The planning document it
replaces is `eval-hardening-plan-2026-09.md`; this one reports outcomes.

Twenty-one commits on `fix/metric-correctness` (PR #39), two on bibtexupdater's
`main`, three issues filed. The suite went from 1,105 tests to 1,302.

## The shape of it

Every defect below is the same shape, and naming it is worth more than the list:
**a mechanism existed, reported honestly, and the report reached nobody.** The
freshness guard printed 46 stale files into a CI log that passed. Nine thousand
API failures were written into result files as verdicts, and counted as coverage.
A ratchet was tuned to the splits CI could see, so it passed by construction
while 112 contradictions sat in the split nobody can inspect. A tool version was
whatever happened to be first on `PATH`.

The other half is a family of commands that answer a question adjacent to the one
asked, and four turned up in one day: `$?` after a pipe reports the last command
in the chain; `pipx list` says what pipx installed, not what `PATH` resolves;
`find | head` turns an incomplete search into a reported absence; and `git push`
saying "Everything up-to-date" is indistinguishable between "nothing to push" and
"your commits went to a different branch". Every one was caught by someone other
than the person who ran it.

## Metric correctness

Four analysis functions were wrong, all in the direction that flatters results,
none pinned by a test. Verified by execution against the committed artifacts.

**The two-sided p-value could only detect a difference in one direction.**
`min(1, 2·P(diff ≤ 0))` returns 1.0 whenever the first-named tool is worse,
however large the gap, and callers enumerate pairs alphabetically — so roughly
half of every pairwise comparison was non-significant by construction. The
in-code comment called this "conservative … a deliberate design choice".

**Per-type F1 and FPR were non-metrics.** Grouping by `hallucination_type` puts
only hallucinated entries in a group, pinning precision at 1.0, FPR at 0.000 and
F1 at the deterministic `2·DR/(1+DR)`. `per_tier_metrics` had been fixed for
exactly this and documents the fix; per-type never got it.

**The tier-weight sweep could not see a false positive.** It read `tm["fpr"]`
while the emitter writes `false_positive_rate`, so the default fired every time
and the swept quantity was weighted recall. Any "rankings are stable under
reweighting" claim rested on it.

**Abstention was free.** `coverage` counted UNCERTAIN as covered while the
confusion matrix, ECE and AUROC all drop it, so a tool abstaining on the hard
entries reported its easy-subset metrics at full coverage. One committed run
answers 68 of 500 entries and records `coverage: 1.0`.

Detection rate, FPR and F1 are unchanged by these fixes — verified by recomputing
two baselines against their released values to three decimals. Per-type metrics
move a great deal, which is why per-type DR now carries a Wilson interval and
leads: with 30–120 hallucinated entries per type against 513 valid ones,
precision is dominated by the shared false-positive count, so per-type F1 tracks
type frequency more than detector quality.

## Provenance

**No released result recorded which tool produced it**, and the tool that
answered was an editable install. `which -a bibtex-check` returns two paths; the
first imports from the bibtexupdater working tree, which was committed into twice
during a three-hour ablation. So entries scored before and after those commits
were scored by different code, and nothing in the output said so. That editable
build also has no single version: `importlib.metadata` says 0.9.2 while
`__version__` says 1.3.1.dev18, from the same interpreter at the same moment.

`EvaluationResult` now carries `tool_version`, `split_sha256` and
`run_timestamp`; `HALLMARK_BIBTEX_CHECK_BIN` pins a binary by absolute path, a
missing pinned path fails loudly rather than falling back, and every run logs
which binary and version answered.

**The freshness guard could not work.** It compared file mtimes, which git does
not preserve, so on a clean checkout it read checkout order and called all 46
released results stale — which is why CI ran it `--warn-only` and the repo test
was `xfail`-ed. Its CI invocation had also kept `--version v1.0` after the data
moved to `data/v1.2`, so it resolved split files under a directory the rename had
removed. It now hashes the split file, results predating the field are
*unverifiable* rather than stale, and it is a hard gate.

**The manifest covered 16 of 45 results**, leaving 29 unchecksummed including
every LLM row in the main table. Extending it immediately failed `--strict` on a
result missing `tier_weighted_f1` entirely; recomputed from its checkpoint
predictions, which reproduce the released DR and F1 to the digit.

**Two directories held two different runs under one set of names.** `results/`
and `data/v1.2/baseline_results/` shared fifteen filenames and every pair
disagreed. `results/` is the default `--results-dir`, so the uncanonical set was
the one being read. Superseded copies moved to `results/superseded_pre_relabel/`;
a test makes the collision impossible.

**A derived table outlived its inputs.** `tables/base_rate_precision.csv` kept
`doi_only` at DR 0.3873 / FPR 0.2788 after the baseline was re-run to 0.1908 /
0.0417, putting its headline "720 flags per true finding" out by a factor of
three. The freshness guard checks results against splits and stops there.

## Data

**The sub-test ratchet measured a different population in CI than locally.**
`verify_subtests` scans `test_hidden`; `data/hidden/` is gitignored. The
CI-visible splits give exactly 99 mismatches and `MAX_MISMATCHES` was 99 — so it
passed by construction while 112 sat in the hidden split, the one split no
external contributor can inspect. Issue #40.

Triage found the two populations failing in *opposite* directions: the public
splits had zero `True → False` mismatches, the hidden split 57 — entries
asserting what their own hallucination type forbids. Also, `fix_doi_resolves_na.py`
listed its splits under a comment reading "Released splits — entries here are
scored against by tools" and omitted `test_hidden`, the split most scored
against, so 43 entries kept the defect that pass existed to remove.

**One case inverted on inspection, and I nearly corrupted correct data.**
`EXPECTED_SUBTESTS[FUTURE_DATE]["fields_complete"]` was `False`, but
`check_fields_complete` returns True for a future-dated entry — every field
present, and "2032" is a well-formed 4-digit year. Running the checker over every
`future_date` entry: True for 14 of 15 in hidden (which assign True, and were
right) and for 29 of 30 in dev_public (which assign False, and were wrong). The
taxonomy contradicted the function that computes the sub-test, and repairing the
hidden entries to match would have destroyed the only correct labels. An audit of
all fourteen types found `future_date` the only one that disagreed.

The gate is now split by direction: contradictions bounded at **zero** — the
public splits prove it achievable — and per-entry disagreements ratcheted against
separate public and hidden baselines, with a skip rather than a pass when the
hidden data is absent.

## Infrastructure failures wearing the shape of measurements

**9,652 `[Error fallback]` records** — API failures written into prediction files
as per-entry verdicts. All carry label UNCERTAIN, so none was scored as a
substantive prediction, which bounds the damage to miscounted coverage. They fall
in `dev_public` (124), `test_public` (183) and the cross-domain splits;
`stress_test` and `hidden` have none. All 180 of DeepSeek-R1's UNCERTAIN records
on `test_public` are of this kind, so its ΔFPR of −0.310 — the largest cross-split
shift in the main table — compares a dev figure over 1,101 answered entries
against a test figure over 651.

**Four pre-screening ablations were null runs**: DR 0.0, FPR 0.0, zero API calls
over a thousand entries. I attributed this to the CLI being absent from `PATH`.
A peer measured the actual mechanism: harcx queries Google Scholar through
`scholarly`, Scholar blocks it, the library retries rather than failing, and at
batch size 20 every batch exceeds its timeout and contributes an empty `checked`
set — while pre-screening's DOI requests still happen, which is exactly why the
failure reads as a clean result.

harcx is on `PATH` now, keyed and pinned, and still cannot produce a verdict —
in two distinct ways, worth keeping separate because they would be cited
differently. Under `-q --threshold 0.75`, the invocation `harc.py` actually uses,
a **single-entry `.bib` did not complete in 150 seconds**. Without `-q` the same
entry returns in seconds but **emits no verdict line at all**, exit 0. The first
is the hang that produces the nulls; the second is a separate parsing-surface
failure that would produce an empty flag set even if the hang were fixed.

**A transient HTTP status was scored as a fabricated citation.** `doi_only` and
`subtests` treated any non-200 as proof a DOI does not exist; `prescreening` had
it right, so three implementations of one check disagreed. Of 150 sampled VALID
entries carrying a DOI, **56 return HTTP 202 and one returns 403** — IEEE and ACM
bot mitigation *after* doi.org redirects successfully. Re-running `doi_only` with
the fix took FPR from **0.279 to 0.043**. Detection rate fell too, which is the
same fix from the other side: flags landing on hallucinated entries for the wrong
reason were counting as detections.

`doi_only` also could not be run through the CLI at all — `run_doi_only` lacked
the `**_kw` catch-all every other runner has, so it raised `TypeError` before
making a request. A plausible reason its result sat stale at 1,068 of 1,119
entries: nobody could regenerate it.

## Source availability is a measurement condition

A peer measured that the same 119 entries, same model, same prompt, cleared 12%
of flags with arXiv starved and 24% with it answering. Reproduced at cascade
scale here. The bibtexupdater ablation began at **53.7% of entries having at least
one source lookup that never completed**, and `bibtex-check`'s own outage guard
refused the run — the guard working exactly as designed.

Three causes, all fixed: the keyless OpenAlex pool (1,000/day/IP) was exhausted
and returned 429 to a bare probe; OpenReview 403s without credentials; and the
wrapper drove `--rate-limit 120` where the tool's documented baseline is 45.
With a key, credentials and a rate of 20, failures fell to **10.0%** — and the run
got **faster**, 2.4 s/entry against 5.7, because pacing avoids the retry storms
that retrying causes. Pacing beats retrying, with the clock going the right way.

Two things worth carrying: **a dead credential is worse than none** (the expired
S2 key 403'd every call while keyless merely throttled — dropping it took failures
from 70% to 27.5%), and the rate is part of the run condition, so
`HALLMARK_BIBTEX_CHECK_RATE_LIMIT` is settable and logged.

## bibtex-updater

**`preprint_only` was the system's most confident wrong answer.** PROBLEM
polarity drawing the CLEARLY-CORRECT anchor, so `p_valid` came out at 0.060 —
more confidently invalid than `title_mismatch` at 0.110 and nearly as extreme as
a DOI resolving to a different paper at 0.035, when a preprint cited as published
is weaker evidence than either. The invariant test then found a third instance I
had not looked for: `url_accessible` asserting VALID polarity at the abstention
anchor. That is the argument for pinning the property rather than the instances.

**Venue comparison reported correct citations as mis-venued.** ISO-4 is the
standard journal abbreviation and the form a large share of real `.bib` files
carry, but the alias map covers ML/CS conferences, so `ACM Trans. Graph.` scored
0.70 against `ACM Transactions on Graphics`, `Proc. Natl. Acad. Sci. U.S.A.` 0.60
and `Annu. Rev. Stat. Appl.` 0.55. And the terminal branch returned MISMATCH for
pairs where neither name was recognised — not recognising a name is not evidence
two names differ. On the workshop corpus, 920 of 5,043 references came back with
venue as the sole disagreement.

Both fixed, with the abstention kept narrow because wrong venue is a real class.
Measuring first corrected the brief twice: the brace-protected ICLR case and COLM
already matched. And the first version broke an existing test — `(CNSM)` against
`NOMS` is a genuine wrong venue because each declares its own acronym.

## Wild data

HALLMARK's own cascade over 5,043 real references from 267 workshop submissions
found **zero fabricated works**. Of the eleven accusations surviving hand audit:
3 corrupt OpenAlex index records (correct DOI, correct authors, wrong title), 3
real papers with unregistered DOIs, 2 real works in humanities venues the indexes
do not cover, 2 unverifiable by construction, and 1 whose existence was proved by
the rebuttal naming it in its own title, read as evidence of absence.

Precision at realistic prevalence, arithmetic over the committed numbers: the
cascade reaches 91.2% at the benchmark's own 62.5% base rate and **5.9% at 1%** —
17 flags per true finding. The benchmark's prevalence is the only point on the
curve where the instrument looks good.

## Corrections I had to make

Recorded because the pattern matters more than the instances, and every one was
caught by someone else or by re-measuring.

- Told Patrik `split_name` was missing from released results. It was present in
  46 of 47; I had read the key `split`.
- Wrote "main is red" into a PR from a local test failure. Main was green; the
  test scans a gitignored split CI cannot see.
- Attributed four null ablations to a missing binary. Inference from
  `mean_api_calls: 0.0`; the mechanism was a subprocess timeout.
- Called the keyed HaRC run "the valid HaRC evaluation". Its coverage is full,
  but it is scored against pre-relabel labels.
- Relayed a wild-corpus breakdown that summed to nine while claiming eleven.
- Believed `pipx list` told me which build would run. It tells you what pipx
  installed.
- Believed my pushes were landing for an hour. They named a branch I was no
  longer on.
- Wrote a venue test whose fixture contained "Workshop", so it exercised the
  satellite-marker branch rather than the one under test.
- Wrote an exemption-staleness test that was itself population-dependent — the
  exact defect the gate was split apart to stop making. CI caught it.
- Wrote my harc tests at `_run_harc_batches`, the level I was editing, when the
  damage lands one level up in `run_with_prescreening`. They would have passed
  while the defect survived. Caught by the session porting them.
- Nearly reported nine `hybrid_fabrication` entries as mislabelled, having
  compared titles by substring containment — which a *modified* title trivially
  satisfies. Checking authors, as the type definition requires, showed all nine
  correct. A looser test would have produced a false accusation against the
  benchmark's own labels.

## The taxonomy result (issue #36)

Measured after the fixes above, and it is the largest single finding of the day.
Scoring every tool three ways — as shipped, with the four real-paper modes
(`wrong_venue`, `preprint_as_published`, `partial_author_list`,
`arxiv_version_mismatch`) removed from the scored set, and with them as negatives
— **the ranking is not stable and the top changes identity.** On `test_public`
14 of 21 tools move under the first fold and 17 of 21 under the second;
`dev_public` agrees. `cascade_db_diagnosis` leads as shipped at MCC 0.897;
`bibtexupdater` takes first when they are scored as false accusations.

The mechanism, verified independently from the committed `per_type_metrics`:
those four modes are **26.8% of the positive class** on `test_public` (139 of
519) and **27.0% of the leading cascade's detections** (139 of 514), every one of
which it converts — DR 1.000. The spread across tools is the part worth keeping:
bibtexupdater 20.0%, Sonnet 4.6 23.2%, `doi_only` 9.1%, median DR 0.755 across 21
tools. A tool's rank depends heavily on how much of its credit comes from
flagging real papers described wrongly, and nothing in the reporting shows it.

`stress_test` is worse: 75 of its 122 entries (61.5%) are real-paper modes, and
**100% of the cascade's misses on that split are on them** — `merged_citation`
sits at DR 1.000 across all 46 entries. So a stress-test detection rate is mostly
a measure of catching correct citations described imprecisely.

`hybrid_fabrication` does **not** fold with them, and folding it alone moves
nothing (Kendall tau 1.000, no tool changes position). The distinction is that
the other four mean the work exists and the entry describes it inaccurately,
while this one means the entry claims one work and the DOI resolves to another —
a fabrication if the index is right, a correct citation if the index is corrupt.

Its population is sound: all 74 entries are generated (53 adversarial, 19
perturbation, 2 LLM), none resolved from an index, and all 35 of the resolvable
DOIs are consistent with the type — 26 to a clearly different paper, 9 to a
similar title with zero author overlap. The exposure is entirely prospective.

**Ruled: report the ablation, defer the relabel.** The instability goes in the
paper as a stated result rather than being settled by relabelling five splits.

## Open

- **The pre-screening ablation** is running version-matched at 1.10.1 with source
  conditions recorded per arm. Four of six ablations were nulls, and the two that
  would show what the layer does for the *co-designed* tool are among them, so
  the "five points of Tier-1 lift" currently rests on `doi_only` alone.
- **`harc_with_s2key_dev_public.json`** is registered known-stale: full coverage,
  but pre-relabel labels. Clearing it needs a keyed current-label run.
- **Is HaRC evaluable at all** on a machine Google Scholar blocks. "Not
  evaluable, here is the mechanism" is a legitimate row.
- **The HaRC exclusion rationale.** It was excluded for Semantic Scholar
  throttling; what is visible is that it hangs on a source bibtex-updater does not
  query. Not checkable from the released artifacts, which record no per-source
  counts.
- **Issue #36**, the largest open item: the taxonomy folds "this work does not
  exist" together with "this entry describes a real work incorrectly". On the
  wild corpus 63% of flags were the second kind, 47% from venue and preprint
  status alone.
- **Regenerating the reference results** under pinned versions, and rebuilding
  the README and site from one run.
