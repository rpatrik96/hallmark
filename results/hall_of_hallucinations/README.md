# Hall of Hallucinations (real-world, evaluation-only)

Source: Hall of Hallucinations by Andrew Wheeler / VerusCite, https://veruscite.com/hallucinations.
Scraped 2026-11-09 with the site owner's permission; credit and link required when used.

Labels are VerusCite's automated verdicts (no row was human-edited on the site).

## Run it

`dataset/` holds three ready HALLMARK data dirs. The CLI only accepts fixed split names, so each subset is its own data dir and the split is always `test_public`.

| Data dir | Entries | Use for |
|---|---|---|
| `dataset/all/` | 1,864 = 932 HALLUCINATED + 932 VALID (shuffled) | main numbers: detection rate, FPR, F1, MCC |
| `dataset/hallucinated/` | 932 HALLUCINATED | detection rate only (FPR undefined) |
| `dataset/valid/` | 932 VALID | FPR only (the detection rate it prints is meaningless, there are no positives) |

```bash
# from the repo root; change all -> hallucinated / valid for the subsets
uv run hallmark stats    --data-dir results/hall_of_hallucinations/dataset/all --split test_public
uv run hallmark evaluate --data-dir results/hall_of_hallucinations/dataset/all --split test_public \
    --baseline doi_only --output results/hall_of_hallucinations/eval/runs/doi_only_all.json
# any registered baseline works the same way: uv run hallmark list-baselines
```

Each data dir has `v1.2/test_public.jsonl` (labels and types, for scoring) and `v1.2/test_public_blind.jsonl` (what a tool sees: key, type, fields).

Rebuild after changing either side: `uv run python scripts/hall_of_hallucinations/make_dataset.py results/hall_of_hallucinations/hallucinated/hall_hallmark_draft.jsonl results/hall_of_hallucinations/valid/hall_valid_draft.jsonl results/hall_of_hallucinations/dataset`

Hallucination types are mostly rule-assigned (see below): report overall detection rate and FPR on this split, not per-type numbers.
Earlier runs in `eval/runs/` used `eval/hm_data/`, which has the same 932 entries as `dataset/hallucinated/` (different order).

Review sheets (the hallucinated-side 100-row sample and decisions, the VALID review page, human decisions and the assistant web check) and the site index snapshot are kept private and not shipped. The rebuild command below needs the two VALID decision CSVs; without them, use `valid/hall_valid_draft.jsonl` as committed.

## Layout

| Folder | What is in it |
|---|---|
| `dataset/` | **runnable data dirs (start here)**: `all/`, `hallucinated/`, `valid/`, see Run it |
| `source/` | scraped site pages (not shipped: index snapshot and `raw/` share reports, 68 MB) |
| `hallucinated/` | HALLUCINATED side: scraped rows, checks, 100-row human review, `hall_hallmark_draft.jsonl` (932) |
| `valid/` | VALID side: verified rows, candidates, review queue/audit, `hall_valid_draft.jsonl` (932) |
| `valid/checks/` | automatic checks on the VALID candidates (Crossref, second check, bibtex-check run, flags) |
| `eval/` | earlier baseline runs on the hallucinated-only split: `hm_data/` and `ra_hallmark/` inputs (both gitignored; same entries as `dataset/hallucinated/`), `runs/` outputs (doi_only, bibtex-updater 1.2.0, refguard, reference-audit) |

### Hallucinated side

| File | Contents |
|---|---|
| `source/raw/` (gitignored, 68 MB) | 160 share-report pages; regenerate with `scripts/hall_of_hallucinations/scrape_hall.py` |
| `hallucinated/hall_hallucinated_rows.jsonl` | 942 rows with status `hallucination` |
| `hallucinated/hall_hallucinated_rows_flagged.jsonl` | + overlap flags: 1 in VerusCite V1, 0 in HALLMARK, 5 duplicates within Hall (936 kept) |
| `hallucinated/hall_autocheck.jsonl` | free DOI (Crossref, doi.org) and Crossref title-search checks for all 936 rows |
| `hallucinated/hall_hallmark_draft.jsonl` | 932 BenchmarkEntry rows (936 minus 4 human n/u) |

Human check of a seed-0 random 100-row sample: 96 hallucinated, 2 not, 2 unsure.
Precision 96/98 = 98.0% (Wilson 95% CI 92.9-99.4%).

Types are not human-verified except for 30 rows. Assignment, as in `scripts/parse_gptzero_csv.py`:
keyword rules over VerusCite's notes (736), fallback `plausible_fabrication` when no rule matches (166).
On the sample the notes rule agreed with the human type on 63/74 rows. Report only overall recall on this split.

### VALID side

VerusCite `verified` rows from the same 160 local share reports (no rescrape), matched 1:1 to the 932 hallucinated entries.

| File | Contents |
|---|---|
| `valid/hall_verified_rows_flagged.jsonl` (gitignored, 13 MB; regenerate with `extract_verified.py`) | 7,766 verified rows + overlap flags: 34 in VerusCite V1, 49 in HALLMARK (any label), 80 share a title with a Hall hallucinated row, 225 duplicates within verified (7,478 kept) |
| `valid/hall_verified_candidates.jsonl` | seed-0 shuffle of the kept rows, first 1,100 (`cand_rank`); the set is the first 932, later ranks backfill rejected rows |
| `valid/checks/hall_autocheck_verified.jsonl` | `hall_autocheck.py` (Crossref DOI + title search) on all 1,100 |
| `valid/checks/hall_autocheck2_verified.jsonl` | second free check on the 233 first-pass "title not found" rows in ranks < 1000: DataCite DOI lookup, OpenAlex and arXiv title search. DBLP not used (API behind a bot challenge, 2026-11-18) |
| `valid/checks/btu_valid337/` | bibtex-check 1.2.0 (same flags as the paper run, no HALLMARK pre-screening) on the 337 title-only passes in the first 932 |
| `valid/checks/hall_btu_flags_verified.jsonl` | 68 of those 337 forced to manual review (`hall_btu_flags.py`) |
| `valid/hall_valid_review_queue_n932.jsonl` | autocheck failures + bibtex-check flags in the first 932 |
| `valid/hall_valid_audit50_n932.jsonl` | uniform random 50 of the first 932 (seed 1) |

Pass rule (`scripts/hall_of_hallucinations/hall_verified.py`): cited DOI in Crossref and DOI target title sim >= 0.9; or, with no DOI / a DOI outside Crossref, a title hit (sim >= 0.9) whose authors include the cited first author, from Crossref or the second check. Unregistered DOIs and DOI-to-other-paper always go to manual review.
Title-only passes (no Crossref DOI match) were also run through bibtex-check 1.2.0: 168 verified, 38 venue abbreviations and 63 with no bibtex-check evidence (25 books skipped, 38 unconfirmed) stay passed; 68 go to manual review (26 venue mismatches not explained as an abbreviation of the Crossref venue, 20 partial_match, 7 given-name substitutions, 6 author, 3 title, 1 year mismatch, 1 invalid year, 4 not found).

On the first 932: 742 pass, 190 to manual review = 122 autocheck failures (95 title not found, mostly reports/web pages/books; 12 DOI points to another paper; 12 first author missing; 3 unregistered DOI) + 68 bibtex-check flags.
bibtex-check is also an evaluated baseline. It only routes rows to a human and never removes one, so its flags do not set its own FPR to 0; the remaining bias is that a wrong citation bibtex-check also misses stays labelled VALID, bounded by the residual label error rate. Backfill rows (rank >= 932) were not run through bibtex-check.
Reviewed failures stay in the set when confirmed, so the set is not skewed toward Crossref-indexed papers; `build` refuses to run while any of the 190 lacks a verdict.
The audit sample bounds the label error rate: 0 errors in 50 -> 95% upper bound ~6% (rule of three); otherwise report a Wilson interval.

Rebuild (from `results/hall_of_hallucinations/`): `python3 -I ../../scripts/hall_of_hallucinations/hall_verified.py build valid/hall_verified_candidates.jsonl valid/checks/hall_autocheck_verified.jsonl valid/hall_valid_draft.jsonl --n 932 --autocheck2 valid/checks/hall_autocheck2_verified.jsonl --btu-flags valid/checks/hall_btu_flags_verified.jsonl --decisions valid/review/hall_valid_review_hall_valid_review_n932.csv valid/review/hall_valid_web_check_u32.csv`

Review (2026-11-20): all 190 queue rows and all 50 audit rows judged; queue 157 y / 32 u / 1 n, audit 50/50 y.
The 32 u rows (mostly reports, web pages, book chapters) were then web-checked by the assistant: 23 confirmed as cited -> y; 3 likely fabricated -> n, plus one real paper cited with an abbreviated title excluded by the user (two of the fabricated cite a journal volume/article number whose DOI belongs to a different paper, one not found anywhere; VerusCite's `verified` verdict looks wrong on these); 5 still unconfirmed -> u.
The hallucinated side is grey-literature heavy (508/942 raw citations carry a URL or "accessed/retrieved"), so confirmed grey literature is kept rather than dropped.
`valid/hall_valid_draft.jsonl`: 932 VALID entries = 194 human y + 23 assistant web-checked y + 715 autocheck passes (10 rejected rows, n 5 + u 5, backfilled from ranks >= 932; 2 unreviewed autocheck failures in the backfill range skipped). The explanation field records which of the three applies.
Audit: 0 errors in the uniform 50 -> 95% upper bound 5.8% on VerusCite `verified` label error; 0 in the 37 audit rows that passed autocheck -> 7.8% on autocheck-pass rows, about 6% on the final set (715/932 of it is autocheck-pass only).
