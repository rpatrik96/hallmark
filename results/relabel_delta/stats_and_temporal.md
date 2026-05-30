# `tab:stats` regeneration + temporal reconciliation (post-relabel)

Generated from the **final** relabeled data on branch `fix/dev-public-mislabel-audit`.
Scope: (a) regenerate the dataset-statistics table for every split; (b) reconcile the
448-vs-858 temporal provenance and verify which printed temporal-table values reproduce
from the shipped prediction files. **No paper or `data/*.jsonl` files were edited.**

Source files (all counts computed directly from these, not from `metadata.json`, which is
stale — it still says `total_entries: 2526` / `build_date: 2026-02-20` and a pre-relabel
hidden split of 200/253):

- `data/v1.0/dev_public.jsonl`
- `data/v1.0/test_public.jsonl`
- `data/v1.0/stress_test.jsonl`
- `data/hidden/test_hidden.jsonl`
- `data/v1.0/test_crossdomain.jsonl` (v1.1 eval-only; **not** part of `tab:stats`)

Conventions matching the paper:
- **Tier distribution counts hallucinated entries only** (paper caption, `benchmark.tex:91`).
- **Types** = number of distinct `hallucination_type` values present among hallucinated entries.
- The `tab:stats` total is the **four main splits** (dev + test + stress + hidden). The
  cross-domain split is a v1.1 eval-only addition and was never in the `tab:stats` total
  (old total 2,525 = 1119 + 831 + 122 + 453).

---

## (a) Dataset statistics — NEW vs OLD

### NEW `tab:stats` (drop-in replacement values)

| Split          | Valid | Halluc. | Total | Tier 1 | Tier 2 | Tier 3 | Types |
|----------------|------:|--------:|------:|-------:|-------:|-------:|------:|
| `dev_public`   |   513 |     606 | 1,119 |    149 |    280 |    177 |    14 |
| `test_public`  |   312 |     519 |   831 |    130 |    238 |    151 |    14 |
| `stress_test`  |     1 |     121 |   122 |    --- |     85 |     36 |     3 |
| `test_hidden`  |   210 |     244 |   454 |     52 |    115 |     77 |    14 |
| **Total**      | 1,036 |   1,490 | 2,526 |    331 |    718 |    441 |    14 |

Per-split internal consistency verified: Tier1+Tier2+Tier3 == Halluc. for every split
(606, 519, 121, 244) and the total tier sum 331+718+441 = 1,490 == total Halluc.

### OLD `tab:stats` (currently in `sections/benchmark.tex:99-104`)

| Split          | Valid | Halluc. | Total | Tier 1 | Tier 2 | Tier 3 | Types |
|----------------|------:|--------:|------:|-------:|-------:|-------:|------:|
| `dev_public`   |   486 |     633 | 1,119 |    155 |    292 |    186 |    14 |
| `test_public`  |   287 |     544 |   831 |    134 |    243 |    167 |    14 |
| `stress_test`  |     1 |     121 |   122 |    --- |     85 |     36 |     3 |
| `test_hidden`  |   200 |     253 |   453 |     69 |    115 |     69 |    14 |
| **Total**      |   974 |   1,551 | 2,525 |    358 |    735 |    458 |    14 |

### Cell-level deltas (NEW − OLD)

| Split          | ΔValid | ΔHalluc | ΔTotal | ΔTier1 | ΔTier2 | ΔTier3 |
|----------------|-------:|--------:|-------:|-------:|-------:|-------:|
| `dev_public`   |    +27 |     −27 |      0 |     −6 |    −12 |     −9 |
| `test_public`  |    +25 |     −25 |      0 |     −4 |     −5 |    −16 |
| `stress_test`  |      0 |       0 |      0 |      0 |      0 |      0 |
| `test_hidden`  |    +10 |      −9 |     +1 |    −17 |      0 |     +8 |
| **Total**      |    +62 |     −61 |     +1 |    −27 |    −17 |    −17 |

Notes on the deltas:
- `dev_public` (+27 valid / −27 halluc) and `test_public` (+25 / −25) are the GT flips:
  entries that moved HALLUCINATED→VALID. The task brief lists 10 dev + 10 test
  `decision==flip_valid` flips in `results/reviewer_experiments/relabel_flips.json`; the
  table shows +27 / +25 because the relabel pass folded additional `flip_valid` corrections
  into the released `dev_public.jsonl` / `test_public.jsonl` beyond the 20 score-changing
  flips highlighted in the brief. (Total moves are valid↔halluc, so split totals are
  conserved at 1,119 and 831.) The 20-flip subset is the relevant one for *tool re-scoring*;
  the full +27/+25 is the relevant one for *these dataset-statistics counts*.
- `test_hidden` total moved 453 → **454** (+1 entry overall, +10 valid / −9 halluc). The
  hidden file now holds 454 entries. The old paper text (`appendix.tex:137`) and the old
  table row both said 453 — the new file is 454.
- `stress_test` is unchanged (1 valid canary + 121 hallucinated; the single valid entry is
  the `__canary__` row, generation_method `canary`).

### Abstract / datasheet totals (the "moved-from-2,525" numbers)

| Quantity                  | OLD   | NEW   |
|---------------------------|------:|------:|
| Total entries (4 splits)  | 2,525 | **2,526** |
| Total valid               |   974 | **1,036** |
| Total hallucinated        | 1,551 | **1,490** |

The "2,525" / "974 valid" / "1,551 hallucinated" string appears in the paper at:
`sections/abstract.tex:4`, `sections/introduction.tex:21`, `sections/limitations.tex:5`,
`sections/benchmark.tex:68` (974 valid) and `:104` (table total row),
`appendix.tex:137` (453→grand total 2,525) and `:899` (datasheet comparison row 2,525),
`checklist.tex:8`. All of these need the 2,526 / 1,036 / 1,490 update (paper-edit stage,
not done here). Cross-domain (500: 200 valid / 300 hallucinated) remains a separate
eval-only v1.1 split and should stay out of this total, matching the old accounting.

### Caption claim to revisit (flag, not a table number)

The `tab:stats` caption (`benchmark.tex:91`) asserts: *"All main splits cover all 14
hallucination types with n ≥ 30 per type."* After relabeling this is **no longer true** for
several types (counts over hallucinated entries):

- `dev_public`: 1 type below 30 — `hybrid_fabrication` = 26.
- `test_public`: 6 types below 30 — `chimeric_title` 23, `fabricated_doi`/`future_date`/
  `merged_citation`/`preprint_as_published`/`hybrid_fabrication` = 29 each.
- `test_hidden`: 13 of 14 types below 30 (min `fabricated_doi` = 7; `hybrid_fabrication` 10).
  Note `test_hidden` was never an n≥30 split — at 244 hallucinated / 14 types it cannot
  satisfy n≥30 for all types regardless of relabeling, so the caption's "all main splits"
  wording was already imprecise about the hidden split.

This is a caption-wording issue for the paper stage (e.g. soften to "≥ 30 per type on
`dev_public`/`test_public` for most types"), not a change to any table number.

---

## (b) Temporal reconciliation — 448 vs 858

### Provenance decision (as instructed, documented here)

**Ship the existing ~448 prediction files as the canonical temporal set; present the
858-entry file as an extended reviewer-experiment supplement.**

Line counts of the released prediction files in `results/temporal_supplement/`:

| Released `*_predictions.jsonl`                                   | Lines | Unique keys |
|-----------------------------------------------------------------|------:|------------:|
| `llm_openrouter_claude_sonnet_4_6_temporal_supplement_*`        |   448 | 448 |
| `llm_openrouter_claude_opus_4_7_temporal_supplement_*`          |   448 | 448 |
| `llm_openrouter_gemini_pro_temporal_supplement_*`               |   448 | 448 |
| `llm_openrouter_llama_4_maverick_temporal_supplement_*`         |   448 | 448 |
| `llm_openrouter_qwen_max_temporal_supplement_*`                 |   448 | 448 |
| `llm_openai_cutoff_aware_temporal_*`                            |   448 | 448 |
| `llm_openrouter_{gemini_flash,qwen}_cutoff_aware_temporal_*`    |   448 | 448 |
| `llm_openai_temporal_*` (GPT-5.1)                               |   480 | **448** (22 dup keys) |
| `llm_openrouter_gemini_flash_temporal_*`                        |   480 | **448** (22 dup keys) |
| `llm_openrouter_mistral_temporal_*`                             |   480 | **448** (22 dup keys) |
| `llm_openrouter_qwen_temporal_*` (Qwen3-235B)                   |   480 | **448** (22 dup keys) |
| `llm_openrouter_deepseek_v3_temporal_*`                         |   480 | **448** (22 dup keys) |
| `llm_openrouter_deepseek_r1_temporal_*`                         |   480 | **448** (22 dup keys) |

**Finding.** Every released file resolves to the **same canonical 448-entry set
(300 VALID / 148 HALLUCINATED, 0 missing ground truth)** — confirmed by the supplement
files (which are literally 448 lines / 448 unique keys, all sharing one identical key set)
and by the stored aggregate JSONs (`*_temporal_supplement_eval.json` and `*_v1subset.json`)
all reporting `num_entries: 448, num_valid: 300, num_hallucinated: 148, coverage: 1.0`. The
six "older" models were re-run with a checkpoint loop that appended 22 duplicate-key lines
(480 total lines), but the unique-key content is exactly the same 448 set. **The paper's
printed N=448 (300/148) is the reproducible canonical set.**

The 858-entry file `results/temporal_supplement/temporal_supplement_2024_2025.jsonl`
(440 VALID / 418 HALLUCINATED) is a later **superset** of the same 2024–2025 DBLP pool. It
was **not relabeled** in this audit, and the canonical 448 keys are a subset of its 858 keys
(every 448 key joins cleanly to a GT label in the 858 file). The 858-set robustness checks
(recall probe, late-cutoff control, GPT-5.1 variance) live in
`results/reviewer_experiments/FINDINGS.md` and should be cited as the **extended
reviewer-experiment supplement**, presented alongside (not overwriting) the canonical 448
temporal table. The paper already carries this exact TODO at `sections/analysis.tex:45` and
`appendix.tex:712`.

### Reproduced canonical temporal-table values (N=448, 300 valid / 148 hallucinated)

All values below are recomputed from the **shipped prediction files** joined to the GT
labels (from the 858 source, restricted to the 448 canonical keys), using the paper's own
metric convention (`hallmark/evaluation/metrics.py`: UNCERTAIN predictions are excluded
from the confusion matrix entirely, so `FPR = fp/(fp+tn)` over non-UNCERTAIN valids and
`DR = tp/(tp+fn)` over non-UNCERTAIN hallucinated). For the six 480-line files,
**first-wins** deduplication of the 22 duplicate keys is required to match the paper (this
matches the stored `*_v1subset.json`; last-wins or all-480-lines does **not** reproduce —
see the flag below). DR21–23 / FPR21–23 are `dev_public` baselines (carried over unchanged;
the temporal supplement was not relabeled).

| Model              | FPR24–25 (printed) | FPR24–25 (reproduced) | DR24–25 | F1 24–25 | MCC 24–25 | reproduces? |
|--------------------|-------------------:|----------------------:|--------:|---------:|----------:|:-----------:|
| Gemini Flash       |              0.595 |                0.5946 |  0.8435 |   0.5548 |    0.2507 | yes (first-wins) |
| GPT-5.1            |              0.759 |                0.7593 |  0.9583 |   0.5455 |    0.2457 | yes (first-wins) |
| Mistral Large      |              0.793 |                0.7932 |  0.9524 |   0.5374 |    0.2078 | yes (first-wins) |
| Qwen3-235B         |              0.809 |                0.8089 |  0.9860 |   0.5413 |    0.2449 | yes (first-wins) |
| DeepSeek-V3.2      |              0.759 |                0.7592 |  0.9797 |   0.5577 |    0.2777 | yes (first-wins) |
| DeepSeek-R1†       |              0.856 |                0.8557 |  0.0000 |   0.0000 |    0.0000 | yes (first-wins) |
| Claude Sonnet 4.6  |          **0.120** |            **0.1200** |  0.8176 |   0.7934 |    0.6877 | **yes (direct)** |
| Claude Opus 4.7    |          **0.073** |            **0.0733** |  0.7162 |   0.7681 |    0.6692 | **yes (direct)** |
| Gemini 2.5 Pro     |              0.250 |                0.2500 |  0.6261 |   0.5882 |    0.3656 | yes (direct) |
| Llama 4 Maverick   |              0.763 |                0.7633 |  0.9392 |   0.5388 |    0.2160 | yes (direct) |
| Qwen3-VL-235B      |              0.887 |                0.8867 |  1.0000 |   0.5269 |    0.2007 | yes (direct) |

† DeepSeek-R1 routes nearly all 2024–2025 entries to UNCERTAIN (150 of 448), so DR=0;
FPR=0.856 is over the few non-UNCERTAIN valids — consistent with the table footnote.

**The two FPR values the task asked us to verify reproduce exactly: Sonnet 4.6 = 0.120
and Opus 4.7 = 0.073** (both from the direct 448-line supplement files, no dedup needed).
**All 11 LLM rows of `tab:temporal_supplement` reproduce from the shipped files.**

### GPT-5.4 FPR 0.41 — reproduces from the stored probe aggregate, NOT from a raw file

GPT-5.4 is **not** a row in `tab:temporal_supplement`; it appears in the takeaway prose
(`appendix.tex:753`, `:762`) and the separate cutoff probe (`app:gpt54-probe`). Its
printed FPR **0.41** matches `hallmark-paper/figures/gpt54_probe_results.json`
(`aggregate.gpt54_fpr = 0.41333`, on the same canonical N=448). **Caveat to flag:** the
per-entry GPT-5.4 predictions are **not** shipped in `results/temporal_supplement/` — only
the aggregate JSON is in the paper's `figures/` directory. So GPT-5.4 FPR 0.41 is verifiable
only against that stored aggregate, not re-derivable from a released prediction file. If the
camera-ready wants the GPT-5.4 row to be reproducible to the same standard as the other 11,
the raw GPT-5.4 448 predictions need to be exported into `results/temporal_supplement/`.

### Temporal values that do NOT reproduce (flags)

1. **GPT-5.4 raw predictions missing** (above). Aggregate FPR 0.41 is consistent with the
   stored probe JSON but is not backed by a shipped per-entry file.

2. **Dedup convention is load-bearing for the six 480-line files.** The printed table values
   (= the stored `*_v1subset.json`) require **first-wins** dedup of the 22 duplicate keys.
   The naive recompute that other readers will reach for does **not** match:
   - all-480-lines (counts duplicates): e.g. GPT-5.1 FPR 0.7446, Gemini Flash 0.5828.
   - last-wins dict dedup: e.g. GPT-5.1 FPR 0.7424, Gemini Flash 0.5859.
   - first-wins dict dedup: GPT-5.1 FPR 0.7593, Gemini Flash 0.5946 — **matches printed**.
   Recommendation: ship the deduplicated 448-line files (or document first-wins explicitly)
   so the table is reproducible without insider knowledge of the checkpoint artifact.

3. **Minor internal inconsistency in the GPT-5.1 probe JSON vs the table.** The table's
   GPT-5.1 FPR24–25 is 0.759 (first-wins, UNCERTAIN-excluded). The
   `gpt54_probe_results.json` reports `gpt51_fpr = 0.7424` for the same model on the same
   448 set — that is the last-wins / UNCERTAIN-handling variant, ~1.7 pp lower. Both are
   "GPT-5.1 on the 448 supplement"; they differ only by dedup/abstention bookkeeping. Worth
   a one-line note so the two GPT-5.1 numbers in the paper aren't read as a discrepancy.

### What does NOT change in the temporal story

- **No temporal tool number changes from the relabel.** The temporal supplement was not
  relabeled, so every FPR/DR/F1/MCC in `tab:temporal_supplement` stands as printed (subject
  only to the reproducibility flags above). The relabel touched only `dev_public` /
  `test_public` / `test_hidden`, not the 2024–2025 supplement.
- The DR21–23 / FPR21–23 baseline columns are `dev_public` numbers and belong to the
  main-results re-scoring task (the 20 GT flips), not to this temporal deliverable.

---

## Summary of action items for the paper stage (not done here)

1. Replace the `tab:stats` body and total row with the NEW values above
   (`benchmark.tex:99-104`).
2. Update the total-count strings 2,525 → **2,526**, 974 → **1,036**, 1,551 → **1,490** at
   the 8 locations listed (`abstract.tex:4`, `introduction.tex:21`, `limitations.tex:5`,
   `benchmark.tex:68`, `appendix.tex:137` and `:899`, `checklist.tex:8`), and 453 → **454**
   for the hidden split (`appendix.tex:137`).
3. Soften the `tab:stats` caption's "n ≥ 30 per type" claim (now violated for several types
   post-relabel; the hidden split never satisfied it).
4. Temporal table is correct as printed; resolve the existing 448-vs-858 TODOs by stating
   448 (300/148) is canonical and citing `results/reviewer_experiments/FINDINGS.md` for the
   858-set extended checks.
5. Optionally export the GPT-5.4 raw 448 predictions and ship deduplicated 448-line files
   for the six checkpoint-appended models, so the whole temporal table is reproducible
   without the first-wins caveat.

`metadata.json` (`data/v1.0/metadata.json`) is stale and should be regenerated from the
relabeled splits (it still encodes the pre-relabel hidden 200/253 and the old totals).
