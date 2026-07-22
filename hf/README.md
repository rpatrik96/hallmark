---
license: mit
language:
- en
pretty_name: HALLMARK
size_categories:
- 1K<n<10K
task_categories:
- text-classification
- text-retrieval
tags:
- citation-verification
- hallucination-detection
- llm-evaluation
- benchmark
- bibtex
- academic-writing
- responsible-ai
- croissant
configs:
  - config_name: default
    data_files:
      - split: dev_public
        path: data/dev_public.parquet
      - split: test_public
        path: data/test_public.parquet
      - split: stress_test
        path: data/stress_test.parquet
  - config_name: blind
    data_files:
      - split: dev_public
        path: blind/dev_public_blind.parquet
      - split: test_public
        path: blind/test_public_blind.parquet
      - split: stress_test
        path: blind/stress_test_blind.parquet
---

# HALLMARK — Citation Hallucination Detection Benchmark

> **Anonymized for NeurIPS 2026 Datasets & Benchmarks Track double-blind review.**
> Source code: <https://anonymous.4open.science/r/hallmark/>
> **Corpus version:** v1.2.2 (2026-07-22) — see the source repository's `CHANGELOG.md` and tagged releases for provenance of every version.

**HALL**ucination bench**MARK** evaluates citation verification tools on detecting hallucinated references in academic papers. The benchmark was motivated by the NeurIPS 2025 incident in which 53 accepted papers were found to contain fabricated citations that passed peer review.

**TL;DR.** 2,072 public bibliography entries (1,950 in `dev_public`+`test_public`, 122 in `stress_test`), labeled `VALID` or `HALLUCINATED` across 14 fine-grained hallucination types and 3 difficulty tiers, with 6 ground-truth sub-tests per entry.

## At a glance

| Split          | Total | Valid | Hallucinated | Tier 1 / 2 / 3 |
|----------------|------:|------:|-------------:|---------------:|
| `dev_public`   | 1,119 |   513 |          606 | 149 / 280 / 177|
| `test_public`  |   831 |   312 |          519 | 130 / 238 / 151|
| `stress_test`  |   122 |     1 |          121 |   — / 85 / 36 |
| `test_hidden`  |   454 |     — |            — |   held out     |

- **14 hallucination types** across 3 difficulty tiers (Easy / Medium / Hard) — see *Taxonomy* below.
- **6 sub-tests per entry**: DOI resolution, title existence, author match, venue correctness, field completeness, cross-database agreement.
- **Sources**: DBLP-scraped real citations + perturbations + LLM-generated fakes + real-world hallucinations recovered from NeurIPS 2025 retracted papers (via GPTZero analysis).
- **License**: MIT.

### Versioning note

The paper's numbers are computed against tag `v1.2.0`. Later releases (`v1.2.1`, `v1.2.2`) replace stale generation-pipeline explanation strings with diagnoses verified against CrossRef/arXiv/DBLP records (per-entry fix logs in the source repository); labels, hallucination types, tiers, sub-tests, and split membership are byte-identical to `v1.2.0`, so every reported number is unaffected. The released corpus is post-relabel throughout: a systematic ground-truth audit corrected entries where real papers were wrongly flagged `HALLUCINATED`.

## Loading

```python
from datasets import load_dataset

# Default config (all labels visible)
ds = load_dataset("hallmark-neurips2026/HALLMARK")

# Blind config (labels stripped — for leaderboard submissions)
blind = load_dataset("hallmark-neurips2026/HALLMARK", "blind")
```

The `default` config exposes the full schema with ground-truth labels. The `blind` config strips
`label`, `hallucination_type`, `difficulty_tier`, `explanation`, `subtests`, and provenance
fields — matching the format an evaluation tool would receive at inference time.

## Schema

Each row is a citation entry with the following fields (default config):

| Field                 | Type           | Description |
|-----------------------|----------------|-------------|
| `bibtex_key`          | str            | Stable hex hash key (12 chars) |
| `bibtex_type`         | str            | `inproceedings`, `article`, `misc`, etc. |
| `fields`              | dict[str, str] | BibTeX fields: `author`, `title`, `year`, `doi`, `venue`, `pages`, … |
| `label`               | str            | `VALID` or `HALLUCINATED` |
| `hallucination_type`  | str / null     | One of 14 types (null for `VALID`) |
| `difficulty_tier`     | int / null     | 1 (easy), 2 (medium), 3 (hard) |
| `explanation`         | str / null     | Human-readable annotation rationale |
| `generation_method`   | str            | `scraped`, `perturbation`, `llm_generated`, `adversarial`, `real_world`, `canary` |
| `source`              | str            | Source pool identifier |
| `source_conference`   | str / null     | Originating venue if applicable |
| `publication_date`    | str / null     | ISO date for temporal contamination analysis |
| `added_to_benchmark`  | str            | ISO date the entry was added |
| `subtests`            | dict[str, bool/null] | Six per-entry sub-test ground truths |
| `raw_bibtex`          | str / null     | Original BibTeX source if preserved |
| `schema_version`      | str            | Currently `"1.0"` |

The `blind` config keeps only `bibtex_key`, `bibtex_type`, `fields`, and `raw_bibtex`.

## Taxonomy

| Tier | Type                       | Notes |
|-----:|----------------------------|-------|
| 1    | `fabricated_doi`           | DOI does not resolve |
| 1    | `nonexistent_venue`        | Venue does not exist |
| 1    | `placeholder_authors`      | "John Smith", "First Last", etc. |
| 1    | `future_date`              | Publication year in the future |
| 2    | `chimeric_title`           | Title combines fragments from multiple real papers |
| 2    | `wrong_venue`              | Real paper, wrong venue |
| 2    | `swapped_authors`          | Authors of a different real paper (a.k.a. *author mismatch*) |
| 2    | `preprint_as_published`    | Cited as conference paper but only on arXiv |
| 2    | `hybrid_fabrication`       | Real DOI but fabricated metadata |
| 3    | `near_miss_title`          | Title close to a real paper, off by a few words |
| 3    | `plausible_fabrication`    | LLM-generated entries that "look real" — sourced from real NeurIPS 2025 retractions |
| †    | `merged_citation`          | Two real citations fused into one |
| †    | `partial_author_list`      | Author list truncated past convention |
| †    | `arxiv_version_mismatch`   | Wrong arXiv version pin |

† Theoretically-motivated stress-test types (present in all splits; concentrated in `stress_test`).

## Repository layout

```
.
├── data/                          # Parquet (default config — full labels)
├── blind/                         # Parquet (blind config — labels stripped)
├── jsonl/                         # Original JSONL (lossless source of truth)
├── sources/                       # Provenance: pre-curation source pools
├── baseline_results/              # Pre-computed baseline predictions on dev_public
├── metadata.json                  # Per-split counts and distributions
├── source_mapping.json            # bibtex_key → source pool mapping
├── valid_entry_verification.json  # Verification trace for VALID entries
└── croissant.json                 # Croissant metadata (distributions with SHA-256, record sets, RAI fields)
```

## Evaluation

Use the [accompanying code repository](https://anonymous.4open.science/r/hallmark/) to run baselines and the official evaluator:

```bash
hallmark evaluate --split dev_public --baseline doi_only

hallmark evaluate --split dev_public \
    --predictions my_predictions.jsonl \
    --tool-name my-tool
```

Predictions follow this format:

```json
{"bibtex_key": "abc123def456", "prediction": "HALLUCINATED", "confidence": 0.91}
```

### Metrics reported

- **Primary**: Detection Rate, FPR, F1-Hallucination, Tier-weighted F1, Expected Calibration Error (ECE)
- **Diagnostic**: detect@k, source-stratified metrics, per-subtest accuracy, Plackett-Luce ranking (ONEBench-inspired)

---

## Datasheet

This section follows the *Datasheets for Datasets* template (Gebru et al., 2021).

### Motivation
- **For what purpose was the dataset created?**
  To enable rigorous, reproducible evaluation of citation-hallucination detection tools — a category of tools that received intense interest after the NeurIPS 2025 retraction incident exposed widespread fabricated citations passing peer review. No prior benchmark covered this task with controlled difficulty stratification and per-entry diagnostic sub-tests.
- **Created by.** Anonymized for double-blind review.

### Composition
- **What do instances represent?** Each instance is a single bibliography entry (BibTeX-style record) drawn from real papers, perturbed real entries, LLM-generated fakes, or real-world hallucinations.
- **How many instances?** 2,526 total (2,072 public + 454 hidden). See *At a glance*.
- **What data does each instance consist of?** Structured BibTeX fields (`author`, `title`, `year`, `venue`, `doi`, …), label, fine-grained hallucination type, difficulty tier, six sub-test ground truths, and provenance metadata.
- **Is there a label or target?** Yes: binary `label ∈ {VALID, HALLUCINATED}` plus `hallucination_type` (14 classes) and `difficulty_tier`.
- **Are there missing values?** `subtests` fields can be `null` when not applicable; `raw_bibtex` is `null` for synthetically constructed entries.
- **Are relationships made explicit?** Yes — `source_mapping.json` traces each `bibtex_key` to its source pool.
- **Recommended train/dev/test split.** `dev_public` for development, `test_public` for evaluation, `stress_test` for robustness analysis. `test_hidden` is reserved for a future leaderboard.

### Collection process
- **How was the data acquired?** Real citations were scraped from DBLP. Perturbed entries were produced by deterministic transformations of real entries (e.g., year shift, author swap, venue substitution). LLM-generated entries were produced by prompting GPT, Claude, DeepSeek, Qwen, Gemini, and Mistral models to fabricate plausible citations. Real-world hallucinations were extracted from the GPTZero NeurIPS 2025 analysis (publicly released).
- **Time frame.** Real citations are dated 1994–2024 (`publication_date` field). Perturbations and LLM generations were produced in 2026.
- **Were any ethical review or IRB processes used?** No human subjects data; no IRB review required.

### Preprocessing / cleaning / labeling
- **Was preprocessing done?** Yes: deduplication via `bibtex_key` hex hash, normalization of BibTeX field names, fuzzy-match guard against title leakage between splits.
- **How were labels assigned?**
  - `scraped` entries from DBLP — labeled `VALID` after positive verification (DOI resolves, title in DBLP/OpenAlex, year matches).
  - `perturbation` entries — labeled `HALLUCINATED` by construction with type recorded.
  - `llm_generated` and `real_world` entries — labeled `HALLUCINATED` after sub-test panel disagreement.
- **Ground-truth audit.** A systematic post-release audit corrected entries where real papers were wrongly flagged `HALLUCINATED` (arXiv-DataCite-DOI root cause and faithful truncated author lists); the released corpus is post-relabel, with the flip log in the source repository (`results/reviewer_experiments/relabel_flips.json`).
- **Verification trace.** `valid_entry_verification.json` records the resolver evidence used for each `VALID` entry.

### Uses
- **Intended uses.**
  - Benchmarking citation-verification tools on hallucination detection.
  - Studying tool calibration (ECE) and tier-stratified failure modes.
  - Comparing LLM-as-judge approaches against rule-based and retrieval-based verifiers.
- **Out-of-scope uses.**
  - **Not** a training set for citation-generation models — using HALLMARK as supervised training data would contaminate downstream evaluation.
  - **Not** a measure of an LLM's general factuality outside the citation domain.
  - **Not** a substitute for end-to-end retraction review of any specific paper.

### Distribution
- **How is the dataset distributed?** HuggingFace Datasets Hub, MIT license, with this card and Croissant metadata (`croissant.json`, including RAI fields).
- **DOI / persistent identifier.** Provided by HuggingFace upon publication.

### Maintenance
- **Who maintains the dataset?** Anonymized for review. Post-acceptance, contact details will be added.
- **Will the dataset be updated?** Yes. Versioning follows semver-style tags on the source repo (`v1.0`, `v1.1`, …). New hallucination instances are added quarterly; the hidden test split rotates annually to mitigate contamination. See the source repository's `CHANGELOG.md` and tagged releases.
- **How can users contribute?** A community contribution interface (`hallmark contribute …` CLI) is documented in the source repo. Submitted entries pass through the same six-sub-test verification pipeline before inclusion.
- **Will old versions be retained?** Yes — every tagged release is permanently retained as a HuggingFace revision.

---

## Responsible AI considerations

Machine-readable Croissant RAI fields are shipped alongside this card inside `croissant.json` (validated against the [MLCommons Croissant RAI spec](https://github.com/mlcommons/croissant)).

### Bias & representational concerns
- **Source skew.** Real citations are over-weighted toward NeurIPS, ICML, and ICLR — venues over-represented in DBLP scraping seeds. Tools tuned for medical or humanities citation conventions may underperform on this benchmark.
- **Author-name distribution.** Names follow real-world author distributions in CS publication corpora; tools relying on phonetic name heuristics may exhibit name-origin disparities.
- **Language.** English-only.

### Harm-mitigation choices
- **No PII beyond authorship metadata.** All author names already appear in publicly indexed bibliographies.
- **DOI prefix safety.** Synthetic `fabricated_doi` entries use a fixed `10.99999/` prefix that is not registered with Crossref, preventing accidental hijack of any real DOI.
- **No attack uplift.** The benchmark is purely *detection-oriented*; we do not release prompts or pipelines that would meaningfully accelerate the production of plausible-looking fake citations beyond what is already in the public domain.

### Limitations & known issues
- **Contamination of LLM training data.** LLM-based detectors trained after the NeurIPS 2025 retraction publicity may have seen overlapping content. We recommend reporting `pre_cutoff` vs `post_cutoff` slices via `hallmark evaluate --temporal`.
- **Stress-test asymmetry.** `stress_test` is overwhelmingly hallucinated (121/122). Use this split for *recall* stress-testing, not for FPR estimation.
- **Tier boundary subjectivity.** Tier assignment for `near_miss_title` is annotator-judged and may differ between reasonable annotators on borderline cases.

---

## Reproducibility & code

- **Source code:** <https://anonymous.4open.science/r/hallmark/> (fully executable; MIT license).
- **Reproduction command:** every split is regeneratable via `python scripts/generate_new_instances.py --reproduce-v1 --seed 8042` against the source pools shipped under `sources/`.
- **Pre-computed baseline results:** `baseline_results/` includes outputs for `harc`, `bibtexupdater`, and the LLM-agentic OpenAI baseline on `dev_public`, with checksums in `manifest.json`.

### Pre-screening transparency

Baseline wrappers shipped with HALLMARK include a lightweight pre-screening layer that runs *before* the external citation-verification tool (DOI resolution, year bounds, author-count heuristics). Some detections in the `baseline_results/` files therefore originate from this pre-screening step rather than from the underlying tool itself. Any result whose `reason` field starts with `[Pre-screening override]` was flagged by the pre-screening layer; consumers who want tool-only performance should filter these entries out before computing metrics.

## Provenance and contamination

- **Temporal split**: `publication_date` is recorded for every real-world entry, enabling pre/post-cutoff contamination analysis.
- **Hidden split**: `test_hidden` (454 entries) is **not** included in this release. It is held server-side for an upcoming leaderboard.
- **Blind format**: Use the `blind` config when reporting numbers from any tool that ingests the dataset directly — it removes any field a tool could shortcut on.

## Citation

```
% Anonymized for NeurIPS 2026 D&B double-blind review.
% Citation BibTeX will be added to this card after acceptance / decision release.
```

## License

MIT — see `LICENSE` in the source repository. Source citations themselves are bibliographic metadata in the public domain; perturbed and synthetic entries are released under MIT.
