# HALLMARK

**HALL**ucination bench**MARK**: A benchmark for evaluating citation hallucination detection tools.

[![Tests](https://github.com/rpatrik96/hallmark/actions/workflows/tests.yml/badge.svg)](https://github.com/rpatrik96/hallmark/actions)
[![Baselines](https://github.com/rpatrik96/hallmark/actions/workflows/baselines.yml/badge.svg)](https://github.com/rpatrik96/hallmark/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.18360-b31b1b.svg)](https://arxiv.org/abs/2607.18360)
[![Website](https://img.shields.io/badge/website-interactive_companion-blue.svg)](https://rpatrik96.github.io/hallmark/)

## Why HALLMARK?

The NeurIPS 2025 incident---where 53 papers were found to contain fabricated citations that passed peer review---exposed a critical gap: **we have no standardized way to measure how well tools detect citation hallucinations.** HALLMARK fills this gap.

HALLMARK draws on best practices from established benchmarks:
- **[HumanEval](https://github.com/openai/human-eval)**: Multi-criteria sub-tests per entry (~6 checks per citation)
- **[SWE-bench](https://swebench.com)**: Contamination awareness via temporal segmentation
- **[LiveCodeBench](https://livecodebench.github.io)**: Continuous updates and post-cutoff evaluation
- **[ONEBench](https://arxiv.org/abs/2412.07689)**: Sample-level atomic evaluation with ever-expanding pool

## Features

- **Hallucination taxonomy**: 14 types across 3 difficulty tiers (Easy / Medium / Hard)
- **2,526 annotated entries**: 826 valid + 1,246 hallucinated with ground truth across the public splits, plus a 454-entry hidden split
- **6 sub-tests per entry**: DOI resolution, title matching, author consistency, venue verification, field completeness, cross-database agreement
- **Evaluation metrics**: Detection Rate, F1, tier-weighted F1, detect@k, ECE
- **Built-in baselines**: DOI-only, bibtex-updater, HaRC, verify-citations, LLM-based (OpenAI, Anthropic, OpenRouter), agentic LLMs with tool use, ensemble, DB-first cascade with hallucination-mode diagnosis, plus ports of two recent papers — `hallucitechecker` ([Sakai et al. 2026](https://arxiv.org/abs/2604.26835)) and `checkifexist` ([Abbonato 2026](https://arxiv.org/abs/2602.15871) Algorithm 1) (CiteVerifier and hallucinator are available as wrapper modules but not registered in the default registry)
- **Baseline registry**: Central discovery, availability checking, and dispatch for all baselines (19+ variants)
- **Reproducible runs**: opt-in `--cache-path` flag wraps HTTP calls in a SQLite-backed `requests-cache` so re-runs reuse frozen API responses; `--timing-breakdown` and `--subtask-diagnostic` surface per-baseline performance + recognition/matching/calibration decomposition
- **Plackett-Luce ranking**: ONEBench-inspired ranking that handles incomplete evaluation data
- **Automated execution**: Orchestrator script and CI workflow for batch baseline evaluation
- **Temporal analysis**: Contamination detection via pre/post-cutoff comparison
- **Community contributions**: ONEBench-style ever-expanding sample pool

## Headline cascade results (v1.1)

`cascade_db_diagnosis` — Stage 1 bibtex-updater + Stage 2 Claude Sonnet 4.6 (via OpenRouter, up to 5 tool calls), conservative vs aggressive scoring of residual `UNCERTAIN`:

| Split          | Mode         |   DR  |  FPR  |   F1  | Tier-3 F1 | AUROC |
|----------------|--------------|------:|------:|------:|----------:|------:|
| `dev_public`   | conservative | 0.976 | 0.559 | 0.760 |     0.417 | 0.833 |
| `dev_public`   | **aggressive** | 0.983 | 0.560 | 0.815 | **0.570** | 0.740 |
| `test_public`  | conservative | 0.972 | 0.456 | 0.854 |     0.596 | 0.867 |
| `test_public`  | **aggressive** | 0.978 | 0.456 | 0.882 | **0.707** | 0.805 |
| `stress_test`  | conservative | 0.969 |   —   | 0.985 |     0.983 |   —   |
| `stress_test`  | **aggressive** | 0.975 |   —   | 0.987 |     0.986 |   —   |

Aggressive promotion of residual `UNCERTAIN` (the "DB-as-gold-standard" stance) lifts Tier-3 F1 by **+11.1 pp on `test_public`** and **+15.3 pp on `dev_public`** at ≤0.1 pp FPR cost; the trade is paid in rank-discrimination (AUROC −6.2 / −9.3 pp). Runner-level (`cascade_db_diagnosis_aggressive`) and evaluator-level (`--eval-mode aggressive`) promotion paths agree to within ~1 pp on every metric. Full JSONs (incl. per-tier/per-type breakdowns) in [`data/v1.0/baseline_results/`](data/v1.0/baseline_results/); see paper §Stage-2 diagnosis cascade for analysis.

## Installation

```bash
# Recommended: clone and install in development mode
git clone https://github.com/rpatrik96/hallmark.git
cd hallmark
uv pip install -e ".[dev]"

# With LLM baseline SDKs (openai, anthropic)
uv pip install -e ".[baselines]"

# With ranking support (Plackett-Luce model via choix)
uv pip install -e ".[ranking]"

# All optional dependencies
uv pip install -e ".[all]"
```

> **Note**: `pip install hallmark` is not yet published to PyPI. Use the clone + install path above.

### Baseline Installation Guide

The `[baselines]` extra installs only the LLM SDKs (`openai`, `anthropic`). External CLI tools require separate installation due to a `bibtexparser` 1.x dependency conflict:

```bash
# HaRC
pipx install harcx

# bibtex-updater (released HALLMARK numbers use tag v1.2.0)
pipx install "bibtex-updater==1.2.0"

# verify-citations
pipx install verify-citations

# CiteVerifier (GhostCite) — clone required
git clone https://github.com/NKU-AOSP-Lab/CiteVerifier

# hallucinator — clone required
git clone https://github.com/gianlucasb/hallucinator
```

Using `pipx` isolates each tool's `bibtexparser` 1.x from your project environment.

## Quick Start

### Evaluate a built-in baseline

```bash
# Run DOI-only baseline on the dev split
hallmark evaluate --split dev_public --baseline doi_only

# Run the v1.1 cascade with aggressive scoring (DB as gold standard).
# Stage 2 LLM diagnoser is routed through OpenRouter — set OPENROUTER_API_KEY.
hallmark evaluate --split dev_public --baseline cascade_db_diagnosis_aggressive \
    --stage2-baseline llm_agentic_openrouter_claude_sonnet_4_6

# Re-score the same predictions under both eval modes (conservative + aggressive)
# in a single payload — the gap quantifies the abstention/indexing-lag tax.
hallmark evaluate --split dev_public --baseline cascade_db_diagnosis \
    --stage2-baseline llm_agentic_openrouter_claude_sonnet_4_6 \
    --eval-mode both

# Run with custom predictions
hallmark evaluate --split dev_public --predictions my_predictions.jsonl --tool-name my-tool
```

### Show dataset statistics

```bash
hallmark stats --split dev_public
```

### Run all baselines at once

```bash
# Run all free baselines and generate leaderboard
python scripts/run_all_baselines.py --split dev_public --output-dir results/

# Run specific baselines in parallel
python scripts/run_all_baselines.py --baselines doi_only,bibtexupdater --parallel

# Run only free (no API key) baselines, skip unavailable
python scripts/run_all_baselines.py --baselines free --skip-unavailable
```

### Resume long-running LLM evaluations in parallel

For LLM-based baselines that take >1 hour sequentially, use the parallel-resume scripts to checkpoint and resume:

```bash
# Resume zero-shot OpenRouter LLM baselines across multiple processes
python scripts/parallel_resume_test_public.py --split test_public --num-workers 4

# Resume agentic verifiers (BTU, multi-tool, tool-augmented) with Sonnet 4.6
python scripts/parallel_agentic_btu_test_public.py --split test_public --verifier agentic_btu_openai
```

Both scripts support checkpointing and can safely resume interrupted runs without recomputing completed entries.

### View the leaderboard

```bash
hallmark leaderboard --results-dir results/
```

See [`examples/`](examples/) for full walkthroughs, including [writing a custom baseline](examples/03_custom_baseline.py) and [per-type analysis](examples/02_per_type_analysis.py).

## Evaluate Your Tool

To evaluate any external tool against HALLMARK, produce a JSONL file with one prediction per line and run:

```bash
hallmark evaluate --predictions my_preds.jsonl --split dev_public
```

Each prediction must include:

```json
{
  "bibtex_key": "a3f9c2b1...",
  "label": "HALLUCINATED",
  "confidence": 0.87,
  "reason": "DOI does not resolve",
  "subtest_results": {"doi_resolves": false},
  "api_sources_queried": ["crossref"],
  "wall_clock_seconds": 1.2,
  "api_calls": 1
}
```

> **bibtex_key format**: Keys in the benchmark are hex hashes (e.g., `a3f9c2b1d4e7...`), not human-readable keys like `vaswani2017attention`. Your predictions **must** use the exact keys from the loaded entries — use `entry.bibtex_key` when iterating over `load_split()` results.

See [`examples/03_custom_baseline.py`](examples/03_custom_baseline.py) for a complete end-to-end example.

### Prediction Fields

| Field | Required | Affects |
|-------|----------|---------|
| `bibtex_key` | Yes | Entry matching |
| `label` | Yes | All metrics |
| `confidence` | Yes | ECE, AUROC, AUPRC |
| `reason` | No | Diagnose output |
| `subtest_results` | No | Subtest accuracy |
| `api_sources_queried` | No | Source-stratified metrics |
| `wall_clock_seconds` | No | Cost efficiency |
| `api_calls` | No | Mean API calls |

**UNCERTAIN label**: `UNCERTAIN` is accepted as a prediction label. `UNCERTAIN` predictions are treated as `VALID` for confusion-matrix metrics (conservative default) and excluded from AUROC/AUPRC. Prefer `VALID` or `HALLUCINATED` with calibrated confidence when possible.

**Confidence semantics**: `confidence` = P(your predicted label is correct). If you predict `HALLUCINATED` with 0.9, you claim 90% certainty it is hallucinated. If you predict `VALID` with 0.8, you claim 80% certainty it is valid. This is NOT P(HALLUCINATED).

## Hallucination Taxonomy

### Tier 1: Easy (detectable by simple API lookup)

| Type | Description | Example |
|------|-------------|---------|
| `fabricated_doi` | DOI that doesn't resolve | `doi = {10.9999/fake.2024.001}` |
| `nonexistent_venue` | Invented journal/conference | `booktitle = {Intl. Conf. on Advanced AI Systems}` |
| `placeholder_authors` | Generic/fake author names | `author = {John Doe and Jane Smith}` |
| `future_date` | Publication year in the future | `year = {2030}` |

### Tier 2: Medium (requires cross-referencing metadata)

| Type | Description | Example |
|------|-------------|---------|
| `chimeric_title` | Real author + fabricated title | Real authors, plausible but non-existent paper |
| `wrong_venue` | Real paper, wrong venue/year | Correct title but at ICML not NeurIPS |
| `author_mismatch` | Author list swapped or fabricated (data value: `swapped_authors`) | Correct title, wrong author list |
| `preprint_as_published` | arXiv paper cited as venue paper | Correct paper, fabricated venue acceptance |
| `hybrid_fabrication` | Real DOI + fabricated metadata | Valid DOI resolves but authors/title don't match |
| `merged_citation` | Metadata from 2-3 papers merged | Authors from paper A, title from paper B |
| `partial_author_list` | Subset of real author list | First and last author only, middle dropped |

### Tier 3: Hard (requires deep verification)

| Type | Description | Example |
|------|-------------|---------|
| `near_miss_title` | Title off by 1-2 words | "Attention Is All You Want" vs "...Need" |
| `plausible_fabrication` | Entirely fabricated but realistic | Realistic author + plausible title |
| `arxiv_version_mismatch` | Mixed preprint/published metadata | arXiv ID with conference venue claim |

## Hosting & Croissant

The dataset is mirrored on HuggingFace (parquet + jsonl + baseline results + RAI Croissant metadata):
<https://huggingface.co/datasets/hallmark-neurips2026/HALLMARK>

A [Croissant 1.0](https://mlcommons.org/croissant/) metadata file is included at the repo root (`croissant.json`). It covers all public splits and includes RAI fields required by NeurIPS 2026 D&B. Validate locally with:

```bash
mlcroissant validate --jsonld croissant.json
```

The data is also shipped in `data/v1.0/` for direct repo-relative access without any external download.

## Dataset

### Splits

| Split | Valid | Hallucinated | Total | Purpose |
|-------|-------|-------------|-------|---------|
| `dev_public` | 513 | 606 | 1,119 | Development and tuning |
| `test_public` | 312 | 519 | 831 | Public leaderboard |
| `test_hidden` | — | — | 454 | Anti-gaming evaluation |
| `stress_test` | 1 | 121 | 122 | Stress-test types depth |

> **stress_test design note**: The `stress_test` split is all-hallucinated by design. It contains
> challenging edge cases (merged citations, partial author lists, arXiv version mismatches) intended
> to stress-test detection robustness beyond the main splits. Because there are no valid entries,
> FPR and specificity are undefined for this split. Use **detection rate** as the primary metric
> when reporting `stress_test` results.

Tier distribution per split: ~27% Tier 1, ~47% Tier 2, ~26% Tier 3 (hallucinated entries).

### Subtest Definitions

| Subtest | Definition |
|---------|------------|
| `doi_resolves` | DOI returns HTTP 200 from doi.org (redirects count as resolved) |
| `title_exists` | Title found in Semantic Scholar or DBLP via exact or fuzzy match (threshold 0.9) |
| `authors_match` | Author last names match the record retrieved via DOI or title lookup |
| `venue_correct` | The venue/journal is correct for this specific paper (not just "a real venue") |
| `fields_complete` | All standard BibTeX fields for this entry type are present and non-empty |
| `cross_db_agreement` | Metadata from DOI resolution matches metadata from title/author search in DBLP/S2 |

### Data Format

Each entry is a JSON object in JSONL format:

> **bibtex_key format**: Keys are hex hashes (e.g., `a3f9c2b1d4e7...`), not human-readable keys. When writing predictions, always use `entry.bibtex_key` directly — do not construct keys manually.

```json
{
  "bibtex_key": "a3f9c2b1d4e76f85",
  "bibtex_type": "inproceedings",
  "fields": {
    "title": "Attention Is All You Need",
    "author": "Ashish Vaswani and Noam Shazeer and ...",
    "year": "2017",
    "booktitle": "NeurIPS",
    "doi": "10.5555/3295222.3295349"
  },
  "label": "VALID",
  "hallucination_type": null,
  "difficulty_tier": null,
  "explanation": "Valid entry scraped from DBLP and verified",
  "subtests": {
    "doi_resolves": true,
    "title_exists": true,
    "authors_match": true,
    "venue_correct": true,
    "fields_complete": true,
    "cross_db_agreement": true
  }
}
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Detection Rate (DR)** | Recall on hallucinated entries |
| **False Positive Rate (FPR)** | Valid entries incorrectly flagged |
| **F1-Hallucination** | Harmonic mean of precision and recall on HALLUCINATED class |
| **Tier-weighted F1** | F1 weighted by difficulty (Tier 3 = 3x weight) |
| **ECE** | Expected Calibration Error — measures confidence calibration quality |
| **detect@k** | Fraction detected using k verification strategies (deterministic and order-dependent, unlike the stochastic pass@k) |
| **MCC** | Matthews Correlation Coefficient — prevalence-invariant; use as primary metric when comparing results across splits |

### Title-Oracle Baseline (Diagnostic)

The `title_oracle` baseline quantifies the ceiling of a perturbation-structure shortcut present in HALLMARK's design.
Because most HALLUCINATED entries are generated by perturbing real (VALID) papers, they inherit the original title.
This means a title that appears as VALID in the dev split almost certainly belongs to a perturbed — hence hallucinated — entry when it reappears in another split.

The oracle exploits this directly: if a blind entry's title matches any VALID title in the dev split, it predicts HALLUCINATED.

Empirical results on v1.0 data:
- ~33% of unique titles appear as both VALID and HALLUCINATED across dev/test splits.
- Applied to the hidden split: F1 = 0.389 at perfect precision (P = 1.0, recall = ~0.24).
- Titles absent from any valid pool are 100% HALLUCINATED in the dataset.

**This is not a legitimate detection method** — it requires access to dev ground-truth labels as a look-up table, which constitutes label leakage when evaluating on dev itself.
Report it alongside real baselines to make the shortcut visible.
Any real tool that achieves F1 below the title oracle on the hidden split is arguably exploiting benchmark structure rather than performing genuine citation verification.

```python
from hallmark.baselines.title_oracle import run_title_oracle
from hallmark.dataset.loader import load_split

dev_entries  = load_split("dev_public")
test_entries = load_split("test_public")
blind_test   = [e.to_blind() for e in test_entries]

predictions = run_title_oracle(blind_test, reference_pool=dev_entries)
```

## Main Results (dev_public, 1,119 entries)

Twelve full-coverage tools evaluated on `dev_public`. All numbers reproduce Table 1 of the paper. **Bold** = best among independent (non-co-designed) full-coverage tools. ΔFPR is the cross-split shift `test_public − dev_public`; `—` means no `test_public` evaluation.

| Tool | DR ↑ | FPR ↓ | F1 ↑ | MCC ↑ | TW-F1 ↑ | ECE ↓ | ΔFPR ↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Citation-database tools (with shared pre-screening)* | | | | | | | |
| DOI-only | .256 | .195 | .361 | .093 | .314 | .143 | +0.108 |
| *Zero-shot LLMs (sorted by FPR)* | | | | | | | |
| Gemini 2.5 Pro | .456 | **.053** | .609 | .446 | .587 | .321 | +0.011 |
| Claude Opus 4.7 | .733 | .060 | .824 | .672 | .840 | .112 | −0.001 |
| Claude Sonnet 4.6 | .777 | .095 | **.840** | **.677** | **.842** | **.066** | +0.023 |
| Gemini 2.5 Flash | .482 | .101 | .617 | .406 | .608 | .286 | +0.010 |
| Llama 4 Maverick | .591 | .150 | .693 | .446 | .688 | .197 | +0.028 |
| GPT-5.4 (zero-shot) | .744 | .228 | .775 | .512 | .792 | .215 | −0.005 |
| Mistral Large | .691 | .258 | .731 | .430 | .743 | .247 | +0.045 |
| GPT-5.1 (zero-shot) | .823 | .405 | .771 | .432 | .818 | .189 | +0.076 |
| Qwen3-235B | .832 | .551 | .737 | .307 | .806 | .294 | +0.080 |
| Qwen3-VL-235B | .834 | .567 | .735 | .294 | .804 | .298 | +0.085 |
| DeepSeek-R1 | .871 | .640 | .737 | .273 | .814 | .247 | −0.310 |
| DeepSeek-V3.2 | **.880** | .730 | .721 | .191 | .805 | .331 | +0.047 |
| *Agentic (tool-use; up to 5 tool calls per entry)* | | | | | | | |
| GPT-5.1 + CrossRef/OpenAlex/arXiv | .956 | .465 | .827 | .556 | .895 | .165 | +0.058 |
| GPT-5.1 + bibtex-updater (tool optional) | .965 | .461 | .832 | .574 | .901 | .113 | −0.116 |
| Sonnet 4.6 + bibtex-updater (tool optional) | .970 | .426 | .845 | .610 | .908 | .110 | −0.092 |
| *Co-designed (reference upper bound)* | | | | | | | |
| bibtex-updater | .946 | .179 | .908 | .781 | .936 | .297 | +0.159 |
| GPT-5.1 + bibtex-updater (always-call; output in prompt) | .818 | .144 | .846 | .670 | .856 | .086 | +0.110 |

DR = Detection Rate · FPR = False Positive Rate · TW-F1 = Tier-weighted F1 · MCC = Matthews Correlation Coefficient · ECE = Expected Calibration Error. The shaded *co-designed* block is a reference upper bound: `bibtex-updater`'s development overlapped with the benchmark's taxonomy design, so its scores risk construct-overfitting and should not be compared head-to-head with independent tools. `HaRC` and `verify-citations` are omitted: Semantic Scholar throttling collapses their effective coverage to <7% on `dev_public`.

### Key Takeaways

1. **LLMs span a wide recall–precision spectrum.** From ultra-conservative (Gemini 2.5 Pro: 46% DR, 5% FPR) to aggressive (DeepSeek-V3.2: 88% DR, 73% FPR). Claude Sonnet 4.6 and Opus 4.7 jointly lead independent tools on F1/calibration (Sonnet F1 = 0.840 / ECE = 0.066), far ahead of GPT-5.1 (F1 0.771) and the recall-aggressive open-weight cohort.

2. **Agentic lookups inflate FPR.** A 5-call budget closes GPT-5.1's recall gap to `bibtex-updater` (DR 0.97 vs. 0.95), but agentic FPR remains ~2.6× higher (0.46 vs. 0.18) because the harness flags an entry whenever any one of CrossRef/OpenAlex/arXiv returns no match. F1 still trails by 7.6 pp. Substituting Sonnet 4.6 reproduces the GPT-5.1 profile within ≤3.5 pp on every metric — the FPR rise is harness-driven, not LLM-driven.

3. **Base-rate precision collapse.** Extrapolated to real-world hallucination rates, every evaluated setting yields roughly one true hallucination per ten flagged citations, so recall-optimized verifiers misallocate reviewer effort.

4. **Post-cutoff calibration breakdown.** On 448 papers from 2024–2025, 8 of 12 LLMs over-flag sharply (FPR up to 0.89). Sonnet 4.6 and Opus 4.7 hold FPR ≤ 0.12; GPT-5.4 (FPR 0.41) and Gemini 2.5 Pro (FPR 0.25) only partially recover.

5. **A capability gap remains.** Even the highest-recall independent model misses 12% of hallucinations, with systematic weaknesses on subtle types (`near_miss_title`: 56%, `author_mismatch`: 58% for GPT-5.1). No tool dominates across regimes: `bibtex-updater` is cheapest and most temporally stable; Sonnet 4.6 / Opus 4.7 lead on FPR and PPV; the rule-based F1 lead collapses on `test_public`.

See the [paper](https://arxiv.org/abs/2607.18360) for the full per-tier, per-type, and temporal-robustness analyses, or explore them interactively on the [companion website](https://rpatrik96.github.io/hallmark/).

### External Tool Baselines

HALLMARK also wraps several external citation verification tools as baselines:

| Baseline | Tool | Databases | Install |
|----------|------|-----------|---------|
| **HaRC** | [harcx](https://pypi.org/project/harcx/) | Semantic Scholar, DBLP, Google Scholar, Open Library | `pip install harcx` |
| **CiteVerifier** | [GhostCite](https://github.com/NKU-AOSP-Lab/CiteVerifier) | DBLP (local), Google Scholar, Google Search | Clone repo |
| **hallucinator** | [hallucinator](https://github.com/gianlucasb/hallucinator) | CrossRef, arXiv, DBLP, Semantic Scholar, ACL Anthology, PubMed, OpenAlex | Clone repo |
| **verify-citations** | [verify-citations](https://pypi.org/project/verify-citations/) | arXiv, ACL Anthology, Semantic Scholar, DBLP, Google Scholar, DuckDuckGo | `pip install verify-citations` |

### LLM Baselines

| Baseline | Model | Provider | API Key Env Var |
|----------|-------|----------|----------------|
| `llm_openai` | GPT-5.1 | OpenAI | `OPENAI_API_KEY` |
| `llm_anthropic` | Claude Sonnet 4.6 | Anthropic | `ANTHROPIC_API_KEY` |
| `llm_openrouter_deepseek_r1` | DeepSeek R1 | OpenRouter | `OPENROUTER_API_KEY` |
| `llm_openrouter_deepseek_v3` | DeepSeek V3.2 | OpenRouter | `OPENROUTER_API_KEY` |
| `llm_openrouter_qwen` | Qwen 3 235B | OpenRouter | `OPENROUTER_API_KEY` |
| `llm_openrouter_mistral` | Mistral Large | OpenRouter | `OPENROUTER_API_KEY` |
| `llm_openrouter_gemini_flash` | Gemini 2.5 Flash | OpenRouter | `OPENROUTER_API_KEY` |

```python
# Use the baseline registry to discover and run any baseline
from hallmark.baselines.registry import list_baselines, check_available, run_baseline
from hallmark.dataset.loader import load_split

entries = load_split("dev_public")

# List all registered baselines (or just the free ones)
print(list_baselines(free_only=True))

# Check if a baseline's dependencies are installed
available, msg = check_available("harc")

# Run a baseline by name
predictions = run_baseline("harc", entries)
```

See also:
- [GhostCite paper](https://arxiv.org/abs/2602.06718) — large-scale analysis of 2.2M citations across 56K papers
- [HalluCitation paper](https://arxiv.org/abs/2601.18724) — analysis of ~300 hallucinated papers in ACL conferences
- [GPTZero Hallucination Detector](https://gptzero.me/hallucination-detector) — commercial API for citation verification


## Python API

```python
from hallmark.dataset.loader import load_split
from hallmark.evaluation.metrics import evaluate
from hallmark.dataset.schema import Prediction

# Load benchmark entries
entries = load_split("dev_public")

# Create predictions (your tool's output)
predictions = [
    Prediction(bibtex_key=e.bibtex_key, label="VALID", confidence=0.5)
    for e in entries
]

# Evaluate
result = evaluate(entries, predictions, tool_name="my-tool", split_name="dev_public")
print(f"F1: {result.f1_hallucination:.3f}")
print(f"Detection Rate: {result.detection_rate:.3f}")
```

## Ranking

HALLMARK includes an ONEBench-inspired ranking system based on the [Plackett-Luce model](https://en.wikipedia.org/wiki/Luce%27s_choice_axiom) that handles incomplete evaluation data (not all tools evaluated on all entries):

```python
from hallmark.evaluation.ranking import rank_tools_plackett_luce, rank_tools_mean_score

# Rank tools using Plackett-Luce (requires choix: pip install hallmark[ranking])
pl_ranking = rank_tools_plackett_luce(entry_keys, tool_names, matrix)

# Fallback: simple mean-score ranking (no extra dependencies)
mean_ranking = rank_tools_mean_score(entry_keys, tool_names, matrix)
```

## CI/CD

HALLMARK includes two GitHub Actions workflows:

- **`tests.yml`**: Runs the full test suite across Python 3.10-3.13 on every push/PR
- **`baselines.yml`**: Runs live free baselines (doi_only, verify_citations) weekly and on demand; harc and bibtexupdater use pre-computed result validation (checksum checks) instead of live re-execution due to API rate limiting

## Contributing Entries

HALLMARK uses an ever-expanding pool inspired by ONEBench. To contribute new entries:

```bash
hallmark contribute --file my_entries.jsonl --contributor "Your Name"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on entry format, validation requirements, and the review process.

## Project Structure

```
hallmark/
├── hallmark/                  # Python package
│   ├── dataset/               # Schema, loader, scraper, generator
│   ├── evaluation/            # Metrics, subtests, aggregator, temporal, ranking
│   ├── baselines/             # Registry + baselines (DOI-only, bibtex-updater, LLM×6, ensemble, HaRC, CiteVerifier, hallucinator, verify-citations)
│   │   └── registry.py        # Central baseline discovery, availability, dispatch
│   ├── contribution/          # Pool manager, entry validation
│   └── cli.py                 # Command-line interface
├── data/
│   ├── v1.0/                  # Benchmark splits (dev_public, test_public)
│   ├── hidden/                # Hidden test set (not public)
│   └── raw/                   # Raw scraped/generated entries
├── scripts/
│   └── run_all_baselines.py   # Batch orchestrator for baseline evaluation
├── .github/workflows/
│   ├── tests.yml              # CI: test suite across Python versions
│   └── baselines.yml          # CI: weekly free baseline evaluation
├── tests/                     # Test suite (562 tests)
├── figures/                   # Evaluation figures
└── examples/                  # Usage examples
```

## Citation

If you use HALLMARK in your research, please cite:

```bibtex
@misc{reizinger2026hallmarkdiagnosingfailuremodes,
      title={HALLMARK: Diagnosing Three Failure Modes in LLM Citation Verifiers},
      author={Patrik Reizinger and Wieland Brendel},
      year={2026},
      eprint={2607.18360},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2607.18360},
}
```

## License

MIT
