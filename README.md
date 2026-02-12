# HALLMARK

**HALL**ucination bench**MARK**: A benchmark for evaluating citation hallucination detection tools.

[![Tests](https://github.com/rpatrik96/hallmark/actions/workflows/tests.yml/badge.svg)](https://github.com/rpatrik96/hallmark/actions)
[![Baselines](https://github.com/rpatrik96/hallmark/actions/workflows/baselines.yml/badge.svg)](https://github.com/rpatrik96/hallmark/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why HALLMARK?

The NeurIPS 2025 incident---where 53 papers were found to contain fabricated citations that passed peer review---exposed a critical gap: **we have no standardized way to measure how well tools detect citation hallucinations.** HALLMARK fills this gap.

HALLMARK draws on best practices from established benchmarks:
- **[HumanEval](https://github.com/openai/human-eval)**: Multi-criteria sub-tests per entry (~6 checks per citation)
- **[SWE-bench](https://swebench.com)**: Contamination awareness via temporal segmentation
- **[LiveCodeBench](https://livecodebench.github.io)**: Continuous updates and post-cutoff evaluation
- **[ONEBench](https://arxiv.org/abs/2412.07689)**: Sample-level atomic evaluation with ever-expanding pool

## Features

- **Hallucination taxonomy**: 12 types across 3 difficulty tiers (Easy / Medium / Hard)
- **1,000 annotated entries**: 900 valid (from DBLP) + 100 hallucinated with ground truth
- **6 sub-tests per entry**: DOI resolution, title matching, author consistency, venue verification, field completeness, cross-database agreement
- **Evaluation metrics**: Detection Rate, F1, tier-weighted F1, detect@k
- **Built-in baselines**: DOI-only, bibtex-updater, LLM-based, ensemble, HaRC, CiteVerifier, hallucinator, verify-citations
- **Baseline registry**: Central discovery, availability checking, and dispatch for all baselines
- **Plackett-Luce ranking**: ONEBench-inspired ranking that handles incomplete evaluation data
- **Automated execution**: Orchestrator script and CI workflow for batch baseline evaluation
- **Temporal analysis**: Contamination detection via pre/post-cutoff comparison
- **Community contributions**: ONEBench-style ever-expanding sample pool

## Installation

```bash
pip install hallmark

# With baseline dependencies
pip install hallmark[baselines]

# With ranking support (Plackett-Luce model)
pip install hallmark[ranking]

# All optional dependencies
pip install hallmark[all]

# Development install (recommended: use uv)
git clone https://github.com/rpatrik96/hallmark.git
cd hallmark
uv pip install -e ".[dev]"
```

## Quick Start

### Evaluate a built-in baseline

```bash
# Run DOI-only baseline on the dev split
hallmark evaluate --split dev_public --baseline doi_only

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

### View the leaderboard

```bash
hallmark leaderboard --results-dir results/
```

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
| `swapped_authors` | Author list from different paper | Correct title, wrong author list |
| `preprint_as_published` | arXiv paper cited as venue paper | Correct paper, fabricated venue acceptance |

### Tier 3: Hard (requires deep verification)

| Type | Description | Example |
|------|-------------|---------|
| `near_miss_title` | Title off by 1-2 words | "Attention Is All You Want" vs "...Need" |
| `plausible_fabrication` | Entirely fabricated but realistic | Realistic author + plausible title |
| `retracted_paper` | Citing retracted work | Paper exists but was retracted |
| `version_confusion` | Wrong version claims | v1 claims corrected in v2 |

## Dataset

### Splits

| Split | Valid | Hallucinated | Total | Purpose |
|-------|-------|-------------|-------|---------|
| `dev_public` | 450 | 50 | 500 | Development and tuning |
| `test_public` | 270 | 30 | 300 | Public leaderboard |
| `test_hidden` | 180 | 20 | 200 | Anti-gaming evaluation |

Tier distribution per split: ~40% Tier 1, ~35% Tier 2, ~25% Tier 3.

### Data Format

Each entry is a JSON object in JSONL format:

```json
{
  "bibtex_key": "vaswani2017attention",
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
    "venue_real": true,
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
| **detect@k** | Fraction detected using k verification strategies (analogous to pass@k) |

## Baseline Results (dev_public, 500 entries)

| Baseline | Detection Rate | F1 | Tier-weighted F1 | FPR |
|----------|:---:|:---:|:---:|:---:|
| HaRC | 0.420 | 0.362 | 0.452 | 0.100 |
| bibtex-updater | 0.280 | 0.394 | 0.500 | 0.016 |
| verify-citations | 0.260 | 0.228 | 0.127 | 0.024 |
| DOI-only | 0.240 | 0.163 | 0.175 | 0.189 |
| Ensemble (doi+btx) | 0.040 | 0.069 | 0.078 | 0.013 |

### External Tool Baselines

HALLMARK also wraps several external citation verification tools as baselines:

| Baseline | Tool | Databases | Install |
|----------|------|-----------|---------|
| **HaRC** | [harcx](https://pypi.org/project/harcx/) | Semantic Scholar, DBLP, Google Scholar, Open Library | `pip install harcx` |
| **CiteVerifier** | [GhostCite](https://github.com/NKU-AOSP-Lab/CiteVerifier) | DBLP (local), Google Scholar, Google Search | Clone repo |
| **hallucinator** | [hallucinator](https://github.com/gianlucasb/hallucinator) | CrossRef, arXiv, DBLP, Semantic Scholar, ACL Anthology, PubMed, OpenAlex | Clone repo |
| **verify-citations** | [verify-citations](https://pypi.org/project/verify-citations/) | arXiv, ACL Anthology, Semantic Scholar, DBLP, Google Scholar, DuckDuckGo | `pip install verify-citations` |

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

## Custom Predictions Format

To evaluate your own tool, produce a JSONL file with one prediction per line:

```json
{
  "bibtex_key": "vaswani2017attention",
  "label": "VALID",
  "confidence": 0.95,
  "reason": "DOI resolves, title found in DBLP"
}
```

Then run:
```bash
hallmark evaluate --split dev_public --predictions predictions.jsonl --tool-name my-tool --output results/my_tool.json
```

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
- **`baselines.yml`**: Runs all free baselines (doi_only, bibtexupdater, harc, verify_citations) weekly and on demand, uploading results as artifacts

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
│   ├── baselines/             # Registry + 9 baselines (DOI-only, bibtex-updater, LLM×2, ensemble, HaRC, CiteVerifier, hallucinator, verify-citations)
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
├── tests/                     # Test suite (125 tests)
├── figures/                   # Evaluation figures
└── examples/                  # Usage examples
```

## Citation

If you use HALLMARK in your research, please cite:

```bibtex
@misc{hallmark2026,
    title={HALLMARK: A HALLucination benchMARK for Citation Verification},
    author={Reizinger, Patrik},
    year={2026},
    url={https://github.com/rpatrik96/hallmark}
}
```

## License

MIT
