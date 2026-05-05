# Contributing to HALLMARK

HALLMARK uses an ever-expanding pool inspired by [ONEBench](https://arxiv.org/abs/2412.07689). We welcome community contributions of new benchmark entries.

## Entry Requirements

### Valid Entries

Valid entries must:
- Be real, published papers verifiable in at least 2 databases (DBLP, CrossRef, Semantic Scholar)
- Include accurate metadata: title, authors, year, venue, and DOI (if available)
- Have `label: "VALID"` and all subtests set to `true`

### Hallucinated Entries

Hallucinated entries must:
- Have `label: "HALLUCINATED"` with a valid `hallucination_type` from the taxonomy
- Include a `difficulty_tier` (1, 2, or 3)
- Include an `explanation` describing what is wrong and how to detect it
- Have accurate `subtests` indicating which verification checks would fail

### Supported Hallucination Types

**Tier 1 (Easy):** `fabricated_doi`, `nonexistent_venue`, `placeholder_authors`, `future_date`

**Tier 2 (Medium):** `chimeric_title`, `wrong_venue`, `author_mismatch` (covers swapped and fabricated authors; data value: `swapped_authors`), `preprint_as_published`, `hybrid_fabrication`

**Tier 3 (Hard):** `near_miss_title`, `plausible_fabrication`, `arxiv_version_mismatch`

## How to Contribute

### 1. Prepare your entries

Create a JSONL file with one entry per line following the schema in [examples/05_contribute_entries.py](examples/05_contribute_entries.py).

### 2. Validate locally

```bash
# Run validation
python -c "
from hallmark.dataset.schema import load_entries
from hallmark.contribution.validate_entry import validate_batch

entries = load_entries('my_entries.jsonl')
results = validate_batch(entries)
n_valid = sum(1 for r in results if r.valid)
print(f'{n_valid}/{len(results)} entries valid')
for entry, r in zip(entries, results):
    if not r.valid:
        print(f'  INVALID {entry.bibtex_key}: {r.errors}')
"
```

### 3. Submit

```bash
hallmark contribute --file my_entries.jsonl --contributor "Your Name"
```

Or open a pull request adding your entries to `data/pool/contributions/`.

### 4. Review process

- Automated validation checks entry format, required fields, and consistency
- Maintainers verify hallucinated entries are genuinely undetectable at claimed difficulty tier
- Accepted entries are added to the validated pool and may appear in future benchmark versions

## Adding a New Hallucination Type

Adding a new type requires changes in 8 locations:

1. **Enum member**: Add to `HallucinationType` in `hallmark/dataset/schema.py`
2. **Tier mapping**: Add to `HALLUCINATION_TIER_MAP` in `hallmark/dataset/schema.py` with the appropriate `DifficultyTier`
3. **Expected subtests**: Add to `EXPECTED_SUBTESTS` in `hallmark/dataset/schema.py` listing which subtest keys should be present for entries of this type
4. **Stress-test flag** (if applicable): Add to `STRESS_TEST_TYPES`
5. **Generator function**: Create in `hallmark/dataset/generators/tier{N}.py` following the signature pattern:
   ```python
   def generate_your_type(entry: BenchmarkEntry, rng: Random) -> BenchmarkEntry:
   ```
6. **Generator exports**: Add to `hallmark/dataset/generators/__init__.py`
7. **Batch dispatcher**: Add dispatch logic in `hallmark/dataset/generators/batch.py`
8. **Tests**: Add generation tests in `tests/test_generator.py`

See `generate_fabricated_doi` in `tier1.py` for the simplest example.

## Adding a New Baseline

New baselines are registered via the central registry in `hallmark/baselines/registry.py`. To add a baseline:

1. Create a wrapper module in `hallmark/baselines/` that maps the tool's output to `Prediction` objects
2. Register it in `registry.py` with a `BaselineInfo` entry (name, description, runner, dependencies)
3. Add tests in `tests/test_baselines.py`
4. If the baseline is free (no API key), add it to the CI matrix in `.github/workflows/baselines.yml`

See existing baselines (e.g., `verify_citations_baseline.py`) for reference.

### Pre-screening Integration

HALLMARK applies a pre-screening layer (DOI resolution, year bounds, author heuristics) before external tools. For fair comparison, your baseline should integrate pre-screening using the provided helper:

```python
from hallmark.baselines.common import run_with_prescreening

def run_my_tool(entries):
    return run_with_prescreening(
        entries=entries,
        run_tool=_my_tool_core,  # your actual tool logic
    )
```

Results are reported both with and without pre-screening for transparency.

## Development Setup

```bash
git clone https://github.com/rpatrik96/hallmark.git
cd hallmark

# Recommended: use uv
uv pip install -e ".[dev]"

# Run tests
uv run python -m pytest

# Run linter and formatter
uv run ruff check hallmark/ tests/
uv run ruff format hallmark/ tests/

# Type checking
uv run mypy --ignore-missing-imports hallmark/
```

## Reporting Issues

If you find errors in existing benchmark entries (e.g., a "valid" entry that is actually hallucinated, or incorrect metadata), please open an issue with:
- The `bibtex_key` of the affected entry
- The split it appears in
- Evidence of the error
