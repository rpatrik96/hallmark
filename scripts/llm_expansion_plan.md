# LLM-Generated Data Expansion Plan

## Goal
Expand LLM-generated entries from ~12.6% to ~50% of hallucinated entries across main splits.

## Current State

| Split | Hallucinated | LLM | Perturbation | Adversarial | Real-world | LLM % |
|-------|-------------|-----|--------------|-------------|------------|-------|
| dev_public | 453 | 72 | 286 | 67 | 28 | 15.9% |
| test_public | 363 | 44 | 256 | 36 | 27 | 12.1% |
| test_hidden | 271 | 46 | 171 | 37 | 17 | 17.0% |
| stress_test | 202 | 0 | 202 | 0 | 0 | 0.0% |
| **Total** | **1,289** | **162** | **915** | **140** | **72** | **12.6%** |

## Target State (main splits only)

Strategy: **Replace** perturbation entries with LLM-generated ones (keep total count stable).

| Split | Hallucinated | LLM target | Additional needed | Perturbation after |
|-------|-------------|-----------|-------------------|-------------------|
| dev_public | 453 | ~227 | ~155 | ~131 |
| test_public | 363 | ~182 | ~138 | ~118 |
| test_hidden | 271 | ~136 | ~90 | ~81 |
| **Total** | **1,087** | **~545** | **~383** | **~330** |

Stress-test split stays all-perturbation (theoretically-motivated types only).

## Per-Type LLM Targets (main types only)

Each of the 11 main types should have ~50% LLM entries. With ~29-66 entries per type per split,
target ~15-33 LLM per type per split.

### Priority ordering (ecological validity impact)
1. **High priority** (types where LLM generation is most natural):
   - `plausible_fabrication` — LLMs naturally produce these
   - `chimeric_title` — prompt with real authors, get fake titles
   - `near_miss_title` — LLMs make subtle title errors
   - `fabricated_doi` — LLMs often hallucinate DOIs

2. **Medium priority** (require careful prompting):
   - `wrong_venue` — prompt for specific paper at wrong venue
   - `author_mismatch` — prompt for paper with wrong authors
   - `nonexistent_venue` — LLMs invent plausible venue names
   - `placeholder_authors` — generic bibliography prompts

3. **Lower priority** (harder to elicit naturally):
   - `preprint_as_published` — ask to cite arXiv papers as published
   - `hybrid_fabrication` — give real DOIs, ask for metadata
   - `future_date` — ask for future conference papers

## Multi-Backend Strategy

Use 5 backends for maximum diversity, rotating across types:

| Backend | Model | Entries | Rationale |
|---------|-------|---------|-----------|
| OpenAI | gpt-5.1 | ~100 | Strongest generation, most diverse |
| Anthropic | claude-sonnet-4-5 | ~80 | Different training data distribution |
| Ollama | llama3.1:70b | ~80 | Open-source, different failure modes |
| Mistral | mistral-large | ~60 | European model, different biases |
| Gemini | gemini-2.0-flash | ~63 | Google model, different knowledge base |

## Execution Commands

```bash
# Step 1: Generate with each backend (run separately, ~30 min each)
python scripts/generate_llm_hallucinations.py --backend openai --model gpt-5.1 \
    --strategy both --target-per-type 10 --output data/v1.0/llm_openai.jsonl

python scripts/generate_llm_hallucinations.py --backend anthropic --model claude-sonnet-4-5-20250929 \
    --strategy both --target-per-type 8 --output data/v1.0/llm_anthropic.jsonl

python scripts/generate_llm_hallucinations.py --backend ollama --model llama3.1:70b \
    --strategy both --target-per-type 8 --output data/v1.0/llm_ollama.jsonl

python scripts/generate_llm_hallucinations.py --backend mistral --model mistral-large-latest \
    --strategy both --target-per-type 6 --output data/v1.0/llm_mistral.jsonl

python scripts/generate_llm_hallucinations.py --backend gemini --model gemini-2.0-flash \
    --strategy both --target-per-type 6 --output data/v1.0/llm_gemini.jsonl

# Step 2: Merge and deduplicate
python scripts/merge_llm_entries.py  # TODO: write this

# Step 3: Replace perturbation entries in splits
python scripts/build_dataset.py  # Re-run pipeline with LLM entries integrated

# Step 4: Validate
uv run python -m pytest tests/test_data_integrity.py -q
```

## Quality Control

1. **CrossRef/DBLP verification**: Every LLM entry verified against bibliographic DBs
   - Title similarity >= 85% against any real paper → reject (it's a real paper, not hallucinated)
   - Author Jaccard >= 0.5 against matched paper → reject
2. **Type classification**: Automated classifier assigns hallucination type
3. **Subtest consistency**: Automated check that subtests match assigned type
4. **Diversity check**: No two LLM entries from same prompt/topic
5. **Backend attribution**: Track which backend generated each entry in `source` field

## Integration into Pipeline

The `integrate` stage in `scripts/stages/integrate.py` already handles external JSONL ingestion.
LLM-generated entries go through the same validation pipeline as all other entries.

After generation:
1. Run `scripts/build_dataset.py` with LLM JSONL files in `data/v1.0/`
2. The integrate stage picks them up automatically
3. Sanitize stage ensures URL presence and field consistency
4. Finalize stage writes updated splits

## Paper Updates After Expansion

- Update `limitations.tex`: expand LLM-generated count from 215 to ~550
- Update `benchmark.tex`: generation method description
- Update `metadata.json`: generation_method_distribution
- Re-run all baselines on updated splits
- Re-generate figures (heatmap, tier plots)
