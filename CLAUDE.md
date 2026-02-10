# HALLMARK - Citation Hallucination Detection Benchmark

## Project Overview
HALLMARK (HALLucination benchMARK) evaluates citation verification tools on detecting hallucinated references in academic papers.

## Development Setup
- Package manager: `uv`
- Run tests: `uv run python -m pytest`
- Lint: `uv run ruff check hallmark/ tests/`
- Format: `uv run ruff format hallmark/ tests/`
- Type check: `uv run mypy hallmark/`
- Pre-commit hooks: ruff, ruff-format, mypy, trailing-whitespace, end-of-file-fixer

## Commit Rules

**CRITICAL: When a commit fails due to pre-commit hooks (mypy, ruff, etc.), you MUST:**
1. Read and understand every error in the output
2. Fix ALL errors — not just some of them
3. Re-run the failing check manually (e.g. `uv run mypy hallmark/`) to confirm the fix
4. Only then attempt the commit again
5. Repeat this cycle until the commit succeeds — do NOT stop or ask the user while errors remain

Never leave a commit in a failed state. Never skip or ignore pre-commit hook errors.

## Project Structure
- `hallmark/` — main package (baselines, dataset, evaluation, contribution)
- `hallmark/baselines/registry.py` — central baseline registry (discovery, availability, dispatch)
- `hallmark/evaluation/ranking.py` — ONEBench-inspired Plackett-Luce ranking
- `scripts/` — orchestrator scripts (run_all_baselines.py, run_evaluation.py)
- `tests/` — pytest test suite
- `data/v1.0/` — benchmark data splits
- `.github/workflows/` — CI (tests.yml, baselines.yml)

## Key Conventions
- Optional dependencies (choix, harcx, openai, anthropic) use lazy imports
- harcx conflicts with bibtexparser>=2.0; install separately with `--no-deps`
- Baselines use the registry pattern in `hallmark/baselines/registry.py`
- Pre-commit hooks run mypy with `--ignore-missing-imports` on `hallmark/`
