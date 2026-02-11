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
3. Re-run the failing check manually (e.g. `uv run mypy --ignore-missing-imports hallmark/`) to confirm the fix
4. Re-stage any files modified by hooks or by your fixes (`git add <file>`)
5. Only then attempt the commit again
6. Repeat this cycle until the commit succeeds — do NOT stop or ask the user while errors remain

Never leave a commit in a failed state. Never skip or ignore pre-commit hook errors.

## Pre-commit Hook Pitfalls

- **ruff-format auto-modifies files**: The hook reformats files in-place, causing the commit to fail. The fix is already applied — just re-stage the modified file and commit again.
- **mypy vs local runs**: The pre-commit hook runs `mypy hallmark/` with `--ignore-missing-imports`. To match locally: `uv run mypy --ignore-missing-imports hallmark/`
- **Type narrowing**: List comprehensions filtering `None` don't narrow types for mypy. Use the walrus operator: `[s for ... if (s := val) is not None]`
- **`callable` vs `Callable`**: Lowercase `callable` is not a valid type annotation. Use `Callable[[ArgType], ReturnType]` from `collections.abc`.
- **Import + annotation edits**: When adding an import and updating the annotation that uses it, edit the annotation first — otherwise ruff may auto-remove the "unused" import before you get to update the annotation.

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
- bibtex-updater, harcx, and verify-citations all require bibtexparser 1.x; install in isolation with `pipx install`
- harc and bibtexupdater baselines time out in CI due to Semantic Scholar API rate-limiting on shared IPs; run locally for real results
- CI evaluates slow baselines on 50-entry stratified samples (`--max-entries 50`)
- Baselines use the registry pattern in `hallmark/baselines/registry.py`
- Pre-commit hooks run mypy with `--ignore-missing-imports` on `hallmark/`
