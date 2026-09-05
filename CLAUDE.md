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

**CRITICAL: When a commit fails due to pre-commit hooks (mypy, ruff, etc.), you MUST immediately fix and retry — NEVER stop, report the failure, or ask the user:**
1. Read and understand every error in the output
2. Fix ALL errors — not just some of them
3. Re-run the failing check manually (e.g. `uv run mypy --ignore-missing-imports hallmark/`) to confirm the fix
4. Re-stage any files modified by hooks or by your fixes (`git add <file>`)
5. Only then attempt the commit again
6. Repeat this cycle until the commit succeeds — do NOT stop or ask the user while errors remain

This is a single atomic workflow. The commit is not done until all hooks pass and git confirms the commit hash. Never leave a commit in a failed state. Never skip or ignore pre-commit hook errors.

## Pre-commit Hook Pitfalls

- **ruff-format auto-modifies files**: The hook reformats files in-place, causing the commit to fail. The fix is already applied — just re-stage the modified file and commit again.
- **mypy vs local runs**: The pre-commit hook runs `mypy hallmark/` with `--ignore-missing-imports`. To match locally: `uv run mypy --ignore-missing-imports hallmark/`
- **Type narrowing**: List comprehensions filtering `None` don't narrow types for mypy. Use the walrus operator: `[s for ... if (s := val) is not None]`
- **`callable` vs `Callable`**: Lowercase `callable` is not a valid type annotation. Use `Callable[[ArgType], ReturnType]` from `collections.abc`.
- **Import + annotation edits**: When adding an import and updating the annotation that uses it, edit the annotation first — otherwise ruff may auto-remove the "unused" import before you get to update the annotation.
- **CI vs pre-commit scope**: Pre-commit hooks only check staged files; CI runs `ruff format --check .` on the entire repo. Always run `uv run ruff format .` before pushing to catch files not covered by the hook.

## Hallucination Taxonomy (11 main + 3 stress-test)
- **Tier 1 (Easy, 4 types):** fabricated_doi, nonexistent_venue, placeholder_authors, future_date
- **Tier 2 (Medium, 5 types):** chimeric_title, wrong_venue, author_mismatch (enum value: `"swapped_authors"`), preprint_as_published, hybrid_fabrication
- **Tier 3 (Hard, 2 main + 1 stress-test):** near_miss_title, plausible_fabrication; arxiv_version_mismatch (stress-test)
- `AUTHOR_MISMATCH` enum member keeps value `"swapped_authors"` for backward compatibility with data files
- `hybrid_fabrication`: real DOI + fabricated metadata — DOI resolves but authors/title don't match the DOI target
- Three theoretically-motivated types (merged_citation, partial_author_list, arxiv_version_mismatch) appear in all splits; `stress_test.jsonl` provides additional depth

## Evaluation Metrics
- **Primary:** Detection Rate, FPR, F1-Hallucination, Tier-weighted F1, ECE
- **Diagnostic:** detect@k, source_stratified_metrics(), subtest_accuracy_table()
- ECE is computed in `evaluate()` and stored in `EvaluationResult.ece`

## Project Structure
- `hallmark/` — main package (baselines, dataset, evaluation, contribution)
- `hallmark/baselines/registry.py` — central baseline registry (discovery, availability, dispatch)
- `hallmark/evaluation/ranking.py` — ONEBench-inspired Plackett-Luce ranking
- `scripts/` — orchestrator scripts (run_all_baselines.py, run_evaluation.py, generate_reference_results.py, generate_new_instances.py)
- `tests/` — pytest test suite (562 tests)
- `data/v1.2/` — benchmark data splits (dev: 1,119, test: 831, hidden: 453, stress: 122; total 2,525 entries)
- `data/v1.2/baseline_results/` — pre-computed reference results for rate-limited baselines
- `.github/workflows/` — CI (tests.yml, baselines.yml)

## Baseline Wrapper Architecture

Baseline wrappers have two distinct layers — keep them clearly separated:

### Tool-level fixes (belong upstream in the external tool)
Changes that fix genuine bugs or missing features in the external tool's wrapper mapping.
Any user of the tool would benefit from these.
- **bibtexupdater**: `year_mismatch`/`venue_mismatch` → HALLUCINATED (was incorrectly suppressed)
- **harc**: false positive filtering for transient API errors, confidence recalibration
- Upstream improvement plan: `~/Documents/GitHub/bibtexupdater/.claude/plans/hallmark-improvements.md`

### Pre-screening layer (benchmark-side addition)
`hallmark/baselines/prescreening.py` adds lightweight local checks (DOI resolution, year bounds, author heuristics) that run before external tools. This is a HALLMARK wrapper addition, not an improvement to the external tools themselves.
- Pre-screening results should be reported transparently (reason strings include `[Pre-screening override]`)
- When reporting results in the paper, clearly attribute which detections come from the tool vs. pre-screening
- The DOI check and year validation are candidates for upstreaming to bibtex-updater (see plan)

## LLM Baselines (OpenAI, Anthropic, OpenRouter, HuggingFace)
- `llm_openai` — GPT-5.1 via `OPENAI_API_KEY`
- `llm_anthropic` — Claude Sonnet 4.6 via `ANTHROPIC_API_KEY`
- `llm_openrouter_{deepseek_r1,deepseek_v3,qwen,mistral,gemini_flash}` — via `OPENROUTER_API_KEY` (uses `openai` SDK with custom `base_url`)
- `llm_openrouter_gemini_flash` — Gemini 2.5 Flash via `OPENROUTER_API_KEY`
- `llm_openrouter_claude_haiku_4_5` — Claude Haiku 4.5 via the OpenRouter mirror
- Model configs live in `OPENROUTER_MODELS` dict in `hallmark/baselines/llm_verifier.py`
- `llm_hf_qwen3_{4b,8b,14b,32b}` — **Qwen3 dense parameter-count sweep, served by Featherless via the HuggingFace router** (`HF_TOKEN`, base URL `https://router.huggingface.co/v1`, configs in `HF_MODELS`). Deregistered from OpenRouter 2026-08-10: OpenRouter has no `qwen3-4b` at all and `qwen3-8b` 404s on the lab key, so it could not serve the full ladder. The `:featherless-ai` suffix pins the provider — without it the router picks whatever is up, which would mix providers across model sizes and confound a sweep whose whole point is that only parameter count varies. The MoE `llm_openrouter_qwen` / `llm_openrouter_qwen_max` are unaffected
- **Thinking-mode guard**: Qwen3 defaults to thinking ON and burns the `max_completion_tokens` budget before emitting the JSON verdict. Model IDs in `_NO_THINK_MODELS` get Qwen3's `/no_think` soft switch appended to the prompt (probed live 2026-08-11 on Featherless: 264 completion tokens → 19). Featherless returns reasoning **inline in `message.content`** as a `<think>...</think>` block, and emits an empty `<think></think>` pair even with `/no_think` — `_strip_think_block()` removes it before parsing, otherwise every reply falls to the regex salvage path that keeps only label+confidence and silently drops `reason` and `predicted_hallucination_type`. The `reasoning_enabled=True/False` kwarg is an explicit opt-in for providers honouring the API-level field; the gpt-5.5 family uses `reasoning_effort="none"` instead
- `_verify_with_openai_compatible()` is the shared helper for all OpenAI-SDK-based providers
- Generation script: `scripts/generate_llm_hallucinations.py --backend openrouter --model deepseek/deepseek-r1`
- Parallel evaluators: `scripts/parallel_resume_test_public.py` (zero-shot), `scripts/parallel_agentic_btu_test_public.py` (agentic) — use for long-running LLM evals with checkpoint recovery

## Key Conventions
- Optional dependencies (choix, harcx, openai, anthropic) use lazy imports
- bibtex-updater, harcx, and verify-citations all require bibtexparser 1.x; install in isolation with `pipx install`
- harc and bibtexupdater baselines time out in CI due to Semantic Scholar API rate-limiting on shared IPs
- Rate-limited baselines use pre-computed reference results: run locally via `python scripts/generate_reference_results.py --baselines harc,bibtexupdater`, commit to `data/v1.2/baseline_results/`, CI validates checksums instead of re-running
- Validate reference results: `hallmark validate-results --results-dir data/v1.2/baseline_results/ --strict`
- CI evaluates live baselines (doi_only, verify_citations) normally; precomputed baselines (harc, bibtexupdater) are validated and copied
- Baselines use the registry pattern in `hallmark/baselines/registry.py`
- Pre-commit hooks run mypy with `--ignore-missing-imports` on `hallmark/`
