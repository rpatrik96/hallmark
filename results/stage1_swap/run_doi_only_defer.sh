#!/bin/zsh
# 63JQ Q1: DOI-only as Stage 1 (only its HALLUCINATED verdicts are final; everything else goes to Stage 2),
# Stage 2 = GPT-6 Luna agent. Stage-2 verdicts already made by earlier Luna runs are reused.
cd /Users/kayoonkim/code/hallmark || exit 1
set -a; source .env; set +a
uv run python scripts/run_cascade_shared_stage1.py --splits test_public --stage2-baseline llm_agentic_openai_gpt_6_luna \
  --stage1-from results/stage1_swap/doi_only_defer --reuse-stage2 results/stage1_swap/luna_stage2_union \
  --out-dir results/stage1_swap/doi_only_luna --prefix luna --workers 4 > results/stage1_swap/doi_only_luna_test_public.log 2>&1
