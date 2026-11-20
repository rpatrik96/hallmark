#!/bin/zsh
cd /Users/kayoonkim/code/hallmark || exit 1
set -a; source .env; set +a
for s in test_public dev_public; do
  uv run python scripts/run_cascade_shared_stage1.py --splits $s --stage2-baseline llm_agentic_openai_gpt_6_luna \
    --stage1-from results/stage1_swap/ra023_nollm_defer --reuse-stage2 results/stage1_swap/luna/luna \
    --out-dir results/stage1_swap/luna_defer --prefix luna --workers 4 > results/stage1_swap/luna_defer_$s.log 2>&1 || { echo "$s FAILED" >> results/stage1_swap/luna_defer.status; exit 1; }
  echo "$s DONE $(date +%T)" >> results/stage1_swap/luna_defer.status
done
