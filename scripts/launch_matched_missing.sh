#!/usr/bin/env bash
# Robustly (re)launch the matched-split LLM sweep for models that are not yet
# complete. Uses setsid so each stream survives the parent shell exiting (the
# earlier background launch lost its stream for-loops to process-group teardown,
# so only each stream's first model ran). Idempotent: --retry-failed re-does only
# errored rows and completed models return instantly from checkpoint.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; . /tmp/.or_env; . /tmp/.openai_env; set +a
PY=./.venv/bin/python
MD=results/crossdomain_matched_llms

run_stream() {  # $1=log ; $2..=model keys
  local log="$1"; shift
  setsid nohup bash -c '
    for m in "$@"; do
      '"$PY"' -u scripts/evaluate_crossdomain_llms.py --split matched --only "$m" --retry-failed
    done
  ' _ "$@" >> "$MD/$log" 2>&1 &
  echo "  stream -> $MD/$log : $*  (pid $!)"
}

# Every model; completed ones (gpt-5.1, flash, qwen) return instantly from
# checkpoint, so this doubles as a full resume. Grouped to keep ~3-4 concurrent.
run_stream relaunch_a.log llm_openai_gpt54 llm_openrouter_gemini_pro llm_openrouter_llama_4_maverick llm_openrouter_mistral
run_stream relaunch_b.log llm_openrouter_qwen_max llm_openrouter_deepseek_v3
run_stream relaunch_c.log llm_openrouter_claude_opus_4_7 llm_openrouter_deepseek_r1
echo "launched. sonnet (already running) left alone; its checkpoint is preserved."
