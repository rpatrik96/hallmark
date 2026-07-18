#!/usr/bin/env bash
# Resume the in-flight HALLMARK reviewer experiments after an interruption
# (laptop disconnect/sleep/reboot). Idempotent and checkpoint-safe:
#   * cross-domain LLM sweep  -> results/crossdomain_llms/checkpoints/<provider_model>.jsonl
#   * DeepSeek-V4-Pro E2 rerun -> results/reviewer_experiments/e2_latecutoff_control/checkpoints/
# Completed models re-read their checkpoint (500/500) and cost no API calls;
# partial models resume from the last saved entry; --retry-failed re-attempts
# any [Error fallback] rows a dropped connection may have written.
#
# SAFETY: refuses to start if the original runners are still alive, so it can
# never double-write a checkpoint. Run it only AFTER a disconnect has killed them
# (or it will tell you they are still going).
#
# Usage:  bash scripts/resume_experiments.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
XD_LOG_DIR="$REPO/results/crossdomain_llms"
E2_LOG="$REPO/results/reviewer_experiments/e2_latecutoff_control/run_v4pro_full_rerun.log"

# --- guard: never run concurrently with the original processes -------------
if pgrep -f "evaluate_crossdomain_llms.py|e2_latecutoff_control.py" >/dev/null 2>&1; then
  echo "Original runners are still alive — not starting a second writer."
  echo "Wait for them to finish, or kill them first, then re-run this script."
  pgrep -fl "evaluate_crossdomain_llms.py|e2_latecutoff_control.py"
  exit 1
fi

# --- API keys ---------------------------------------------------------------
set -a
[ -f /tmp/.or_env ] && . /tmp/.or_env
[ -f /tmp/.openai_env ] && . /tmp/.openai_env
set +a
: "${OPENROUTER_API_KEY:?source /tmp/.or_env first}"
: "${OPENAI_API_KEY:?source /tmp/.openai_env first}"

cd "$REPO"
mkdir -p "$XD_LOG_DIR"

# --- cross-domain sweep: 4 streams, same grouping as the original launch ----
# Each model is resumable and retry_failed=True re-does interrupted calls.
run_stream () {  # $1 = env-list, $2... = model keys ; writes to a per-stream log
  local log="$1"; shift
  for m in "$@"; do
    "$PY" -u scripts/evaluate_crossdomain_llms.py --only "$m" --retry-failed
  done >> "$XD_LOG_DIR/$log" 2>&1
}

echo "[resume] launching cross-domain sweep (4 streams) ..."
run_stream resume_openai.log      llm_openai llm_openai_gpt54 &
run_stream resume_or_fast.log     llm_openrouter_gemini_flash llm_openrouter_gemini_pro llm_openrouter_llama_4_maverick llm_openrouter_mistral &
run_stream resume_or_qwen_ds.log  llm_openrouter_qwen llm_openrouter_qwen_max llm_openrouter_deepseek_v3 &
run_stream resume_or_claude_r1.log llm_openrouter_claude_sonnet_4_6 llm_openrouter_claude_opus_4_7 llm_openrouter_deepseek_r1 &

# --- V4-Pro E2 control (its own retry_failed is hard-coded in the script) ----
echo "[resume] resuming DeepSeek-V4-Pro E2 control ..."
"$PY" -u scripts/reviewer_experiments/e2_latecutoff_control.py --only deepseek-v4-pro >> "$E2_LOG" 2>&1 &

# --- recency-matched cross-domain sweep (only if the split has been built) ---
MATCHED_SPLIT="$REPO/data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl"
MATCHED_LOG_DIR="$REPO/results/crossdomain_matched_llms"
if [ -f "$MATCHED_SPLIT" ]; then
  echo "[resume] resuming recency-matched cross-domain sweep (4 streams) ..."
  mkdir -p "$MATCHED_LOG_DIR"
  run_matched () {  # $1 = log ; $2... = model keys
    local log="$1"; shift
    for m in "$@"; do
      "$PY" -u scripts/evaluate_crossdomain_llms.py --split matched --only "$m" --retry-failed
    done >> "$MATCHED_LOG_DIR/$log" 2>&1
  }
  run_matched resume_openai.log      llm_openai llm_openai_gpt54 &
  run_matched resume_or_fast.log     llm_openrouter_gemini_flash llm_openrouter_gemini_pro llm_openrouter_llama_4_maverick llm_openrouter_mistral &
  run_matched resume_or_qwen_ds.log  llm_openrouter_qwen llm_openrouter_qwen_max llm_openrouter_deepseek_v3 &
  run_matched resume_or_claude_r1.log llm_openrouter_claude_sonnet_4_6 llm_openrouter_claude_opus_4_7 llm_openrouter_deepseek_r1 &
else
  echo "[resume] matched split not built yet ($MATCHED_SPLIT missing) — skipping matched sweep."
fi

wait
echo "[resume] all resume jobs exited. Aggregate with:"
echo "  $PY scripts/aggregate_crossdomain_llms.py --json results/crossdomain_llms/decomposition.json"
echo "  $PY scripts/analyze_domain_recency_2x2.py --json results/crossdomain_matched_llms/domain_recency_2x2.json"
