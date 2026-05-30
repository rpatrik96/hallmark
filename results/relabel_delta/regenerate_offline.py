"""Regenerate all OFFLINE-RESCORABLE aggregate JSONs against the NEW labels.

Each tool/split here has a full-coverage per-entry prediction file that
reproduces the published aggregate against the pre-relabel (7a52362) labels
(verified separately). We recompute on the current labels and write the result
into data/v1.0/baseline_results/, preserving the published JSON's key set.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _rescore import rescore
from _writer import write_aggregate

REPO = Path("/Users/patrik.reizinger/Documents/GitHub/hallmark")
OUT = REPO / "data/v1.0/baseline_results"


def load_template(pub_rel: str) -> dict:
    """Read the PRISTINE published aggregate from git HEAD (idempotent vs working tree)."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{pub_rel}"],
        capture_output=True,
        text=True,
    )
    if out.returncode == 0 and out.stdout.strip():
        return json.loads(out.stdout)
    return json.loads((REPO / pub_rel).read_text())


# (out_filename, pred_path, split, tool_name, published_template_path, eval_mode)
JOBS = [
    # ---- dev_public zero-shot LLMs (offline) ----
    (
        "llm_openrouter_deepseek_r1_dev_public.json",
        "results/llm_openrouter_deepseek_r1_dev_public_predictions.jsonl",
        "dev_public",
        "llm_openrouter_deepseek_r1",
        "results/llm_openrouter_deepseek_r1_dev_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_deepseek_v3_dev_public.json",
        "results/llm_openrouter_deepseek_v3_dev_public_predictions.jsonl",
        "dev_public",
        "llm_openrouter_deepseek_v3",
        "results/llm_openrouter_deepseek_v3_dev_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_gemini_flash_dev_public.json",
        "results/llm_openrouter_gemini_flash_dev_public_predictions.jsonl",
        "dev_public",
        "llm_openrouter_gemini_flash",
        "results/llm_openrouter_gemini_flash_dev_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_mistral_dev_public.json",
        "results/llm_openrouter_mistral_dev_public_predictions.jsonl",
        "dev_public",
        "llm_openrouter_mistral",
        "results/llm_openrouter_mistral_dev_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_qwen_dev_public.json",
        "results/llm_openrouter_qwen_dev_public_predictions.jsonl",
        "dev_public",
        "llm_openrouter_qwen",
        "results/llm_openrouter_qwen_dev_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_gemini_pro_dev_public.json",
        "results/new_models/gemini_pro.jsonl",
        "dev_public",
        "llm_openrouter_gemini_pro",
        "results/new_models/gemini_pro.json",
        "conservative",
    ),
    (
        "llm_openrouter_qwen_max_dev_public.json",
        "results/new_models/qwen_max.jsonl",
        "dev_public",
        "llm_openrouter_qwen_max",
        "results/new_models/qwen_max.json",
        "conservative",
    ),
    (
        "llm_openrouter_llama_4_maverick_dev_public.json",
        "results/new_models/llama4_maverick.jsonl",
        "dev_public",
        "llm_openrouter_llama_4_maverick",
        "results/new_models/llama4_maverick.json",
        "conservative",
    ),
    (
        "llm_openai_dev_public.json",
        "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
        "dev_public",
        "llm_openai",
        "data/v1.0/baseline_results/llm_openai_dev_public.json",
        "conservative",
    ),
    (
        "llm_openai_gpt54_dev_public.json",
        "results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl",
        "dev_public",
        "llm_openai_gpt-5.4",
        "data/v1.0/baseline_results/llm_openai_gpt54_dev_public.json",
        "conservative",
    ),
    # ---- dev_public agentic / co-designed (offline) ----
    (
        "llm_agentic_btu_sonnet_4_6_dev_public.json",
        "results/checkpoints/llm_agentic_btu_sonnet_4_6_dev_public_v2/"
        "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
        "dev_public",
        "llm_agentic_btu_sonnet_4_6",
        "data/v1.0/baseline_results/llm_agentic_btu_sonnet_4_6_dev_public.json",
        "conservative",
    ),
    (
        "llm_agentic_btu_openai_dev_public.json",
        "results/temporal_checkpoints/agentic_btu_openai_gpt-5.1.jsonl",
        "dev_public",
        "llm_agentic_btu_openai",
        "data/v1.0/baseline_results/llm_agentic_btu_openai_dev_public.json",
        "conservative",
    ),
    (
        "llm_agentic_openai_dev_public.json",
        "results/temporal_checkpoints/agentic_openai_gpt-5.1.jsonl",
        "dev_public",
        "llm_agentic_openai",
        "data/v1.0/baseline_results/llm_agentic_openai_dev_public.json",
        "conservative",
    ),
    # ---- test_public (offline; all verified) ----
    (
        "llm_openrouter_deepseek_r1_test_public.json",
        "results/checkpoints/llm_openrouter_deepseek_r1_test_public/"
        "openrouter_deepseek_deepseek-r1.jsonl",
        "test_public",
        "llm_openrouter_deepseek_r1",
        "data/v1.0/baseline_results/llm_openrouter_deepseek_r1_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_deepseek_v3_test_public.json",
        "results/checkpoints/llm_openrouter_deepseek_v3/openrouter_deepseek_deepseek-v3.2.jsonl",
        "test_public",
        "llm_openrouter_deepseek_v3",
        "results/llm_openrouter_deepseek_v3_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_gemini_flash_test_public.json",
        "results/checkpoints/llm_openrouter_gemini_flash_test_public/"
        "openrouter_google_gemini-2.5-flash.jsonl",
        "test_public",
        "llm_openrouter_gemini_flash",
        "data/v1.0/baseline_results/llm_openrouter_gemini_flash_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_gemini_pro_test_public.json",
        "results/checkpoints/llm_openrouter_gemini_pro/openrouter_google_gemini-2.5-pro.jsonl",
        "test_public",
        "llm_openrouter_gemini_pro",
        "data/v1.0/baseline_results/llm_openrouter_gemini_pro_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_mistral_test_public.json",
        "results/checkpoints/llm_openrouter_mistral_test_public/"
        "openrouter_mistralai_mistral-large-2512.jsonl",
        "test_public",
        "llm_openrouter_mistral",
        "data/v1.0/baseline_results/llm_openrouter_mistral_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_qwen_test_public.json",
        "results/checkpoints/llm_openrouter_qwen_test_public/openrouter_qwen_qwen3-235b-a22b-2507.jsonl",
        "test_public",
        "llm_openrouter_qwen",
        "data/v1.0/baseline_results/llm_openrouter_qwen_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_qwen_max_test_public.json",
        "results/checkpoints/llm_openrouter_qwen_max_test_public/"
        "openrouter_qwen_qwen3-vl-235b-a22b-instruct.jsonl",
        "test_public",
        "llm_openrouter_qwen_max",
        "data/v1.0/baseline_results/llm_openrouter_qwen_max_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_llama_4_maverick_test_public.json",
        "results/checkpoints/llm_openrouter_llama_4_maverick_test_public/"
        "openrouter_meta-llama_llama-4-maverick.jsonl",
        "test_public",
        "llm_openrouter_llama_4_maverick",
        "data/v1.0/baseline_results/llm_openrouter_llama_4_maverick_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_claude_opus_4_7_test_public.json",
        "results/checkpoints/llm_openrouter_claude_opus_4_7_test_public/"
        "openrouter_anthropic_claude-opus-4.7.jsonl",
        "test_public",
        "llm_openrouter_claude_opus_4_7",
        "data/v1.0/baseline_results/llm_openrouter_claude_opus_4_7_test_public.json",
        "conservative",
    ),
    (
        "llm_openrouter_claude_sonnet_4_6_test_public.json",
        "results/checkpoints/llm_openrouter_claude_sonnet_4_6/"
        "openrouter_anthropic_claude-sonnet-4.6.jsonl",
        "test_public",
        "llm_openrouter_claude_sonnet_4_6",
        "results/llm_openrouter_claude_sonnet_4_6_test_public.json",
        "conservative",
    ),
    (
        "llm_openai_test_public.json",
        "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl",
        "test_public",
        "llm_openai",
        "results/llm_openai_test_public.json",
        "conservative",
    ),
    (
        "llm_openai_gpt54_test_public.json",
        "results/checkpoints/llm_openai_gpt54_test_public/openai_gpt-5.4.jsonl",
        "test_public",
        "llm_openai_gpt-5.4",
        "results/gpt54/llm_openai_gpt54_test_public.json",
        "conservative",
    ),
    (
        "llm_agentic_btu_openai_test_public.json",
        "results/checkpoints/llm_agentic_btu_openai_test_public/"
        "agentic_btu_openai_openai_gpt-5.1.jsonl",
        "test_public",
        "llm_agentic_btu_openai",
        "data/v1.0/baseline_results/llm_agentic_btu_openai_test_public.json",
        "conservative",
    ),
    (
        "llm_agentic_openai_test_public.json",
        "results/checkpoints/llm_agentic_openai_test_public/agentic_openai_openai_gpt-5.1.jsonl",
        "test_public",
        "llm_agentic_openai",
        "data/v1.0/baseline_results/llm_agentic_openai_test_public.json",
        "conservative",
    ),
    (
        "llm_agentic_btu_sonnet_4_6_test_public.json",
        "results/checkpoints/llm_agentic_btu_sonnet_4_6_test_public/"
        "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
        "test_public",
        "llm_agentic_btu_sonnet_4_6",
        "data/v1.0/baseline_results/llm_agentic_btu_sonnet_4_6_test_public.json",
        "conservative",
    ),
    (
        "llm_tool_augmented_test_public.json",
        "results/checkpoints/llm_tool_augmented_test_public/"
        "openai_tool_augmented_openai_gpt-5.1.jsonl",
        "test_public",
        "llm_tool_augmented",
        "data/v1.0/baseline_results/llm_tool_augmented_test_public.json",
        "conservative",
    ),
]


def main() -> None:
    rows = []
    for fname, pp, sp, tool, pub_rel, mode in JOBS:
        pub = load_template(pub_rel)
        r_old, r_new, _ = rescore(REPO / pp, sp, tool, eval_mode=mode)
        # sanity: r_old must match published primary metrics
        assert abs(r_old.detection_rate - pub["detection_rate"]) < 3e-3, (
            f"{tool}/{sp} DR old-mismatch {r_old.detection_rate} vs {pub['detection_rate']}"
        )
        assert (
            abs((r_old.false_positive_rate or 0) - (pub.get("false_positive_rate") or 0)) < 3e-3
        ), f"{tool}/{sp} FPR old-mismatch"
        merged = write_aggregate(OUT / fname, pub, r_new)
        rows.append(
            (
                tool,
                sp,
                pub["detection_rate"],
                merged["detection_rate"],
                (pub.get("false_positive_rate") or 0),
                (merged.get("false_positive_rate") or 0),
                pub["f1_hallucination"],
                merged["f1_hallucination"],
                (pub.get("mcc") or 0),
                (merged.get("mcc") or 0),
            )
        )
    print(f"{'tool':36s}{'split':12s} DR_old->new   FPR_old->new  F1_old->new   MCC_old->new")
    for t, sp, dro, drn, fpro, fprn, f1o, f1n, mco, mcn in rows:
        print(
            f"{t:36s}{sp:12s} {dro:.3f}->{drn:.3f} {fpro:.3f}->{fprn:.3f} "
            f"{f1o:.3f}->{f1n:.3f} {mco:.3f}->{mcn:.3f}"
        )
    print(f"\nWROTE {len(rows)} aggregates to {OUT}")


if __name__ == "__main__":
    main()
