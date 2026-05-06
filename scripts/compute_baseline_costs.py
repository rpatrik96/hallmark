"""Compute cost and latency statistics for each main-table baseline.

Reads checkpoint JSONLs under results/checkpoints/ for dev_public runs.
For tools with no timing checkpoint (precomputed-only results), uses
literature/documented estimates flagged with an asterisk in the output.

Output: tables/baseline_cost_latency.csv  +  pipe-separated stdout table.

Usage:
    uv run python scripts/compute_baseline_costs.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Price table: (prompt_$/M_tok, completion_$/M_tok)
# Source: OpenRouter list price as of 2026-05-04
# ---------------------------------------------------------------------------
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-5.1": (1.25, 10.0),
    "openai/gpt-5.4": (1.25, 10.0),
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
    "anthropic/claude-opus-4.7": (5.0, 25.0),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "deepseek/deepseek-v3.2": (0.27, 1.10),
    "qwen/qwen3-235b-a22b-2507": (0.20, 0.60),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.60),
    "mistralai/mistral-large-2512": (2.0, 6.0),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-pro": (3.5, 15.0),
    "meta-llama/llama-4-maverick": (0.20, 0.60),
    # DOI-only and bibtex-updater: $0 LLM cost (no LLM calls)
}

# Assumed token counts when not logged (prompt ~600 tokens, completion varies)
ASSUMED_PROMPT_TOKENS = 600
ASSUMED_COMPLETION_TOKENS = 80  # ~320 chars / 4 chars-per-token for short verdicts

# ---------------------------------------------------------------------------
# Checkpoint registry: map display name -> (checkpoint_dir, jsonl_filename,
#                                           model_id_for_pricing, is_agentic)
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
CKPT = REPO / "results" / "checkpoints"

BASELINES: list[dict] = [
    # --- Citation-database ---
    {
        "tool": "DOI-only",
        "ckpt_dir": None,  # no JSONL timing; DOI lookup is ~0.3s per CrossRef call
        "model_id": None,
        "is_agentic": False,
        "estimated_mean_wc": 0.3,
        "estimated_p95_wc": 1.2,
        "note": "estimated (CrossRef HTTP round-trip)",
    },
    # --- Zero-shot LLMs ---
    {
        "tool": "GPT-5.1 (zero-shot)",
        "ckpt_dir": "llm_openai",
        "jsonl": "openai_gpt-5.1.jsonl",
        "model_id": "openai/gpt-5.1",
        "is_agentic": False,
        "note": "measured",
    },
    {
        "tool": "GPT-5.4 (zero-shot)",
        "ckpt_dir": "llm_openai_gpt54_dev_public_v3",
        "jsonl": "openai_gpt-5.4.jsonl",
        "model_id": "openai/gpt-5.4",
        "is_agentic": False,
        "note": "measured",
    },
    {
        "tool": "Claude Sonnet 4.6",
        "ckpt_dir": "llm_openrouter_claude_sonnet_4_6",
        "jsonl": "openrouter_anthropic_claude-sonnet-4.6.jsonl",
        "model_id": "anthropic/claude-sonnet-4.6",
        "is_agentic": False,
        "note": "measured (test_public run; dev n=1119 via precomputed)",
    },
    {
        "tool": "Claude Opus 4.7",
        "ckpt_dir": None,
        "model_id": "anthropic/claude-opus-4.7",
        "is_agentic": False,
        "estimated_mean_wc": 6.0,
        "estimated_p95_wc": 14.0,
        "note": "estimated (no timing checkpoint; Opus ~1.2x Sonnet latency)",
    },
    {
        "tool": "DeepSeek-R1",
        "ckpt_dir": None,
        "model_id": "deepseek/deepseek-r1",
        "is_agentic": False,
        # Documented in paper as ~25s/entry for chain-of-thought
        "estimated_mean_wc": 25.0,
        "estimated_p95_wc": 55.0,
        "note": "estimated (paper §4: ~25 s/entry for CoT)",
    },
    {
        "tool": "DeepSeek-V3.2",
        "ckpt_dir": "llm_openrouter_deepseek_v3",
        "jsonl": "openrouter_deepseek_deepseek-v3.2.jsonl",
        "model_id": "deepseek/deepseek-v3.2",
        "is_agentic": False,
        "note": "measured (test_public run)",
    },
    {
        "tool": "Qwen3-235B",
        "ckpt_dir": None,
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "is_agentic": False,
        "estimated_mean_wc": 5.0,
        "estimated_p95_wc": 12.0,
        "note": "estimated (no timing checkpoint)",
    },
    {
        "tool": "Qwen3-VL-235B",
        "ckpt_dir": None,
        "model_id": "qwen/qwen3-vl-235b-a22b-instruct",
        "is_agentic": False,
        "estimated_mean_wc": 5.0,
        "estimated_p95_wc": 12.0,
        "note": "estimated (no timing checkpoint)",
    },
    {
        "tool": "Mistral Large",
        "ckpt_dir": None,
        "model_id": "mistralai/mistral-large-2512",
        "is_agentic": False,
        "estimated_mean_wc": 3.5,
        "estimated_p95_wc": 8.0,
        "note": "estimated (no timing checkpoint)",
    },
    {
        "tool": "Gemini 2.5 Flash",
        "ckpt_dir": None,
        "model_id": "google/gemini-2.5-flash",
        "is_agentic": False,
        "estimated_mean_wc": 3.0,
        "estimated_p95_wc": 7.0,
        "note": "estimated (no timing checkpoint)",
    },
    {
        "tool": "Llama 4 Maverick",
        "ckpt_dir": None,
        "model_id": "meta-llama/llama-4-maverick",
        "is_agentic": False,
        "estimated_mean_wc": 3.0,
        "estimated_p95_wc": 7.0,
        "note": "estimated (no timing checkpoint)",
    },
    {
        "tool": "Gemini 2.5 Pro",
        "ckpt_dir": "llm_openrouter_gemini_pro",
        "jsonl": "openrouter_google_gemini-2.5-pro.jsonl",
        "model_id": "google/gemini-2.5-pro",
        "is_agentic": False,
        "note": "measured (test_public run)",
    },
    # --- Agentic ---
    {
        "tool": "GPT-5.1 + CrossRef/OA/arXiv",
        "ckpt_dir": None,
        "model_id": "openai/gpt-5.1",
        "is_agentic": True,
        "estimated_mean_wc": 18.0,
        "estimated_p95_wc": 60.0,
        "estimated_mean_api": 3.5,
        "note": "estimated (no timing checkpoint; ~3-5 tool calls per entry)",
    },
    {
        "tool": "GPT-5.1 + BTU (agentic)",
        "ckpt_dir": None,
        "model_id": "openai/gpt-5.1",
        "is_agentic": True,
        "estimated_mean_wc": 18.0,
        "estimated_p95_wc": 60.0,
        "estimated_mean_api": 3.5,
        "note": "estimated (no timing checkpoint)",
    },
    {
        "tool": "Sonnet 4.6 + BTU (agentic)",
        "ckpt_dir": "llm_agentic_btu_sonnet_4_6_dev_public_v2",
        "jsonl": "agentic_btu_openai_anthropic_claude-sonnet-4.6.jsonl",
        "model_id": "anthropic/claude-sonnet-4.6",
        "is_agentic": True,
        "note": "measured",
    },
    # --- Co-designed ---
    {
        "tool": "bibtex-updater",
        "ckpt_dir": None,
        "model_id": None,  # no LLM
        "is_agentic": False,
        "estimated_mean_wc": 8.0,
        "estimated_p95_wc": 30.0,
        "note": "estimated (Semantic Scholar API; no timing in precomputed JSONL)",
    },
    {
        "tool": "GPT-5.1 + BTU (always-call)",
        "ckpt_dir": None,
        "model_id": "openai/gpt-5.1",
        "is_agentic": True,
        "estimated_mean_wc": 20.0,
        "estimated_p95_wc": 65.0,
        "estimated_mean_api": 4.0,
        "note": "estimated (no timing checkpoint)",
    },
]


def load_checkpoint(ckpt_dir: str, jsonl: str) -> list[dict]:
    fp = CKPT / ckpt_dir / jsonl
    if not fp.exists():
        print(f"  WARNING: checkpoint not found: {fp}", file=sys.stderr)
        return []
    with fp.open() as f:
        return [json.loads(line) for line in f]


def compute_stats(
    entries: list[dict],
    model_id: str | None,
    is_agentic: bool,
    estimated_mean_wc: float | None = None,
    estimated_p95_wc: float | None = None,
    estimated_mean_api: float | None = None,
) -> dict:
    """Return dict with latency and cost stats."""
    if entries:
        wcs = [e["wall_clock_seconds"] for e in entries if "wall_clock_seconds" in e]
        n_missing = len(entries) - len(wcs)
        if n_missing > 0:
            pct = 100 * n_missing / len(entries)
            print(
                f"  WARNING: {n_missing}/{len(entries)} ({pct:.0f}%) entries missing wall_clock_seconds",
                file=sys.stderr,
            )
        mean_wc = float(np.mean(wcs)) if wcs else (estimated_mean_wc or 0.0)
        p95_wc = float(np.percentile(wcs, 95)) if wcs else (estimated_p95_wc or 0.0)
        total_min = float(sum(wcs) / 60) if wcs else (mean_wc * 1119 / 60)

        api_vals = [e["api_calls"] for e in entries if "api_calls" in e]
        mean_api = float(np.mean(api_vals)) if api_vals else (estimated_mean_api or 1.0)
    else:
        # Fully estimated
        mean_wc = estimated_mean_wc or 0.0
        p95_wc = estimated_p95_wc or 0.0
        total_min = mean_wc * 1119 / 60  # estimate for n=1119
        mean_api = estimated_mean_api or (3.5 if is_agentic else 1.0)

    # Token estimation: not logged in most checkpoints; use assumed values
    # Agentic calls multiply by mean api_calls
    call_multiplier = mean_api if is_agentic else 1.0
    mean_prompt_tok = ASSUMED_PROMPT_TOKENS * call_multiplier
    mean_completion_tok = ASSUMED_COMPLETION_TOKENS * call_multiplier

    # Cost estimation
    if model_id and model_id in PRICES:
        p_in, p_out = PRICES[model_id]
        cost_per_entry = (mean_prompt_tok * p_in + mean_completion_tok * p_out) / 1_000_000
    else:
        cost_per_entry = 0.0  # DOI-only / bibtex-updater: no LLM cost

    return {
        "mean_sec_per_entry": round(mean_wc, 2),
        "p95_sec_per_entry": round(p95_wc, 2),
        "total_min": round(total_min, 1),
        "mean_prompt_tok": round(mean_prompt_tok, 0),
        "mean_completion_tok": round(mean_completion_tok, 0),
        "est_cost_per_entry_usd": round(cost_per_entry, 5),
        "est_cost_per_1k_entries_usd": round(cost_per_entry * 1000, 3),
    }


def main() -> None:
    print("Price table used for cost estimation:")
    for model, (pin, pout) in PRICES.items():
        print(f"  {model}: ${pin}/M prompt + ${pout}/M completion")
    print()

    rows: list[dict] = []

    for bl in BASELINES:
        tool = bl["tool"]
        print(f"Processing: {tool}")

        entries: list[dict] = []
        if bl.get("ckpt_dir"):
            entries = load_checkpoint(bl["ckpt_dir"], bl["jsonl"])
            if not entries:
                print("  SKIP: no entries loaded", file=sys.stderr)
                continue

        stats = compute_stats(
            entries=entries,
            model_id=bl.get("model_id"),
            is_agentic=bl.get("is_agentic", False),
            estimated_mean_wc=bl.get("estimated_mean_wc"),
            estimated_p95_wc=bl.get("estimated_p95_wc"),
            estimated_mean_api=bl.get("estimated_mean_api"),
        )
        note = bl.get("note", "")
        rows.append({"tool": tool, **stats, "note": note})

    # Write CSV
    out_dir = REPO / "tables"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "baseline_cost_latency.csv"
    fieldnames = [
        "tool",
        "mean_sec_per_entry",
        "p95_sec_per_entry",
        "total_min",
        "mean_prompt_tok",
        "mean_completion_tok",
        "est_cost_per_entry_usd",
        "est_cost_per_1k_entries_usd",
        "note",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to: {csv_path}")

    # Print pipe-separated table
    header = " | ".join(fieldnames)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[k]) for k in fieldnames))
    print(sep)


if __name__ == "__main__":
    main()
