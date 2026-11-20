"""Splice the v1.2.4 truncated-field re-runs into the per-entry prediction files.

Nine public entries (7 dev, 2 test) had an author list cut off at a LaTeX brace
until v1.2.4 restored it, so every tool judged a truncated citation. Rule
(issue #97 follow-up): re-run a tool on such an entry only where it said VALID;
HALLUCINATED and UNCERTAIN calls stand. 50 cells were re-run on 2026-10-16/17:
38 LLM calls on current main, and 12 bibtex-check-based calls (bibtex-updater,
tool-augmented, Sonnet agentic+BTU) with bibtex-check 1.2.0 and the wrapper as of
67a08c8^ (before #69 added --outage-threshold), the setup of the released run.
bibtex-updater looks citations up live, so its re-run reflects today's databases.

Each re-run row replaces the old row for its key; the old label/confidence and
the setup are recorded in rerun_log.json next to this script.

Usage: uv run python results/rerun_truncated_fields/splice_reruns.py RERUN_DIR
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_LOG = Path(__file__).with_name("rerun_log.json")
NOTE = "v1.2.4 truncated-field re-run; see results/rerun_truncated_fields/rerun_log.json"
MAIN = "current main (80ce8de)"
PRE69 = "bibtex-check 1.2.0 + wrapper at 67a08c8^ (pre-#69)"

# rerun file stem -> (target per split, model/route, code setup)
TARGETS = {
    "llm_openai": (
        {"test_public": "results/checkpoints/llm_openai/openai_gpt-5.1.jsonl"},
        "gpt-5.1 (OpenAI)",
        MAIN,
    ),
    "llm_openai_gpt54": (
        {
            "dev_public": "results/checkpoints/llm_openai_gpt54_dev_public_v3/openai_gpt-5.4.jsonl",
            "test_public": "results/checkpoints/llm_openai_gpt54_test_public/openai_gpt-5.4.jsonl",
        },
        "gpt-5.4 (OpenAI)",
        MAIN,
    ),
    "llm_openrouter_claude_opus_4_7": (
        {
            s: f"results/checkpoints/llm_openrouter_claude_opus_4_7_{s}/openrouter_anthropic_claude-opus-4.7.jsonl"
            for s in ("dev_public", "test_public")
        },
        "anthropic/claude-opus-4.7 (OpenRouter)",
        MAIN,
    ),
    **{
        f"llm_openrouter_{m}": (
            {
                s: f"results/llm_openrouter_{m}_{s}_predictions.jsonl"
                for s in ("dev_public", "test_public")
            },
            f"{mid} (OpenRouter)",
            MAIN,
        )
        for m, mid in {
            "deepseek_r1": "deepseek/deepseek-r1",
            "deepseek_v3": "deepseek/deepseek-v3.2",
            "gemini_flash": "google/gemini-2.5-flash",
            "mistral": "mistralai/mistral-large-2512",
        }.items()
    },
    "llm_openrouter_gemini_pro": (
        {
            "dev_public": "results/new_models/gemini_pro.jsonl",
            "test_public": "results/llm_openrouter_gemini_pro_test_public_predictions.jsonl",
        },
        "google/gemini-2.5-pro (OpenRouter)",
        MAIN,
    ),
    "llm_openrouter_llama_4_maverick": (
        {
            "dev_public": "results/new_models/llama4_maverick.jsonl",
            "test_public": "results/llm_openrouter_llama_4_maverick_test_public_predictions.jsonl",
        },
        "meta-llama/llama-4-maverick (OpenRouter)",
        MAIN,
    ),
    "llm_agentic_openai": (
        {"dev_public": "results/temporal_checkpoints/agentic_openai_gpt-5.1.jsonl"},
        "gpt-5.1 agentic (OpenAI)",
        MAIN,
    ),
    "cascade_sonnet46_stage2": (
        {"dev_public": "results/cascade_sonnet46_shared/sonnet46_dev_public_preds.jsonl"},
        "Stage 2 anthropic/claude-sonnet-4.6 agentic (OpenRouter); Stage 1 kept",
        MAIN,
    ),
    "llm_tool_augmented": (
        {
            "dev_public": "data/v1.2/baseline_results/llm_tool_augmented_dev_public.jsonl",
            "test_public": "results/llm_tool_augmented_test_public_predictions.jsonl",
        },
        "gpt-5.1 + bibtex-updater evidence (OpenAI)",
        PRE69,
    ),
    "llm_agentic_btu_sonnet_4_6": (
        {"test_public": "results/llm_agentic_btu_sonnet_4_6_test_public_predictions.jsonl"},
        "anthropic/claude-sonnet-4.6 agentic + bibtex-updater (OpenRouter)",
        PRE69,
    ),
    "bibtexupdater": (
        {
            s: f"results/relabel_delta/btu_v1_2_0_prescreen_fix/bibtexupdater_{s}_per_entry.jsonl"
            for s in ("dev_public", "test_public")
        },
        "bibtex-check 1.2.0",
        PRE69,
    ),
}


def btu_row(old: dict, new: dict) -> dict:
    """bibtex-updater per-entry schema; abstentions resolve to VALID (paper convention)."""
    reason = new.get("reason") or ""
    status = m.group(1) if (m := re.match(r"Status: (\w+)", reason)) else old.get("btu_status")
    mism = re.search(r"Mismatched: (\[[^\]]*\])", reason)
    row = dict(old)
    row.update(
        pred_label="VALID" if new["label"] == "UNCERTAIN" else new["label"],
        pred_confidence=new["confidence"],
        source=new.get("source", old.get("source")),
        btu_status=status,
        btu_abstained=new["label"] == "UNCERTAIN",
        btu_mismatched_fields=json.loads(mism.group(1).replace("'", '"')) if mism else [],
        btu_api_sources=new.get("api_sources_queried", old.get("btu_api_sources")),
        reason=reason,
        rerun=NOTE,
    )
    return row


def splice(target: Path, new_rows: dict[str, dict], is_btu: bool) -> dict[str, dict]:
    lines = target.read_text().splitlines()
    old_by_key: dict[str, dict] = {}
    out = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("bibtex_key")
        if key not in new_rows:
            out.append(line)  # untouched rows stay byte-identical
            continue
        old_by_key[key] = row
        row = btu_row(row, new_rows[key]) if is_btu else {**row, **new_rows[key], "rerun": NOTE}
        out.append(json.dumps(row, ensure_ascii=False))
    missing = set(new_rows) - set(old_by_key)
    if missing:
        raise SystemExit(f"{target}: keys not found {sorted(missing)}")
    target.write_text("\n".join(out) + "\n")
    return old_by_key


def main(rerun_dir: Path) -> None:
    log = []
    for stem, (targets, model, setup) in TARGETS.items():
        for split, rel in targets.items():
            src = rerun_dir / f"{stem}_{split}.jsonl"
            if not src.exists():  # no VALID call to re-run on this split
                continue
            new_rows = {
                (d := json.loads(line))["bibtex_key"]: d
                for line in src.read_text().splitlines()
                if line.strip()
            }
            olds = splice(REPO / rel, new_rows, stem == "bibtexupdater")
            for key, new in new_rows.items():
                old = olds[key]
                log.append(
                    {
                        "tool": stem,
                        "split": split,
                        "bibtex_key": key,
                        "file": rel,
                        "old_label": old.get("label", old.get("pred_label")),
                        "old_confidence": old.get("confidence", old.get("pred_confidence")),
                        "new_label": new["label"],
                        "new_confidence": new["confidence"],
                        "model": model,
                        "setup": setup,
                        "run_date": "2026-10-16/17",
                    }
                )
    OUT_LOG.write_text(
        json.dumps(
            {"note": __doc__.split("\n\n")[1].replace("\n", " "), "cells": log},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"spliced {len(log)} cells; log -> {OUT_LOG.relative_to(REPO)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
