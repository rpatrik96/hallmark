#!/usr/bin/env python3
"""Re-run an agentic baseline on its DOI re-run targets, on the v1.2.5 input.

Targets come from ``results/relabel_recheck/doi_rerun_targets.json`` (the
entries whose recorded tool calls hit a Crossref-only ``resolve_doi`` on an
arXiv DOI). Each split's targets are sharded over ``--workers`` threads; each
shard has its own resume checkpoint and tool cache, so an interrupted run picks
up where it stopped. Output, one Prediction per line:
``results/relabel_recheck/doi_suffix_reruns/<tool>_<split>.jsonl``.

Wired tools (all native OpenAI, ``OPENAI_API_KEY``; see ``CONFIGS``): the
standalone ``llm_agentic_openai`` (April prompt and four tools of the released
run) and Stage 2 of ``cascade_gpt51`` (harness defaults of its 2026-08-06 run).
The only intended difference from the original runs is the DataCite fallback
in ``resolve_doi``.

Usage:
    set -a && source .env && set +a
    uv run python scripts/rerun_doi_targets.py llm_agentic_openai --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# HALLMARK_CODE_DIR runs the harness from another checkout (the luna Stage 2
# needs the gpt-6-luna sampling override that only exists there); data,
# targets and outputs still come from this repo.
sys.path.insert(0, os.environ.get("HALLMARK_CODE_DIR", str(REPO)))

from hallmark.baselines._agentic_tools import TOOL_DEFINITIONS  # noqa: E402
from hallmark.baselines.llm_agentic import verify_agentic_openai  # noqa: E402
from hallmark.dataset.schema import BenchmarkEntry, Prediction  # noqa: E402

#: The released llm_agentic_openai run (85977b4, 2026-04-22) predates the 14-type
#: taxonomy block and UNCERTAIN in SYSTEM_PROMPT, and the fifth tool
#: (search_semantic_scholar, c81debe). Its rows carry no predicted type, no
#: UNCERTAIN and no S2 call, so a re-run pins that prompt and the four tools.
APRIL_PROMPT = (
    REPO / "results/relabel_recheck/prompts/llm_agentic_openai_system_prompt_85977b4.txt"
).read_text()
FOUR_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] != "search_semantic_scholar"]
assert [t["name"] for t in FOUR_TOOLS] == [
    "resolve_doi",
    "search_crossref",
    "search_openalex",
    "search_arxiv",
]

TARGETS = REPO / "results/relabel_recheck/doi_rerun_targets.json"
OUT = REPO / "results/relabel_recheck/doi_suffix_reruns"
WORK = OUT / "_work"


#: tool -> (model, extra kwargs, Stage-2 tag or None). The cascade Stage-2 runs
#: (2026-08-06) used the harness defaults -- today's SYSTEM_PROMPT and the five
#: TOOL_DEFINITIONS, unchanged since -- so they pass no overrides; their rows are
#: tagged exactly as cascade.run_cascade tags them.
CONFIGS: dict[str, tuple[str, dict, str | None]] = {
    "llm_agentic_openai": (
        "gpt-5.1",
        {"tool_defs": FOUR_TOOLS, "system_prompt": APRIL_PROMPT},
        None,
    ),
    "cascade_gpt51": ("gpt-5.1", {}, "llm_agentic_openai"),
    # One GPT-5.4 Stage-2 run backs both cascade_gpt54 and cascade_gpt54_shared
    # (identical Stage-2 rows on every shared key), so it re-runs their union.
    "cascade_gpt54": ("gpt-5.4", {}, "llm_agentic_openai_gpt_5_4"),
    "cascade_luna_shared": ("gpt-6-luna", {}, "llm_agentic_openai_gpt_6_luna"),
    # Sonnet 4.6 Stage 2 (2026-09-21) went through OpenRouter, as the registry
    # entry llm_agentic_openrouter_claude_sonnet_4_6 routes it.
    "cascade_sonnet46_shared": (
        "anthropic/claude-sonnet-4.6",
        {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
        "llm_agentic_openrouter_claude_sonnet_4_6",
    ),
}

#: Re-run tool -> target stems whose keys it covers (default: itself).
TARGET_STEMS: dict[str, tuple[str, ...]] = {
    "cascade_gpt54": ("cascade_gpt54", "cascade_gpt54_shared"),
}


def tag_stage2(p: Prediction, stage2_baseline: str) -> Prediction:
    """Wrap a raw Stage-2 prediction exactly as ``cascade.run_cascade`` does."""
    return Prediction(
        bibtex_key=p.bibtex_key,
        label=p.label,
        confidence=p.confidence,
        reason=f"[Stage 2: {stage2_baseline}] {p.reason}",
        subtest_results=dict(p.subtest_results),
        api_sources_queried=list(p.api_sources_queried),
        wall_clock_seconds=p.wall_clock_seconds,
        api_calls=p.api_calls,
        source=p.source or "tool",
        predicted_hallucination_type=p.predicted_hallucination_type,
        cascade_stage="stage2_diagnosis",
    )


def run_shard(tool: str, split: str, idx: int, entries: list[BenchmarkEntry]) -> list[Prediction]:
    if tool not in CONFIGS:
        raise SystemExit(f"{tool}: not wired in this script")
    model, extra, stage2_tag = CONFIGS[tool]
    extra = dict(extra)
    key_env = extra.pop("key_env", "OPENAI_API_KEY")
    base_url = extra.pop("base_url", None)
    if model == "gpt-6-luna":
        from hallmark.baselines import llm_agentic

        overrides = getattr(llm_agentic, "_OPENAI_SAMPLING_OVERRIDES", {})
        if overrides.get(model) != {"reasoning_effort": "none"}:
            raise SystemExit("gpt-6-luna needs HALLMARK_CODE_DIR with its sampling override")
    shard_dir = WORK / f"{tool}_{split}_shard{idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    preds = verify_agentic_openai(
        [e.to_blind() for e in entries],
        model=model,
        api_key=os.environ[key_env],
        base_url=base_url,
        checkpoint_dir=shard_dir,
        cache_db_path=shard_dir / "tools.sqlite",
        checkpoint_name=f"{tool}_shard{idx}",
        **extra,
    )
    return [tag_stage2(p, stage2_tag) for p in preds] if stage2_tag else preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tool")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["dev_public", "test_public", "stress_test"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    targets = json.loads(TARGETS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        keys = {
            k
            for stem in TARGET_STEMS.get(args.tool, (args.tool,))
            for k in targets.get(f"{stem}_{split}", {}).get("keys", [])
        }
        if not keys:
            continue
        rows = (REPO / f"data/v1.2/{split}.jsonl").read_text().splitlines()
        entries = [
            e
            for e in (BenchmarkEntry.from_json(r) for r in rows if r.strip())
            if e.bibtex_key in keys
        ]
        assert len(entries) == len(keys), (
            f"{split}: {len(keys) - len(entries)} target keys not in split"
        )
        shards = [entries[i :: args.workers] for i in range(args.workers)]
        logging.info(
            "%s %s: %d entries over %d shards", args.tool, split, len(entries), len(shards)
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_shard, args.tool, split, i, shard) for i, shard in enumerate(shards)
            ]
            parts = [f.result() for f in futures]
        preds = {p.bibtex_key: p for part in parts for p in part}
        unavailable = [k for k, p in preds.items() if "unavailable" in (p.reason or "")]
        if unavailable:
            raise SystemExit(
                f"{split}: {len(unavailable)} rows are 'unavailable' fallbacks; not writing"
            )
        missing = keys - preds.keys()
        out = OUT / f"{args.tool}_{split}.jsonl"
        out.write_text("".join(preds[k].to_json() + "\n" for k in sorted(preds)))
        logging.info(
            "wrote %s (%d rows, %d missing)", out.relative_to(REPO), len(preds), len(missing)
        )


if __name__ == "__main__":
    main()
