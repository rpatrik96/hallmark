#!/usr/bin/env python3
"""Re-run an agentic baseline on its DOI re-run targets, on the v1.2.5 input.

Targets come from ``results/relabel_recheck/doi_rerun_targets.json`` (the
entries whose recorded tool calls hit a Crossref-only ``resolve_doi`` on an
arXiv DOI). Each split's targets are sharded over ``--workers`` threads; each
shard has its own resume checkpoint and tool cache, so an interrupted run picks
up where it stopped. Output, one Prediction per line:
``results/relabel_recheck/doi_suffix_reruns/<tool>_<split>.jsonl``.

Only ``llm_agentic_openai`` is wired here: GPT-5.1, native OpenAI API
(``OPENAI_API_KEY``), the April system prompt and four tools of the released
run; the only intended difference is the DataCite fallback in ``resolve_doi``.

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
sys.path.insert(0, str(REPO))

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


def run_shard(tool: str, split: str, idx: int, entries: list[BenchmarkEntry]) -> list[Prediction]:
    if tool != "llm_agentic_openai":
        raise SystemExit(f"{tool}: not wired in this script")
    shard_dir = WORK / f"{tool}_{split}_shard{idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return verify_agentic_openai(
        [e.to_blind() for e in entries],
        model="gpt-5.1",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=None,
        checkpoint_dir=shard_dir,
        cache_db_path=shard_dir / "tools.sqlite",
        checkpoint_name=f"agentic_openai_shard{idx}",
        tool_defs=FOUR_TOOLS,
        system_prompt=APRIL_PROMPT,
    )


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
        spec = targets.get(f"{args.tool}_{split}")
        if not spec or not spec["keys"]:
            continue
        keys = set(spec["keys"])
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
