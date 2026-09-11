"""Re-query the entries retyped in v1.2.3, for models with no stored per-entry predictions.

Background
----------
v1.2.3 retyped four real-world entries to ``plausible_fabrication`` / Tier 3. The
tool's *prediction* on those entries cannot have changed -- the bibtex fields are
byte-identical and ``BlindEntry`` never exposes the type or tier -- but the tier
metrics built on top of them did change, because the entries moved between tier
buckets.

Recomputing those metrics needs one bit per entry per model: did that model flag
it? The published aggregates in ``baseline_results/`` store only totals, so for
most cells that bit was never recorded. Where ``per_type_metrics`` shows the old
type was detected at DR == 1.000 *and* the cell has no UNCERTAIN predictions, the
bit is deducible (every entry of that type was flagged, so ours was too) and no
query is needed. Everywhere else it has to be measured.

This script measures exactly those unknown bits and nothing else.

Protocol
--------
Matches the original zero-shot runs exactly:

* ``k=1`` -- one call per (model, entry), as in the original run
* ``temperature=0.0``, ``seed=42``, ``max_completion_tokens=1024`` -- library
  defaults, unchanged since before the earliest zero-shot run (seed added
  2026-02-21; temperature was a hardcoded 0.0 before 2026-05-04 and defaults to
  0.0 after, for every non-gpt-5.5 model)
* prompt and parsing come from ``llm_verifier`` via ``run_baseline`` -- not
  reimplemented here, so the call path is identical to the original

Usage
-----
    # show the query plan, spend nothing
    uv run python scripts/requery_retyped_entries.py --split dev_public --dry-run

    # run it
    uv run python scripts/requery_retyped_entries.py --split dev_public
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallmark.baselines.registry import run_baseline
from hallmark.dataset.schema import BenchmarkEntry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# The four entries retyped in v1.2.3, with the type they carried when the
# published aggregates were scored. That old type is what per_type_metrics is
# keyed by, so it -- not the new plausible_fabrication -- drives the deduction.
RETYPED: dict[str, list[tuple[str, str]]] = {
    "dev_public": [
        ("ffe715a15b16", "fabricated_doi"),
        ("ec01d96455e0", "preprint_as_published"),
    ],
    "test_public": [
        ("ab9c13051a56", "placeholder_authors"),
        ("cad998c28243", "placeholder_authors"),
    ],
}

# tool_name in baseline_results -> (registered baseline, model override or None).
# gpt-5.4 has no baseline of its own: scripts/run_gpt54_splits.py runs the
# llm_openai baseline with model="gpt-5.4", so we do the same.
TOOL_TO_BASELINE: dict[str, tuple[str, str | None]] = {
    "llm_openai": ("llm_openai", None),  # gpt-5.1, the verify_with_openai default
    "llm_openai_gpt-5.4": ("llm_openai", "gpt-5.4"),
    "llm_openrouter_claude_opus_4_7": ("llm_openrouter_claude_opus_4_7", None),
    "llm_openrouter_claude_sonnet_4_6": ("llm_openrouter_claude_sonnet_4_6", None),
    "llm_openrouter_deepseek_r1": ("llm_openrouter_deepseek_r1", None),
    "llm_openrouter_deepseek_v3": ("llm_openrouter_deepseek_v3", None),
    "llm_openrouter_gemini_flash": ("llm_openrouter_gemini_flash", None),
    "llm_openrouter_gemini_pro": ("llm_openrouter_gemini_pro", None),
    "llm_openrouter_llama_4_maverick": ("llm_openrouter_llama_4_maverick", None),
    "llm_openrouter_mistral": ("llm_openrouter_mistral", None),
    "llm_openrouter_qwen": ("llm_openrouter_qwen", None),
    "llm_openrouter_qwen_max": ("llm_openrouter_qwen_max", None),
}

# Per-model call overrides.
#
# deepseek-r1: thinking tokens exhaust the default 1024-token completion budget
# before the JSON verdict is emitted, so the reply falls to "[Error fallback]
# Parse error: None" and is scored UNCERTAIN. The original dev run hit this on
# only 18/1119 entries (1.6%), so failing 2 of 2 today indicates the model behind
# this unversioned alias became more verbose.
#
# Disabling reasoning is not an option -- OpenRouter rejects it for this endpoint
# with "400 Reasoning is mandatory for this endpoint and cannot be disabled" --
# so instead give the reasoning room and let the verdict follow it.
#
# This is a deviation from the original run's 1024-token budget. It changes only
# how much room the model has to finish, not what it is asked or how it decodes
# (temperature and seed are untouched), but it belongs in the writeup.
MODEL_KWARGS: dict[str, dict[str, object]] = {
    "llm_openrouter_deepseek_r1": {"max_completion_tokens": 8192},
}


@dataclass(frozen=True)
class Query:
    """One unknown bit: did ``tool_name`` flag ``bibtex_key``?"""

    tool_name: str
    bibtex_key: str
    old_type: str
    reason: str


def _is_frontier(tool_name: str) -> bool:
    """Frontier zero-shot LLMs only -- no agentic, cascade, or small-model runs."""
    return tool_name.startswith(("llm_openai", "llm_openrouter")) and not any(
        excluded in tool_name
        for excluded in ("agentic", "tool_augmented", "cascade", "haiku", "cutoff_aware")
    )


def _deducible(aggregate: dict, old_type: str) -> tuple[bool, str]:
    """Can the model's answer on this entry be read off the stored aggregate?

    Two conditions must hold:

    * the old type was detected at DR == 1.000, i.e. zero entries of that type
      were missed, so ours cannot have been among the misses
    * the cell has no UNCERTAIN predictions -- ``build_confusion_matrix`` drops
      those entirely while ``count`` still includes them, so with abstentions
      present DR == 1.000 means "all *answered* entries were flagged", which
      leaves "ours was the abstention" open
    """
    per_type = aggregate.get("per_type_metrics", {})
    metrics = per_type.get(old_type)
    if not metrics:
        return False, f"no per_type_metrics for {old_type}"

    detection_rate = metrics["detection_rate"]
    count = metrics["count"]
    missed = round((1 - detection_rate) * count)
    num_uncertain = aggregate.get("num_uncertain")

    if missed != 0:
        return False, f"{old_type} DR={detection_rate:.3f}x{count} -> {missed} missed"
    if num_uncertain != 0:
        return False, f"{old_type} DR=1.000 but num_uncertain={num_uncertain}"
    return True, f"{old_type} DR=1.000, num_uncertain=0 -> proven flagged"


def build_plan(baseline_results: Path, split: str) -> tuple[list[Query], list[str]]:
    """Return the queries needed for ``split``, plus notes on skipped cells."""
    queries: list[Query] = []
    notes: list[str] = []

    for path in sorted(baseline_results.glob("*.json")):
        if path.name == "manifest.json":
            continue
        aggregate = json.loads(path.read_text())
        tool_name = aggregate.get("tool_name", "")
        if not _is_frontier(tool_name) or aggregate.get("split_name") != split:
            continue
        if tool_name not in TOOL_TO_BASELINE:
            notes.append(f"{tool_name}: no baseline mapping, skipped")
            continue
        if not aggregate.get("per_tier_metrics"):
            notes.append(
                f"{tool_name}: no per_tier_metrics -- re-score from per-entry file instead"
            )
            continue

        for bibtex_key, old_type in RETYPED[split]:
            known, reason = _deducible(aggregate, old_type)
            if known:
                notes.append(f"{tool_name} {bibtex_key}: {reason}")
            else:
                queries.append(Query(tool_name, bibtex_key, old_type, reason))

    return queries, notes


def load_entries(data_dir: Path, split: str, keys: set[str]) -> dict[str, BenchmarkEntry]:
    """Load just the retyped entries. run_baseline blinds them before dispatch."""
    entries = {}
    for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        entry = BenchmarkEntry.from_json(line)
        if entry.bibtex_key in keys:
            entries[entry.bibtex_key] = entry
    missing = keys - set(entries)
    if missing:
        raise SystemExit(f"entries not found in {split}: {sorted(missing)}")
    return entries


def load_dotenv(path: Path) -> list[str]:
    """Populate os.environ from a KEY=VALUE file, without overriding exported vars.

    The repo has no python-dotenv dependency and other scripts just read
    os.environ, expecting the caller to have sourced .env. This keeps that
    contract -- an already-exported value always wins -- while letting the
    script run straight from a checkout that has .env sitting untracked.

    Returns the names (never the values) of the variables it set.
    """
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if os.environ.get(key):  # exported value takes precedence
            continue
        os.environ[key] = value.strip().strip("'\"")
        loaded.append(key)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev_public", choices=sorted(RETYPED))
    parser.add_argument("--data-dir", type=Path, default=Path("data/v1.2"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/reviewer_experiments/tier_retype_requeries.jsonl")
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--only", help="restrict to tool_names containing this substring")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, make no API calls")
    args = parser.parse_args()

    baseline_results = args.data_dir / "baseline_results"
    queries, notes = build_plan(baseline_results, args.split)
    if args.only:
        queries = [q for q in queries if args.only in q.tool_name]

    print(f"\n=== {args.split}: {len(queries)} queries ===")
    for query in queries:
        print(f"  {query.tool_name:34} {query.bibtex_key}  ({query.reason})")
    print(f"\n=== deduced from stored aggregates, not queried: {len(notes)} ===")
    for note in notes:
        print(f"  {note}")

    if args.dry_run:
        print("\n--dry-run: no API calls made")
        return

    if loaded := load_dotenv(args.env_file):
        logger.info("loaded from %s: %s", args.env_file, ", ".join(loaded))

    missing_keys = [
        var for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY") if not os.environ.get(var)
    ]
    if missing_keys:
        raise SystemExit(
            f"missing API key(s): {', '.join(missing_keys)}. "
            f"Export them or put them in {args.env_file}."
        )

    entries = load_entries(args.data_dir, args.split, {q.bibtex_key for q in queries})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # One call per (model, entry): k=1, matching the original run. Entries are
    # sent one at a time so a failure isolates to a single query rather than
    # taking the model's other entry down with it.
    written = 0
    with args.out.open("a") as handle:
        for index, query in enumerate(queries, 1):
            baseline, model_override = TOOL_TO_BASELINE[query.tool_name]
            kwargs: dict = {"model": model_override} if model_override else {}
            kwargs.update(MODEL_KWARGS.get(query.tool_name, {}))
            logger.info("[%d/%d] %s %s", index, len(queries), query.tool_name, query.bibtex_key)
            try:
                predictions = run_baseline(
                    baseline, [entries[query.bibtex_key]], split=args.split, **kwargs
                )
            except Exception as exc:
                logger.error("  failed: %s", exc)
                record = {
                    "tool_name": query.tool_name,
                    "bibtex_key": query.bibtex_key,
                    "old_type": query.old_type,
                    "error": str(exc),
                }
            else:
                prediction = predictions[0]
                logger.info("  -> %s (conf %.2f)", prediction.label, prediction.confidence)
                record = {
                    "tool_name": query.tool_name,
                    "bibtex_key": query.bibtex_key,
                    "old_type": query.old_type,
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "predicted_hallucination_type": prediction.predicted_hallucination_type,
                    "reason": prediction.reason,
                }
            record["split"] = args.split
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += 1

    print(f"\nwrote {written} records to {args.out}")


if __name__ == "__main__":
    main()
