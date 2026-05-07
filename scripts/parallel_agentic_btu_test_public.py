"""Parallel agentic BTU runner for test_public via OpenRouter.

Mirror of parallel_resume_test_public.py but uses the existing agentic-BTU
verifier (`verify_agentic_btu_openai`) routed through OpenRouter so a Sonnet
model can be used without an Anthropic API key. Each thread calls the
verifier with a single-entry list, producing one Prediction per entry, and
appends to the shared checkpoint JSONL atomically.

Usage:

    source /tmp/.or_env  # OPENROUTER_API_KEY
    source /tmp/.s2_env  # SEMANTIC_SCHOLAR_API_KEY (used by BTU)
    uv run python scripts/parallel_agentic_btu_test_public.py \\
        --checkpoint-dir results/checkpoints/llm_agentic_btu_sonnet_4_6_test_public \\
        --model anthropic/claude-sonnet-4.6 \\
        --workers 6

Resume semantics: skips bibtex_keys present in the existing checkpoint
JSONL (matching pattern `agentic_btu_openai_<safe_model>.jsonl` written by
the underlying verifier). Smoke-test with `--max-entries 3 --workers 2`.

Rate-limit notes
================
OpenRouter:        ~200 RPM cap on chat completions for paid users.
Semantic Scholar:  100 RPM on the BTU-side tool calls (paid key).
Each agentic entry triggers up to ~3-5 BTU tool calls, so 6 workers in
steady state ~= 6 entries/min * 3-5 BTU = 18-30 RPM upstream of S2 ---
well under the 100 RPM cap. If 429s appear, the underlying client's
exponential-backoff retry handles them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Route the openai SDK through OpenRouter BEFORE importing the agentic module.
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
or_key = os.environ.get("OPENROUTER_API_KEY")
if or_key and not os.environ.get("OPENAI_API_KEY", "").startswith("sk-or-"):
    # Override OPENAI_API_KEY with the OpenRouter key so the openai SDK auths
    # against OpenRouter. Existing OPENAI_API_KEY is preserved if it already
    # looks like an OpenRouter key.
    os.environ["OPENAI_API_KEY"] = or_key

from hallmark.baselines.llm_agentic import (  # noqa: E402
    verify_agentic_btu_openai,
    verify_agentic_openai,
)
from hallmark.baselines.llm_tool_augmented import verify_tool_augmented  # noqa: E402
from hallmark.dataset.loader import load_split  # noqa: E402

VERIFIERS = {
    "agentic_btu_openai": (verify_agentic_btu_openai, "agentic_btu_openai"),
    "agentic_openai": (verify_agentic_openai, "agentic_openai"),
    "tool_augmented": (verify_tool_augmented, "tool_augmented"),
}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _safe_model(model: str) -> str:
    return model.replace("/", "_")


def _checkpoint_path(checkpoint_dir: Path, model: str) -> Path:
    safe = _safe_model(model)
    return checkpoint_dir / f"agentic_btu_openai_{safe}.jsonl"


def _read_done_keys(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    keys: set[str] = set()
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        keys.add(rec["bibtex_key"])
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        choices=sorted(VERIFIERS),
        default="agentic_btu_openai",
        help="Which verifier function to call",
    )
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--split", default="test_public")
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--cache-db-path", type=Path, default=Path(".cache/agentic_tools.sqlite"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.cache_db_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")
    if not (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")):
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY not set; BTU will run on free tier with"
            " 100 RPS / 5000 RPM throttling that may collapse coverage."
        )

    verifier_fn, verifier_prefix = VERIFIERS[args.verifier]
    jsonl_path = args.checkpoint_dir / f"{verifier_prefix}_{_safe_model(args.model)}.jsonl"
    done_keys = _read_done_keys(jsonl_path)

    entries = load_split(split=args.split, version=args.version)
    remaining = [e for e in entries if e.bibtex_key not in done_keys]
    if args.max_entries > 0:
        remaining = remaining[: args.max_entries]

    logger.info(
        "Found %d existing predictions; remaining %d of %d (workers=%d)",
        len(done_keys),
        len(remaining),
        len(entries),
        args.workers,
    )
    logger.info("  data-split:     %s (v%s)", args.split, args.version)
    logger.info("  checkpoint:     %s", jsonl_path)
    logger.info("  model:          %s", args.model)
    logger.info("  cache-db:       %s", args.cache_db_path)

    if args.dry_run:
        return
    if not remaining:
        logger.info("Nothing to do.")
        return

    lock = threading.Lock()
    completed = 0
    start = time.time()

    def call_one(entry: object) -> dict:
        # Each thread issues an independent verify call with a single-entry
        # list. The verifier writes its own checkpoint line; we ALSO append
        # here for redundancy.
        kw: dict[str, object] = {
            "model": args.model,
            "checkpoint_dir": args.checkpoint_dir,
        }
        # tool_augmented has no cache_db_path arg
        if args.verifier != "tool_augmented":
            kw["cache_db_path"] = args.cache_db_path
        preds = verifier_fn([entry], **kw)  # type: ignore[arg-type, list-item]
        if not preds:
            return {
                "bibtex_key": getattr(entry, "bibtex_key", "?"),
                "label": "UNCERTAIN",
                "confidence": 0.5,
                "reason": "[Error fallback] verify returned empty list",
                "wall_clock_seconds": 0.0,
                "api_calls": 0,
                "api_sources_queried": [],
            }
        p = preds[0]
        return {
            "bibtex_key": p.bibtex_key,
            "label": p.label,
            "confidence": p.confidence,
            "reason": p.reason,
            "wall_clock_seconds": getattr(p, "wall_clock_seconds", 0.0) or 0.0,
            "api_calls": getattr(p, "api_calls", 0) or 0,
            "api_sources_queried": getattr(p, "api_sources_queried", []) or [],
        }

    # The underlying verify_agentic_btu_openai writes its own per-entry
    # checkpoint line; we just track completion here for progress logging.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(call_one, e): e for e in remaining}
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                _ = fut.result()
            except Exception as e:
                logger.exception("Unhandled error on %s: %s", entry.bibtex_key, e)
            with lock:
                pass
            completed += 1
            if completed % 5 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta_min = (len(remaining) - completed) / max(rate, 1e-9) / 60
                logger.info(
                    "[%d/%d] %.2f entries/s, ETA %.1f min",
                    completed,
                    len(remaining),
                    rate,
                    eta_min,
                )

    logger.info("Done. Processed %d entries.", completed)


if __name__ == "__main__":
    main()
