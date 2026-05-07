"""Resume an interrupted hallmark evaluate run in parallel.

Picks up an existing checkpoint dir, reads predictions already there, and
fans out remaining entries across N concurrent threads to OpenRouter using
the same verification prompt the main baseline uses.

# Rate-limit design notes
# ========================
# OpenRouter paid-tier cap:   ~200 RPM for chat completions
# DeepSeek-R1 upstream:       4-10 concurrent per provider; OpenRouter routes
#                             across Targon / Together / DeepInfra / native.
# 8 workers x ~40 s/call ~= 12 RPM steady state -> well under the 200 RPM cap.
# Realistic speedup:          4-6x (some calls queue at upstream).
# If 429s appear in logs, the exponential-backoff retry handles them naturally.

Usage:

    uv run python scripts/parallel_resume_test_public.py \\
        --checkpoint-dir results/checkpoints/llm_openrouter_deepseek_r1_test_public \\
        --model deepseek/deepseek-r1 \\
        --jsonl-name openrouter_deepseek_deepseek-r1.jsonl \\
        --workers 8

    # Dry-run: see resume plan without API calls
    uv run python scripts/parallel_resume_test_public.py \\
        --checkpoint-dir results/checkpoints/llm_openrouter_deepseek_r1_test_public \\
        --model deepseek/deepseek-r1 \\
        --jsonl-name openrouter_deepseek_deepseek-r1.jsonl \\
        --workers 8 --dry-run

    # Smoke-test: 3 entries against a cheap model to confirm the run path works
    uv run python scripts/parallel_resume_test_public.py \\
        --checkpoint-dir /tmp/smoke_resume_test \\
        --model meta-llama/llama-4-maverick \\
        --jsonl-name smoke.jsonl \\
        --max-entries 3 \\
        --workers 2

After completion, run `hallmark evaluate --predictions <jsonl>` to compute
the eval.json from the assembled predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hallmark.baselines.llm_verifier import VERIFICATION_PROMPT  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def entry_to_bibtex(entry: dict) -> str:
    """Reconstruct a BibTeX string from the schema's `fields` dict."""
    btype = entry.get("bibtex_type", "article")
    key = entry["bibtex_key"]
    flds = entry["fields"]
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in sorted(flds.items()))
    return f"@{btype}{{{key},\n  {body}\n}}"


def parse_response(content: str, bibtex_key: str) -> tuple[str, float, str]:
    """Parse LLM JSON response into (label, confidence, reason).

    Mirrors the logic in hallmark.baselines.llm_verifier._parse_llm_response
    so predictions are consistent with the sequential run:
    1. Try bare JSON.
    2. Try each fenced code block.
    3. Regex-salvage label+confidence from truncated output.
    4. Return UNCERTAIN fallback.
    """
    if not content or not content.strip():
        return "UNCERTAIN", 0.5, "[Error fallback] Empty response"

    original = content

    def _try_parse(text: str) -> dict | None:
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "label" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # Try bare JSON first (no fences)
    data: dict | None = None
    if "```" not in content:
        data = _try_parse(content)
    else:
        blocks = content.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            if block.lower().startswith("json"):
                block = block[4:]
            data = _try_parse(block)
            if data is not None:
                break

    if data is None:
        # Salvage path: extract label/confidence via regex from truncated output
        label_match = re.search(
            r'"label"\s*:\s*"(VALID|HALLUCINATED|UNCERTAIN)"',
            original,
            re.IGNORECASE,
        )
        conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', original)
        if label_match is not None:
            salvaged_label = label_match.group(1).upper()
            if salvaged_label not in {"VALID", "HALLUCINATED", "UNCERTAIN"}:
                salvaged_label = "UNCERTAIN"
            salvaged_conf = float(conf_match.group(1)) if conf_match else 0.5
            salvaged_conf = max(0.0, min(1.0, salvaged_conf))
            return (
                salvaged_label,
                salvaged_conf,
                "[Salvaged] label/confidence extracted from truncated JSON",
            )

        logger.warning("Failed to parse LLM response for %s", bibtex_key)
        return "UNCERTAIN", 0.5, f"[Error fallback] Parse error: {original[:100]}"

    try:
        label = data.get("label", "UNCERTAIN").upper()
        if label not in {"VALID", "HALLUCINATED", "UNCERTAIN"}:
            label = "UNCERTAIN"
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason", ""))
        return label, confidence, reason
    except (ValueError, TypeError) as e:
        return "UNCERTAIN", 0.5, f"[Error fallback] Parse error: {e}"


def call_one(
    client: openai.OpenAI,
    model: str,
    entry: dict,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> dict:
    """Make one verification call with per-request timeout and exponential backoff.

    IMPORTANT: `timeout` is passed to the client constructor AND to each
    individual `chat.completions.create()` call.  This is the primary defence
    against the 9.5-hour-hang failure mode observed when the network dropped
    mid-stream on the sequential DS-R1 run.

    Retries (with 2s/4s/8s backoff) on:
      - openai.RateLimitError  (HTTP 429 from OpenRouter or upstream)
      - openai.APITimeoutError (per-request timeout hit)
    Other errors fall through to the error-fallback record after max_retries.
    """
    bibtex = entry_to_bibtex(entry)
    prompt = VERIFICATION_PROMPT.format(bibtex=bibtex)

    start = time.time()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
                timeout=timeout,  # per-request timeout — critical to prevent hangs
            )
            content = resp.choices[0].message.content or ""
            label, conf, reason = parse_response(content, entry["bibtex_key"])
            elapsed = time.time() - start
            return {
                "bibtex_key": entry["bibtex_key"],
                "label": label,
                "confidence": conf,
                "reason": reason,
                "wall_clock_seconds": elapsed,
                "api_calls": 1,
                "api_sources_queried": [],
            }
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            last_err = e
            backoff = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            logger.warning(
                "Rate-limit/timeout on %s (attempt %d/%d): %s; backoff %ds",
                entry["bibtex_key"],
                attempt + 1,
                max_retries,
                type(e).__name__,
                backoff,
            )
            time.sleep(backoff)
        except Exception as e:
            last_err = e
            logger.warning(
                "API error on %s (attempt %d/%d): %s",
                entry["bibtex_key"],
                attempt + 1,
                max_retries,
                e,
            )
            # Only retry on rate-limit/timeout; bail on other errors immediately
            break

    elapsed = time.time() - start
    return {
        "bibtex_key": entry["bibtex_key"],
        "label": "UNCERTAIN",
        "confidence": 0.5,
        "reason": f"[Error fallback] API error: {last_err}",
        "wall_clock_seconds": elapsed,
        "api_calls": max_retries,
        "api_sources_queried": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel resume runner for hallmark LLM evaluations."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing (or to create) the JSONL checkpoint.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model id (e.g., deepseek/deepseek-r1).",
    )
    parser.add_argument(
        "--jsonl-name",
        required=True,
        help="Filename within --checkpoint-dir to read existing and append new predictions.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=ROOT / "data" / "v1.0" / "test_public.jsonl",
        help="Benchmark JSONL file to evaluate (default: test_public).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API threads.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (passed to both client and each call).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resume plan (done/remaining/total) and exit without API calls.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Cap the number of entries processed this run (good for testing).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the first 5 remaining entries with workers=2 to verify the run path.",
    )
    args = parser.parse_args()

    # --smoke-test overrides workers and max-entries
    if args.smoke_test:
        args.workers = 2
        args.max_entries = 5 if args.max_entries is None else min(args.max_entries, 5)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY not set")

    # Create client once; timeout is set at client level as default and also
    # passed per-call to catch hangs even if the SDK default changes.
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or "dry-run",
        timeout=args.timeout,
        max_retries=0,  # We handle retries manually to log backoffs
    )

    jsonl_path = args.checkpoint_dir / args.jsonl_name
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume: build set of already-completed bibtex_keys
    done_keys: set[str] = set()
    existing = load_jsonl(jsonl_path)
    for r in existing:
        done_keys.add(r["bibtex_key"])
    n_done = len(done_keys)

    # Load all entries, filter to remaining
    all_entries = load_jsonl(args.data_file)
    remaining = [e for e in all_entries if e["bibtex_key"] not in done_keys]
    n_total = len(all_entries)
    n_remaining = len(remaining)

    if args.dry_run:
        print(f"Found {n_done} existing predictions; remaining {n_remaining} of {n_total}")
        print(f"  data-file:      {args.data_file}")
        print(f"  checkpoint:     {jsonl_path}")
        print(f"  model:          {args.model}")
        print(f"  workers:        {args.workers}")
        print(f"  timeout:        {args.timeout}s")
        if args.max_entries:
            print(f"  max-entries:    {args.max_entries}")
        return

    logger.info(
        "Found %d existing predictions; remaining %d of %d; workers: %d",
        n_done,
        n_remaining,
        n_total,
        args.workers,
    )

    # Apply max-entries cap after logging the full remaining count
    if args.max_entries is not None:
        remaining = remaining[: args.max_entries]
        logger.info("Capped to %d entries (--max-entries)", len(remaining))

    if not remaining:
        logger.info("Nothing to do.")
        return

    lock = threading.Lock()
    completed = 0
    run_start = time.time()

    with jsonl_path.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(call_one, client, args.model, e, args.timeout): e for e in remaining}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                bk = futures[fut]["bibtex_key"]
                logger.exception("Unhandled error on %s: %s", bk, e)
                rec = {
                    "bibtex_key": bk,
                    "label": "UNCERTAIN",
                    "confidence": 0.5,
                    "reason": f"[Error fallback] Unhandled: {e}",
                    "wall_clock_seconds": 0.0,
                    "api_calls": 0,
                    "api_sources_queried": [],
                }

            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()

            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - run_start
                rate = completed / elapsed
                eta_min = (len(remaining) - completed) / max(rate, 1e-9) / 60
                logger.info(
                    "[%d/%d] %.2f entries/s, ETA %.1f min",
                    completed,
                    len(remaining),
                    rate,
                    eta_min,
                )

    logger.info("Done. Wrote %d new predictions to %s", completed, jsonl_path)


if __name__ == "__main__":
    main()
