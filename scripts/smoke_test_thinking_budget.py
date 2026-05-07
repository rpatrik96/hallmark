"""Smoke-test runner: thinking-budget sensitivity for post-snapshot frontier models.

Runs zero-shot citation verification on a stratified n=100 subsample of
dev_public across three models under two budget regimes (a=low, b=high) and
saves per-entry JSONL predictions + summary JSON to results/checkpoints/.

Usage:
    uv run python scripts/smoke_test_thinking_budget.py \\
        --regime both --models all --n 100 \\
        --output-dir results/checkpoints/smoke_test_thinking_budget

    # Dry run (no API calls):
    uv run python scripts/smoke_test_thinking_budget.py \\
        --regime both --models all --n 100 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure hallmark package is importable when run as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from hallmark.baselines.llm_verifier import (
    _build_verification_prompt,
    _parse_llm_response,
)
from hallmark.dataset.schema import BenchmarkEntry, load_entries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model x regime cohort definition
# ---------------------------------------------------------------------------

COHORT: dict[str, dict[str, Any]] = {
    "gpt-5-5": {
        "model": "openai/gpt-5.5",
        "tier": "non-thinking",
        "regime_a": {"max_tokens": 256},
        "regime_b": {"max_tokens": 1024},
        "regime_c": {"max_tokens": 4096},
    },
    "gemini-3-1-pro": {
        "model": "google/gemini-3.1-pro-preview",
        "tier": "thinking",
        "regime_a": {"max_tokens": 2048, "reasoning": {"max_tokens": 1024}},
        "regime_b": {"max_tokens": 8192, "reasoning": {"max_tokens": 4096}},
        "regime_c": {"max_tokens": 16384, "reasoning": {"max_tokens": 8192}},
    },
    "deepseek-v4-pro": {
        "model": "deepseek/deepseek-v4-pro",
        "tier": "thinking",
        "regime_a": {"max_tokens": 4096, "reasoning": {"effort": "low"}},
        "regime_b": {"max_tokens": 8192, "reasoning": {"effort": "high"}},
        "regime_c": {"max_tokens": 16384, "reasoning": {"effort": "high"}},
    },
}

# Price table: USD per 1M tokens (prompt, completion)
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-5.5": (5.0, 30.0),
    "google/gemini-3.1-pro-preview": (3.5, 15.0),
    "deepseek/deepseek-v4-pro": (0.30, 1.20),
}

DEFAULT_PROMPT_TOKENS = 600  # fallback when usage.prompt_tokens unavailable

# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------


def stratified_sample(
    entries: list[BenchmarkEntry],
    n: int,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Return a stratified sample of exactly *n* entries with a floor of 5 per bucket.

    Buckets are defined by ``hallucination_type`` for hallucinated entries and
    ``"valid"`` for VALID entries.  Allocation is proportional to bucket size
    with a hard floor of 5 per bucket; remaining slots are distributed
    proportionally over buckets that still have room.
    """

    rng = random.Random(seed)

    # Build buckets, sorted by bibtex_key for determinism
    buckets: dict[str, list[BenchmarkEntry]] = {}
    for e in entries:
        key = "valid" if e.label == "VALID" else (e.hallucination_type or "unknown")
        buckets.setdefault(key, [])
        buckets[key].append(e)
    for v in buckets.values():
        v.sort(key=lambda e: e.bibtex_key)

    num_buckets = len(buckets)
    floor = 5
    floor_total = floor * num_buckets
    if floor_total > n:
        raise ValueError(
            f"floor={floor} x {num_buckets} buckets = {floor_total} > n={n}; "
            "increase n or reduce floor"
        )

    # Proportional allocation with floor
    total = sum(len(v) for v in buckets.values())
    alloc: dict[str, int] = {}
    for bk, bv in buckets.items():
        alloc[bk] = max(floor, round(n * len(bv) / total))

    # Adjust to hit exactly n
    diff = sum(alloc.values()) - n
    # Sort by allocation descending so we trim/add to largest buckets first
    sorted_keys = sorted(alloc, key=lambda k: alloc[k], reverse=True)
    i = 0
    while diff != 0:
        bk = sorted_keys[i % len(sorted_keys)]
        if diff > 0 and alloc[bk] > floor:
            alloc[bk] -= 1
            diff -= 1
        elif diff < 0 and alloc[bk] < len(buckets[bk]):
            alloc[bk] += 1
            diff += 1
        i += 1

    # Sample from each bucket
    sampled: list[BenchmarkEntry] = []
    for bk, cnt in alloc.items():
        pool = buckets[bk]
        cnt = min(cnt, len(pool))
        sampled.extend(rng.sample(pool, cnt))

    return sampled


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Return {bibtex_key: record_dict} from an existing JSONL checkpoint."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["bibtex_key"]] = rec
    return out


def append_record(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def flush_summary(
    path: Path,
    model_id: str,
    regime: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    n = len(records)
    parse_failures = sum(1 for r in records if r.get("parse_failure", False))
    output_tokens_list = [r.get("output_tokens", 0) for r in records]
    wall_times = [r.get("wall_clock_seconds", 0.0) for r in records]
    prompt_tokens_list = [r.get("prompt_tokens", DEFAULT_PROMPT_TOKENS) for r in records]

    p95_output_tokens = int(np.percentile(output_tokens_list, 95)) if output_tokens_list else 0

    mean_output = statistics.mean(output_tokens_list) if output_tokens_list else 0.0
    mean_prompt = (
        statistics.mean(prompt_tokens_list) if prompt_tokens_list else DEFAULT_PROMPT_TOKENS
    )

    price_prompt, price_compl = PRICES.get(model_id, (0.0, 0.0))
    estimated_cost = n * (mean_prompt * price_prompt + mean_output * price_compl) / 1e6

    summary = {
        "model": model_id,
        "regime": regime,
        "n": n,
        "parse_failure_rate": parse_failures / n if n else 0.0,
        "mean_output_tokens": mean_output,
        "p95_output_tokens": p95_output_tokens,
        "total_wall_clock_seconds": sum(wall_times),
        "estimated_cost_usd": estimated_cost,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-entry API call
# ---------------------------------------------------------------------------


def call_openrouter(
    client: Any,
    model_id: str,
    prompt: str,
    max_tokens: int,
    reasoning: dict[str, Any] | None,
) -> tuple[str, int, int, int]:
    """Call OpenRouter and return (content, prompt_tokens, output_tokens, reasoning_tokens)."""
    extra_body: dict[str, Any] = {}
    if reasoning:
        extra_body["reasoning"] = reasoning

    # GPT-5.5 via OpenRouter: temperature must be 1.0 (same constraint as native OpenAI)
    temperature = 1.0 if "gpt-5.5" in model_id else 0.0

    call_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 42,
    }
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**call_kwargs)
    content = str(resp.choices[0].message.content or "").strip()

    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", DEFAULT_PROMPT_TOKENS) or DEFAULT_PROMPT_TOKENS
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    # OpenRouter surfaces reasoning tokens under completion_tokens_details
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

    return content, prompt_tokens, completion_tokens, reasoning_tokens


def run_entry(
    client: Any,
    entry: BenchmarkEntry,
    model_id: str,
    max_tokens: int,
    reasoning: dict[str, Any] | None,
    max_retries: int = 3,
    failure_log: Path | None = None,
) -> dict[str, Any]:
    """Verify one entry, return a record dict ready for JSONL output."""
    import openai

    blind = entry.to_blind()
    prompt = _build_verification_prompt(blind)

    backoff = [2, 4, 8]
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        start = time.monotonic()
        try:
            content, prompt_tokens, output_tokens, reasoning_tokens = call_openrouter(
                client, model_id, prompt, max_tokens, reasoning
            )
            elapsed = time.monotonic() - start

            pred = _parse_llm_response(content, entry.bibtex_key)
            # [Salvaged] predictions are partial successes, NOT parse failures.
            # Only treat [Error fallback] + "Parse error" as a true parse failure.
            parse_failure = (
                pred.reason.startswith("[Error fallback]") and "Parse error" in pred.reason
            )

            return {
                "bibtex_key": entry.bibtex_key,
                "label": pred.label,
                "confidence": pred.confidence,
                "reason": pred.reason,
                "wall_clock_seconds": elapsed,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "prompt_tokens": prompt_tokens,
                "raw_response_preview": content[:500],
                "parse_failure": parse_failure,
            }

        except openai.RateLimitError as e:
            last_exc = e
            if attempt < max_retries:
                sleep_sec = backoff[min(attempt, len(backoff) - 1)]
                logger.warning(
                    "Rate limit on %s/%s (attempt %d/%d), sleeping %ds",
                    model_id,
                    entry.bibtex_key,
                    attempt + 1,
                    max_retries + 1,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
            continue

        except openai.APIStatusError as e:
            elapsed = time.monotonic() - start
            # 404 → model not available, log and bail
            if e.status_code == 404:
                msg = f"Model {model_id} returned 404: {e.body}"
                logger.error(msg)
                if failure_log is not None:
                    with open(failure_log, "a") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "model": model_id,
                                    "status_code": e.status_code,
                                    "error": str(e.body),
                                }
                            )
                            + "\n"
                        )
                raise  # propagate so the outer loop can skip the whole model

            last_exc = e
            if attempt < max_retries:
                sleep_sec = backoff[min(attempt, len(backoff) - 1)]
                logger.warning(
                    "API error %d for %s (attempt %d/%d), sleeping %ds: %s",
                    e.status_code,
                    entry.bibtex_key,
                    attempt + 1,
                    max_retries + 1,
                    sleep_sec,
                    e,
                )
                time.sleep(sleep_sec)
            continue

        except Exception as e:
            elapsed = time.monotonic() - start
            last_exc = e
            break

    # All retries exhausted / non-retryable error
    elapsed = time.monotonic() - start
    reason = f"[Parse failure] API error after {max_retries} retries: {str(last_exc)[:200]}"
    return {
        "bibtex_key": entry.bibtex_key,
        "label": "UNCERTAIN",
        "confidence": 0.0,
        "reason": reason,
        "wall_clock_seconds": elapsed,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "prompt_tokens": DEFAULT_PROMPT_TOKENS,
        "raw_response_preview": "",
        "parse_failure": True,
    }


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def run_cell(
    model_key: str,
    regime: str,
    entries: list[BenchmarkEntry],
    output_dir: Path,
    failure_log: Path,
) -> None:
    """Run one (model, regime) cell, writing JSONL + summary JSON."""
    import openai

    cfg = COHORT[model_key]
    model_id: str = cfg["model"]
    regime_cfg: dict[str, Any] = dict(cfg[f"regime_{regime}"])
    max_tokens: int = regime_cfg.pop("max_tokens")
    reasoning: dict[str, Any] | None = regime_cfg.pop("reasoning", None)

    cell_name = f"{model_key}_{regime}"
    jsonl_path = output_dir / f"{cell_name}.jsonl"
    summary_path = output_dir / f"{cell_name}.summary.json"

    # Resume semantics
    completed = load_checkpoint(jsonl_path)
    records: list[dict[str, Any]] = list(completed.values())
    pending = [e for e in entries if e.bibtex_key not in completed]

    if not pending:
        logger.info("[%s] All %d entries already completed, skipping.", cell_name, len(records))
        flush_summary(summary_path, model_id, regime, records)
        return

    logger.info(
        "[%s] Starting: %d total, %d pending (model=%s, max_tokens=%d, reasoning=%s)",
        cell_name,
        len(entries),
        len(pending),
        model_id,
        max_tokens,
        reasoning,
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OSError("OPENROUTER_API_KEY not set")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=0,  # we handle retries ourselves
        timeout=120.0,
    )

    model_unavailable = False
    for i, entry in enumerate(pending):
        try:
            rec = run_entry(
                client,
                entry,
                model_id,
                max_tokens,
                reasoning,
                failure_log=failure_log,
            )
        except openai.APIStatusError as e:
            if e.status_code == 404:
                logger.error("[%s] Model unavailable (404), skipping entire cell.", cell_name)
                model_unavailable = True
                break
            # Other status errors: record as parse failure
            rec = {
                "bibtex_key": entry.bibtex_key,
                "label": "UNCERTAIN",
                "confidence": 0.0,
                "reason": f"[Parse failure] APIStatusError {e.status_code}: {str(e)[:200]}",
                "wall_clock_seconds": 0.0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "prompt_tokens": DEFAULT_PROMPT_TOKENS,
                "raw_response_preview": "",
                "parse_failure": True,
            }

        records.append(rec)
        append_record(jsonl_path, rec)

        # Flush summary every 10 entries
        if (i + 1) % 10 == 0:
            flush_summary(summary_path, model_id, regime, records)
            logger.info("[%s] %d/%d done", cell_name, len(records), len(entries))

    if not model_unavailable:
        flush_summary(summary_path, model_id, regime, records)
        logger.info(
            "[%s] Done: %d entries, %.1f%% parse failures",
            cell_name,
            len(records),
            100 * sum(1 for r in records if r.get("parse_failure")) / max(len(records), 1),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test thinking-budget sensitivity on dev_public subsample."
    )
    p.add_argument(
        "--regime",
        choices=["a", "b", "c", "both", "all"],
        default="both",
        help=(
            "Budget regime to run. 'a' = main-table-matched, 'b' = paired headroom, "
            "'c' = ceiling-of-headroom (conditional, run only if regime B p95 saturates "
            "the cap on a given cell). 'both' = a+b. 'all' = a+b+c. Default: both."
        ),
    )
    p.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all' (choices: " + ", ".join(COHORT) + ")",
    )
    p.add_argument("--n", type=int, default=100, help="Sample size (default: 100)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/checkpoints/smoke_test_thinking_budget"),
        help="Output directory",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/v1.0"), help="Data directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and exit without API calls",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    # Resolve model list
    if args.models == "all":
        model_keys = list(COHORT)
    else:
        model_keys = [m.strip() for m in args.models.split(",")]
        unknown = [m for m in model_keys if m not in COHORT]
        if unknown:
            logger.error("Unknown model keys: %s. Valid: %s", unknown, list(COHORT))
            sys.exit(1)

    # Resolve regime list
    if args.regime == "both":
        regimes = ["a", "b"]
    elif args.regime == "all":
        regimes = ["a", "b", "c"]
    else:
        regimes = [args.regime]

    # Output directory
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Failure log
    failure_log = output_dir / "provider_failures.log"

    # ---------------------------------------------------------------------------
    # Build or load sample
    # ---------------------------------------------------------------------------
    sample_keys_path = output_dir / "sample_keys.json"
    data_path = args.data_dir / "dev_public.jsonl"

    entries_all = load_entries(data_path)

    if sample_keys_path.exists():
        logger.info("Loading existing sample from %s", sample_keys_path)
        sample_keys: list[str] = json.loads(sample_keys_path.read_text())
        key_set = set(sample_keys)
        entries_map = {e.bibtex_key: e for e in entries_all}
        missing = key_set - set(entries_map)
        if missing:
            logger.warning("Sample references %d keys not in data file", len(missing))
        sample_entries = [entries_map[k] for k in sample_keys if k in entries_map]
    else:
        sample_entries = stratified_sample(entries_all, n=args.n, seed=args.seed)
        sample_keys = [e.bibtex_key for e in sample_entries]
        if not args.dry_run:
            sample_keys_path.write_text(json.dumps(sample_keys, indent=2) + "\n")
            logger.info("Saved sample keys to %s", sample_keys_path)

    # ---------------------------------------------------------------------------
    # Bucket distribution report
    # ---------------------------------------------------------------------------
    bucket_counts: dict[str, int] = {}
    for e in sample_entries:
        bk = "valid" if e.label == "VALID" else (e.hallucination_type or "unknown")
        bucket_counts[bk] = bucket_counts.get(bk, 0) + 1

    print("\n=== Stratified sample bucket distribution ===")
    print(f"{'Bucket':<35} {'Count':>6}")
    print("-" * 43)
    for bk in sorted(bucket_counts):
        print(f"{bk:<35} {bucket_counts[bk]:>6}")
    print("-" * 43)
    print(f"{'TOTAL':<35} {sum(bucket_counts.values()):>6}")
    note = (
        f"  (floor=5 per bucket; proportional allocation; "
        f"total={sum(bucket_counts.values())} / requested={args.n})"
    )
    print(note)

    # ---------------------------------------------------------------------------
    # Model x regime grid report
    # ---------------------------------------------------------------------------
    print("\n=== Model x regime execution plan ===")
    header = f"{'Cell':<30} {'Model ID':<40} {'max_tokens':>12} {'reasoning'}"
    print(header)
    print("-" * 100)
    for mk in model_keys:
        cfg = COHORT[mk]
        for r in regimes:
            rc = dict(cfg[f"regime_{r}"])
            mt = rc.get("max_tokens", "?")
            rea = rc.get("reasoning")
            cell = f"{mk}_{r}"
            print(f"{cell:<30} {cfg['model']:<40} {mt:>12} {rea}")

    print(f"\nTotal cells: {len(model_keys) * len(regimes)}")
    print(f"Entries per cell: {len(sample_entries)}")
    print(f"Total API calls (approx): {len(model_keys) * len(regimes) * len(sample_entries)}")

    if args.dry_run:
        print("\n[dry-run] Exiting without API calls.")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Save sample keys now (after dry-run check, if not already saved)
    # ---------------------------------------------------------------------------
    if not sample_keys_path.exists():
        sample_keys_path.write_text(json.dumps(sample_keys, indent=2) + "\n")
        logger.info("Saved sample keys to %s", sample_keys_path)

    # ---------------------------------------------------------------------------
    # Run cells
    # ---------------------------------------------------------------------------
    for mk in model_keys:
        for r in regimes:
            try:
                run_cell(mk, r, sample_entries, output_dir, failure_log)
            except Exception as exc:
                logger.error("Cell %s_%s failed with unexpected error: %s", mk, r, exc)
                # Log to provider failures and continue
                with open(failure_log, "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "model_key": mk,
                                "regime": r,
                                "error": str(exc),
                            }
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
