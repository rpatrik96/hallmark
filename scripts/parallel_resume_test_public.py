"""Resume an interrupted hallmark evaluate run in parallel.

Picks up an existing checkpoint dir, reads predictions already there, and fans
out remaining entries across N concurrent threads to the selected provider
(--provider openrouter|huggingface).

Prompt building and response parsing are IMPORTED from
hallmark.baselines.llm_verifier rather than reimplemented here, so this path
cannot drift from the sequential one. An earlier revision kept local copies and
they did drift: the copy never extracted predicted_hallucination_type, and it
had no <think>-block stripping, so Qwen3 replies fell to the lossy salvage path.

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

    # HuggingFace router (Qwen3 dense sweep) — needs HF_TOKEN
    uv run python scripts/parallel_resume_test_public.py \\
        --provider huggingface \\
        --checkpoint-dir results/checkpoints/llm_hf_qwen3_4b_test_public \\
        --model Qwen/Qwen3-4B:featherless-ai \\
        --jsonl-name huggingface_qwen3-4b.jsonl \\
        --workers 4

After completion, run `hallmark evaluate --predictions <jsonl>` to compute
the eval.json from the assembled predictions.

Checkpoint guard
================
Records are written through ``GuardedCheckpointWriter``: a record that carries no
verdict (an ``[Error fallback]`` reason) never enters the checkpoint, so its key
is absent on the next run and gets retried instead of being trusted forever.  It
is parked in a ``<jsonl>.rejected-<timestamp>.jsonl`` sidecar.  Once the failed
share of the run crosses the batch-health threshold the run is refused outright
and exits non-zero, on the reasoning that a share that high is a transport outage
rather than a property of the bibliography.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hallmark.baselines.llm_verifier import (  # noqa: E402
    _NO_THINK_MODELS,
    _NO_THINK_SUFFIX,
    HF_ROUTER_BASE_URL,
    _build_verification_prompt,
    _parse_llm_response,
)
from hallmark.dataset.schema import BlindEntry  # noqa: E402

# Provider routing: (base_url, env var holding the key, api_sources_queried prefix).
# Keep every provider-specific value here so adding one is a single-line change.
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter"),
    "huggingface": (HF_ROUTER_BASE_URL, "HF_TOKEN", "huggingface"),
}
from hallmark.baselines.checkpoint_guard import (  # noqa: E402
    GuardedCheckpointWriter,
    PoisonedBatchError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_prompt(entry: dict, model: str) -> str:
    """Build the verification prompt for one entry.

    Delegates to the baseline's own prompt builder rather than reconstructing
    the BibTeX locally, so the parallel and sequential paths cannot drift.

    Qwen3 defaults to thinking ON and burns the completion-token budget before
    emitting the JSON verdict; models in _NO_THINK_MODELS get Qwen3's own
    ``/no_think`` soft switch, exactly as the baseline does.
    """
    blind = BlindEntry(
        bibtex_key=entry["bibtex_key"],
        bibtex_type=entry.get("bibtex_type", "article"),
        fields=entry["fields"],
        raw_bibtex=entry.get("raw_bibtex", ""),
    )
    prompt = _build_verification_prompt(blind)
    if model in _NO_THINK_MODELS:
        prompt = f"{prompt}\n\n{_NO_THINK_SUFFIX}"
    return prompt


def call_one(
    client: openai.OpenAI,
    model: str,
    entry: dict,
    timeout: float = 120.0,
    max_retries: int = 3,
    max_completion_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
    source_prefix: str = "openrouter",
) -> dict:
    """Make one verification call with per-request timeout and exponential backoff.

    IMPORTANT: `timeout` is passed to the client constructor AND to each
    individual `chat.completions.create()` call.  This is the primary defence
    against the 9.5-hour-hang failure mode observed when the network dropped
    mid-stream on the sequential DS-R1 run.

    Retries (with 2s/4s/8s backoff) on:
      - openai.RateLimitError    (HTTP 429 from OpenRouter or upstream)
      - openai.APITimeoutError   (per-request timeout hit)
      - openai.APIConnectionError (network blip / connection reset)
      - openai.APIStatusError    only when status_code >= 500 (server-side
                                  errors are usually transient)
    4xx (auth, quota, malformed request) bails immediately.
    """
    prompt = build_prompt(entry, model)

    start = time.time()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,  # match the main verifier path (default 0.0)
                seed=seed,  # determinism parity with hallmark.baselines.llm_verifier
                timeout=timeout,  # per-request timeout — critical to prevent hangs
            )
            content = resp.choices[0].message.content or ""
            pred = _parse_llm_response(content, entry["bibtex_key"])
            elapsed = time.time() - start
            return {
                "bibtex_key": entry["bibtex_key"],
                "label": pred.label,
                "confidence": pred.confidence,
                "reason": pred.reason,
                "predicted_hallucination_type": pred.predicted_hallucination_type,
                "wall_clock_seconds": elapsed,
                "api_calls": 1,
                "api_sources_queried": [f"{source_prefix}/{model}"],
            }
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            last_err = e
            backoff = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            logger.warning(
                "Transient error on %s (attempt %d/%d): %s; backoff %ds",
                entry["bibtex_key"],
                attempt + 1,
                max_retries,
                type(e).__name__,
                backoff,
            )
            time.sleep(backoff)
        except openai.APIStatusError as e:
            last_err = e
            # Retry 5xx (server-side, usually transient); bail on 4xx.
            status = getattr(e, "status_code", 0) or 0
            if status >= 500:
                backoff = 2 ** (attempt + 1)
                logger.warning(
                    "Server error %d on %s (attempt %d/%d); backoff %ds",
                    status,
                    entry["bibtex_key"],
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.warning(
                    "Client error %d on %s — bailing without retry: %s",
                    status,
                    entry["bibtex_key"],
                    e,
                )
                break
        except Exception as e:
            last_err = e
            logger.warning(
                "Non-API error on %s (attempt %d/%d): %s — bailing",
                entry["bibtex_key"],
                attempt + 1,
                max_retries,
                e,
            )
            break

    elapsed = time.time() - start
    return {
        "bibtex_key": entry["bibtex_key"],
        "label": "UNCERTAIN",
        "confidence": 0.5,
        "reason": f"[Error fallback] API error: {last_err}",
        # Present on the failure path too, so every checkpoint row has the same
        # shape and readers can index it unconditionally. Note this is NOT how a
        # failure is detected — null is also the required value for a successful
        # VALID verdict; --retry-failed keys on the reason prefix instead.
        "predicted_hallucination_type": None,
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
        "--provider",
        choices=sorted(PROVIDERS),
        default="openrouter",
        help=(
            "Inference provider. 'openrouter' (default) uses OPENROUTER_API_KEY; "
            "'huggingface' uses HF_TOKEN and the HF Inference Providers router, "
            "which is where the Qwen3 dense sweep runs."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Provider model id — e.g. 'deepseek/deepseek-r1' for openrouter, "
            "'Qwen/Qwen3-4B:featherless-ai' for huggingface."
        ),
    )
    parser.add_argument(
        "--jsonl-name",
        required=True,
        help="Filename within --checkpoint-dir to read existing and append new predictions.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=ROOT / "data" / "v1.2" / "test_public.jsonl",
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
        "--max-completion-tokens",
        type=int,
        default=1024,
        help=(
            "Max completion tokens per request. Reasoning models "
            "(DeepSeek-R1, GPT-5.5, Gemini 3.x Pro) may need 4096+ to fit "
            "the JSON verdict after the reasoning trace."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0, matching the main verifier path).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed for determinism parity with the main verifier (default 42).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Re-attempt entries whose only checkpoint row is an '[Error fallback]' "
            "(API outage, depleted credits). Without this they count as done and "
            "are skipped forever. Successful retries append a new row, which wins "
            "on load since predictions dedupe by bibtex_key, last-one-wins."
        ),
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

    base_url, env_var, source_prefix = PROVIDERS[args.provider]
    api_key = os.environ.get(env_var)
    if not api_key and not args.dry_run:
        raise SystemExit(f"{env_var} not set (required for --provider {args.provider})")

    # Create client once; timeout is set at client level as default and also
    # passed per-call to catch hangs even if the SDK default changes.
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key or "dry-run",
        timeout=args.timeout,
        max_retries=0,  # We handle retries manually to log backoffs
    )

    jsonl_path = args.checkpoint_dir / args.jsonl_name
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume: build set of already-completed bibtex_keys
    # An entry counts as done if it has any non-failed row. With --retry-failed,
    # "[Error fallback]" rows do NOT count, so a run killed mid-sweep (outage,
    # depleted credits) can be resumed rather than leaving those keys looking
    # complete forever. Mirrors _load_checkpoint(skip_failed=) in llm_verifier.
    done_keys: set[str] = set()
    existing = load_jsonl(jsonl_path)
    n_failed = 0
    for r in existing:
        if str(r.get("reason", "")).startswith("[Error fallback]"):
            n_failed += 1
            if args.retry_failed:
                continue
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
        print(
            f"  failed rows:    {n_failed} ({'retrying' if args.retry_failed else 'counted as done'})"
        )
        print(f"  provider:       {args.provider} ({base_url})")
        print(f"  model:          {args.model}")
        print(f"  /no_think:      {args.model in _NO_THINK_MODELS}")
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

    completed = 0
    run_start = time.time()
    poisoned: PoisonedBatchError | None = None

    with (
        GuardedCheckpointWriter(jsonl_path) as writer,
        ThreadPoolExecutor(max_workers=args.workers) as ex,
    ):
        futures = {
            ex.submit(
                call_one,
                client,
                args.model,
                e,
                args.timeout,
                3,
                args.max_completion_tokens,
                args.temperature,
                args.seed,
                source_prefix,
            ): e
            for e in remaining
        }
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

            try:
                writer.add(rec)
            except PoisonedBatchError as exc:
                # Stop paying for calls that cannot produce evidence. Only usable
                # verdicts reached the checkpoint, so every refused key is
                # retried by the next run.
                poisoned = exc
                logger.error("%s", exc)
                for pending in futures:
                    pending.cancel()
                break

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

    if poisoned is not None:
        raise SystemExit(f"{poisoned}\nRefused records: {writer.rejected_path}")

    logger.info("Done. Wrote %d new predictions to %s", writer.written, jsonl_path)


if __name__ == "__main__":
    main()
