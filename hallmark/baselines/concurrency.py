"""Parallel-fan-out wrapper for any registered HALLMARK baseline.

Most LLM-shaped baselines (zero-shot OpenAI/Anthropic/OpenRouter, agentic
tool-use, BTU-as-tool, tool-augmented) are *embarrassingly parallel at the
entry level*: each BibTeX entry is verified by an independent API call
sequence, with no cross-entry state beyond the on-disk JSONL checkpoint.

This module exposes :func:`parallel_run_baseline`, a thin
``ThreadPoolExecutor`` fan-out around the existing baseline runners.  The
sequential runners themselves implement the per-entry contract; we just
issue ``workers`` of them concurrently, each with a one-element entry list,
and rely on each runner's own checkpoint-resume logic to dedup.

Trade-offs / when to use ``--workers > 1``
-----------------------------------------
*   **Use it for**:  rate-limited remote APIs (OpenRouter, Anthropic) where
    a single entry takes 5-60 s and the upstream cap is well above your
    sequential RPM (e.g. 8 workers x 30 s/call ~ 16 RPM, vs. OpenRouter's
    ~200 RPM ceiling).  Realistic speedups: 4-6x.
*   **Don't use it for**:  CLI-spawning baselines (``bibtexupdater``,
    ``harc``, ``verify_citations``) — those shell out to a long-running
    subprocess that already saturates the upstream and would step on each
    other's tmpdirs/caches.  The CLI has no intrinsic safeguard against
    this, so it is the caller's responsibility to leave ``--workers=1`` for
    CLI baselines.
*   **Per-thread vs shared verifier state**:  each worker thread builds its
    own openai/anthropic client (fast, the SDK is thread-safe) and shares
    the SQLite tool cache across threads (also safe — see
    ``_agentic_cache.py`` ``check_same_thread=False``).  The on-disk JSONL
    is appended to under a module-level lock (see :func:`add_checkpoint_lock`
    + ``_CHECKPOINT_LOCK``) because macOS's ``O_APPEND`` is *not* atomic
    for writes >512 bytes, and our prediction records routinely exceed
    that threshold once the ``reason`` is populated by an agentic loop.

Rate-limit considerations
-------------------------
Pick ``workers`` so that ``workers x median_latency_seconds <= upstream_rpm
/ 60``.  In practice 4-8 workers is a safe default for OpenRouter paid tier
(200 RPM) on agentic baselines that average 30-60 s per entry.  If you see
HTTP 429s in the worker logs, the underlying SDK's exponential-backoff
retry handles them transparently — the only failure mode that *isn't*
auto-handled is when ``max_consecutive_failures`` (in the agentic
verifiers) is set too low and a 429 burst trips the abort path; the
agentic verifiers default this to 10 to absorb such bursts.

Entry-points
------------
* :func:`parallel_run_baseline` — top-level fan-out around any registered
  baseline.
* :func:`add_checkpoint_lock` — decorator that wraps a checkpoint-write
  callable in ``_CHECKPOINT_LOCK`` so concurrent writes serialise.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from hallmark.baselines import registry
from hallmark.baselines.common import fallback_predictions
from hallmark.dataset.schema import BenchmarkEntry, BlindEntry, Prediction

logger = logging.getLogger(__name__)

# Module-level lock used by every checkpoint writer that wants to be safe
# against concurrent appends. Importable for use inside other baseline
# modules (e.g. llm_agentic.py) so they share a single lock with this fan-out.
_CHECKPOINT_LOCK = threading.Lock()


F = TypeVar("F", bound=Callable[..., Any])


def add_checkpoint_lock(fn: F) -> F:
    """Decorator: wrap ``fn`` so it executes under ``_CHECKPOINT_LOCK``.

    Use this on any function that appends to a JSONL checkpoint to keep
    concurrent appends from interleaving on macOS, where ``O_APPEND`` is
    only atomic for writes <=512 bytes.

    The decorated function preserves its original signature.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _CHECKPOINT_LOCK:
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _to_blind_list(entries: Sequence[BenchmarkEntry | BlindEntry]) -> list[BlindEntry]:
    """Coerce a heterogeneous entry sequence to a list of BlindEntry.

    Accepts both BenchmarkEntry (call sites that haven't blinded yet) and
    BlindEntry (call sites that already did the blinding, e.g. the
    parallel-script shims that load from JSONL).
    """
    out: list[BlindEntry] = []
    for e in entries:
        if isinstance(e, BlindEntry):
            out.append(e)
        elif isinstance(e, BenchmarkEntry):
            out.append(e.to_blind())
        else:  # pragma: no cover — defensive
            raise TypeError(f"parallel_run_baseline: unsupported entry type {type(e).__name__}")
    return out


def _collect_completed_keys(checkpoint_dir: Path) -> set[str]:
    """Read every JSONL file in ``checkpoint_dir`` and collect bibtex_keys.

    Each registered LLM baseline writes its checkpoint as
    ``<source_prefix>_<safe_model>.jsonl``.  Different baselines or model
    runs land in different files, but for a single run we expect exactly
    one file to grow.  We union over all files in the dir so resume works
    even when a directory is reused across runs.
    """
    if not checkpoint_dir.exists():
        return set()

    import json

    done: set[str] = set()
    for jsonl in sorted(checkpoint_dir.glob("*.jsonl")):
        try:
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bk = rec.get("bibtex_key")
                if isinstance(bk, str):
                    done.add(bk)
        except OSError as exc:
            logger.warning("Could not read checkpoint %s: %s", jsonl, exc)
    return done


def _fallback_for_unhandled(entry: BlindEntry, exc: BaseException) -> Prediction:
    """Build an UNCERTAIN fallback for a worker that raised."""
    return Prediction(
        bibtex_key=entry.bibtex_key,
        label="UNCERTAIN",
        confidence=0.5,
        reason=f"[Error fallback] Unhandled in parallel worker: {exc}",
        wall_clock_seconds=0.0,
        api_calls=0,
        api_sources_queried=[],
    )


def parallel_run_baseline(
    name: str,
    entries: Sequence[BenchmarkEntry | BlindEntry],
    *,
    workers: int,
    checkpoint_dir: Path,
    split: str | None = None,
    progress_every: int = 10,
    **baseline_kwargs: Any,
) -> list[Prediction]:
    """Fan-out wrapper around a registered baseline using ``ThreadPoolExecutor``.

    For each entry not already present in any JSONL under ``checkpoint_dir``,
    submits a single-entry call to ``registry.run_baseline(name, [entry], ...)``
    on a worker thread.  The baseline's own checkpoint-write path is
    responsible for persisting the prediction; this function additionally
    aggregates the in-memory predictions and returns them.

    Args:
        name:           Registered baseline name (e.g. ``llm_openrouter_gemini_flash``).
        entries:        Benchmark or blind entries to verify.
        workers:        Number of concurrent worker threads.  Use ``1`` to
                        force the sequential path through ``run_baseline``.
        checkpoint_dir: Directory where the underlying baseline writes its
                        per-entry JSONL checkpoint.  Must be writable.
        split:          Forwarded to ``run_baseline`` (some baselines branch
                        on split — title_oracle warns when split=="dev_public").
        progress_every: Log a progress line every ``N`` completions.
        **baseline_kwargs: Forwarded verbatim to the baseline runner.  Note
                        that ``checkpoint_dir`` is *always* injected from
                        the explicit kwarg; passing it again here is a
                        TypeError.

    Returns:
        Concatenated list of ``Prediction`` records covering every entry
        in ``entries`` (resumed-from-disk + freshly produced in this run).
        Records are returned in completion order, not input order.
    """
    if "checkpoint_dir" in baseline_kwargs:
        raise TypeError(
            "parallel_run_baseline: pass checkpoint_dir as a keyword argument, "
            "not via **baseline_kwargs"
        )

    blind_entries = _to_blind_list(entries)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Sequential fast path. Used when --workers 1 (or omitted) to keep the
    # same call shape as the legacy sequential CLI path. We call the
    # registry's runner directly (instead of registry.run_baseline) because
    # our entries are already blinded and run_baseline would re-blind them
    # under a different type signature.
    if workers <= 1:
        info = registry.get_registry()[name]
        merged_kwargs: dict[str, Any] = {
            **info.runner_kwargs,
            **baseline_kwargs,
            "checkpoint_dir": checkpoint_dir,
        }
        if split is not None:
            merged_kwargs.setdefault("split", split)
        if info.requires_api_key and info.env_var and "api_key" not in merged_kwargs:
            import os

            env_key = os.environ.get(info.env_var)
            if env_key:
                merged_kwargs["api_key"] = env_key
        return list(info.runner(blind_entries, **merged_kwargs))

    # Resume: skip entries that already have a record on disk.
    done_keys = _collect_completed_keys(checkpoint_dir)
    remaining = [e for e in blind_entries if e.bibtex_key not in done_keys]
    n_total = len(blind_entries)
    n_done = len(blind_entries) - len(remaining)
    n_remaining = len(remaining)
    logger.info(
        "parallel_run_baseline(%s): %d done, %d remaining of %d (workers=%d)",
        name,
        n_done,
        n_remaining,
        n_total,
        workers,
    )

    # Pre-existing predictions from disk are loaded by each worker's call
    # into the underlying runner; here we just need the freshly produced ones
    # for the in-memory return value.  We also load any disk-existing records
    # ourselves so the returned list covers the full input set.
    pre_existing = _load_predictions_from_dir(checkpoint_dir, done_keys)

    if not remaining:
        logger.info("parallel_run_baseline(%s): nothing to do.", name)
        return pre_existing

    new_preds: list[Prediction] = []
    completed = 0
    started = time.time()

    def _call_one(entry: BlindEntry) -> list[Prediction]:
        # Each worker calls the registered runner with a single-entry list.
        # The runner re-blinds on its own (registry.run_baseline does
        # _to_blind), but our entries are already BlindEntry, so we bypass
        # the registry to avoid double-blinding and pass directly to the
        # runner. We still want the registry-level api_key auto-injection,
        # however, so we replicate that minimal logic here.
        info = registry.get_registry()[name]
        merged_kwargs: dict[str, Any] = {
            **info.runner_kwargs,
            **baseline_kwargs,
            "checkpoint_dir": checkpoint_dir,
        }
        if split is not None:
            merged_kwargs.setdefault("split", split)
        if info.requires_api_key and info.env_var and "api_key" not in merged_kwargs:
            import os

            env_key = os.environ.get(info.env_var)
            if env_key:
                merged_kwargs["api_key"] = env_key
        try:
            return list(info.runner([entry], **merged_kwargs))
        except Exception as exc:
            logger.exception(
                "parallel_run_baseline worker failed on %s: %s",
                entry.bibtex_key,
                exc,
            )
            return [_fallback_for_unhandled(entry, exc)]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_call_one, e): e for e in remaining}
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                preds = fut.result()
            except Exception as exc:
                logger.exception(
                    "parallel_run_baseline future raised on %s: %s",
                    entry.bibtex_key,
                    exc,
                )
                preds = [_fallback_for_unhandled(entry, exc)]

            new_preds.extend(preds)
            completed += 1
            if completed % progress_every == 0:
                elapsed = time.time() - started
                rate = completed / max(elapsed, 1e-9)
                eta_min = (n_remaining - completed) / max(rate, 1e-9) / 60.0
                logger.info(
                    "parallel_run_baseline(%s): [%d/%d] %.2f entries/s, ETA %.1f min",
                    name,
                    completed,
                    n_remaining,
                    rate,
                    eta_min,
                )

    logger.info(
        "parallel_run_baseline(%s): done. %d new predictions in %.1f s.",
        name,
        len(new_preds),
        time.time() - started,
    )
    # Note: a single bibtex_key may appear in both pre_existing (loaded
    # from disk) and new_preds (a worker that raced past resume-dedup
    # because the file didn't exist when we read it).  De-dup by key,
    # preferring the freshly produced record.
    by_key: dict[str, Prediction] = {p.bibtex_key: p for p in pre_existing}
    for p in new_preds:
        by_key[p.bibtex_key] = p
    return list(by_key.values())


def _load_predictions_from_dir(checkpoint_dir: Path, done_keys: set[str]) -> list[Prediction]:
    """Load Prediction records from every JSONL in ``checkpoint_dir``.

    Returns one Prediction per (bibtex_key) found; if the same key appears
    in multiple files, the last-read record wins.
    """
    import json

    by_key: dict[str, Prediction] = {}
    for jsonl in sorted(checkpoint_dir.glob("*.jsonl")):
        try:
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bk = rec.get("bibtex_key")
                if not isinstance(bk, str) or bk not in done_keys:
                    continue
                by_key[bk] = Prediction(
                    bibtex_key=bk,
                    label=rec.get("label", "UNCERTAIN"),
                    confidence=float(rec.get("confidence", 0.5)),
                    reason=rec.get("reason", ""),
                    wall_clock_seconds=float(rec.get("wall_clock_seconds", 0.0)),
                    api_calls=int(rec.get("api_calls", 0)),
                    api_sources_queried=list(rec.get("api_sources_queried", []) or []),
                )
        except OSError as exc:
            logger.warning("Could not read checkpoint %s: %s", jsonl, exc)
    if not by_key:
        # Should not happen if done_keys was non-empty, but stay defensive.
        return list(fallback_predictions([], reason="no checkpoint records found"))
    return list(by_key.values())
