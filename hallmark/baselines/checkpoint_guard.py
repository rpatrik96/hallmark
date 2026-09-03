"""Refuse to checkpoint a run that produced no usable verdicts.

The long-running evaluation scripts (``scripts/parallel_resume_test_public.py``,
``scripts/parallel_agentic_btu_test_public.py``) append one JSONL record per
entry and resume by skipping the keys already in that file.  That makes a failed
call permanent: on 2026-09-02 a wifi outage turned 2,500 consecutive lookups into
no-evidence verdicts, and anything checkpointed during an outage is trusted by
every later run.

This module applies the same batch-level sanity check the bibtex-check wrapper
uses (``hallmark.baselines.bibtexupdater.BatchHealth``) to the records a runner
is about to persist:

- an error record never reaches the checkpoint at all — it goes to a
  ``.rejected-<timestamp>.jsonl`` sidecar, so the next run re-runs that key;
- once error records cross the batch threshold the run is refused outright with
  ``PoisonedBatchError``, before it burns more API budget.

The checkpoint is therefore left holding only usable verdicts, and a refused
batch is retried cleanly rather than resumed from.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hallmark.baselines.bibtexupdater import (
    MIN_BATCH_FOR_HEALTH_CHECK,
    NOT_FOUND_SHARE_THRESHOLD,
    BatchHealth,
)

logger = logging.getLogger(__name__)

# Reason prefixes the runners and the agentic verifier use for a record that
# carries no verdict.  ``[Salvaged]`` is deliberately absent: a salvaged record
# has a real label parsed out of truncated output.
ERROR_REASON_PREFIXES: tuple[str, ...] = (
    "[Error fallback]",
    "[Agentic error]",
)

# Substrings that identify an error record as a failed request rather than an
# answer that could not be parsed.  Matched case-insensitively against the
# reason, which embeds the exception class or message.
TRANSPORT_ERROR_HINTS: tuple[str, ...] = (
    "apierror",
    "api error",
    "apiconnection",
    "apitimeout",
    "connection",
    "timeout",
    "timed out",
    "rate limit",
    "ratelimit",
    "429",
    "network",
    "dns",
    "ssl",
    "reset by peer",
    "unreachable",
    "temporarily",
)


class PoisonedBatchError(RuntimeError):
    """Raised when a run's error share crosses the batch-health threshold."""

    def __init__(self, health: BatchHealth, message: str) -> None:
        super().__init__(message)
        self.health = health


def _reason(record: Mapping[str, Any]) -> str:
    return str(record.get("reason") or "")


def is_error_record(record: Mapping[str, Any]) -> bool:
    """True when the record carries no verdict, only a failure."""
    return _reason(record).startswith(ERROR_REASON_PREFIXES)


def is_transport_error_record(record: Mapping[str, Any]) -> bool:
    """True when the record's failure was a failed request, not a bad answer."""
    if not is_error_record(record):
        return False
    reason = _reason(record).lower()
    return any(hint in reason for hint in TRANSPORT_ERROR_HINTS)


def assess_run_health(
    records: Iterable[Mapping[str, Any]],
    *,
    threshold: float = NOT_FOUND_SHARE_THRESHOLD,
    min_batch_size: int = MIN_BATCH_FOR_HEALTH_CHECK,
) -> BatchHealth:
    """Score a runner's records with the bibtex-check batch-health thresholds.

    Error records map onto the ``BatchHealth`` counters the wrapper already uses:
    a failed request counts as ``transport_error``, and an error record that
    completed but yielded nothing usable counts as ``coverage_incomplete``.
    ``not_found`` stays zero — these runners never make that claim.
    """
    recs = list(records)
    transport = sum(1 for r in recs if is_transport_error_record(r))
    unusable = sum(1 for r in recs if is_error_record(r)) - transport
    return BatchHealth(
        total=len(recs),
        not_found=0,
        transport_error=transport,
        coverage_incomplete=unusable,
        threshold=threshold,
        min_batch_size=min_batch_size,
    )


def refusal_message(health: BatchHealth, checkpoint_path: Path) -> str:
    """Operator-facing description of why the run was refused."""
    return (
        f"Refusing to checkpoint this run: {health.no_evidence}/{health.total} "
        f"entries came back with no verdict "
        f"({health.no_evidence_share:.1%}: {health.transport_error} failed "
        f"requests, {health.coverage_incomplete} unusable answers), above the "
        f"{health.threshold:.0%} threshold. A share this high is almost always a "
        f"broken transport path — a DNS, network or proxy outage, or sustained "
        f"throttling — rather than a property of the bibliography. Nothing was "
        f"written to {checkpoint_path}; fix connectivity and re-run, and the "
        f"refused entries will be retried."
    )


def rejected_path_for(checkpoint_path: Path, timestamp: float | None = None) -> Path:
    """Sidecar path that holds the records this run refused to checkpoint."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(timestamp or time.time()))
    return checkpoint_path.with_name(f"{checkpoint_path.name}.rejected-{stamp}.jsonl")


class RunHealthTracker:
    """Thread-safe running tally of a run's error share.

    ``add`` raises ``PoisonedBatchError`` as soon as the run crosses the batch
    threshold, so a caller can cancel the remaining work instead of paying for
    calls that cannot produce evidence.  Below ``min_batch_size`` records the
    share is treated as noise and never trips (see ``BatchHealth``).
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        threshold: float = NOT_FOUND_SHARE_THRESHOLD,
        min_batch_size: int = MIN_BATCH_FOR_HEALTH_CHECK,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self._threshold = threshold
        self._min_batch_size = min_batch_size
        self._lock = threading.Lock()
        self._total = 0
        self._transport = 0
        self._unusable = 0

    @property
    def health(self) -> BatchHealth:
        with self._lock:
            return self._health_unlocked()

    def _health_unlocked(self) -> BatchHealth:
        return BatchHealth(
            total=self._total,
            not_found=0,
            transport_error=self._transport,
            coverage_incomplete=self._unusable,
            threshold=self._threshold,
            min_batch_size=self._min_batch_size,
        )

    def add(self, record: Mapping[str, Any]) -> BatchHealth:
        """Fold one record into the tally; raise once the run is poisoned."""
        with self._lock:
            self._total += 1
            if is_transport_error_record(record):
                self._transport += 1
            elif is_error_record(record):
                self._unusable += 1
            health = self._health_unlocked()
        if health.suspected_transport_failure:
            raise PoisonedBatchError(health, refusal_message(health, self.checkpoint_path))
        return health


class GuardedCheckpointWriter:
    """Append usable records to the checkpoint and park failures in a sidecar.

    Usable verdicts are written and flushed as they arrive, so a long run stays
    resumable.  Error records never enter the checkpoint, which is what makes a
    refused entry retryable: the next run does not see its key and re-runs it.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        threshold: float = NOT_FOUND_SHARE_THRESHOLD,
        min_batch_size: int = MIN_BATCH_FOR_HEALTH_CHECK,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.rejected_path = rejected_path_for(checkpoint_path)
        self.tracker = RunHealthTracker(
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            min_batch_size=min_batch_size,
        )
        self.written = 0
        self.rejected = 0
        self._lock = threading.Lock()
        self._checkpoint_file: Any = None
        self._rejected_file: Any = None

    def __enter__(self) -> GuardedCheckpointWriter:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_file = self.checkpoint_path.open("a")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def add(self, record: Mapping[str, Any]) -> None:
        """Persist one record, or park it; raise when the run turns poisoned."""
        error = is_error_record(record)
        with self._lock:
            if error:
                self._write_rejected(record)
                self.rejected += 1
            else:
                self._write_checkpoint(record)
                self.written += 1
        # Outside the write lock: the tracker has its own.
        self.tracker.add(record)

    def _write_checkpoint(self, record: Mapping[str, Any]) -> None:
        if self._checkpoint_file is None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_file = self.checkpoint_path.open("a")
        self._checkpoint_file.write(json.dumps(dict(record)) + "\n")
        self._checkpoint_file.flush()
        os.fsync(self._checkpoint_file.fileno())

    def _write_rejected(self, record: Mapping[str, Any]) -> None:
        if self._rejected_file is None:
            self.rejected_path.parent.mkdir(parents=True, exist_ok=True)
            self._rejected_file = self.rejected_path.open("a")
        self._rejected_file.write(json.dumps(dict(record)) + "\n")
        self._rejected_file.flush()

    def close(self) -> None:
        for handle in (self._checkpoint_file, self._rejected_file):
            if handle is not None:
                handle.close()
        self._checkpoint_file = None
        self._rejected_file = None
        if self.rejected:
            logger.warning(
                "%d record(s) carried no verdict and were parked in %s — their "
                "keys are absent from %s, so the next run retries them.",
                self.rejected,
                self.rejected_path,
                self.checkpoint_path,
            )


def quarantine_error_records(checkpoint_path: Path, keys: set[str]) -> Path | None:
    """Move error lines for ``keys`` out of a checkpoint into a sidecar.

    For runners whose checkpoint is written by the verifier itself, so the guard
    cannot intercept the write.  The checkpoint is rewritten atomically via a
    temporary file and ``os.replace``; every line that is not an error record for
    one of ``keys`` survives unchanged and in order.

    Returns the sidecar path, or None when there was nothing to quarantine.
    """
    if not keys or not checkpoint_path.exists():
        return None

    kept: list[str] = []
    moved: list[str] = []
    for line in checkpoint_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if record.get("bibtex_key") in keys and is_error_record(record):
            moved.append(line)
        else:
            kept.append(line)

    if not moved:
        return None

    sidecar = rejected_path_for(checkpoint_path)
    with sidecar.open("a") as f:
        f.write("\n".join(moved) + "\n")

    tmp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in kept))
    os.replace(tmp, checkpoint_path)
    logger.warning(
        "Quarantined %d error record(s) from %s into %s — those keys will be "
        "retried on the next run.",
        len(moved),
        checkpoint_path,
        sidecar,
    )
    return sidecar
