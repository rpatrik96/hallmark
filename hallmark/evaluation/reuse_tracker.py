"""Test set reuse tracking for HALLMARK benchmark splits.

Tracks how many times each split has been evaluated and estimates
the remaining statistical budget using Dwork et al. (2015) adaptive
data analysis bounds.

Key insight: non-adaptive queries degrade at O(1/sqrt(n)) but
adaptive queries degrade at O(sqrt(k*log(n)/n)), exponentially worse.

Reference: Dwork et al. (2015). The reusable holdout: Preserving validity
in adaptive data analysis.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Default split sizes for HALLMARK v1.0
DEFAULT_SPLIT_SIZES: dict[str, int] = {
    "dev_public": 1119,
    "test_public": 831,
    "hidden": 453,
    "stress_test": 122,
}


@dataclass
class SplitReuseBudget:
    """Statistical budget status for a benchmark split."""

    split_name: str
    n: int
    k: int  # evaluations so far
    unique_tools: int
    non_adaptive_bound: float  # 1 / sqrt(n)
    adaptive_bound: float  # sqrt(k * ln(n) / n)
    budget_ratio: float  # adaptive / non_adaptive
    remaining_evaluations: int  # evals before ratio exceeds threshold
    first_evaluation: str | None
    last_evaluation: str | None


def compute_reuse_budget(
    history_path: str | Path,
    split_sizes: dict[str, int] | None = None,
    max_budget_ratio: float = 2.0,
) -> dict[str, SplitReuseBudget]:
    """Compute statistical budget for each split from evaluation history.

    Reads history.jsonl (one JSON object per line with keys:
    timestamp, tool_name, split_name).

    Args:
        history_path: path to results/history.jsonl
        split_sizes: override split sizes. Defaults to HALLMARK v1.0.
        max_budget_ratio: maximum tolerable degradation factor.

    Returns:
        dict mapping split_name -> SplitReuseBudget
    """
    if split_sizes is None:
        split_sizes = dict(DEFAULT_SPLIT_SIZES)

    history_path = Path(history_path)
    entries: list[dict] = []
    if history_path.exists():
        with open(history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    # Group by split
    split_entries: dict[str, list[dict]] = {}
    for entry in entries:
        sname = entry.get("split_name", "")
        if sname not in split_entries:
            split_entries[sname] = []
        split_entries[sname].append(entry)

    results: dict[str, SplitReuseBudget] = {}
    for split_name, n in split_sizes.items():
        spl_entries = split_entries.get(split_name, [])
        k = len(spl_entries)
        unique_tools = len({e.get("tool_name", "") for e in spl_entries})

        timestamps = [e.get("timestamp", "") for e in spl_entries if e.get("timestamp")]
        first_eval = min(timestamps) if timestamps else None
        last_eval = max(timestamps) if timestamps else None

        non_adaptive = 1.0 / math.sqrt(n) if n > 0 else float("inf")
        adaptive = math.sqrt(k * math.log(n) / n) if n > 1 and k > 0 else 0.0
        ratio = adaptive / non_adaptive if non_adaptive > 0 and adaptive > 0 else 0.0

        remaining = estimate_remaining_budget(n, k, max_budget_ratio)

        results[split_name] = SplitReuseBudget(
            split_name=split_name,
            n=n,
            k=k,
            unique_tools=unique_tools,
            non_adaptive_bound=non_adaptive,
            adaptive_bound=adaptive,
            budget_ratio=ratio,
            remaining_evaluations=remaining,
            first_evaluation=first_eval,
            last_evaluation=last_eval,
        )

    return results


def estimate_remaining_budget(
    n: int,
    k_current: int,
    max_budget_ratio: float = 2.0,
) -> int:
    """How many more evaluations before budget_ratio exceeds threshold.

    Solves: sqrt(k_max * ln(n) / n) = max_budget_ratio * (1 / sqrt(n))
    => k_max = max_budget_ratio^2 / ln(n)

    Returns 0 if already exceeded or n <= 1.
    """
    if n <= 1:
        return 0
    log_n = math.log(n)
    if log_n <= 0:
        return 0
    k_max = max_budget_ratio**2 / log_n
    remaining = int(k_max) - k_current
    return max(0, remaining)


def log_evaluation(
    history_path: str | Path,
    tool_name: str,
    split_name: str,
    metrics: dict[str, float] | None = None,
) -> None:
    """Append an evaluation event to history.jsonl.

    Creates the file and parent directories if they don't exist.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, str | dict[str, float]] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "split_name": split_name,
    }
    if metrics:
        record["metrics"] = metrics

    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
