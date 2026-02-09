"""ONEBench-style aggregation for sparse/incomplete evaluation results.

Handles the case where different tools evaluate different subsets of criteria,
producing reliable rankings even with ~95% missing measurements.

Key insight from ONEBench: sample-level atomic evaluation allows meaningful
comparison even when tools have very different capabilities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ToolScore:
    """Aggregated score for a single tool."""

    tool_name: str
    overall_score: float
    num_evaluated: int
    num_total: int
    coverage: float  # fraction of entries evaluated
    subtest_scores: dict[str, float] = field(default_factory=dict)
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class SparseEvaluation:
    """Sparse evaluation matrix: tools x entries x subtests.

    Not all tools evaluate all entries on all subtests. This structure
    tracks what was evaluated and enables robust aggregation.
    """

    # scores[tool_name][entry_key][subtest_name] = score (0.0-1.0) or None
    scores: dict[str, dict[str, dict[str, float | None]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    entry_keys: set[str] = field(default_factory=set)
    subtest_names: set[str] = field(default_factory=set)
    tool_names: set[str] = field(default_factory=set)

    def add_score(
        self,
        tool_name: str,
        entry_key: str,
        subtest_name: str,
        score: float | None,
    ) -> None:
        """Record a single evaluation result."""
        self.scores[tool_name][entry_key][subtest_name] = score
        self.entry_keys.add(entry_key)
        self.subtest_names.add(subtest_name)
        self.tool_names.add(tool_name)

    def get_score(
        self,
        tool_name: str,
        entry_key: str,
        subtest_name: str,
    ) -> float | None:
        """Retrieve a score, returning None if not evaluated."""
        return self.scores.get(tool_name, {}).get(entry_key, {}).get(subtest_name)

    def coverage(self, tool_name: str) -> float:
        """Fraction of (entry, subtest) pairs that this tool evaluated."""
        total = len(self.entry_keys) * len(self.subtest_names)
        if total == 0:
            return 0.0
        evaluated = sum(
            1
            for entry_key in self.entry_keys
            for subtest_name in self.subtest_names
            if self.get_score(tool_name, entry_key, subtest_name) is not None
        )
        return evaluated / total


def aggregate_scores(
    sparse_eval: SparseEvaluation,
    method: str = "mean_of_means",
) -> list[ToolScore]:
    """Aggregate sparse evaluation results into tool rankings.

    Methods:
        - "mean_of_means": For each tool, compute mean score per subtest,
          then average across subtests. Handles missing data gracefully.
        - "pairwise": Compare tools head-to-head on entries they both evaluated.
          More robust to selection bias.
        - "entry_mean": For each entry, compute mean subtest score, then
          average across entries. Simple and interpretable.
    """
    if method == "mean_of_means":
        return _aggregate_mean_of_means(sparse_eval)
    elif method == "pairwise":
        return _aggregate_pairwise(sparse_eval)
    elif method == "entry_mean":
        return _aggregate_entry_mean(sparse_eval)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _aggregate_mean_of_means(sparse_eval: SparseEvaluation) -> list[ToolScore]:
    """Mean-of-means aggregation: average across subtests, then across entries."""
    results = []

    for tool_name in sparse_eval.tool_names:
        subtest_means: dict[str, float] = {}
        total_evaluated = 0

        for subtest in sparse_eval.subtest_names:
            scores = []
            for entry_key in sparse_eval.entry_keys:
                s = sparse_eval.get_score(tool_name, entry_key, subtest)
                if s is not None:
                    scores.append(s)
                    total_evaluated += 1

            if scores:
                subtest_means[subtest] = sum(scores) / len(scores)

        if subtest_means:
            overall = sum(subtest_means.values()) / len(subtest_means)
            ci = _bootstrap_ci(list(subtest_means.values()))
        else:
            overall = 0.0
            ci = (0.0, 0.0)

        total_possible = len(sparse_eval.entry_keys) * len(sparse_eval.subtest_names)
        results.append(
            ToolScore(
                tool_name=tool_name,
                overall_score=overall,
                num_evaluated=total_evaluated,
                num_total=total_possible,
                coverage=total_evaluated / total_possible if total_possible > 0 else 0.0,
                subtest_scores=subtest_means,
                confidence_interval=ci,
            )
        )

    results.sort(key=lambda x: x.overall_score, reverse=True)
    return results


def _aggregate_pairwise(sparse_eval: SparseEvaluation) -> list[ToolScore]:
    """Pairwise aggregation: compare tools on shared entries."""
    tools = list(sparse_eval.tool_names)
    wins: dict[str, int] = defaultdict(int)
    total_comparisons: dict[str, int] = defaultdict(int)

    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            t1, t2 = tools[i], tools[j]
            t1_score, t2_score = 0.0, 0.0
            shared = 0

            for entry_key in sparse_eval.entry_keys:
                for subtest in sparse_eval.subtest_names:
                    s1 = sparse_eval.get_score(t1, entry_key, subtest)
                    s2 = sparse_eval.get_score(t2, entry_key, subtest)
                    if s1 is not None and s2 is not None:
                        t1_score += s1
                        t2_score += s2
                        shared += 1

            if shared > 0:
                if t1_score > t2_score:
                    wins[t1] += 1
                elif t2_score > t1_score:
                    wins[t2] += 1
                total_comparisons[t1] += 1
                total_comparisons[t2] += 1

    results = []
    for tool_name in tools:
        win_rate = wins[tool_name] / total_comparisons[tool_name] if total_comparisons[tool_name] > 0 else 0.0
        total_possible = len(sparse_eval.entry_keys) * len(sparse_eval.subtest_names)
        evaluated = sum(
            1
            for ek in sparse_eval.entry_keys
            for st in sparse_eval.subtest_names
            if sparse_eval.get_score(tool_name, ek, st) is not None
        )
        results.append(
            ToolScore(
                tool_name=tool_name,
                overall_score=win_rate,
                num_evaluated=evaluated,
                num_total=total_possible,
                coverage=evaluated / total_possible if total_possible > 0 else 0.0,
            )
        )

    results.sort(key=lambda x: x.overall_score, reverse=True)
    return results


def _aggregate_entry_mean(sparse_eval: SparseEvaluation) -> list[ToolScore]:
    """Entry-mean aggregation: average subtests per entry, then average entries."""
    results = []

    for tool_name in sparse_eval.tool_names:
        entry_scores = []
        total_evaluated = 0

        for entry_key in sparse_eval.entry_keys:
            scores = []
            for subtest in sparse_eval.subtest_names:
                s = sparse_eval.get_score(tool_name, entry_key, subtest)
                if s is not None:
                    scores.append(s)
                    total_evaluated += 1

            if scores:
                entry_scores.append(sum(scores) / len(scores))

        if entry_scores:
            overall = sum(entry_scores) / len(entry_scores)
            ci = _bootstrap_ci(entry_scores)
        else:
            overall = 0.0
            ci = (0.0, 0.0)

        total_possible = len(sparse_eval.entry_keys) * len(sparse_eval.subtest_names)
        results.append(
            ToolScore(
                tool_name=tool_name,
                overall_score=overall,
                num_evaluated=total_evaluated,
                num_total=total_possible,
                coverage=total_evaluated / total_possible if total_possible > 0 else 0.0,
                confidence_interval=ci,
            )
        )

    results.sort(key=lambda x: x.overall_score, reverse=True)
    return results


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        mean = values[0] if values else 0.0
        return (mean, mean)

    arr = np.array(values)
    rng = np.random.default_rng(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    alpha = (1 - confidence) / 2
    lower = float(np.quantile(bootstrap_means, alpha))
    upper = float(np.quantile(bootstrap_means, 1 - alpha))
    return (lower, upper)
