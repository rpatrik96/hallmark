"""Evaluation metrics for HALLMARK.

Primary: Detection Rate (DR), False Positive Rate (FPR), F1-Hallucination, Tier-weighted F1.
Secondary: detect@k, temporal robustness, cost efficiency.
Inspired by HumanEval (multi-criteria, pass@k) and ONEBench (sample-level atomic eval).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from hallmark.dataset.schema import BenchmarkEntry, EvaluationResult, Prediction


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix for hallucination detection."""

    tp: int = 0  # Correctly detected hallucinations
    fp: int = 0  # Valid entries incorrectly flagged as hallucinated
    tn: int = 0  # Correctly identified valid entries
    fn: int = 0  # Missed hallucinations

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Same as Detection Rate (DR)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        return self.recall

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0


def build_confusion_matrix(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> ConfusionMatrix:
    """Build confusion matrix from entries and predictions."""
    cm = ConfusionMatrix()
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        if pred is None:
            # Missing prediction treated as "VALID" (conservative)
            if entry.label == "HALLUCINATED":
                cm.fn += 1
            else:
                cm.tn += 1
            continue

        if entry.label == "HALLUCINATED":
            if pred.label == "HALLUCINATED":
                cm.tp += 1
            else:
                cm.fn += 1
        else:
            if pred.label == "HALLUCINATED":
                cm.fp += 1
            else:
                cm.tn += 1
    return cm


def tier_weighted_f1(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
    tier_weights: dict[int, float] | None = None,
) -> float:
    """Compute F1 weighted by difficulty tier (Tier 3 worth 3x Tier 1).

    Each hallucinated entry contributes to F1 proportionally to its tier weight.
    """
    if tier_weights is None:
        tier_weights = {1: 1.0, 2: 2.0, 3: 3.0}

    weighted_tp = 0.0
    weighted_fn = 0.0
    weighted_fp = 0.0
    total_weight = 0.0

    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        tier = entry.difficulty_tier or 1
        w = tier_weights.get(tier, 1.0)

        if entry.label == "HALLUCINATED":
            total_weight += w
            if pred is not None and pred.label == "HALLUCINATED":
                weighted_tp += w
            else:
                weighted_fn += w
        else:
            if pred is not None and pred.label == "HALLUCINATED":
                weighted_fp += w

    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def detect_at_k(
    entries: list[BenchmarkEntry],
    predictions_per_strategy: list[dict[str, Prediction]],
    k: int | None = None,
) -> dict[int, float]:
    """Compute detect@k: fraction of hallucinations detected using k strategies.

    Analogous to HumanEval's pass@k. A hallucination is "detected" if ANY of the
    first k strategies flags it.

    Args:
        entries: Benchmark entries.
        predictions_per_strategy: List of prediction dicts, one per verification strategy.
        k: If provided, compute only for this k. Otherwise compute for k=1..len(strategies).

    Returns:
        Dict mapping k -> fraction of hallucinations detected by at least one of k strategies.
    """
    hallucinated = [e for e in entries if e.label == "HALLUCINATED"]
    if not hallucinated:
        return {}

    n_strategies = len(predictions_per_strategy)
    ks = [k] if k is not None else list(range(1, n_strategies + 1))
    results = {}

    for ki in ks:
        if ki > n_strategies:
            break
        detected = 0
        for entry in hallucinated:
            for strategy_preds in predictions_per_strategy[:ki]:
                pred = strategy_preds.get(entry.bibtex_key)
                if pred is not None and pred.label == "HALLUCINATED":
                    detected += 1
                    break
        results[ki] = detected / len(hallucinated)

    return results


def per_tier_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[int, dict[str, float]]:
    """Compute metrics broken down by difficulty tier."""
    tier_entries: dict[int, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        tier = entry.difficulty_tier or 0
        tier_entries[tier].append(entry)

    result = {}
    for tier, tier_e in sorted(tier_entries.items()):
        cm = build_confusion_matrix(tier_e, predictions)
        result[tier] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "precision": cm.precision,
            "count": len(tier_e),
        }
    return result


def per_type_metrics(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction],
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by hallucination type."""
    type_entries: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for entry in entries:
        h_type = entry.hallucination_type or "valid"
        type_entries[h_type].append(entry)

    result = {}
    for h_type, type_e in sorted(type_entries.items()):
        cm = build_confusion_matrix(type_e, predictions)
        result[h_type] = {
            "detection_rate": cm.detection_rate,
            "false_positive_rate": cm.false_positive_rate,
            "f1": cm.f1,
            "count": len(type_e),
        }
    return result


def cost_efficiency(predictions: list[Prediction]) -> dict[str, float]:
    """Compute cost efficiency metrics."""
    if not predictions:
        return {"entries_per_second": 0.0, "mean_api_calls": 0.0}

    total_time = sum(p.wall_clock_seconds for p in predictions)
    total_api_calls = sum(p.api_calls for p in predictions)
    n = len(predictions)

    return {
        "entries_per_second": n / total_time if total_time > 0 else float("inf"),
        "mean_api_calls": total_api_calls / n,
        "total_wall_clock_seconds": total_time,
    }


def evaluate(
    entries: list[BenchmarkEntry],
    predictions: list[Prediction],
    tool_name: str = "unknown",
    split_name: str = "unknown",
    predictions_per_strategy: list[dict[str, Prediction]] | None = None,
) -> EvaluationResult:
    """Run full evaluation and return aggregated results.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool's predictions.
        tool_name: Name of the tool being evaluated.
        split_name: Name of the benchmark split.
        predictions_per_strategy: Optional list of prediction dicts for detect@k.
    """
    pred_map = {p.bibtex_key: p for p in predictions}

    cm = build_confusion_matrix(entries, pred_map)
    tw_f1 = tier_weighted_f1(entries, pred_map)
    tier_metrics = per_tier_metrics(entries, pred_map)
    type_metrics = per_type_metrics(entries, pred_map)
    cost = cost_efficiency(predictions)

    dat_k: dict[int, float] = {}
    if predictions_per_strategy:
        dat_k = detect_at_k(entries, predictions_per_strategy)

    num_hallucinated = sum(1 for e in entries if e.label == "HALLUCINATED")
    num_valid = sum(1 for e in entries if e.label == "VALID")

    return EvaluationResult(
        tool_name=tool_name,
        split_name=split_name,
        num_entries=len(entries),
        num_hallucinated=num_hallucinated,
        num_valid=num_valid,
        detection_rate=cm.detection_rate,
        false_positive_rate=cm.false_positive_rate,
        f1_hallucination=cm.f1,
        tier_weighted_f1=tw_f1,
        detect_at_k=dat_k,
        cost_efficiency=cost.get("entries_per_second"),
        mean_api_calls=cost.get("mean_api_calls"),
        per_tier_metrics=tier_metrics,
        per_type_metrics=type_metrics,
    )
